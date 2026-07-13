import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x = x + inputs
            x = self.ln(x)
        return x


class TemEmbedding(nn.Module):
    def __init__(self, D, cfg):
        super(TemEmbedding, self).__init__()
        self.cfg = cfg
        self.ff_te = FeedForward([self.cfg.slots_per_day + 7, D, D])
        self.ff_se = FeedForward([self.cfg.nodes, D, D])

    def forward(self, TE):
        '''
        TE: [B,T,2]
        return: [B,T,N,D]
        '''
        T = self.cfg.slots_per_day
        pos = torch.arange(self.cfg.nodes).unsqueeze(0).to(TE.device)
        SE = F.one_hot(pos.to(torch.int64), num_classes=self.cfg.nodes).unsqueeze(0).float()  # [1,N,N]
        SE = self.ff_se(SE)  # [1,1,N,D]

        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[:, :, 0, 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[:, :, 0, 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B,T,295]
        TE = TE.unsqueeze(dim=2)  # [B,T,1,295]
        TE = self.ff_te(TE)  # [B,T,1,F]

        return TE + SE  # [B,T,N,F]


class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=3, dropout=0.1, levels=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh.transpose(1, 3)).transpose(1, 3)
        return xh


class TemporalAttention(nn.Module):
    def __init__(self, heads, dims):
        super(TemporalAttention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True):
        '''
        x: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        if te is not None:
            x = x + te

        query = self.qfc(x)  # [B,T,N,F]
        key = self.kfc(x)  # [B,T,N,F]
        value = self.vfc(x)  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,T,N,d]
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attention = torch.matmul(query, key)  # [k*B,N,T,T]
        attention /= (self.d ** 0.5)  # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attention).to(x.device)  # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1)  # [k*B,N,T,T]

        value = torch.matmul(attention, value)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value = value + x

        value = self.ln(value)

        return self.ff(value)


class ThreeSTBLN(nn.Module):
    """
    输入:
        X: [B, P, N, 3]
           X[..., 0] = 历史交通速度/流量
           X[..., 1] = minute_of_day
           X[..., 2] = day_of_week
        E: [7, slots_per_day, N, d_model]
           外部提供的时空嵌入场

    输出:
        {
          'recon_past': [B, P, N, 1],
          'pred_future': [B, Q, N, 1],
          'full':       [B, P+Q, N, 1],
        }

    说明:
        1) 保持 3S-TBLN 的主体结构不变：Semantic Transformer + ST-Blocks +
           Bridge Attention + 双向重构/预测。
        2) 与原工程不同的是，forward 接口改为只接收 X 和 E。
        3) 原先由 last_week_speed + STEmbedding 提供的模式/时空条件，
           现在改为从外部时空嵌入场 E 中按时间索引检索得到。
        4) 具体卷积 / 线性映射参考 GMAN：统一改为 GMAN 风格的 Conv2d / FC。
    """

    def __init__(self, cfg, mean=0.0, std=1.0):
        super().__init__()
        self.cfg = cfg
        self.mean = mean
        self.std = std

        self.STEmbedding = TemEmbedding(cfg.d_model, self.cfg)
        self.STBlock_1 = TemporalAttention(cfg.num_heads, cfg.d_model // cfg.num_heads)
        self.STBlock_2 = TemporalAttention(cfg.num_heads, cfg.d_model // cfg.num_heads)
        # self.STBlock_1 = nn.Sequential(*[TemporalConvNet(cfg.d_model, kernel_size=3, dropout=0., levels=1) for _ in range(cfg.num_blocks)])
        # self.STBlock_2 = nn.Sequential(*[TemporalConvNet(cfg.d_model, kernel_size=3, dropout=0., levels=1) for _ in range(2)])
        self.FC_1 = FeedForward([1, cfg.d_model, cfg.d_model])
        self.FC_2 = FeedForward([cfg.d_model, cfg.d_model, 1])

    def _gather_from_field(self, E, minute_of_day, day_of_week):
        """
        根据 day_of_week / minute_of_day 从时空嵌入场 E 中检索对应先验。

        minute_of_day: [B, T, N]
        day_of_week:   [B, T, N]
        返回:          [B, T, N, D]
        """
        b, t, n = minute_of_day.shape
        minute_idx = minute_of_day.long().clamp(0, self.cfg.slots_per_day - 1)
        day_idx = day_of_week.long().clamp(0, 7 - 1)
        node_idx = torch.arange(n, device=E.device).view(1, 1, n).expand(b, t, n)
        return E[day_idx, minute_idx, node_idx]

    def _infer_future_calendar_from_last_step(self, X):
        """
        根据历史最后一个时刻，推理未来 Q 步的 minute_of_day 和 day_of_week。
        """
        b, _, n, _ = X.shape
        q = self.cfg.output_len
        slots_per_day = self.cfg.slots_per_day
        days_per_week = 7

        last_minute_all = X[:, -1, :, 1].long()
        last_day_all = X[:, -1, :, 2].long()

        base_minute = last_minute_all[:, :1]
        base_day = last_day_all[:, :1]

        if not torch.all(last_minute_all == base_minute):
            raise ValueError("For each sample, minute_of_day at the last history step must be consistent across nodes")
        if not torch.all(last_day_all == base_day):
            raise ValueError("For each sample, day_of_week at the last history step must be consistent across nodes")

        last_minute = base_minute[:, 0]
        last_day = base_day[:, 0]

        offsets = torch.arange(1, q + 1, device=X.device).view(1, q)
        total_index = last_day[:, None] * slots_per_day + last_minute[:, None] + offsets

        future_day = (total_index // slots_per_day) % days_per_week
        future_minute = total_index % slots_per_day

        future_day = future_day.unsqueeze(-1).expand(b, q, n)
        future_minute = future_minute.unsqueeze(-1).expand(b, q, n)
        return future_minute, future_day

    def forward(self, X, E):

        p = self.cfg.input_len
        q = self.cfg.output_len

        if X.size(1) != p:
            raise ValueError('The temporal length of X must equal cfg.input_len')

        hist_speed = X[..., 0:1].float()
        hist_minute = X[..., 1].long()
        hist_day = X[..., 2].long()

        start_time = datetime.datetime.now()
        ste_p = self._gather_from_field(E, hist_minute, hist_day)
        future_minute, future_day = self._infer_future_calendar_from_last_step(X)
        ste_q = self._gather_from_field(E, future_minute, future_day)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time

        # input
        x = self.FC_1(hist_speed)
        # STE
        ME, DE = torch.concatenate([hist_minute, future_minute], dim=1), torch.concatenate([hist_day, future_day],
                                                                                           dim=1)
        STE = self.STEmbedding(torch.cat((DE.unsqueeze(dim=-1), ME.unsqueeze(dim=-1)), dim=-1))
        STE_his = STE[:, :p]
        STE_pred = STE[:, p:]

        # encoder
        enc_out = self.STBlock_1(x, ste_p + STE_his, Mask=False)  # [B, P, N, C_enc]
        B, P, N, D = enc_out.shape
        enc_out = enc_out[:, -1:]  # [B, 1, N, C_enc]
        enc_out = enc_out.expand(B, P, N, D)  # [B, Q, N, C_enc]
        # decoder
        dec_out = self.STBlock_2(torch.concat([enc_out, ste_q + STE_pred], dim=1), None, Mask=True)  # [B, P, N, C_enc]
        # output future
        pre_future = self.FC_2(dec_out[:, p:]) * self.std + self.mean

        return {
            'pred_future': pre_future,
            'infer_time': total_time
        }