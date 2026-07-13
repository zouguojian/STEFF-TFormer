import os
import json
import math
import argparse
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

"""
AlphaTraffic Foundation - 最终版本（NPZ读取 + 详细注释 + shape说明）
=========================================================================

本版本满足以下要求：
1. A 和 X 按如下方式读取：
   A = np.load(args.adj_path, allow_pickle=True)['data'].astype(np.float64)
   X = np.load(args.x_path, allow_pickle=True)['data'].astype(np.float64)

2. 输入 X 的最后一维定义固定为：
   - X[..., 0]：交通速度或流量
   - X[..., 1]：Minute of Day，范围通常为 0~287
   - X[..., 2]：Day of Week，范围通常为 0~6

3. 输出：
   - 完整动态场 dynamic_field: [Q, N, 64]
   - 周期模板场 weekly_template_field: [2016, N, 64]
   - 四维周期模板场 weekly_template_field_4d: [7, 288, N, 64]

4. 最终把 weekly_template_field_4d 保存到 npz：
   weekly_template_field_4d.npz
   且 key 为 'data'

5. 每个函数都带有：
   - 功能说明
   - 输入 shape
   - 输出 shape

说明：
- A 和 X 在加载时保持 np.float64
- 进入模型训练前统一转换成 torch.float32，以减少显存和训练开销
"""


# ============================================================
# 1. 统一类型转换
# ============================================================

def as_torch_tensor(x: Any, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    功能：
        把输入统一转换成 torch.Tensor。

    输入：
        x:
            可以是 torch.Tensor / numpy.ndarray / list / tuple / 标量
        dtype:
            目标 torch dtype，例如 torch.float32
        device:
            目标设备，例如 'cpu' 或 'cuda'

    输出：
        t: torch.Tensor

    shape：
        输出 shape 与输入数据本身一致，不改变维度结构。
    """
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)

    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


# ============================================================
# 2. 基础工具函数
# ============================================================

def normalize_adjacency(A: Any) -> torch.Tensor:
    """
    功能：
        对邻接矩阵做对称归一化：
        A_hat = D^{-1/2}(A+I)D^{-1/2}

    输入：
        A: [N, N]
            邻接矩阵，可为 numpy.ndarray 或 torch.Tensor

    输出：
        A_hat: [N, N]
            对称归一化后的邻接矩阵，torch.Tensor
    """
    A = as_torch_tensor(A, dtype=torch.float32)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square [N, N], got shape {tuple(A.shape)}")

    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)  # [N, N]
    A = A + I

    deg = A.sum(dim=-1).clamp_min(1.0)  # [N]
    deg_inv_sqrt = deg.pow(-0.5)        # [N]

    A_hat = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]  # [N, N]
    return A_hat



def encode_time_raw(dow: Any, minute_slot: Any) -> torch.Tensor:
    """
    功能：
        把时间元信息编码成周期特征。

    输入：
        dow: [...]
            Day of Week，范围通常为 0~6
        minute_slot: [...]
            Minute of Day slot，范围通常为 0~287

    输出：
        time_feat: [..., 4]
            = [sin(dow), cos(dow), sin(slot), cos(slot)]

    shape说明：
        - 若 dow.shape = [B, T, N]
        - 则输出 shape = [B, T, N, 4]
    """
    dow = as_torch_tensor(dow, dtype=torch.float32)
    minute_slot = as_torch_tensor(minute_slot, dtype=torch.float32, device=str(dow.device))

    two_pi = 2.0 * math.pi
    dow_angle = two_pi * dow / 7.0
    slot_angle = two_pi * minute_slot / 288.0

    return torch.stack(
        [
            torch.sin(dow_angle),
            torch.cos(dow_angle),
            torch.sin(slot_angle),
            torch.cos(slot_angle),
        ],
        dim=-1,
    )



def encode_period_raw(period: Any) -> torch.Tensor:
    """
    功能：
        把 valid period 编码成模型可用的 period feature。

    输入：
        period: [B, 4]
            每一行为：
            [start_dow, start_minute_slot, end_dow, end_minute_slot]

    输出：
        period_feat: [B, 9]
            由以下部分拼接而成：
            - start time 编码: [B, 4]
            - end time 编码:   [B, 4]
            - duration:        [B, 1]
    """
    period = as_torch_tensor(period, dtype=torch.float32)

    if period.ndim != 2 or period.shape[-1] != 4:
        raise ValueError(f"period must be [B, 4], got shape {tuple(period.shape)}")

    s_dow, s_min, e_dow, e_min = period.unbind(dim=-1)

    start_feat = encode_time_raw(s_dow, s_min)  # [B, 4]
    end_feat = encode_time_raw(e_dow, e_min)    # [B, 4]

    start_idx = s_dow * 288.0 + s_min
    end_idx = e_dow * 288.0 + e_min
    delta = (end_idx - start_idx) % (7.0 * 288.0)
    delta = delta / (7.0 * 288.0)

    return torch.cat([start_feat, end_feat, delta.unsqueeze(-1)], dim=-1)  # [B, 9]



def build_support_features(x: Any, mask: Any) -> torch.Tensor:
    """
    功能：
        把 support window 构造成模型输入特征。

    输入：
        x: [B, T, N, 3]
            - x[..., 0] = traffic
            - x[..., 1] = minute_slot
            - x[..., 2] = day_of_week
        mask: [B, T, N, 1]
            观测掩码，1 表示可观测，0 表示缺失

    输出：
        feat: [B, T, N, 6]
            6 = traffic(1) + cyc_time(4) + obs_mask(1)
    """
    x = as_torch_tensor(x, dtype=torch.float32)
    mask = as_torch_tensor(mask, dtype=torch.float32, device=str(x.device))

    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"x must be [B, T, N, 3], got shape {tuple(x.shape)}")
    if mask.ndim != 4 or mask.shape[-1] != 1:
        raise ValueError(f"mask must be [B, T, N, 1], got shape {tuple(mask.shape)}")

    traffic = x[..., 0:1] * mask                # [B, T, N, 1]
    minute_slot = x[..., 1]                     # [B, T, N]
    dow = x[..., 2]                             # [B, T, N]
    time_feat = encode_time_raw(dow, minute_slot)  # [B, T, N, 4]

    return torch.cat([traffic, time_feat, mask], dim=-1)  # [B, T, N, 6]


class TrafficScaler(object):
    """
    功能：
        只对 X[..., 0]（交通值）做标准化。

    输入：
        X: [B_total, N, 3]

    输出：
        X_norm: [B_total, N, 3]
            仅 traffic 通道被标准化，时间通道保持原值
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: Any):
        X = as_torch_tensor(X, dtype=torch.float32)
        values = X[..., 0]  # [B_total, N]
        self.mean = values.mean()
        self.std = values.std().clamp_min(1e-6)
        return self

    def transform(self, X: Any) -> torch.Tensor:
        X = as_torch_tensor(X, dtype=torch.float32)
        X = X.clone()
        X[..., 0] = (X[..., 0] - self.mean) / self.std
        return X

    def inverse_value(self, y: Any) -> torch.Tensor:
        y = as_torch_tensor(y, dtype=torch.float32)
        return y * self.std + self.mean

    def state_dict(self) -> Dict[str, float]:
        return {
            "mean": float(self.mean.item()) if isinstance(self.mean, torch.Tensor) else float(self.mean),
            "std": float(self.std.item()) if isinstance(self.std, torch.Tensor) else float(self.std),
        }


# ============================================================
# 3. 文件保存函数
# ============================================================

def ensure_dir(path: str):
    """
    功能：
        若目录不存在，则创建目录。

    输入：
        path: str

    输出：
        无
    """
    os.makedirs(path, exist_ok=True)



def save_tensor_pt_and_npy(tensor: Any, out_dir: str, name: str):
    """
    功能：
        把张量同时保存成 .pt 和 .npy。

    输入：
        tensor: 任意 shape 的张量
        out_dir: 输出目录
        name: 文件名主干

    输出：
        无，直接写文件
    """
    tensor = as_torch_tensor(tensor)
    ensure_dir(out_dir)
    torch.save(tensor.cpu(), os.path.join(out_dir, f"{name}.pt"))
    np.save(os.path.join(out_dir, f"{name}.npy"), tensor.detach().cpu().numpy())



def save_npz_with_data(array: Any, path: str):
    """
    功能：
        把数组保存成 .npz，并固定使用 key='data'。

    输入：
        array: 任意 shape 的数组或张量
        path:  保存路径，例如 './outputs/weekly_template_field_4d.npz'

    输出：
        无，直接写文件
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    else:
        array = np.asarray(array)

    np.savez(path, data=array)



def save_json(data: Dict, path: str):
    """
    功能：
        把字典保存成 json。

    输入：
        data: dict
        path: 保存路径

    输出：
        无
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 4. Dataset
# ============================================================

class TrafficWindowDataset(Dataset):
    """
    功能：
        把原始时间序列 X[B_total, N, 3] 切成训练样本。

    单个样本输出：
        support: [T_s, N, 3]
        period:  [4]
        target:  [N]

    其中：
        support = X[t-support_len : t]
        valid   = X[t : t+valid_len]
        target  = mean(valid[..., 0], dim=0)
    """
    def __init__(self, X: Any, support_len: int = 288, valid_len: int = 1, stride: int = 1):
        super().__init__()
        X = as_torch_tensor(X, dtype=torch.float32)
        if X.ndim != 3 or X.shape[-1] != 3:
            raise ValueError(f"X must be [B_total, N, 3], got shape {tuple(X.shape)}")

        self.X = X
        self.support_len = support_len
        self.valid_len = valid_len
        self.indices = list(range(support_len, X.shape[0] - valid_len + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.indices[idx]

        support = self.X[t - self.support_len : t]  # [T_s, N, 3]
        valid = self.X[t : t + self.valid_len]      # [T_v, N, 3]

        period = torch.tensor(
            [
                valid[0, 0, 2],  # start_dow
                valid[0, 0, 1],  # start_minute_slot
                valid[-1, 0, 2], # end_dow
                valid[-1, 0, 1], # end_minute_slot
            ],
            dtype=self.X.dtype,
        )  # [4]

        target = valid[..., 0].mean(dim=0)  # [N]

        return {
            "support": support,
            "period": period,
            "target": target,
        }


# ============================================================
# 5. 模型模块
# ============================================================

class ObservationProjector(nn.Module):
    """
    功能：
        把输入特征从 6 维投影到隐空间 C=d_model。

    输入：
        feat: [B, T, N, 6]

    输出：
        h:    [B, T, N, C]
    """
    def __init__(self, num_nodes: int, in_dim: int = 6, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.obs_mlp = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.node_emb = nn.Embedding(num_nodes, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.obs_mlp(feat)  # [B, T, N, C]
        node_bias = self.node_emb.weight.unsqueeze(0).unsqueeze(0)  # [1, 1, N, C]
        return self.dropout(self.norm(h + node_bias))               # [B, T, N, C]


class GraphSpatialMix(nn.Module):
    """
    功能：
        图空间分支，利用邻接矩阵做节点间传播。

    输入：
        x:     [B, T, N, C]
        A_hat: [N, N]

    输出：
        y:     [B, T, N, C]
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_nei = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("nm,btmc->btnc", A_hat, x)  # [B, T, N, C]
        y = self.lin_self(x) + self.lin_nei(agg)        # [B, T, N, C]
        return self.out(y)                              # [B, T, N, C]


class TimeAxialAttention(nn.Module):
    """
    功能：
        时间轴注意力分支，对每个节点沿时间轴做 self-attention。

    输入：
        x: [B, T, N, C]

    输出：
        y: [B, T, N, C]
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        y = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N, T, C]
        attn_in = self.norm1(y)
        y2, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        y = y + y2
        y = y + self.ffn(self.norm2(y))
        return y.reshape(B, N, T, C).permute(0, 2, 1, 3)  # [B, T, N, C]


class PrecisionOperator(nn.Module):
    """
    功能：
        局部精细分支，用卷积提取短时间局部模式。

    输入：
        x: [B, T, N, C]

    输出：
        y: [B, T, N, C]
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 1), padding=(1, 0), groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 3, 1, 2)  # [B, C, T, N]
        y = self.net(y)            # [B, C, T, N]
        return y.permute(0, 2, 3, 1)  # [B, T, N, C]


class GTPBlock(nn.Module):
    """
    功能：
        Graph + Time + Precision 三路特征融合模块。

    输入：
        x:     [B, T, N, C]
        A_hat: [N, N]

    输出：
        x_new: [B, T, N, C]
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_g = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        self.graph = GraphSpatialMix(d_model, dropout)
        self.time = TimeAxialAttention(d_model, nhead, dropout)
        self.precision = PrecisionOperator(d_model, dropout)
        self.fuse = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        hg = self.graph(self.norm_g(x), A_hat)  # [B, T, N, C]
        ht = self.time(self.norm_t(x))          # [B, T, N, C]
        hp = self.precision(self.norm_p(x))     # [B, T, N, C]

        cat = torch.cat([hg, ht, hp], dim=-1)  # [B, T, N, 3C]
        w = self.fuse(cat).softmax(dim=-1)     # [B, T, N, 3]
        y = w[..., 0:1] * hg + w[..., 1:2] * ht + w[..., 2:3] * hp  # [B, T, N, C]

        x = x + y
        x = x + self.ffn(self.out_norm(x))
        return x  # [B, T, N, C]


class PeriodConditionedSummarizer(nn.Module):
    """
    功能：
        用 valid period 的时间条件去汇聚 support 时序特征。

    输入：
        x:           [B, T, N, C]
        period_feat: [B, 9]

    输出：
        summary:     [B, N, C]
    """
    def __init__(self, d_model: int, period_dim: int = 9, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.period_mlp = nn.Sequential(nn.Linear(period_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.post = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, period_feat: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        q = self.period_mlp(period_feat)[:, None, None, :].expand(B, N, 1, C).reshape(B * N, 1, C)  # [B*N, 1, C]
        kv = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # [B*N, T, C]
        summary, _ = self.attn(q, kv, kv, need_weights=False)
        summary = summary.squeeze(1).reshape(B, N, C)  # [B, N, C]
        return self.post(self.norm(summary))


class SphericalBottleneck(nn.Module):
    """
    功能：
        把 summary 映射到单位球 embedding。

    输入：
        x: [B, N, C]

    输出：
        mu: [B, N, E]
    """
    def __init__(self, d_model: int, emb_dim: int = 64):
        super().__init__()
        self.mu = nn.Linear(d_model, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mu(x), p=2, dim=-1)


class PeriodConditionedTrafficDecoder(nn.Module):
    """
    功能：
        根据 embedding 和 valid period，解码出该窗口的 traffic 统计值。

    输入：
        mu:          [B, N, E]
        period_feat: [B, 9]

    输出：
        pred:        [B, N]
    """
    def __init__(self, emb_dim: int = 64, period_dim: int = 9, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.period_mlp = nn.Sequential(nn.Linear(period_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(emb_dim + hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, mu: torch.Tensor, period_feat: torch.Tensor) -> torch.Tensor:
        B, N, _ = mu.shape
        p = self.period_mlp(period_feat).unsqueeze(1).expand(B, N, -1)  # [B, N, hidden]
        return self.net(torch.cat([mu, p], dim=-1)).squeeze(-1)         # [B, N]


class AlphaTrafficFoundation(nn.Module):
    """
    功能：
        AlphaTraffic Foundation 主模型。

    输入：
        support:      [B, T, N, 3]
        support_mask: [B, T, N, 1]
        period_raw:   [B, 4]

    输出：
        out['mu']:          [B, N, E]
        out['period_feat']: [B, 9]
        out['pred']:        [B, N]
    """
    def __init__(self, A: Any, num_nodes: int, d_model: int = 128, emb_dim: int = 64,
                 depth: int = 4, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A_hat", normalize_adjacency(A))
        self.projector = ObservationProjector(num_nodes=num_nodes, in_dim=6, d_model=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([GTPBlock(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(depth)])
        self.summarizer = PeriodConditionedSummarizer(d_model=d_model, period_dim=9, nhead=nhead, dropout=dropout)
        self.bottleneck = SphericalBottleneck(d_model=d_model, emb_dim=emb_dim)
        self.decoder = PeriodConditionedTrafficDecoder(emb_dim=emb_dim, period_dim=9, hidden=d_model, dropout=dropout)

    def encode(self, support: Any, support_mask: Any, period_raw: Any) -> Dict[str, torch.Tensor]:
        """
        功能：
            只做编码，不做最终 traffic 解码。

        输入：
            support:      [B, T, N, 3]
            support_mask: [B, T, N, 1]
            period_raw:   [B, 4]

        输出：
            out['mu']:          [B, N, E]
            out['period_feat']: [B, 9]
        """
        support = as_torch_tensor(support, dtype=torch.float32)
        support_mask = as_torch_tensor(support_mask, dtype=torch.float32, device=str(support.device))
        period_raw = as_torch_tensor(period_raw, dtype=torch.float32, device=str(support.device))

        feat = build_support_features(support, support_mask)  # [B, T, N, 6]
        period_feat = encode_period_raw(period_raw).to(support.device)  # [B, 9]
        h = self.projector(feat)  # [B, T, N, C]
        for blk in self.blocks:
            h = blk(h, self.A_hat)  # [B, T, N, C]
        summary = self.summarizer(h, period_feat)  # [B, N, C]
        mu = self.bottleneck(summary)              # [B, N, E]
        return {"mu": mu, "period_feat": period_feat}

    def forward(self, support: Any, support_mask: Any, period_raw: Any) -> Dict[str, torch.Tensor]:
        out = self.encode(support, support_mask, period_raw)
        out["pred"] = self.decoder(out["mu"], out["period_feat"])  # [B, N]
        return out


# ============================================================
# 6. 损失函数
# ============================================================

def make_student_mask(support: Any, drop_prob: float = 0.25, block_drop_prob: float = 0.5) -> torch.Tensor:
    """
    功能：
        构造 student 分支的观测缺失 mask。

    输入：
        support: [B, T, N, 3]

    输出：
        mask:    [B, T, N, 1]
    """
    support = as_torch_tensor(support, dtype=torch.float32)
    B, T, N, _ = support.shape
    device = support.device
    dtype = support.dtype

    mask = torch.ones(B, T, N, 1, device=device, dtype=dtype)
    time_keep = (torch.rand(B, T, 1, 1, device=device) > drop_prob).to(dtype)
    mask = mask * time_keep

    for b in range(B):
        if torch.rand(1, device=device).item() < block_drop_prob:
            span = max(1, T // 4)
            start = int(torch.randint(0, T - span + 1, (1,), device=device).item())
            mask[b, start : start + span] = 0.0

    return mask



def batch_uniformity_loss(mu: Any) -> torch.Tensor:
    """
    功能：
        约束 embedding 在单位球面上的分布更均匀，避免塌缩。

    输入：
        mu: [B, N, E]

    输出：
        scalar loss
    """
    mu = as_torch_tensor(mu, dtype=torch.float32)
    u = F.normalize(mu.reshape(-1, mu.shape[-1]), dim=-1)  # [B*N, E]
    if u.shape[0] < 2:
        return u.new_tensor(0.0)
    u_roll = torch.roll(u, shifts=1, dims=0)
    return (u * u_roll).sum(dim=-1).abs().mean()



def consistency_loss(mu_teacher: Any, mu_student: Any) -> torch.Tensor:
    """
    功能：
        teacher / student 一致性损失。

    输入：
        mu_teacher: [B, N, E]
        mu_student: [B, N, E]

    输出：
        scalar loss
    """
    mu_teacher = as_torch_tensor(mu_teacher, dtype=torch.float32)
    mu_student = as_torch_tensor(mu_student, dtype=torch.float32, device=str(mu_teacher.device))
    cos = (F.normalize(mu_teacher, dim=-1) * F.normalize(mu_student, dim=-1)).sum(dim=-1)
    return ((1.0 - cos) / 2.0).mean()


# ============================================================
# 7. 训练与推理
# ============================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str,
                    lambda_cons: float = 0.02, lambda_uni: float = 0.005) -> Dict[str, float]:
    """
    功能：
        执行一轮训练。

    输入：
        loader 中每个 batch 包含：
            support: [B, T, N, 3]
            period:  [B, 4]
            target:  [B, N]

    输出：
        metrics: dict
            包括 loss / rec / cons / uni
    """
    model.train()
    meters = {"loss": 0.0, "rec": 0.0, "cons": 0.0, "uni": 0.0}
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"训练进度：第 {batch_idx + 1}/{len(loader)} 个 batch")
        support = as_torch_tensor(batch["support"], dtype=torch.float32, device=device)
        period = as_torch_tensor(batch["period"], dtype=torch.float32, device=device)
        target = as_torch_tensor(batch["target"], dtype=torch.float32, device=device)

        teacher_mask = torch.ones_like(support[..., :1])
        student_mask = make_student_mask(support)

        out_teacher = model(support, teacher_mask, period)
        out_student = model(support, student_mask, period)

        rec = F.smooth_l1_loss(out_teacher["pred"], target)
        cons = consistency_loss(out_teacher["mu"], out_student["mu"])
        uni = batch_uniformity_loss(out_teacher["mu"])
        loss = rec + lambda_cons * cons + lambda_uni * uni

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters["loss"] += float(loss.item())
        meters["rec"] += float(rec.item())
        meters["cons"] += float(cons.item())
        meters["uni"] += float(uni.item())
        num_batches += 1

    for k in meters:
        meters[k] /= max(1, num_batches)
    return meters


@torch.no_grad()
def build_embedding_field(model: AlphaTrafficFoundation, X: Any, support_len: int = 288, valid_len: int = 1,
                          stride: int = 1, batch_size: int = 16, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：
        构建完整动态场。

    输入：
        X: [B_total, N, 3]

    输出：
        embedding_field: [Q, N, E]
        period_index:    [Q, 4]
    """
    model.eval()
    X = as_torch_tensor(X, dtype=torch.float32)
    ds = TrafficWindowDataset(X=X, support_len=support_len, valid_len=valid_len, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embeddings = []
    periods = []
    for batch in loader:
        support = as_torch_tensor(batch["support"], dtype=torch.float32, device=device)
        period = as_torch_tensor(batch["period"], dtype=torch.float32, device=device)
        mask = torch.ones_like(support[..., :1])
        out = model.encode(support, mask, period)
        embeddings.append(out["mu"].cpu())
        periods.append(period.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(periods, dim=0)


@torch.no_grad()
def build_network_level_embedding(embedding_field: Any, mode: str = "mean") -> torch.Tensor:
    """
    功能：
        从节点级 embedding 聚合成网络级 embedding。

    输入：
        embedding_field: [Q, N, E] 或 [S, N, E] 或 [7, 288, N, E]

    输出：
        若输入为 [Q, N, E] 或 [S, N, E]：
            [Q, E] 或 [S, E]
        若输入为 [7, 288, N, E]：
            [7, 288, E]
    """
    embedding_field = as_torch_tensor(embedding_field, dtype=torch.float32)

    if embedding_field.ndim == 3:
        if mode == "mean":
            return embedding_field.mean(dim=1)
        if mode == "max":
            return embedding_field.max(dim=1).values
    elif embedding_field.ndim == 4:
        if mode == "mean":
            return embedding_field.mean(dim=2)
        if mode == "max":
            return embedding_field.max(dim=2).values

    raise ValueError("Unsupported shape for build_network_level_embedding")


@torch.no_grad()
def build_full_dynamic_field(model: AlphaTrafficFoundation, X: Any, support_len: int = 288, valid_len: int = 1,
                             stride: int = 1, batch_size: int = 16, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：
        显式命名的完整动态场构建函数。

    输入：
        X: [B_total, N, 3]

    输出：
        dynamic_field: [Q, N, E]
        period_index:  [Q, 4]
    """
    return build_embedding_field(model=model, X=X, support_len=support_len, valid_len=valid_len,
                                 stride=stride, batch_size=batch_size, device=device)



def period_index_to_week_slot(period_index: Any, slots_per_day: int = 288) -> torch.Tensor:
    """
    功能：
        把每个 dynamic_field 对应的 period_index 映射到一周中的固定时间片编号。

    输入：
        period_index: [Q, 4]
            [start_dow, start_minute_slot, end_dow, end_minute_slot]
        slots_per_day: int
            一天有多少个时间片，5分钟粒度时为 288

    输出：
        week_slot: [Q]
            范围为 0 ~ (7 * slots_per_day - 1)
    """
    period_index = as_torch_tensor(period_index, dtype=torch.float32)
    if period_index.ndim != 2 or period_index.shape[-1] != 4:
        raise ValueError(f"period_index must be [Q, 4], got shape {tuple(period_index.shape)}")

    start_dow = period_index[:, 0].long()
    start_slot = period_index[:, 1].long()

    if ((start_slot < 0) | (start_slot >= slots_per_day)).any():
        raise ValueError(f"start_slot should be in [0, {slots_per_day - 1}]")

    return start_dow * slots_per_day + start_slot  # [Q]


@torch.no_grad()
def build_weekly_template_field(dynamic_field: Any, period_index: Any, slots_per_day: int = 288) -> Dict[str, torch.Tensor]:
    """
    功能：
        把完整动态场聚合成周期模板场。

    输入：
        dynamic_field: [Q, N, E]
        period_index:  [Q, 4]
        slots_per_day: int，默认 288

    输出：
        weekly_template_field:      [S, N, E]
        weekly_template_field_4d:   [7, 288, N, E]
        weekly_template_std:        [S, N, E]
        weekly_template_std_4d:     [7, 288, N, E]
        weekly_slot_count:          [S]
        weekly_slot_count_2d:       [7, 288]
        weekly_slot_mask:           [S]
        weekly_slot_mask_2d:        [7, 288]
        week_slot_index:            [Q]

    其中：
        S = 7 * slots_per_day
        当 slots_per_day=288 时：
        S = 2016
    """
    dynamic_field = as_torch_tensor(dynamic_field, dtype=torch.float32)
    period_index = as_torch_tensor(period_index, dtype=torch.float32)

    if dynamic_field.ndim != 3:
        raise ValueError(f"dynamic_field must be [Q, N, E], got shape {tuple(dynamic_field.shape)}")

    Q, N, E = dynamic_field.shape
    week_slot_index = period_index_to_week_slot(period_index, slots_per_day=slots_per_day)  # [Q]

    S = 7 * slots_per_day
    device = dynamic_field.device
    dtype = dynamic_field.dtype

    template_sum = torch.zeros(S, N, E, device=device, dtype=dtype)
    template_sq_sum = torch.zeros(S, N, E, device=device, dtype=dtype)
    template_count = torch.zeros(S, device=device, dtype=dtype)

    for q in range(Q):
        s = int(week_slot_index[q].item())
        template_sum[s] += dynamic_field[q]
        template_sq_sum[s] += dynamic_field[q] * dynamic_field[q]
        template_count[s] += 1.0

    denom = template_count.clamp_min(1.0).view(S, 1, 1)
    weekly_template_field = template_sum / denom                       # [S, N, E]
    weekly_template_sq_mean = template_sq_sum / denom                  # [S, N, E]
    weekly_template_var = (weekly_template_sq_mean - weekly_template_field * weekly_template_field).clamp_min(0.0)
    weekly_template_std = torch.sqrt(weekly_template_var)              # [S, N, E]
    weekly_slot_mask = template_count > 0                              # [S]

    weekly_template_field_4d = weekly_template_field.reshape(7, slots_per_day, N, E)  # [7, 288, N, E]
    weekly_template_std_4d = weekly_template_std.reshape(7, slots_per_day, N, E)       # [7, 288, N, E]
    weekly_slot_count_2d = template_count.reshape(7, slots_per_day)                     # [7, 288]
    weekly_slot_mask_2d = weekly_slot_mask.reshape(7, slots_per_day)                    # [7, 288]

    return {
        "weekly_template_field": weekly_template_field,
        "weekly_template_field_4d": weekly_template_field_4d,
        "weekly_template_std": weekly_template_std,
        "weekly_template_std_4d": weekly_template_std_4d,
        "weekly_slot_count": template_count,
        "weekly_slot_count_2d": weekly_slot_count_2d,
        "weekly_slot_mask": weekly_slot_mask,
        "weekly_slot_mask_2d": weekly_slot_mask_2d,
        "week_slot_index": week_slot_index,
    }


@torch.no_grad()
def build_dynamic_and_weekly_fields(model: AlphaTrafficFoundation, X: Any, support_len: int = 288, valid_len: int = 1,
                                    stride: int = 1, batch_size: int = 16, device: str = "cpu",
                                    slots_per_day: int = 288) -> Dict[str, torch.Tensor]:
    """
    功能：
        同时输出完整动态场和周期模板场。

    输入：
        X: [B_total, N, 3]

    输出：
        dynamic_field:             [Q, N, E]
        period_index:              [Q, 4]
        dynamic_network_field:     [Q, E]
        weekly_template_field:     [S, N, E]
        weekly_template_field_4d:  [7, 288, N, E]
        weekly_template_std:       [S, N, E]
        weekly_template_std_4d:    [7, 288, N, E]
        weekly_slot_count:         [S]
        weekly_slot_count_2d:      [7, 288]
        weekly_slot_mask:          [S]
        weekly_slot_mask_2d:       [7, 288]
        week_slot_index:           [Q]
        weekly_network_field:      [S, E]
        weekly_network_field_4d:   [7, 288, E]
    """
    dynamic_field, period_index = build_full_dynamic_field(
        model=model,
        X=X,
        support_len=support_len,
        valid_len=valid_len,
        stride=stride,
        batch_size=batch_size,
        device=device,
    )

    weekly_outputs = build_weekly_template_field(
        dynamic_field=dynamic_field,
        period_index=period_index,
        slots_per_day=slots_per_day,
    )

    dynamic_network_field = build_network_level_embedding(dynamic_field, mode="mean")
    weekly_network_field = build_network_level_embedding(weekly_outputs["weekly_template_field"], mode="mean")
    weekly_network_field_4d = build_network_level_embedding(weekly_outputs["weekly_template_field_4d"], mode="mean")

    outputs = {
        "dynamic_field": dynamic_field,
        "period_index": period_index,
        "dynamic_network_field": dynamic_network_field,
        "weekly_network_field": weekly_network_field,
        "weekly_network_field_4d": weekly_network_field_4d,
    }
    outputs.update(weekly_outputs)
    return outputs


# ============================================================
# 8. 主函数
# ============================================================

def parse_args():
    """
    功能：
        解析命令行参数。

    输出：
        args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="AlphaTraffic Foundation - 最终版本")
    parser.add_argument("--adj_path", type=str, required=True)
    parser.add_argument("--x_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--support_len", type=int, default=288)
    parser.add_argument("--valid_len", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--slots_per_day", type=int, default=288)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_cons", type=float, default=0.02)
    parser.add_argument("--lambda_uni", type=float, default=0.005)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()



def validate_and_convert_inputs(A: Any, X: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    功能：
        校验输入数据并统一转换成 torch.Tensor。

    输入：
        A: [N, N]
        X: [B_total, N, 3]

    输出：
        A_torch: [N, N]
        X_torch: [B_total, N, 3]
    """
    A = as_torch_tensor(A, dtype=torch.float32)
    X = as_torch_tensor(X, dtype=torch.float32)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square [N, N], got shape {tuple(A.shape)}")
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must be [B_total, N, 3], got shape {tuple(X.shape)}")
    if A.shape[0] != X.shape[1]:
        raise ValueError(f"A and X node dimension mismatch: A {tuple(A.shape)}, X {tuple(X.shape)}")

    return A, X



def main():
    """
    功能：
        完整训练与输出流程入口。

    主流程：
        1. 读取 A / X（按指定 npz['data'] 方式）
        2. 校验输入 shape
        3. 标准化 X[..., 0]
        4. 训练模型
        5. 输出 dynamic_field / weekly_template_field / weekly_template_field_4d
        6. 保存结果，其中 weekly_template_field_4d 保存到 npz
    """
    args = parse_args()
    ensure_dir(args.out_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Loading data...")
    print("=" * 80)

    # --------------------------------------------------------
    # 用户指定的最终读取方式
    # --------------------------------------------------------
    A = np.load(args.adj_path, allow_pickle=True)['data'].astype(np.float64)  # [N, N]
    X = np.load(args.x_path, allow_pickle=True)['data'].astype(np.float64)    # [B_total, N, 3]

    A, X = validate_and_convert_inputs(A, X)

    B_total, N, D = X.shape
    print(f"A shape: {tuple(A.shape)}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"B_total={B_total}, N={N}, D={D}, device={device}")

    print("=" * 80)
    print("Preprocessing...")
    print("=" * 80)

    scaler = TrafficScaler().fit(X)
    X_norm = scaler.transform(X)  # [B_total, N, 3]

    train_ds = TrafficWindowDataset(
        X=X_norm,
        support_len=args.support_len,
        valid_len=args.valid_len,
        stride=args.stride,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    print(f"Number of training windows: {len(train_ds)}")

    print("=" * 80)
    print("Building model...")
    print("=" * 80)

    model = AlphaTrafficFoundation(
        A=A,
        num_nodes=N,
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        depth=args.depth,
        nhead=args.nhead,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Training...")
    print("=" * 80)

    history = []
    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, train_loader, optimizer, device, args.lambda_cons, args.lambda_uni)
        history.append({"epoch": epoch + 1, **metrics})
        print(f"Epoch {epoch + 1:03d} | {metrics}")

    print("=" * 80)
    print("Building dynamic field + weekly template field...")
    print("=" * 80)

    outputs = build_dynamic_and_weekly_fields(
        model=model,
        X=X_norm,
        support_len=args.support_len,
        valid_len=args.valid_len,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
        slots_per_day=args.slots_per_day,
    )

    print(f"dynamic_field shape:            {tuple(outputs['dynamic_field'].shape)}")
    print(f"period_index shape:             {tuple(outputs['period_index'].shape)}")
    print(f"weekly_template_field shape:    {tuple(outputs['weekly_template_field'].shape)}")
    print(f"weekly_template_field_4d shape: {tuple(outputs['weekly_template_field_4d'].shape)}")
    print(f"weekly_template_std shape:      {tuple(outputs['weekly_template_std'].shape)}")
    print(f"weekly_template_std_4d shape:   {tuple(outputs['weekly_template_std_4d'].shape)}")
    print(f"weekly_slot_count shape:        {tuple(outputs['weekly_slot_count'].shape)}")
    print(f"dynamic_network_field shape:    {tuple(outputs['dynamic_network_field'].shape)}")
    print(f"weekly_network_field shape:     {tuple(outputs['weekly_network_field'].shape)}")
    print(f"weekly_network_field_4d shape:  {tuple(outputs['weekly_network_field_4d'].shape)}")

    print("=" * 80)
    print("Saving outputs...")
    print("=" * 80)

    # torch.save(model.state_dict(), os.path.join(args.out_dir, "alphatraffic_model_state_dict.pt"))
    # save_json(scaler.state_dict(), os.path.join(args.out_dir, "scaler.json"))
    # save_json(vars(args), os.path.join(args.out_dir, "config.json"))
    # save_json({"history": history}, os.path.join(args.out_dir, "train_history.json"))

    # save_tensor_pt_and_npy(X_norm, args.out_dir, "X_normalized")
    # save_tensor_pt_and_npy(outputs["dynamic_field"], args.out_dir, "dynamic_field")
    # save_tensor_pt_and_npy(outputs["period_index"], args.out_dir, "period_index")
    # save_tensor_pt_and_npy(outputs["dynamic_network_field"], args.out_dir, "dynamic_network_field")

    # save_tensor_pt_and_npy(outputs["weekly_template_field"], args.out_dir, "weekly_template_field")
    # save_tensor_pt_and_npy(outputs["weekly_template_field_4d"], args.out_dir, "weekly_template_field_4d")
    # save_tensor_pt_and_npy(outputs["weekly_template_std"], args.out_dir, "weekly_template_std")
    # save_tensor_pt_and_npy(outputs["weekly_template_std_4d"], args.out_dir, "weekly_template_std_4d")
    # save_tensor_pt_and_npy(outputs["weekly_slot_count"], args.out_dir, "weekly_slot_count")
    # save_tensor_pt_and_npy(outputs["weekly_network_field"], args.out_dir, "weekly_network_field")
    # save_tensor_pt_and_npy(outputs["weekly_network_field_4d"], args.out_dir, "weekly_network_field_4d")
    # save_tensor_pt_and_npy(outputs["week_slot_index"], args.out_dir, "week_slot_index")

    # --------------------------------------------------------
    # 按要求额外保存 weekly_template_field_4d 到 npz，并固定 key='data'
    # --------------------------------------------------------
    save_npz_with_data(
        outputs["weekly_template_field_4d"],
        os.path.join(args.out_dir, "weekly_template_field_4d.npz"),
    )

    print(f"All outputs have been saved to: {args.out_dir}")


if __name__ == "__main__":
    main()