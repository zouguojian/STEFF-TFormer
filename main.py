import copy
import random
from model.model import STEFFTFormer
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from utils_.until import load_dataset, masked_mae, masked_mse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

parser.add_argument("--input_len", type=int, default=48)
parser.add_argument("--output_len", type=int, default=48)
parser.add_argument("--slots_per_day", type=int, default=288)
parser.add_argument("--minute_of_day_size", type=int, default=288)
parser.add_argument("--day_of_week_size", type=int, default=7)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--spatial_top_k", type=int, default=32)
parser.add_argument("--eta", type=float, default=0.3)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--loss_mode", type=str, default="paper", choices=["paper", "repo"])
parser.add_argument("--temporal_causal_mask", action="store_true")
parser.add_argument("--semantic_kernel_sizes", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--semantic_use_sum", type=bool, default=True)

parser.add_argument('--cudas', type=int, default=0, help='cuda id')
parser.add_argument('--nodes', type=int, default=325, help='number of nodes 170, 207, 108, or 325')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--emb_size', type=int, default=32, help='size of embedding')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--patience', type=int, default=5, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--datafile', default='./data/', help='traffic file')
parser.add_argument('--dataset', default='PEMS-BAY', help='traffic file')
parser.add_argument('--model_file', default='best_prior_pems08_model_0.pth', help='save the model to disk')
parser.add_argument('--log_file', default='./data/log', help='log file')
parser.add_argument("--test", action="store_true", help="test program")
args = parser.parse_args()

# ============================================================
# 0. 随机种子
# ============================================================
def set_seed(seed=42):
    """
    设置随机种子，提升实验可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    # random.seed(seed)


# ============================================================
# 1. 数据集
# ============================================================
class TrafficForecastDataset(Dataset):
    """
    交通预测数据集。

    输入：
        X_data: [num_samples, P, N, 3]
            X[..., 0] = 交通速度/流量
            X[..., 1] = Minute of Day, 范围 0~287
            X[..., 2] = Day of Week, 范围 0~6

        y_data: [num_samples, Q, N, 1]

    输出：
        X: [P, N, 3]
        y: [Q, N, 1]
    """
    def __init__(self, X_data, y_data):
        super().__init__()

        if isinstance(X_data, np.ndarray):
            X_data = torch.from_numpy(X_data.astype(np.float64))
        if isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data.astype(np.float64))

        self.X_data = X_data.float()
        self.y_data = y_data.float()

        assert self.X_data.ndim == 4, "X_data 必须是 [num_samples, P, N, 3]"
        assert self.y_data.ndim == 4, "y_data 必须是 [num_samples, Q, N, 1]"
        assert self.X_data.shape[0] == self.y_data.shape[0], "X_data 和 y_data 样本数不一致"
        assert self.X_data.shape[-1] == 3, "X_data 最后一维必须为 3"

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


# ============================================================
# 2. 指标函数
# ============================================================
def compute_regression_metrics(y_pred, y_true, mask_value=None, eps=1e-6):
    """
    计算 MAE / RMSE / MAPE

    输入：
        y_pred: [B, Q, N, 1] 或 [num_samples, Q, N, 1]
        y_true: [B, Q, N, 1] 或 [num_samples, Q, N, 1]
        mask_value: 若不为 None，则忽略 y_true == mask_value 的位置

    输出：
        dict:
            {
                "MAE": ...,
                "RMSE": ...,
                "MAPE": ...
            }
    """
    assert y_pred.shape == y_true.shape, "y_pred 和 y_true 的 shape 必须一致"

    if mask_value is None:
        mask = torch.ones_like(y_true, dtype=torch.bool)
    else:
        mask = (y_true != mask_value)

    valid_num = mask.sum().item()
    if valid_num == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MAPE": float("nan")
        }

    abs_error = torch.abs(y_pred - y_true)
    sq_error = (y_pred - y_true) ** 2
    ape = abs_error / (torch.abs(y_true) + eps)

    mae = abs_error[mask].mean().item()
    rmse = torch.sqrt(sq_error[mask].mean()).item()
    mape = ape[mask].mean().item() * 100.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


# ============================================================
# 3. batch 解析函数
# ============================================================
def unpack_batch(batch, device, E_global=None):
    """
    兼容两种 DataLoader 输出：
        1) (X, y)      -> 使用外部 E_global
        2) (X, E, y)   -> 直接使用 batch 自带的 E

    输入：
        X: [B, P, N, 3]
        y: [B, Q, N, 1]
        E: [7, 288, N, e_dim] 或 [B, 7, 288, N, e_dim]

    输出：
        X: [B, P, N, 3]
        E: [7, 288, N, e_dim]
        y: [B, Q, N, 1]
    """
    if len(batch) == 2:
        X, y = batch
        if E_global is None:
            raise ValueError("当前 batch 只有 (X, y)，因此必须传入 E_global")
        E = torch.from_numpy(E_global.astype(np.float64))
    elif len(batch) == 3:
        X, E, y = batch
    else:
        raise ValueError("batch 必须是 (X, y) 或 (X, E, y)")

    X = X.float().to(device)
    y = y.float().to(device)
    E = E.float().to(device)

    # 如果 E 被复制到了 batch 维，默认所有样本共用同一个 E
    if E.ndim == 5:
        E = E[0]

    return X, E, y


# ============================================================
# 4. 训练 / 验证 / 测试
# ============================================================
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    E_global=None,
    grad_clip=None,
    log_interval=50
):
    """
    单轮训练。
    """
    model.train()

    total_loss = 0.0
    total_samples = 0
    start_time = datetime.datetime.now()
    for batch_idx, batch in enumerate(train_loader):
        X, E, y = unpack_batch(batch, device=device, E_global=E_global)

        optimizer.zero_grad()

        out_puts = model(X, E)
        # loss_past = masked_mae(out_puts["recon_past"], X[...,0:1], 0.0)
        loss = masked_mae(out_puts["pred_future"], y, 0.0)

        # loss = args.eta * loss_past + (1.0 - args.eta) * loss_future

        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = X.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx == 100:
            end_time = datetime.datetime.now()
            total_time = end_time - start_time
            print("Total training time is : %f for each 100 iterations." % total_time.total_seconds())

        if (batch_idx + 1) % log_interval == 0:
            print("[Train] Batch {}/{} | Loss = {:.6f}".format(
                batch_idx + 1, len(train_loader), loss.item()
            ))

    epoch_loss = total_loss / max(total_samples, 1)
    return epoch_loss


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    E_global=None,
    mask_value=None
):
    """
    验证函数。
    返回：
        {
            "val_loss": ...,
            "MAE": ...,
            "RMSE": ...,
            "MAPE": ...
        }
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    preds_list = []
    trues_list = []

    for batch in val_loader:
        X, E, y = unpack_batch(batch, device=device, E_global=E_global)

        out_puts = model(X, E)
        # loss_past = masked_mae(out_puts["recon_past"], X[...,0:1], 0.0)
        loss = masked_mae(out_puts["pred_future"], y, 0.0)
        # loss = args.eta * loss_past + (1.0 - args.eta) * loss_future

        batch_size = X.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds_list.append(out_puts["pred_future"].detach().cpu())
        trues_list.append(y.detach().cpu())

    val_loss = total_loss / max(total_samples, 1)

    y_pred = torch.cat(preds_list, dim=0)
    y_true = torch.cat(trues_list, dim=0)

    metrics = compute_regression_metrics(
        y_pred=y_pred,
        y_true=y_true,
        mask_value=mask_value
    )

    results = {
        "val_loss": val_loss,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"]
    }
    return results


@torch.no_grad()
def test(
    model,
    test_loader,
    device,
    E_global=None,
    mask_value=None,
    return_outputs=True
):
    """
    测试函数。
    返回：
        {
            "MAE": ...,
            "RMSE": ...,
            "MAPE": ...,
            "y_pred": ...,
            "y_true": ...
        }
    """
    if args.model_file is not None:
        model.load_state_dict(torch.load(args.model_file, map_location=device))
        model.to(device)
    model.eval()

    preds_list = []
    trues_list = []
    start_time = datetime.datetime.now()
    ablation_time = datetime.timedelta(0)
    for batch in test_loader:
        X, E, y = unpack_batch(batch, device=device, E_global=E_global)
        out_puts = model(X, E)

        preds_list.append(out_puts["pred_future"].detach().cpu())
        ablation_time += out_puts["infer_time"].total_seconds()
        trues_list.append(y.detach().cpu())
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total testing time is : %f." % (total_time.total_seconds() - ablation_time.total_seconds()))

    y_pred = torch.cat(preds_list, dim=0)
    y_true = torch.cat(trues_list, dim=0)

    print(y_pred.shape, y_true.shape)

    print('                MAE\t\tRMSE\t\tMAPE')
    for i in range(48):
        metrics = compute_regression_metrics(y_pred=y_pred[:,i],
                                            y_true=y_true[:,i],
                                            mask_value=mask_value)
        print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, metrics["MAE"], metrics["RMSE"], metrics["MAPE"]))

    metrics = compute_regression_metrics(
        y_pred=y_pred,
        y_true=y_true,
        mask_value=mask_value
    )

    results = {
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"]
    }

    if return_outputs:
        results["y_pred"] = y_pred
        results["y_true"] = y_true

    return results


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    E_global=None,
    grad_clip=None,
    mask_value=None,
    early_stop_patience=20,
    save_path="best_model.pth"
):
    """
    完整训练流程：
        - 每轮训练
        - 每轮验证
        - 保存最佳模型
        - 早停
        - 学习率调度
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_MAE": [],
        "val_RMSE": [],
        "val_MAPE": []
    }

    best_val_loss = float("inf")
    best_state_dict = None
    wait = 0

    # --------------------------------------------------------
    # 损失函数 / 优化器 / 调度器
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            E_global=E_global,
            grad_clip=grad_clip,
            log_interval=50
        )

        val_results = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            E_global=E_global,
            mask_value=mask_value
        )

        val_loss = val_results["val_loss"]
        val_mae = val_results["MAE"]
        val_rmse = val_results["RMSE"]
        val_mape = val_results["MAPE"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_MAE"].append(val_mae)
        history["val_RMSE"].append(val_rmse)
        history["val_MAPE"].append(val_mape)

        print(
            "Epoch [{:03d}/{:03d}] | Train Loss: {:.6f} | Val Loss: {:.6f} | "
            "Val MAE: {:.6f} | Val RMSE: {:.6f} | Val MAPE: {:.3f}%".format(
                epoch, epochs, train_loss, val_loss, val_mae, val_rmse, val_mape
            )
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            wait = 0

            if save_path is not None:
                torch.save(best_state_dict, save_path)
                print("Best model saved to: {}".format(save_path))
        else:
            wait += 1

        if early_stop_patience is not None and wait >= early_stop_patience:
            print("Early stopping triggered at epoch {}.".format(epoch))
            break

    return history


# ============================================================
# 5. 示例主函数
# ============================================================
if __name__ == "__main__":
    set_seed(42)

    # --------------------------------------------------------
    # 设备
    # --------------------------------------------------------
    device = torch.device(f"cuda:"+str(args.cudas) if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_X, train_y, val_X, val_y, test_X, test_y, E_global, mean, std = load_dataset(args)
    train_dataset = TrafficForecastDataset(train_X, train_y)
    val_dataset = TrafficForecastDataset(val_X, val_y)
    test_dataset = TrafficForecastDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------
    model = STEFFTFormer(args, mean, std).to(device)

    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    if not args.test:  # training
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            E_global=E_global,
            grad_clip=5.0,
            mask_value=0.0,
            early_stop_patience=50,
            save_path=args.model_file
        )
    else: # test
        # --------------------------------------------------------
        # 测试
        # --------------------------------------------------------
        test_results = test(
            model=model,
            test_loader=test_loader,
            device=device,
            E_global=E_global,
            mask_value=0.0,
            return_outputs=True
        )

        print("\n========== Test Results ==========")
        print("Test MAE  : {:.3f}".format(test_results["MAE"]))
        print("Test RMSE : {:.3f}".format(test_results["RMSE"]))
        print("Test MAPE : {:.3f}%".format(test_results["MAPE"]))

        y_pred = test_results["y_pred"]
        y_true = test_results["y_true"]

        y_pred = torch.squeeze(y_pred).transpose(2, 1).cpu().numpy()
        y_true = torch.squeeze(y_true).transpose(2, 1).cpu().numpy()
        np.savez_compressed('data/STEFF-' + 'PEMS-BAY', **{'prediction': y_pred, 'truth': y_true})

        print("y_pred.shape =", y_pred.shape)
        print("y_true.shape =", y_true.shape)