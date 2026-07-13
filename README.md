# AlphaTraffic Foundation 最终版本

## 读取方式
代码中按如下方式读取数据：

```python
A = np.load(args.adj_path, allow_pickle=True)['data'].astype(np.float64)
X = np.load(args.x_path, allow_pickle=True)['data'].astype(np.float64)
```

## 输入定义
- `A.shape = [N, N]`
- `X.shape = [B_total, N, 3]`

其中：
- `X[..., 0]`：交通速度或流量
- `X[..., 1]`：Minute of Day，范围 `0~287`
- `X[..., 2]`：Day of Week，范围 `0~6`

## 主要输出
- `dynamic_field: [Q, N, 64]`
- `weekly_template_field: [2016, N, 64]`
- `weekly_template_field_4d: [7, 288, N, 64]`

## 额外保存
- `weekly_template_field_4d.npz`
- 其中固定使用 key = `data`

## 推荐运行命令
```bash
python AlphaTraffic.py \
  --adj_path data/adjacent.npz \
  --x_path data/data_train.npz \
  --out_dir outputs \
  --support_len 288 \
  --valid_len 1 \
  --stride 1 \
  --epochs 20 \
  --batch_size 2 \
  --slots_per_day 288
```

## 训练运行命令
```bash
python main.py \
  --cudas 0 \
  --nodes 170 \
  --dataset PEMS08 \
  --batch_size 32 \
  --epochs 100 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --num_blocks 1 \
  --channels 1 \
  --spatial_top_k 32 \
  --eta 0.2 \
  --dropout 0.0 \
  --loss_mode paper \
  --model_file best_prior_pems08_model_0.pth \
  --temporal_causal_mask --test
```

```bash
python main.py \
  --cudas 1 \
  --nodes 207 \
  --dataset METR-LA \
  --batch_size 32 \
  --epochs 100 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --num_blocks 1 \
  --channels 1 \
  --spatial_top_k 32 \
  --eta 0.2 \
  --dropout 0.0 \
  --loss_mode paper \
  --model_file best_prior_metr_model_0.pth \
  --temporal_causal_mask --test
```

```bash
python main.py \
  --cudas 1 \
  --nodes 325 \
  --dataset PEMS-BAY \
  --batch_size 32 \
  --epochs 100 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --num_blocks 1 \
  --channels 1 \
  --spatial_top_k 32 \
  --eta 0.2 \
  --dropout 0.0 \
  --loss_mode paper \
  --model_file best_prior_pems_bay_model_0.pth \
  --temporal_causal_mask --test
```