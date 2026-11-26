# STEFF-TFormer

STEFF-TFormer is a PyTorch implementation for traffic forecasting with an external spatio-temporal embedding field. The repository contains two complementary workflows:

1. `AlphaTraffic.py` trains an AlphaTraffic Foundation encoder and exports a weekly spatio-temporal template field.
2. `main.py` trains or evaluates `STEFFTFormer`, a supervised traffic forecasting model that consumes the exported weekly field as prior knowledge.

The expected temporal resolution is 5 minutes, so one day contains `288` slots and one week contains `7 x 288 = 2016` slots.

## Repository Structure

```text
.
├── AlphaTraffic.py        # AlphaTraffic Foundation training and weekly field generation
├── main.py                # STEFFTFormer training and testing entry point
├── model/
│   └── model.py           # STEFFTFormer model definition
├── data/                  # Dataset directory
├── results/               # Suggested output directory
└── README.md
```

## Environment

The code is written for Python and PyTorch. Install the core dependencies before running the scripts:

```bash
pip install numpy torch
```

If you use CUDA, install the PyTorch build that matches your GPU and driver environment.

## Data Format

### AlphaTraffic Foundation Input

`AlphaTraffic.py` loads both the adjacency matrix and the traffic sequence from `.npz` files using the fixed key `data`:

```python
A = np.load(args.adj_path, allow_pickle=True)["data"].astype(np.float64)
X = np.load(args.x_path, allow_pickle=True)["data"].astype(np.float64)
```

Expected shapes:

```text
A: [N, N]
X: [B_total, N, 3]
```

The last dimension of `X` must follow this convention:

```text
X[..., 0] = traffic speed or flow value
X[..., 1] = minute-of-day slot, usually in [0, 287]
X[..., 2] = day-of-week index, usually in [0, 6]
```

Only the traffic channel `X[..., 0]` is standardized. Calendar channels are kept in their original integer slot format.

### STEFFTFormer Input

`main.py` expects `load_dataset(args)` from `utils_.until` to return:

```text
train_X, train_y, val_X, val_y, test_X, test_y, E_global, mean, std
```

Expected tensor shapes:

```text
train_X / val_X / test_X: [num_samples, P, N, 3]
train_y / val_y / test_y: [num_samples, Q, N, 1]
E_global:                 [7, 288, N, d_model]
```

For the input sequence `X`, the channel convention is the same:

```text
X[..., 0] = historical traffic value
X[..., 1] = minute-of-day slot
X[..., 2] = day-of-week index
```

## Workflow 1: Build the Weekly Embedding Field

Run `AlphaTraffic.py` to train the AlphaTraffic Foundation model and generate the weekly template field:

```bash
python AlphaTraffic.py \
  --adj_path data/adjacent.npz \
  --x_path data/data_train.npz \
  --out_dir outputs \
  --support_len 288 \
  --valid_len 1 \
  --stride 1 \
  --slots_per_day 288 \
  --d_model 128 \
  --emb_dim 64 \
  --depth 4 \
  --nhead 4 \
  --dropout 0.1 \
  --epochs 20 \
  --batch_size 16 \
  --lr 0.001 \
  --weight_decay 0.0001
```

Main arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--adj_path` | Path to adjacency `.npz` file with key `data` | required |
| `--x_path` | Path to traffic sequence `.npz` file with key `data` | required |
| `--out_dir` | Output directory | required |
| `--support_len` | Historical support window length | `288` |
| `--valid_len` | Target validation window length used for period conditioning | `1` |
| `--stride` | Sliding-window stride | `1` |
| `--slots_per_day` | Number of time slots per day | `288` |
| `--d_model` | Hidden feature dimension | `128` |
| `--emb_dim` | Output embedding dimension | `64` |
| `--depth` | Number of GTP blocks | `4` |
| `--nhead` | Number of temporal attention heads | `4` |
| `--lambda_cons` | Teacher-student consistency loss weight | `0.02` |
| `--lambda_uni` | Embedding uniformity loss weight | `0.005` |

`AlphaTraffic.py` builds:

```text
dynamic_field:             [Q, N, emb_dim]
period_index:              [Q, 4]
weekly_template_field:     [2016, N, emb_dim]
weekly_template_field_4d:  [7, 288, N, emb_dim]
weekly_template_std:       [2016, N, emb_dim]
weekly_template_std_4d:    [7, 288, N, emb_dim]
weekly_slot_count:         [2016]
dynamic_network_field:     [Q, emb_dim]
weekly_network_field:      [2016, emb_dim]
weekly_network_field_4d:   [7, 288, emb_dim]
```

The current script saves the weekly template field as:

```text
<out_dir>/weekly_template_field_4d.npz
```

with the fixed key:

```text
data
```

The saved array has shape:

```text
[7, 288, N, emb_dim]
```

## Workflow 2: Train STEFFTFormer for Traffic Forecasting

Run `main.py` for supervised forecasting. The model loads data through `utils_.until.load_dataset`, builds `STEFFTFormer`, and trains with masked MAE on `pred_future`.

Example for PEMS-BAY:

```bash
python main.py \
  --cudas 0 \
  --nodes 325 \
  --dataset PEMS-BAY \
  --datafile ./data/ \
  --batch_size 32 \
  --epochs 100 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --num_blocks 1 \
  --channels 3 \
  --spatial_top_k 32 \
  --eta 0.2 \
  --dropout 0.0 \
  --loss_mode paper \
  --model_file best_prior_pems_bay_model_0.pth \
  --temporal_causal_mask
```

Example for METR-LA:

```bash
python main.py \
  --cudas 0 \
  --nodes 207 \
  --dataset METR-LA \
  --datafile ./data/ \
  --batch_size 32 \
  --epochs 100 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --num_blocks 1 \
  --channels 3 \
  --spatial_top_k 32 \
  --eta 0.2 \
  --dropout 0.0 \
  --loss_mode paper \
  --model_file best_prior_metr_model_0.pth \
  --temporal_causal_mask
```

Example for PEMS08:

```bash
python main.py \
  --cudas 0 \
  --nodes 170 \
  --dataset PEMS08 \
  --datafile ./data/ \
  --batch_size 32 \
  --epochs 100 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --num_blocks 1 \
  --channels 3 \
  --spatial_top_k 32 \
  --eta 0.2 \
  --dropout 0.0 \
  --loss_mode paper \
  --model_file best_prior_pems08_model_0.pth \
  --temporal_causal_mask
```

Important `main.py` arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--input_len` | Historical sequence length `P` | `48` |
| `--output_len` | Forecast horizon `Q` | `48` |
| `--slots_per_day` | Number of time slots per day | `288` |
| `--d_model` | Hidden dimension used by temporal and embedding modules | `64` |
| `--num_heads` | Number of temporal attention heads | `8` |
| `--num_blocks` | Number of model blocks configured by the script | `2` |
| `--nodes` | Number of traffic sensors/nodes | `325` |
| `--dataset` | Dataset name passed to the loader | `PEMS-BAY` |
| `--datafile` | Base data directory | `./data/` |
| `--model_file` | Checkpoint path for saving or loading | `best_prior_pems08_model_0.pth` |
| `--test` | Run evaluation instead of training | disabled |

During training, the best validation checkpoint is saved to `--model_file`.

## Testing

Use `--test` to load `--model_file` and evaluate the model:

```bash
python main.py \
  --cudas 0 \
  --nodes 325 \
  --dataset PEMS-BAY \
  --datafile ./data/ \
  --batch_size 32 \
  --input_len 48 \
  --output_len 48 \
  --slots_per_day 288 \
  --d_model 64 \
  --num_heads 8 \
  --model_file best_prior_pems_bay_model_0.pth \
  --test
```

The test loop reports MAE, RMSE, and MAPE for each forecasting step and for the full prediction horizon. It also saves compressed prediction and truth arrays to:

```text
data/STEFF-PEMS-BAY.npz
```

with keys:

```text
prediction
truth
```

## Model Notes

### AlphaTraffic Foundation

`AlphaTrafficFoundation` uses:

- normalized graph propagation over `A + I`;
- cyclic day-of-week and minute-of-day encodings;
- node embeddings;
- Graph-Time-Precision (`GTPBlock`) fusion;
- period-conditioned temporal summarization;
- spherical bottleneck embeddings;
- teacher-student consistency and embedding uniformity regularization.

Its main purpose in this repository is to produce the weekly embedding prior `weekly_template_field_4d`.

### STEFFTFormer

`STEFFTFormer` consumes:

```text
X: [B, P, N, 3]
E: [7, slots_per_day, N, d_model]
```

It retrieves external spatio-temporal priors from `E` by `day_of_week`, `minute_of_day`, and node index. The model then combines those priors with learned temporal embeddings and temporal attention blocks to predict:

```text
pred_future: [B, Q, N, 1]
```

## Notes and Caveats

- `AlphaTraffic.py` currently saves only `weekly_template_field_4d.npz`; additional save calls for model weights, histories, and intermediate fields are present but commented out.
- `main.py` depends on `utils_.until.load_dataset`, `masked_mae`, and `masked_mse`. Make sure that module is available in your Python path before running supervised training.
- In the current test code, the saved result filename is hard-coded as `data/STEFF-PEMS-BAY.npz`.
- For each sample, the last historical `minute_of_day` and `day_of_week` must be consistent across all nodes because future calendar slots are inferred from the last history step.

## License

This repository includes a `LICENSE` file. Please refer to it for usage terms.
