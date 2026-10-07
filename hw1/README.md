# HW1 Mini-LLM Exercise (nanoGPT)

The code is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
(commit `3adf61e`, MIT license, see `LICENSE`). I only changed `model.py` and `train.py`,
and added `run_all.sh` and `plot_results.py`.

## What is changed

New options in `GPTConfig` / `train.py`. With the default values the code is the same as the original nanoGPT.

| option | values | used for |
|---|---|---|
| `norm_type` | `layernorm` (default), `rmsnorm` | Q2, replace all LayerNorm with RMSNorm |
| `mlp_type` | `gelu` (default), `swiglu` | Q3, SwiGLU MLP with hidden size 8d/3 |
| `pos_type` | `learned` (default), `none`, `rope` | Q4, NoPE and RoPE |
| `n_kv_head` | `0` (= n_head, MHA), `2` | Q5, GQA with group size 2 |

`train.py` also writes `loss_log.json` (eval losses every 250 iters and train loss every 10 iters) into the out dir.

## How to run

```
pip install torch numpy matplotlib requests
python data/shakespeare_char/prepare.py
bash run_all.sh          # 6 models x 5 learning rates, about 45 s per run on a MacBook Air M3 (CPU)
python plot_results.py   # figures/ and figures/summary.md
```

All runs use `config/train_shakespeare_char.py` with the MacBook settings from the README quick start
(4 layers, 4 heads, n_embd 128, block size 64, batch size 12, 2000 iters, dropout 0, device cpu).
Only one thing is changed per run. Seed (1337) and iteration count (2000) are the same for all runs.
The only difference to the README command is `eval_iters=200` (instead of 20), so the loss estimate is less noisy.

Example, one run:
```
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=200 --log_interval=10 \
  --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 \
  --dropout=0.0 --learning_rate=1e-3 --out_dir=runs/rope_lr1e-3 --pos_type=rope
```

## Results

Best validation loss (mean over 200 batches). The main comparison uses the default lr = 1e-3.
I also tried other peak learning rates (min lr is 1e-4 for all).

| model | params | lr=5e-4 | lr=1e-3 (default) | lr=2e-3 | lr=4e-3 | lr=8e-3 | best lr |
|---|---|---|---|---|---|---|---|
| Baseline | 804,096 | 2.0058 | 1.9189 | 1.8278 | 1.7756 | 1.7701 | 8e-3 |
| RMSNorm | 804,096 | 2.0034 | 1.9131 | 1.8263 | 1.7576 | 1.7740 | 4e-3 |
| SwiGLU | 803,584 | 1.8648 | 1.8084 | 1.7556 | 1.7434 | 1.7738 | 4e-3 |
| NoPE | 795,904 | 2.1240 | 2.0567 | 2.0305 | 2.0507 | 2.2046 | 2e-3 |
| RoPE | 795,904 | 1.8606 | 1.7969 | 1.7663 | 1.7461 | 1.7750 | 4e-3 |
| GQA | 738,560 | 1.9983 | 1.9113 | 1.8354 | 1.7590 | 1.7579 | 8e-3 |

Figures are in `figures/`, and the logs of every run are in `runs/<model>_lr<lr>/`
(`train.log` is the full stdout, `loss_log.json` has the loss curves).
