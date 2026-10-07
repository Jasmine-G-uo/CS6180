# HW1 Mini-LLM Exercise (nanoGPT)

The code is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
(commit `3adf61e`, MIT license, see `LICENSE`). I only changed `model.py` and `train.py`,
and added the run and plot scripts.

## What is changed

New options in `GPTConfig` / `train.py`. With the default values the code is the same as the original nanoGPT.

| option | values | used for |
|---|---|---|
| `norm_type` | `layernorm` (default), `rmsnorm` | Q2, replace all LayerNorm with RMSNorm |
| `mlp_type` | `gelu` (default), `swiglu` | Q3, SwiGLU MLP with hidden size 8d/3 |
| `pos_type` | `learned` (default), `none`, `rope` | Q4, NoPE and RoPE |
| `n_kv_head` | `0` (= n_head, MHA), `n_head/2` | Q5, GQA with group size 2 |

`train.py` also writes `loss_log.json` (eval losses every 250 iters and train loss every 10 iters) into the out dir.

## Main experiment: default config on GPU

All runs use `config/train_shakespeare_char.py` without change (6 layers, 6 heads, n_embd 384,
block size 256, batch 64, dropout 0.2, 5000 iters). They run on one A100 GPU of the Northeastern
Explorer cluster (`run_full.sbatch`). Only one thing is changed per run, and the seed (1337) and the
iteration count (5000) are the same for all runs. I also tried peak lr 5e-4 and 2e-3.

```
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --out_dir=runs_full/baseline_lr1e-3
python train.py config/train_shakespeare_char.py --out_dir=runs_full/rope_lr1e-3 --pos_type=rope
# other variants: --norm_type=rmsnorm, --mlp_type=swiglu, --pos_type=none, --n_kv_head=3
python plot_full.py      # figures_full/ and figures_full/summary.md
```

Best validation loss (iteration of the best checkpoint in brackets):

| model | params | lr=5e-4 | lr=1e-3 (default) | lr=2e-3 |
|---|---|---|---|---|
| Baseline | 10,745,088 | 1.4803 (2500) | 1.4697 (1750) | 1.4626 (1750) |
| RMSNorm | 10,745,088 | 1.4747 (2500) | 1.4737 (1750) | 1.4664 (2000) |
| SwiGLU | 10,745,088 | 1.5016 (1750) | 1.4893 (1500) | 1.4853 (2000) |
| NoPE | 10,646,784 | 1.5132 (2750) | 1.5310 (2250) | 1.5296 (3000) |
| RoPE | 10,646,784 | 1.4722 (1750) | 1.4791 (1500) | 1.4708 (1500) |
| GQA | 9,860,352 | 1.4769 (2250) | 1.4715 (2000) | not finished (job time limit) |

Logs: `runs_full/<model>_lr<lr>/` (`train.log` is the full stdout, `loss_log.json` has the curves),
Slurm outputs: `runs_full/slurm_*.out`, figures: `figures_full/`.

## Extra experiment: MacBook setting from the README quick start

Same variants with the small MacBook command from the README (4 layers, 4 heads, n_embd 128,
block size 64, batch 12, 2000 iters, dropout 0, CPU, `eval_iters=200`), 5 peak learning rates.

```
bash run_all.sh          # 6 models x 5 learning rates, about 45 s per run on a MacBook Air M3
python plot_results.py   # figures/ and figures/summary.md
```

| model | params | lr=5e-4 | lr=1e-3 (default) | lr=2e-3 | lr=4e-3 | lr=8e-3 |
|---|---|---|---|---|---|---|
| Baseline | 804,096 | 2.0058 | 1.9189 | 1.8278 | 1.7756 | 1.7701 |
| RMSNorm | 804,096 | 2.0034 | 1.9131 | 1.8263 | 1.7576 | 1.7740 |
| SwiGLU | 803,584 | 1.8648 | 1.8084 | 1.7556 | 1.7434 | 1.7738 |
| NoPE | 795,904 | 2.1240 | 2.0567 | 2.0305 | 2.0507 | 2.2046 |
| RoPE | 795,904 | 1.8606 | 1.7969 | 1.7663 | 1.7461 | 1.7750 |
| GQA | 738,560 | 1.9983 | 1.9113 | 1.8354 | 1.7590 | 1.7579 |

Logs: `runs/<model>_lr<lr>/`, figures: `figures/`.
