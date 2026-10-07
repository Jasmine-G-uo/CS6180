#!/bin/bash
# Runs all experiments for the mini-LLM exercise.
# Small config from the README quick start ("I only have a macbook"), same seed (1337)
# and same number of iterations (2000) for every run. Each variant changes one thing only.
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON="config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=200 --log_interval=10 \
 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0"

run() {
  name=$1; lr=$2; shift 2
  out=runs/${name}_lr${lr}
  mkdir -p $out
  echo "$(date +%H:%M:%S) start $out"
  $PY -u train.py $COMMON --learning_rate=$lr --out_dir=$out "$@" > $out/train.log 2>&1
  echo "$(date +%H:%M:%S) done  $out  $(grep 'step 2000' $out/train.log)"
}

for lr in ${LRS:-1e-3 5e-4 2e-3 4e-3 8e-3}; do
  run baseline $lr
  run rmsnorm  $lr --norm_type=rmsnorm
  run swiglu   $lr --mlp_type=swiglu
  run nope     $lr --pos_type=none
  run rope     $lr --pos_type=rope
  run gqa      $lr --n_kv_head=2
done
