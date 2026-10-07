#!/bin/bash
#SBATCH --job-name=nanogpt-full
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_%j.out
# Full default config (config/train_shakespeare_char.py) on one GPU.
# Usage: LRS="1e-3" sbatch --export=ALL --gres=gpu:a100:1 run_full.sbatch baseline rmsnorm swiglu
cd /scratch/$USER/nanoGPT
LRS=${LRS:-1e-3}
module load anaconda3/2024.06
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"

for LR in $LRS; do
for name in "$@"; do
  case $name in
    baseline) flags="" ;;
    rmsnorm)  flags="--norm_type=rmsnorm" ;;
    swiglu)   flags="--mlp_type=swiglu" ;;
    nope)     flags="--pos_type=none" ;;
    rope)     flags="--pos_type=rope" ;;
    gqa)      flags="--n_kv_head=3" ;;
  esac
  out=runs_full/${name}_lr${LR}
  mkdir -p $out
  echo "$(date +%T) start $name lr=$LR"
  python -u train.py config/train_shakespeare_char.py --learning_rate=$LR --out_dir=$out $flags > $out/train.log 2>&1
  echo "$(date +%T) done  $name lr=$LR  $(grep 'step 5000' $out/train.log)"
done
done
