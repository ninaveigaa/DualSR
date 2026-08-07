#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 04:00:00
#SBATCH -J dualedsr_train
#SBATCH -o slurm/logs/dualedsr_%j.out
#SBATCH -e slurm/logs/dualedsr_%j.err

set -euo pipefail

mkdir -p slurm/logs outputs/logs outputs/checkpoints

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/DualSR
source venv/bin/activate

python -u src/train.py \
    --data_source wang \
    --iterations 100000 \
    --n_train 200 \
    --n_valid 40 \
    --val_every 1000 \
    --checkpoint_every 5000 \
    --log_every 100 \
    --domain_size_lr 32 \
    --scale 4 \
    --base_filters 64 \
    --n_res_blocks 16 \
    --lr 1e-4 \
    --log_path outputs/logs/dual_train_log_${SLURM_JOB_ID}.csv \
    --checkpoint_dir outputs/checkpoints/${SLURM_JOB_ID}
