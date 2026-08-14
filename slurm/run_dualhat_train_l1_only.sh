#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 16:00:00
#SBATCH -J dualhat_l1_only
#SBATCH -o slurm/logs/dualhat_l1_only_%j.out
#SBATCH -e slurm/logs/dualhat_l1_only_%j.err

set -euo pipefail

mkdir -p slurm/logs outputs/logs outputs/checkpoints

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/DualSR
source venv/bin/activate

python -u src/train.py \
    --model_type dualhat \
    --data_source deeprock \
    --iterations 100000 \
    --n_train 200 \
    --n_valid 40 \
    --val_every 1000 \
    --checkpoint_every 5000 \
    --log_every 100 \
    --domain_size_lr 32 \
    --scale 4 \
    --loss_type l1 \
    --embed_dim 60 \
    --window_size 5 \
    --hat_depths 6,6 \
    --hat_num_heads 6,6 \
    --lr 1e-4 \
    --log_path outputs/logs/dualhat_train_log_l1_only_${SLURM_JOB_ID}.csv \
    --checkpoint_dir outputs/checkpoints/dualhat_l1_only_${SLURM_JOB_ID}
