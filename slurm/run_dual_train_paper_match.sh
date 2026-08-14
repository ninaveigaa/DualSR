#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 10:00:00
#SBATCH -J dualedsr_paper_match
#SBATCH -o slurm/logs/dualedsr_paper_match_%j.out
#SBATCH -e slurm/logs/dualedsr_paper_match_%j.err

set -euo pipefail

mkdir -p slurm/logs outputs/logs outputs/checkpoints

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/DualSR
source venv/bin/activate

python -u src/train.py \
    --data_source deeprock \
    --iterations 100000 \
    --n_train 200 \
    --n_valid 40 \
    --val_every 1000 \
    --checkpoint_every 5000 \
    --log_every 100 \
    --domain_size_lr 32 \
    --scale 4 \
    --batch_size 4 \
    --loss_type l1 \
    --base_filters 64 \
    --n_res_blocks 16 \
    --lr 1e-4 \
    --log_path outputs/logs/dual_train_log_paper_match_${SLURM_JOB_ID}.csv \
    --checkpoint_dir outputs/checkpoints/paper_match_${SLURM_JOB_ID}
