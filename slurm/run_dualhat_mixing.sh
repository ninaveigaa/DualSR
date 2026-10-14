#!/usr/bin/env bash
#SBATCH -A NAISS2025-3-39
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 48:00:00
#SBATCH -J dualhat_mixing
#SBATCH -o slurm/logs/dualhat_mixing_%j.out
#SBATCH -e slurm/logs/dualhat_mixing_%j.err

set -euo pipefail

mkdir -p slurm/logs outputs/logs outputs/checkpoints

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/DualSR
source venv/bin/activate

python -u src/train.py \
    --model_type dualhat \
    --data_source jhtdb_mixing \
    --mixing_h5 data/jhtdb_mixing/mixing_density.h5 \
    --train_times 4 9 14 20 27 \
    --val_times 36 \
    --patches_per_crop 8 \
    --num_workers 4 \
    --iterations 100000 \
    --val_every 1000 \
    --val_batches 64 \
    --checkpoint_every 5000 \
    --log_every 100 \
    --domain_size_lr 32 \
    --scale 4 \
    --loss_type l1 \
    --embed_dim 60 \
    --window_size 8 \
    --hat_depths 6,6 \
    --hat_num_heads 6,6 \
    --lr 1e-4 \
    --log_path outputs/logs/dualhat_mixing_${SLURM_JOB_ID}.csv \
    --checkpoint_dir outputs/checkpoints/dualhat_mixing_${SLURM_JOB_ID}
