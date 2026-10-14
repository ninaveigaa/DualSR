#!/usr/bin/env bash
# DualHAT on JHTDB mixing, LR patch 32^3 (HR 128^3), 1 x GH200.
# Submit from the repo root:
#     sbatch -A <project>-gpu slurm/arrhenius/run_dualhat_mixing.sh
# No `-N`: on Arrhenius it allocates (and bills) the whole 4-GPU node.
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 36:00:00
#SBATCH -J dualhat_mixing
#SBATCH -o slurm/logs/dualhat_mixing_%j.out
#SBATCH -e slurm/logs/dualhat_mixing_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/logs outputs/checkpoints
source venv-gpu/bin/activate

python -u src/train.py \
    --model_type dualhat \
    --data_source jhtdb_mixing \
    --mixing_h5 data/jhtdb_mixing/mixing_density.h5 \
    --train_times 4 9 14 20 27 \
    --val_times 36 \
    --patches_per_crop 8 \
    --num_workers 8 \
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
