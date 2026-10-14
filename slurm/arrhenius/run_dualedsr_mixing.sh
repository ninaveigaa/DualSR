#!/usr/bin/env bash
# DualEDSR on JHTDB mixing, LR patch 32^3 (HR 128^3), 1 x GH200.
# Optional: ITERATIONS=30000 MIXING_H5=data/... sbatch ... to override.
# Submit from the repo root:
#     sbatch -A <project>-gpu slurm/arrhenius/run_dualedsr_mixing.sh
# No `-N`: on Arrhenius it allocates (and bills) the whole 4-GPU node.
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J dualedsr_mixing
#SBATCH -o slurm/logs/dualedsr_mixing_%j.out
#SBATCH -e slurm/logs/dualedsr_mixing_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs/logs outputs/checkpoints
source venv-gpu/bin/activate

python -u src/train.py \
    --model_type dualedsr \
    --data_source jhtdb_mixing \
    --mixing_h5 ${MIXING_H5:-data/jhtdb_mixing/mixing_density.h5} \
    --train_times 4 9 14 20 27 \
    --val_times 36 \
    --patches_per_crop 8 \
    --num_workers 8 \
    --iterations ${ITERATIONS:-100000} \
    --val_every 1000 \
    --val_batches 64 \
    --checkpoint_every 5000 \
    --log_every 100 \
    --domain_size_lr 32 \
    --scale 4 \
    --loss_type l1 \
    --base_filters 64 \
    --n_res_blocks 16 \
    --lr 1e-4 \
    --log_path outputs/logs/dualedsr_mixing_${SLURM_JOB_ID}.csv \
    --checkpoint_dir outputs/checkpoints/dualedsr_mixing_${SLURM_JOB_ID}
