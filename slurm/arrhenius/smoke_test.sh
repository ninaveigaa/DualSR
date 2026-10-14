#!/usr/bin/env bash
# 2-minute GPU check: DualHAT on random data, LR patch 32, window 8.
#     sbatch -A <project>-gpu slurm/arrhenius/smoke_test.sh
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -t 00:15:00
#SBATCH -J smoke
#SBATCH -o slurm/logs/smoke_%j.out
#SBATCH -e slurm/logs/smoke_%j.err

set -eo pipefail  # no -u: Lmod module functions use unset variables
type ml >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true
cd "$SLURM_SUBMIT_DIR"
# Start clean: the job inherits the submitting shell's environment, which on
# the (x86) login node may hold an x86 venv or CPU-only modules that cannot
# run on the Arm GPU nodes.
if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate 2>/dev/null || true; fi
module purge 2>/dev/null || true
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -E '/venv|/software/sse2' | paste -sd: -)"
mkdir -p outputs/logs outputs/checkpoints
# Load the GPU Python module venv-gpu was built on (it provides numpy & co.)
GPU_PY_MOD="$(cat venv-gpu/.module 2>/dev/null || true)"
if [ -z "$GPU_PY_MOD" ]; then
    GPU_PY_MOD="$(module -t avail 2>&1 | grep -E '^GPU/.*Python/3' | sort -V | tail -1 || true)"
fi
if [ -n "$GPU_PY_MOD" ]; then ml "$GPU_PY_MOD" && echo "Loaded $GPU_PY_MOD"; fi
source venv-gpu/bin/activate
python -c "import numpy, h5py, einops, torch; print('env OK, CUDA:', torch.cuda.is_available())"

python -u src/train.py --model_type dualhat --data_source dummy \
    --iterations 20 --log_every 10 --val_every 10 --val_batches 2 \
    --checkpoint_every 20 --domain_size_lr 32 --window_size 8 \
    --hat_depths 6,6 --hat_num_heads 6,6 \
    --log_path outputs/smoke/log.csv --checkpoint_dir outputs/smoke
