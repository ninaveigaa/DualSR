#!/usr/bin/env bash
# Error maps |SR - HR| on the validation volume (t = 36).
#     sbatch -A <project>-gpu slurm/arrhenius/make_error_figure.sh
# Output: mixing_errors.png in the repo root (and a PSNR table in the log).
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -J error_fig
#SBATCH -o slurm/logs/error_fig_%j.out
#SBATCH -e slurm/logs/error_fig_%j.err

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

python -u src/visualize_mixing_errors.py --h5 ${MIXING_H5:-data/jhtdb_mixing/mixing_density_quick.h5} --out mixing_errors.png
