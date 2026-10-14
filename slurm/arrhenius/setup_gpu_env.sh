#!/usr/bin/env bash
# Creates venv-gpu for training on the Arrhenius `gpu` partition.
# The GPU nodes have Arm (aarch64) Grace CPUs + NVIDIA GH200, so this venv
# MUST be built on a GPU node -- an x86 venv (login node, Alvis) will not run.
#
# Submit ONCE from the repo root:
#     sbatch -A <project>-gpu slurm/arrhenius/setup_gpu_env.sh
# and check slurm/logs/setup_gpu_<jobid>.out ends with "CUDA available: True".
#
# If NAISS provides a GPU PyTorch module (check `ml spider pytorch` on a GPU
# node), you can load it instead and skip the torch pip install below.
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J setup_gpu_env
#SBATCH -o slurm/logs/setup_gpu_%j.out
#SBATCH -e slurm/logs/setup_gpu_%j.err

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

echo "Architecture: $(uname -m)"
# Modules visible on the GPU (Arm) nodes, for reference if something fails:
module avail 2>&1 | grep -i -E "python|pytorch|torch|cuda" || true
# CPU-partition modules do not work here; use a GPU/ Python module if one is
# available, otherwise fall back to the system python3.
GPU_PY_MOD="$(module -t avail 2>&1 | grep -E '^GPU/.*Python/3' | sort -V | tail -1 || true)"
if [ -n "$GPU_PY_MOD" ]; then ml "$GPU_PY_MOD" && echo "Loaded $GPU_PY_MOD"; fi
PY=$(command -v python3)
echo "Using $PY ($($PY --version 2>&1))"
rm -rf venv-gpu
"$PY" -m venv venv-gpu
printf '%s\n' "${GPU_PY_MOD:-}" > venv-gpu/.module
source venv-gpu/bin/activate
pip install --upgrade pip
# CUDA 12.8 wheels exist for aarch64 (GH200)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy h5py einops pandas matplotlib givernylocal
python -c "import numpy, h5py, einops" && echo "numpy/h5py/einops OK"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
