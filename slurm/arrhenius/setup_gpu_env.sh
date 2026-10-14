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

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Architecture: $(uname -m)"
# Modules visible on the GPU (Arm) nodes, for reference if something fails:
module avail 2>&1 | grep -i -E "python|pytorch|torch|cuda" || true
# CPU-partition modules do not work here; use a GPU/ Python module if one is
# available, otherwise fall back to the system python3.
for m in $(module -t avail 2>&1 | grep -E "^GPU/.*Python/3" | sort -V | tail -1); do ml "$m" && echo "Loaded $m"; done
python3 -m venv venv-gpu
source venv-gpu/bin/activate
pip install --upgrade pip
# CUDA 12.8 wheels exist for aarch64 (GH200)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy h5py einops pandas matplotlib givernylocal

python - <<'PY'
import torch
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
