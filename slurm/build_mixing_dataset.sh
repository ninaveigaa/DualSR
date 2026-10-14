#!/usr/bin/env bash
# Downloads the 6 JHTDB mixing volumes and builds the HR/LR HDF5 file
# (src/build_mixing_dataset.py). CPU only, but needs internet access to
# web.idies.jhu.edu -- if the compute nodes have no outbound access, run the
# same python command on a login node inside tmux/screen instead.
# Resumable: re-submitting continues from the last downloaded block.
#SBATCH -A NAISS2025-3-39
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH -C NOGPU
#SBATCH -t 48:00:00
#SBATCH -J mixing_build
#SBATCH -o slurm/logs/mixing_build_%j.out
#SBATCH -e slurm/logs/mixing_build_%j.err

set -euo pipefail

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/kthmech/nvlmds/DualSR
source venv/bin/activate
mkdir -p slurm/logs data/jhtdb_mixing

# export JHTDB_TOKEN=<your personal token> before sbatch (sbatch passes the env)
python -u src/build_mixing_dataset.py \
    --out data/jhtdb_mixing/mixing_density.h5 \
    --variable density \
    --times 4 9 14 20 27 36 \
    --crop_size 256 \
    --scale 4 \
    --block_size 128
