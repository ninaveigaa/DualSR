#!/usr/bin/env bash
# Downloads the 6 JHTDB mixing volumes and builds the HR/LR HDF5 file.
# CPU only (x86 `cpu` partition). Needs internet access to web.idies.jhu.edu;
# if compute nodes have none, run the python command below on the login node
# inside tmux instead. Resumable: re-submitting continues where it stopped.
#
# Submit from the repo root (inside /nobackup/proj/disk/<project>/DualSR):
#     export JHTDB_TOKEN=<your token>
#     sbatch -A <project>-cpu slurm/arrhenius/build_mixing_dataset.sh
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH -J mixing_build
#SBATCH -o slurm/logs/mixing_build_%j.out
#SBATCH -e slurm/logs/mixing_build_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source venv-cpu/bin/activate
mkdir -p data/jhtdb_mixing

python -u src/build_mixing_dataset.py \
    --out data/jhtdb_mixing/mixing_density.h5 \
    --variable density \
    --times 4 9 14 20 27 36 \
    --crop_size 256 \
    --scale 4 \
    --block_size 128
