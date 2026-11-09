#!/usr/bin/env bash
# Builds a JHTDB mixing dataset of independent HR 64^3 volumes (LR 16^3 by
# 4x4x4 block averaging). Each training sample is one whole 64^3 volume,
# taken at a random, non-overlapping position of the 16x16x16 grid of 64^3
# cubes in the 1024^3 domain (a different subset for each snapshot).
#
# Default: 128 volumes per snapshot x 6 snapshots = 768 HR/LR pairs
# (640 for training at t = 4..27, 128 for validation at t = 36), about
# 0.8 GB of HR data instead of ~26 GB for the full 256^3 dataset.
# Override with CROPS_PER_VOLUME=... at submit time.
#
# CPU only (x86 `cpu` partition). Needs internet access to web.idies.jhu.edu;
# if compute nodes have none, run the python command below on the login node
# inside tmux instead. Resumable: re-submitting continues where it stopped.
#
# Submit from the repo root:
#     export JHTDB_TOKEN=<your token>
#     sbatch -A <project>-cpu slurm/arrhenius/build_mixing64_dataset.sh
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -J mixing64_build
#SBATCH -o slurm/logs/mixing64_build_%j.out
#SBATCH -e slurm/logs/mixing64_build_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ml Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source venv-cpu/bin/activate
mkdir -p data/jhtdb_mixing

CPV=${CROPS_PER_VOLUME:-128}
python -u src/build_mixing_dataset.py \
    --out ${MIXING_H5:-data/jhtdb_mixing/mixing_density_hr64_cpv${CPV}.h5} \
    --variable density \
    --times 4 9 14 20 27 36 \
    --crop_size 64 \
    --scale 4 \
    --block_size 64 \
    --crops_per_volume ${CPV}
