#!/usr/bin/env bash
# Creates venv-cpu (x86-64) for building the dataset on the Arrhenius
# `cpu` partition / login node. Run ONCE, on the login node, from the repo root:
#     bash slurm/arrhenius/setup_cpu_env.sh
set -euo pipefail

python3 -m venv venv-cpu
source venv-cpu/bin/activate
pip install --upgrade pip
pip install numpy h5py givernylocal
python -c "import numpy, h5py, givernylocal; print('venv-cpu OK')"
