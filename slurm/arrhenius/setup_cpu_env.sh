#!/usr/bin/env bash
# Creates venv-cpu (x86-64) for building the dataset on the Arrhenius
# `cpu` partition / login node. Run ONCE, on the login node, from the repo root:
#     bash slurm/arrhenius/setup_cpu_env.sh
set -euo pipefail

# Arrhenius has no `python` until a Python module is loaded
ml Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
python -m venv --system-site-packages venv-cpu
source venv-cpu/bin/activate
pip install --upgrade pip
pip install h5py givernylocal
python -c "import numpy, h5py, givernylocal; print('venv-cpu OK')"
