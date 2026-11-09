# Running on Arrhenius (NAISS)

Scripts for the JHTDB mixing dataset on Arrhenius. Differences from Alvis:

- GPU nodes (`-p gpu`): 4 x NVIDIA GH200 with **Arm (aarch64)** Grace CPUs.
  CPU nodes (`-p cpu`): AMD x86-64. A venv built on one does not run on the other,
  hence `venv-cpu` (dataset download) and `venv-gpu` (training).
- Account: `<project>-gpu` for GPU jobs, `<project>-cpu` for CPU jobs (see `projinfo`).
  It is passed on the command line (`sbatch -A ...`), not in the scripts.
- Never use `-N`: it allocates and bills the full node (4 GPUs).
- Put the repo and data in `/nobackup/proj/disk/<project>/` (home quota is 30 GiB;
  the HDF5 file is ~26 GB). All scripts run from the directory you submit from.

## Steps (from the repo root, e.g. /nobackup/proj/disk/<project>/DualSR)

    mkdir -p slurm/logs
    bash slurm/arrhenius/setup_cpu_env.sh                      # once, login node (loads the Python module)
    sbatch -A <project>-gpu slurm/arrhenius/setup_gpu_env.sh   # once; check log says CUDA available: True

    export JHTDB_TOKEN=<your token>
    sbatch -A <project>-cpu slurm/arrhenius/build_mixing_dataset.sh

    # after mixing_build_<jobid>.out ends with "Finalized ..."
    sbatch -A <project>-gpu slurm/arrhenius/run_dualedsr_mixing.sh
    sbatch -A <project>-gpu slurm/arrhenius/run_dualhat_mixing.sh

Training uses LR patches of 32^3 (HR 128^3); DualHAT uses `--window_size 8`
(must divide 32). Check the partition's walltime limit with `sinfo` and adjust `-t`.

## HR 64^3 volumes (smaller dataset, shorter runs)

`build_mixing64_dataset.sh` downloads independent HR 64^3 volumes (LR 16^3 by
4x4x4 block averaging) at random, non-overlapping positions of each snapshot:
128 per snapshot by default (640 train at t = 4..27, 128 valid at t = 36),
~0.8 GB instead of ~26 GB. Each training sample is one whole volume.

    export JHTDB_TOKEN=<your token>
    sbatch -A <project>-cpu slurm/arrhenius/build_mixing64_dataset.sh   # CROPS_PER_VOLUME=... to change

    H5=data/jhtdb_mixing/mixing_density_hr64_cpv128.h5
    DOMAIN_LR=16 MIXING_H5=$H5 ITERATIONS=20000 sbatch -A <project>-gpu -t 01:00:00 slurm/arrhenius/run_dualedsr_mixing.sh
    DOMAIN_LR=16 MIXING_H5=$H5 ITERATIONS=20000 sbatch -A <project>-gpu -t 03:00:00 slurm/arrhenius/run_dualhat_mixing.sh

`--window_size 8` still works for DualHAT (it divides 16 and 64).
