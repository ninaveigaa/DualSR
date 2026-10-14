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
    bash slurm/arrhenius/setup_cpu_env.sh                      # once, login node
    sbatch -A <project>-gpu slurm/arrhenius/setup_gpu_env.sh   # once; check log says CUDA available: True

    export JHTDB_TOKEN=<your token>
    sbatch -A <project>-cpu slurm/arrhenius/build_mixing_dataset.sh

    # after mixing_build_<jobid>.out ends with "Finalized ..."
    sbatch -A <project>-gpu slurm/arrhenius/run_dualedsr_mixing.sh
    sbatch -A <project>-gpu slurm/arrhenius/run_dualhat_mixing.sh

Training uses LR patches of 32^3 (HR 128^3); DualHAT uses `--window_size 8`
(must divide 32). Check the partition's walltime limit with `sinfo` and adjust `-t`.
