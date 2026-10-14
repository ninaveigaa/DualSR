"""
src/build_mixing_dataset.py

Builds the HR/LR training pairs from the JHTDB "mixing" dataset
(Homogeneous Buoyancy Driven turbulence, 1024^3, 1012 snapshots,
dt = 0.04, t in [0, 40.44]), following the protocol agreed at the
5 Oct 2026 progress meeting:

  1. Select 6 volumes one redundancy horizon (T_E) apart:
     t = 4, 9, 14, 20, 27, 36.
  2. Download the full 1024^3 volume of ONE channel per snapshot
     (default: density). Each snapshot is treated as a static 3D
     greyscale image, as in micro-CT.
  3. Cut 256^3 HR crops on a 4 x 4 x 4 grid (64 crops per volume,
     384 pairs in total).
  4. Generate the 64^3 LR crops by 4 x 4 x 4 block averaging.
  5. Scale globally to [0, 1] (one min/max over all selected volumes).
  6. Store everything in one HDF5 file.

The download is done crop by crop (and each crop in sub-blocks), so the
full 1024^3 volume never has to fit in memory, and it is RESUMABLE:
re-running the script skips every block already downloaded.

Output HDF5 layout:
    hr        float32 [N, 256, 256, 256]   (x, y, z), scaled to [0, 1]
    lr        float32 [N,  64,  64,  64]   (x, y, z), scaled to [0, 1]
    t         float64 [N]   physical time of the source snapshot
    t_index   int64   [N]   JHTDB getCutout time index (1-based)
    origin    int64   [N, 3] 0-based (x, y, z) start of the crop in the 1024^3 grid
    attrs: variable, component, scale, crop_size, global_min,
           global_max, normalized (bool)

Undo the scaling with:  raw = x * (global_max - global_min) + global_min

Usage (on a machine with JHTDB access):
    python src/build_mixing_dataset.py --out data/jhtdb_mixing/mixing_density.h5 \
        --variable density --token <your JHTDB token>

The JHTDB public testing token is limited in volume; for 6 x 1024^3
values use your personal token (request one at turbulence.pha.jhu.edu).
It can also be passed via the JHTDB_TOKEN environment variable.
"""

import os
import time
import argparse

import numpy as np
import h5py
import torch
import torch.nn.functional as F


DATASET_TITLE = "mixing"
GRID_SIZE = 1024
DT = 0.04  # time between stored snapshots
N_SNAPSHOTS = 1012
DEFAULT_TIMES = (4, 9, 14, 20, 27, 36)

# number of values per grid point for each JHTDB variable
VARIABLE_COMPONENTS = {"density": 1, "pressure": 1, "velocity": 3}


def time_to_index(t):
    """Physical time -> 1-based getCutout time index (t = 0 -> 1)."""
    idx = int(round(t / DT)) + 1
    if not 1 <= idx <= N_SNAPSHOTS:
        raise ValueError(f"t = {t} is outside the stored range [0, 40.44]")
    return idx


def parse_variable(spec):
    """
    'density' / 'pressure'           -> (variable, component 0)
    'velocity_u' / '_v' / '_w'       -> ('velocity', 0 / 1 / 2)
    """
    if spec in ("density", "pressure"):
        return spec, 0
    if spec.startswith("velocity_") and spec[-1] in "uvw":
        return "velocity", "uvw".index(spec[-1])
    raise ValueError(f"unknown variable '{spec}'. Use density, pressure, "
                     f"velocity_u, velocity_v or velocity_w.")


def fetch_block(dataset, variable, component, t_index, x0, y0, z0, size,
                max_retries=5):
    """
    Downloads one size^3 block starting at the 0-based grid point
    (x0, y0, z0) and returns it as float32 [x, y, z].

    givernylocal returns data in (z, y, x, values) order, so it is
    transposed here to (x, y, z).
    """
    from givernylocal.turbulence_toolkit import getCutout

    axes_ranges = np.array([[x0 + 1, x0 + size],
                            [y0 + 1, y0 + size],
                            [z0 + 1, z0 + size],
                            [t_index, t_index]])
    strides = np.array([1, 1, 1, 1])

    for attempt in range(1, max_retries + 1):
        try:
            result = getCutout(dataset, variable, axes_ranges, strides,
                               verbose=False)
            # exactly one data variable is returned (one variable, one time)
            name = list(result.data_vars)[0]
            arr = np.asarray(result[name].values, dtype=np.float32)  # [z, y, x, c]
            return np.ascontiguousarray(arr[..., component].transpose(2, 1, 0))
        except Exception as e:  # network / server errors: back off and retry
            if attempt == max_retries:
                raise
            wait = 10 * attempt
            print(f"    download failed ({e}); retry {attempt}/{max_retries - 1} "
                  f"in {wait}s", flush=True)
            time.sleep(wait)


def create_file(path, n_crops, crop_size, lr_size, variable_spec, times, scale):
    with h5py.File(path, "w") as f:
        chunk = min(64, crop_size)
        f.create_dataset("hr", shape=(n_crops, crop_size, crop_size, crop_size),
                         dtype="float32", chunks=(1, chunk, chunk, chunk))
        f.create_dataset("lr", shape=(n_crops, lr_size, lr_size, lr_size),
                         dtype="float32", chunks=(1, lr_size, lr_size, lr_size))
        f.create_dataset("t", shape=(n_crops,), dtype="float64")
        f.create_dataset("t_index", shape=(n_crops,), dtype="int64")
        f.create_dataset("origin", shape=(n_crops, 3), dtype="int64")
        # one flag per HR sub-block, so an interrupted run resumes cleanly
        f.attrs["variable"] = variable_spec
        f.attrs["scale"] = scale
        f.attrs["crop_size"] = crop_size
        f.attrs["times"] = np.asarray(times, dtype=np.float64)
        f.attrs["normalized"] = False


def download_all(args, dataset_factory=None):
    variable, component = parse_variable(args.variable)
    times = list(args.times)
    crop = args.crop_size
    n_per_axis = GRID_SIZE // crop
    lr_size = crop // args.scale
    sub = args.block_size
    if crop % sub != 0:
        raise ValueError("--block_size must divide --crop_size")
    n_sub = crop // sub
    n_crops = len(times) * n_per_axis ** 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not os.path.exists(args.out):
        create_file(args.out, n_crops, crop, lr_size, args.variable, times,
                    args.scale)

    with h5py.File(args.out, "a") as f:
        if f.attrs["variable"] != args.variable or \
                list(f.attrs["times"]) != [float(t) for t in times] or \
                f.attrs["crop_size"] != crop:
            raise ValueError(f"{args.out} was created with different settings; "
                             f"delete it or choose another --out.")
        if f.attrs["normalized"]:
            print("File is already complete and normalized; nothing to do.")
            return
        done = f.require_dataset("done", shape=(n_crops, n_sub, n_sub, n_sub),
                                 dtype="bool", fillvalue=False)

        dataset = None  # created lazily, only if something is left to fetch
        t_start = time.time()
        for ti, t in enumerate(times):
            t_index = time_to_index(t)
            for ix in range(n_per_axis):
                for iy in range(n_per_axis):
                    for iz in range(n_per_axis):
                        n = ((ti * n_per_axis + ix) * n_per_axis + iy) * n_per_axis + iz
                        x0, y0, z0 = ix * crop, iy * crop, iz * crop
                        f["t"][n] = t
                        f["t_index"][n] = t_index
                        f["origin"][n] = (x0, y0, z0)
                        if done[n].all():
                            continue
                        if dataset is None:
                            dataset = (dataset_factory() if dataset_factory
                                       else make_dataset(args.token, args.cache_dir))
                        for a in range(n_sub):
                            for b in range(n_sub):
                                for c in range(n_sub):
                                    if done[n, a, b, c]:
                                        continue
                                    block = fetch_block(
                                        dataset, variable, component, t_index,
                                        x0 + a * sub, y0 + b * sub, z0 + c * sub, sub)
                                    f["hr"][n, a*sub:(a+1)*sub, b*sub:(b+1)*sub,
                                            c*sub:(c+1)*sub] = block
                                    done[n, a, b, c] = True
                        f.flush()
                        elapsed = time.time() - t_start
                        print(f"[t={t:g} idx={t_index}] crop {n + 1}/{n_crops} "
                              f"origin=({x0},{y0},{z0}) done | {elapsed / 60:.1f} min",
                              flush=True)


def finalize(path, scale):
    """Global [0, 1] scaling of the HR crops + LR by block averaging."""
    with h5py.File(path, "a") as f:
        if f.attrs["normalized"]:
            print("Already normalized.")
            return
        if "done" in f and not f["done"][()].all():
            raise RuntimeError("Download incomplete; re-run without --finalize_only "
                               "to fetch the missing blocks first.")
        hr = f["hr"]
        n_crops = hr.shape[0]

        # pass 1: global min / max over every crop of every volume
        gmin, gmax = np.inf, -np.inf
        for n in range(n_crops):
            x = hr[n]
            gmin = min(gmin, float(x.min()))
            gmax = max(gmax, float(x.max()))
        span = gmax - gmin
        if span <= 0:
            raise RuntimeError("Constant field: cannot scale to [0, 1].")
        print(f"global min = {gmin:.6g}, global max = {gmax:.6g}")

        # pass 2: scale HR, then LR = 4x4x4 block mean of the scaled HR
        # (block averaging is linear, so this equals scaling the raw LR)
        for n in range(n_crops):
            x = (hr[n] - gmin) / span
            hr[n] = x
            lr = F.avg_pool3d(torch.from_numpy(x)[None, None],
                              kernel_size=scale, stride=scale)[0, 0]
            f["lr"][n] = lr.numpy()

        f.attrs["global_min"] = gmin
        f.attrs["global_max"] = gmax
        f.attrs["normalized"] = True
        if "done" in f:
            del f["done"]
    print(f"Finalized {path}: {n_crops} HR/LR pairs scaled to [0, 1].")


def make_dataset(token, cache_dir):
    from givernylocal.turbulence_dataset import turb_dataset
    os.makedirs(cache_dir, exist_ok=True)
    return turb_dataset(dataset_title=DATASET_TITLE, output_path=cache_dir,
                        auth_token=token)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/jhtdb_mixing/mixing_density.h5")
    p.add_argument("--variable", default="density",
                   help="density, pressure, velocity_u, velocity_v or velocity_w")
    p.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES),
                   help="physical times of the snapshots to use")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--block_size", type=int, default=128,
                   help="edge of each download request; givernylocal works "
                        "best up to ~200^3 per request")
    p.add_argument("--token", default=os.environ.get("JHTDB_TOKEN",
                                                    "edu.jhu.pha.turbulence.testing-201406"))
    p.add_argument("--cache_dir", default="./giverny_output")
    p.add_argument("--finalize_only", action="store_true",
                   help="skip downloading; only scale and build the LR crops")
    args = p.parse_args()

    if GRID_SIZE % args.crop_size != 0 or args.crop_size % args.scale != 0:
        raise ValueError("crop_size must divide 1024 and be divisible by scale")

    if not args.finalize_only:
        download_all(args)
    finalize(args.out, args.scale)


if __name__ == "__main__":
    main()
