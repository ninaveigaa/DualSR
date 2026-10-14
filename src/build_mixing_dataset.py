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

Only needs numpy, h5py and givernylocal (no PyTorch), so it can run on
CPU nodes.

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


DATASET_TITLE = "mixing"
GRID_SIZE = 1024
DT = 0.04  # time between stored snapshots
N_SNAPSHOTS = 1012
DEFAULT_TIMES = (4, 9, 14, 20, 27, 36)
TESTING_TOKEN = "edu.jhu.pha.turbulence.testing-201406"

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


def crop_positions(n_per_axis, crops_per_volume, seed, ti):
    """
    Grid positions (ix, iy, iz) of the crops taken from volume number ti.
    All positions when crops_per_volume is None; otherwise a seeded random
    subset, different for each volume (more spatial variety).
    """
    allpos = [(ix, iy, iz) for ix in range(n_per_axis)
              for iy in range(n_per_axis) for iz in range(n_per_axis)]
    if crops_per_volume is None or crops_per_volume >= len(allpos):
        return allpos
    rng = np.random.default_rng(seed + ti)
    pick = sorted(rng.choice(len(allpos), size=crops_per_volume, replace=False))
    return [allpos[i] for i in pick]


def create_file(path, n_crops, crop_size, lr_size, variable_spec, times, scale,
                crops_per_volume, crop_seed):
    with h5py.File(path, "w") as f:
        chunk = min(64, crop_size)
        f.create_dataset("hr", shape=(n_crops, crop_size, crop_size, crop_size),
                         dtype="float32", chunks=(1, chunk, chunk, chunk))
        f.create_dataset("lr", shape=(n_crops, lr_size, lr_size, lr_size),
                         dtype="float32", chunks=(1, lr_size, lr_size, lr_size))
        f.create_dataset("t", shape=(n_crops,), dtype="float64")
        f.create_dataset("t_index", shape=(n_crops,), dtype="int64")
        f.create_dataset("origin", shape=(n_crops, 3), dtype="int64")
        f.attrs["variable"] = variable_spec
        f.attrs["scale"] = scale
        f.attrs["crop_size"] = crop_size
        f.attrs["times"] = np.asarray(times, dtype=np.float64)
        f.attrs["crops_per_volume"] = -1 if crops_per_volume is None else crops_per_volume
        f.attrs["crop_seed"] = crop_seed
        f.attrs["normalized"] = False


def download_all(args, dataset_factory=None):
    """
    dataset_factory: callable returning a givernylocal turb_dataset. Each
    download thread gets its own (getCutout mutates the dataset object).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    variable, component = parse_variable(args.variable)
    times = list(args.times)
    crop = args.crop_size
    n_per_axis = GRID_SIZE // crop
    lr_size = crop // args.scale
    sub = args.block_size
    if crop % sub != 0:
        raise ValueError("--block_size must divide --crop_size")
    n_sub = crop // sub
    cpv = args.crops_per_volume
    plan = [(ti, t, pos) for ti, t in enumerate(times)
            for pos in crop_positions(n_per_axis, cpv, args.crop_seed, ti)]
    n_crops = len(plan)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not os.path.exists(args.out):
        create_file(args.out, n_crops, crop, lr_size, args.variable, times,
                    args.scale, cpv, args.crop_seed)

    if dataset_factory is None:
        dataset_factory = lambda: make_dataset(args.token, args.cache_dir)
    local = threading.local()

    def fetch(t_index, x, y, z):
        if not hasattr(local, "dataset"):
            local.dataset = dataset_factory()
        return fetch_block(local.dataset, variable, component, t_index, x, y, z, sub)

    # a large chunk cache, so many small block writes do not re-read chunks
    with h5py.File(args.out, "a", rdcc_nbytes=512 * 1024 ** 2) as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        stored_cpv = int(f.attrs.get("crops_per_volume", -1))
        if f.attrs["variable"] != args.variable or \
                list(f.attrs["times"]) != [float(t) for t in times] or \
                f.attrs["crop_size"] != crop or \
                stored_cpv != (-1 if cpv is None else cpv) or \
                int(f.attrs.get("crop_seed", args.crop_seed)) != args.crop_seed:
            raise ValueError(f"{args.out} was created with different settings; "
                             f"delete it or choose another --out.")
        if f.attrs["normalized"]:
            print("File is already complete and normalized; nothing to do.")
            return
        if "done" in f and f["done"].shape[1] != n_sub:
            raise ValueError(f"{args.out} was started with a different --block_size; "
                             f"use the same value to resume.")
        done = f.require_dataset("done", shape=(n_crops, n_sub, n_sub, n_sub),
                                 dtype="bool", fillvalue=False)

        print(f"{n_crops} crops of {crop}^3 ({len(times)} volumes x "
              f"{n_crops // len(times)} crops), requests of {sub}^3, "
              f"{args.workers} parallel downloads", flush=True)
        t_start = time.time()
        for n, (ti, t, (ix, iy, iz)) in enumerate(plan):
            t_index = time_to_index(t)
            x0, y0, z0 = ix * crop, iy * crop, iz * crop
            f["t"][n] = t
            f["t_index"][n] = t_index
            f["origin"][n] = (x0, y0, z0)
            done_n = done[n]
            todo = [(a, b, c) for a in range(n_sub) for b in range(n_sub)
                    for c in range(n_sub) if not done_n[a, b, c]]
            if not todo:
                continue
            futures = {pool.submit(fetch, t_index, x0 + a * sub, y0 + b * sub,
                                   z0 + c * sub): (a, b, c) for a, b, c in todo}
            t_crop = time.time()
            try:
                completed = as_completed(futures)
                for k, fut in enumerate(completed, 1):
                    a, b, c = futures[fut]
                    block = fut.result()  # re-raises a download error
                    f["hr"][n, a*sub:(a+1)*sub, b*sub:(b+1)*sub, c*sub:(c+1)*sub] = block
                    done_n[a, b, c] = True
                    if k % 256 == 0 or k == len(todo):
                        done[n] = done_n  # checkpoint progress for resuming
                        f.flush()
                        rate = k / (time.time() - t_crop)
                        print(f"    crop {n + 1}: {k}/{len(todo)} blocks "
                              f"({rate:.1f} blocks/s)", flush=True)
            except BaseException:
                # give up on the queued requests so the script stops quickly;
                # everything written so far is kept and resumes on the next run
                for other in futures:
                    other.cancel()
                done[n] = done_n
                f.flush()
                raise
            elapsed = time.time() - t_start
            left = (n_crops - n - 1) * (time.time() - t_crop)
            print(f"[t={t:g} idx={t_index}] crop {n + 1}/{n_crops} "
                  f"origin=({x0},{y0},{z0}) done | {elapsed / 60:.1f} min elapsed, "
                  f"~{left / 60:.0f} min left", flush=True)


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
            c = x.shape[0] // scale
            lr = x.reshape(c, scale, c, scale, c, scale).mean(axis=(1, 3, 5))
            f["lr"][n] = lr.astype(np.float32)

        f.attrs["global_min"] = gmin
        f.attrs["global_max"] = gmax
        f.attrs["normalized"] = True
        if "done" in f:
            del f["done"]
    print(f"Finalized {path}: {n_crops} HR/LR pairs scaled to [0, 1].")


def finalize_partial(src, dst, scale):
    """
    Plan B when the download cannot finish: copy only the crops whose
    download is complete from `src` into a new file `dst`, then scale and
    build the LR crops there (same as finalize). `src` is left untouched,
    so the full download can still be resumed later.
    """
    with h5py.File(src, "r") as f:
        if f.attrs["normalized"]:
            raise RuntimeError(f"{src} is already complete; use it directly.")
        done = f["done"][()]
        complete = [n for n in range(done.shape[0]) if done[n].all()]
        if not complete:
            raise RuntimeError("No complete crop yet.")
        crop = int(f.attrs["crop_size"])
        lr_size = crop // scale
        times_done = sorted(set(float(f["t"][n]) for n in complete))
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        create_file(dst, len(complete), crop, lr_size, f.attrs["variable"],
                    times_done, scale, -1, int(f.attrs.get("crop_seed", 0)))
        with h5py.File(dst, "a") as g:
            for i, n in enumerate(complete):
                g["hr"][i] = f["hr"][n]
                g["t"][i] = f["t"][n]
                g["t_index"][i] = f["t_index"][n]
                g["origin"][i] = f["origin"][n]
            g.attrs["partial_of"] = os.path.abspath(src)
    counts = {}
    with h5py.File(dst, "r") as g:
        for t in g["t"][()]:
            counts[float(t)] = counts.get(float(t), 0) + 1
    print(f"Copied {len(complete)} complete crops: " +
          ", ".join(f"t={t:g}: {c}" for t, c in sorted(counts.items())))
    finalize(dst, scale)


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
    p.add_argument("--block_size", type=int, default=None,
                   help="edge of each download request. Default: 128 with a "
                        "personal token, 16 with the public testing token "
                        "(which allows at most 4096 points per request)")
    p.add_argument("--crops_per_volume", type=int, default=None,
                   help="take only this many crops per volume (seeded random "
                        "subset of the 4x4x4 grid). Default: all 64")
    p.add_argument("--crop_seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8,
                   help="parallel download requests")
    p.add_argument("--token", default=os.environ.get("JHTDB_TOKEN", TESTING_TOKEN))
    p.add_argument("--cache_dir", default="./giverny_output")
    p.add_argument("--partial_out", default=None,
                   help="plan B: write only the crops downloaded so far to this "
                        "new file (scaled, with LR) and exit; --out is kept "
                        "as is so the download can be resumed")
    p.add_argument("--finalize_only", action="store_true",
                   help="skip downloading; only scale and build the LR crops")
    args = p.parse_args()

    if args.block_size is None:
        args.block_size = 16 if args.token == TESTING_TOKEN else 128
    if args.token == TESTING_TOKEN and args.block_size ** 3 > 4096:
        raise ValueError("the public testing token allows at most 4096 points "
                         "per request: use --block_size 16 (or a personal token)")
    if args.token == TESTING_TOKEN:
        print("NOTE: using the public testing token (16^3 requests). For the "
              "full dataset, request a personal token and set JHTDB_TOKEN.")

    if GRID_SIZE % args.crop_size != 0 or args.crop_size % args.scale != 0:
        raise ValueError("crop_size must divide 1024 and be divisible by scale")

    if args.partial_out:
        finalize_partial(args.out, args.partial_out, args.scale)
        return
    if not args.finalize_only:
        download_all(args)
    finalize(args.out, args.scale)


if __name__ == "__main__":
    main()
