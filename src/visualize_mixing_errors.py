"""
src/visualize_mixing_errors.py

Error maps on one crop of the validation volume (default t = 36):
    top row:    trilinear | DualEDSR | DualHAT | HR target
    bottom row: |trilinear - HR| | |DualEDSR - HR| | |DualHAT - HR|  (+ colour bar)
Same XY slice as visualize_mixing.py, one shared error colour scale, and
the mean absolute error (MAE) of the whole 3D crop in each title.

    python src/visualize_mixing_errors.py --h5 data/jhtdb_mixing/mixing_density_quick.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_mixing import latest_ckpt, psnr, run
from DualEDSR_arch import DualEDSR
from HATDualSR_arch import DualHAT

INK, SOFT, BG = "#1F2430", "#5B6270", "#FBFBF9"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/jhtdb_mixing/mixing_density_quick.h5")
    p.add_argument("--edsr", default=latest_ckpt("outputs/checkpoints/dualedsr_mixing_*"))
    p.add_argument("--hat", default=latest_ckpt("outputs/checkpoints/dualhat_mixing_*"))
    p.add_argument("--time", type=float, default=36)
    p.add_argument("--out", default="mixing_errors.png")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(a.h5, "r") as f:
        n = int(np.flatnonzero(np.isclose(f["t"][()], a.time))[0])
        lr = f["lr"][n].astype(np.float32)
        hr = f["hr"][n].astype(np.float32)

    preds = {"Trilinear": F.interpolate(torch.from_numpy(lr)[None, None], scale_factor=4,
                                        mode="trilinear", align_corners=False)[0, 0].numpy()}
    if a.edsr:
        m = DualEDSR(scale=4, base_filters=64, n_res_blocks=16, in_channels=1)
        m.load_state_dict(torch.load(a.edsr, map_location="cpu"))
        preds["DualEDSR"] = run(m.to(device).eval(), lr, device)
        del m
    if a.hat:
        m = DualHAT(scale=4, in_channels=1, embed_dim=60, depths=(6, 6),
                    num_heads=(6, 6), window_size=8)
        m.load_state_dict(torch.load(a.hat, map_location="cpu"))
        preds["DualHAT"] = run(m.to(device).eval(), lr, device)

    z = hr.shape[2] // 2
    errs = {k: np.abs(v - hr) for k, v in preds.items()}
    emax = float(np.percentile(np.concatenate([e[:, :, z].ravel() for e in errs.values()]), 99.5))
    vmin, vmax = float(hr[:, :, z].min()), float(hr[:, :, z].max())

    k = len(preds)
    fig, axes = plt.subplots(2, k + 1, figsize=(3.1 * (k + 1), 7.3), facecolor=BG)
    print(f"crop {n}, t = {a.time:g}")
    for c, (name, v) in enumerate(preds.items()):
        mae = float(errs[name].mean())
        axes[0, c].imshow(v[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, c].set_title(f"{name}\n{psnr(v, hr):.1f} dB", fontsize=11, color=INK)
        im = axes[1, c].imshow(errs[name][:, :, z].T, cmap="magma", vmin=0, vmax=emax, origin="lower")
        axes[1, c].set_title(f"|{name} − HR|\nMAE {mae:.4f}", fontsize=11, color=INK)
        print(f"  {name:<10} PSNR {psnr(v, hr):5.1f} dB   MAE {mae:.4f}")
    axes[0, k].imshow(hr[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, k].set_title("HR target\n", fontsize=11, color=INK)
    axes[1, k].axis("off")
    cb = fig.colorbar(im, ax=axes[1, k], fraction=0.5, pad=0.0, location="left")
    cb.set_label("absolute error (scaled density, [0, 1])", color=SOFT, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#D5D9E0")
    axes[0, 0].set_ylabel(f"t = {a.time:g}\nvalidation volume", fontsize=11, color=INK)
    axes[1, 0].set_ylabel("absolute error", fontsize=11, color=INK)
    fig.text(0.01, 0.01, f"Density, XY slice through the middle of a 256³ crop of the validation volume "
             f"(never seen in training). Error colour scale shared; PSNR and MAE over the whole 3D crop.",
             fontsize=9, color=SOFT)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out, dpi=200)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
