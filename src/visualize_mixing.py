"""
src/visualize_mixing.py

Example super-resolution figure for the JHTDB mixing runs:
    LR input | trilinear | DualEDSR | DualHAT | HR target
one row per crop: by default one crop of the validation volume (t = 36)
and one crop of a training volume (t = 9), both clearly labelled. Each
row shares the HR grey scale; titles give the PSNR of the full 3D crop
against HR (same metric as in train.py).

The models are fully convolutional, so they run on the whole 64^3 LR crop
(256^3 HR); if that does not fit in GPU memory, the centre 32^3 patch
(the training patch size) is used instead.

Usage (GPU node, see slurm/arrhenius/make_example_figure.sh):
    python src/visualize_mixing.py --h5 data/jhtdb_mixing/mixing_density_quick.h5
"""

import argparse
import glob
import math
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
from DualEDSR_arch import DualEDSR
from HATDualSR_arch import DualHAT


def latest_ckpt(pattern):
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    for d in reversed(dirs):
        pts = sorted(glob.glob(os.path.join(d, "model_iter*.pt")))
        if pts:
            return pts[-1]
    return None


def psnr(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    rng = float(target.max() - target.min())
    return 20 * math.log10(max(rng, 1e-8)) - 10 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def run(model, lr, device):
    x = torch.from_numpy(lr)[None, None].to(device)
    _, pred = model(x)
    return pred[0, 0].float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/jhtdb_mixing/mixing_density_quick.h5")
    p.add_argument("--edsr", default=latest_ckpt("outputs/checkpoints/dualedsr_mixing_*"))
    p.add_argument("--hat", default=latest_ckpt("outputs/checkpoints/dualhat_mixing_*"))
    p.add_argument("--times", type=float, nargs="+", default=[36, 9],
                   help="one row per time: first crop of that volume")
    p.add_argument("--val_times", type=float, nargs="+", default=[36])
    p.add_argument("--out", default="mixing_example.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| DualEDSR:", args.edsr, "| DualHAT:", args.hat)

    models = {}
    if args.edsr:
        m = DualEDSR(scale=4, base_filters=64, n_res_blocks=16, in_channels=1)
        m.load_state_dict(torch.load(args.edsr, map_location="cpu"))
        models["DualEDSR"] = m.to(device).eval()
    if args.hat:
        m = DualHAT(scale=4, in_channels=1, embed_dim=60, depths=(6, 6),
                    num_heads=(6, 6), window_size=8)
        m.load_state_dict(torch.load(args.hat, map_location="cpu"))
        models["DualHAT"] = m.to(device).eval()
    if not models:
        raise SystemExit("No checkpoints found in outputs/checkpoints/")

    with h5py.File(args.h5, "r") as f:
        t_all = f["t"][()]
        rows = []
        for t in args.times:
            idx = np.flatnonzero(np.isclose(t_all, t))
            if len(idx) == 0:
                print(f"t = {t:g} not in the file, skipped")
                continue
            n = int(idx[0])
            rows.append((t, n, f["lr"][n].astype(np.float32), f["hr"][n].astype(np.float32)))

    cols = ["LR input", "Trilinear"] + list(models) + ["HR target"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols), 3.7 * len(rows)),
                             facecolor="#FBFBF9", squeeze=False)
    print(f"{'crop':<28}" + "".join(f"{c:>12}" for c in cols[1:-1]))
    for r, (t, n, lr, hr) in enumerate(rows):
        preds = {}
        for name, m in models.items():
            try:
                preds[name] = run(m, lr, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                c0 = (lr.shape[0] - 32) // 2
                lr, hr = lr[c0:c0+32, c0:c0+32, c0:c0+32], hr[4*c0:4*c0+128, 4*c0:4*c0+128, 4*c0:4*c0+128]
                print(f"  out of memory on the full crop: using the centre 32^3 patch")
                preds = {k: run(mm, lr, device) for k, mm in models.items()}
                break
        tri = F.interpolate(torch.from_numpy(lr)[None, None], scale_factor=4,
                            mode="trilinear", align_corners=False)[0, 0].numpy()
        near = np.repeat(np.repeat(np.repeat(lr, 4, 0), 4, 1), 4, 2)
        panels = [("LR input", near, None), ("Trilinear", tri, psnr(tri, hr))]
        panels += [(k, v, psnr(v, hr)) for k, v in preds.items()]
        panels += [("HR target", hr, None)]
        z = hr.shape[2] // 2
        vmin, vmax = float(hr[:, :, z].min()), float(hr[:, :, z].max())
        role = "validation" if any(np.isclose(t, v) for v in args.val_times) else "training"
        for c, (name, vol, ps) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(vol[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#D5D9E0")
            title = name if ps is None else f"{name}\n{ps:.1f} dB"
            ax.set_title(title, fontsize=11, color="#1F2430")
        axes[r, 0].set_ylabel(f"t = {t:g}\n{role} volume", fontsize=11, color="#1F2430")
        print(f"{f't={t:g} ({role}) crop {n}':<28}" +
              "".join(f"{ps:>9.1f} dB" for _, _, ps in panels[1:-1]))

    fig.text(0.01, 0.01, "Density, XY slice through the middle of a 256³ HR crop; PSNR over the "
             "whole 3D crop. Models trained on the reduced subset (10 crops).",
             fontsize=9, color="#5B6270")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=200)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
