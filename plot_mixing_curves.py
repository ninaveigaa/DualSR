"""
plot_mixing_curves.py

Training curves of DualEDSR and DualHAT on the JHTDB mixing data, on a
light background to match the progress-meeting slides, plus a short
summary (iterations reached, best / last validation PSNR, wall time) to
fill in the slide text.

Usage (picks the newest log of each model when no file is given):
    python plot_mixing_curves.py
    python plot_mixing_curves.py --edsr outputs/logs/dualedsr_mixing_123.csv \
        --hat outputs/logs/dualhat_mixing_124.csv --out mixing_curves.png
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {"DualEDSR": "#2B5FA8", "DualHAT": "#C2621B"}  # blue / orange
INK, SOFT, GRID = "#1F2430", "#5B6270", "#D5D9E0"


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load(path):
    df = pd.read_csv(path)
    val = df.dropna(subset=["val_psnr_xyz"]).drop_duplicates(
        subset=["val_loss_xyz", "val_psnr_xyz"], keep="first")
    return df, val


def smooth(s, n):
    return s.rolling(n, min_periods=1, center=True).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edsr", default=newest("outputs/logs/dualedsr_mixing_*.csv"))
    p.add_argument("--hat", default=newest("outputs/logs/dualhat_mixing_*.csv"))
    p.add_argument("--out", default="mixing_curves_v2.png")
    args = p.parse_args()

    runs = {k: v for k, v in (("DualEDSR", args.edsr), ("DualHAT", args.hat)) if v}
    if not runs:
        raise SystemExit("No logs found in outputs/logs/ (dualedsr_mixing_*.csv, dualhat_mixing_*.csv)")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                         "axes.edgecolor": GRID, "axes.labelcolor": SOFT,
                         "xtick.color": SOFT, "ytick.color": SOFT, "text.color": INK})
    fig, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="#FBFBF9")

    print(f"{'model':<9} {'iters':>7} {'val PSNR, last 10 evals':>24} "
          f"{'train PSNR (end)':>17} {'time':>7}")
    for name, path in runs.items():
        df, val = load(path)
        c = MODELS[name]
        n = max(1, len(df) // 50)
        ax_l.plot(df["iteration"], smooth(df["train_loss_xyz"], n), color=c, alpha=0.35,
                  lw=1.2, label=f"{name} train")
        ax_l.plot(val["iteration"], val["val_loss_xyz"], color=c, lw=0.8, alpha=0.35)
        ax_l.plot(val["iteration"], smooth(val["val_loss_xyz"], 5), color=c, lw=2.4,
                  label=f"{name} validation (5-eval mean)")
        ax_p.plot(df["iteration"], smooth(df["train_psnr_xyz"], n), color=c, alpha=0.35,
                  lw=1.2, label=f"{name} train")
        ax_p.plot(val["iteration"], val["val_psnr_xyz"], color=c, lw=0.8, alpha=0.35,
                  marker="o", ms=2)
        ax_p.plot(val["iteration"], smooth(val["val_psnr_xyz"], 5), color=c, lw=2.4,
                  label=f"{name} validation (5-eval mean)")
        last = val["val_psnr_xyz"].tail(10)
        print(f"{name:<9} {int(df['iteration'].iloc[-1]):>7} "
              f"{last.mean():>14.1f} ± {last.std():.1f} dB "
              f"{smooth(df['train_psnr_xyz'], n).iloc[-1]:>13.1f} dB "
              f"{df['elapsed_s'].iloc[-1] / 3600:>5.1f} h")

    for ax, title, ylabel in ((ax_l, "L1 loss (XYZ stage)", "L1 loss"),
                              (ax_p, "PSNR (XYZ stage)", "PSNR [dB]")):
        ax.set_facecolor("#FBFBF9")
        ax.set_title(title, color=INK, fontsize=14, loc="left")
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_l.set_yscale("log")
    ax_p.legend(frameon=False, fontsize=10, loc="lower right")
    fig.text(0.01, 0.01, "Reduced subset: 12 pairs (10 train, 2 validation); "
             "train curves smoothed; validation (t = 36, 2 crops) every 1000 iterations, "
             "thin = raw, bold = 5-evaluation mean",
             fontsize=10, color=SOFT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
