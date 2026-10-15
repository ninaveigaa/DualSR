"""
plot_mixing_psnr.py

One clean panel for the slides: validation PSNR of DualEDSR and DualHAT
(5-evaluation running mean, shaded ±1 std), training PSNR as thin dashed
lines, and the final values written at the end of each line.

    python plot_mixing_psnr.py            # newest logs, writes mixing_psnr.png
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {"DualEDSR": "#2B5FA8", "DualHAT": "#C2621B"}
INK, SOFT, GRID, BG = "#1F2430", "#5B6270", "#E3E6EA", "#FBFBF9"


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edsr", default=newest("outputs/logs/dualedsr_mixing_*.csv"))
    p.add_argument("--hat", default=newest("outputs/logs/dualhat_mixing_*.csv"))
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out", default="mixing_psnr.png")
    a = p.parse_args()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 13,
                         "axes.edgecolor": GRID, "xtick.color": SOFT, "ytick.color": SOFT})
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=BG)
    ax.set_facecolor(BG)

    placed = []
    for name, path in (("DualEDSR", a.edsr), ("DualHAT", a.hat)):
        if not path:
            continue
        c = MODELS[name]
        df = pd.read_csv(path)
        val = df.dropna(subset=["val_psnr_xyz"]).drop_duplicates(
            subset=["val_loss_xyz", "val_psnr_xyz"], keep="first")
        it = val["iteration"] / 1000
        roll = val["val_psnr_xyz"].rolling(a.window, min_periods=1, center=True)
        mean, std = roll.mean(), roll.std().fillna(0)
        tr = df["train_psnr_xyz"].rolling(max(1, len(df) // 40), min_periods=1, center=True).mean()

        ax.plot(df["iteration"] / 1000, tr, color=c, lw=1.3, ls="--", alpha=0.7)
        ax.fill_between(it, mean - std, mean + std, color=c, alpha=0.15, lw=0)
        ax.plot(it, mean, color=c, lw=3)
        last10 = val["val_psnr_xyz"].tail(10)
        x_end = it.iloc[-1]
        y_lab = mean.iloc[-1]
        for y_prev in placed:  # keep the two labels from overlapping
            if abs(y_lab - y_prev) < 1.6:
                y_lab = y_prev - 1.6 if y_lab <= y_prev else y_prev + 1.6
        placed.append(y_lab)
        ax.text(x_end + 0.8, y_lab, f"{name}\nval {last10.mean():.1f} dB",
                color=c, fontsize=12, va="center", fontweight="bold")
        ax.text(x_end + 0.8, tr.iloc[-1], f"train {tr.iloc[-1]:.1f} dB",
                color=c, fontsize=11, va="center", alpha=0.85)
        print(f"{name}: val {last10.mean():.1f} ± {last10.std():.1f} dB (last 10 evals), "
              f"train {tr.iloc[-1]:.1f} dB, {int(df['iteration'].iloc[-1])} it, "
              f"{df['elapsed_s'].iloc[-1] / 3600:.1f} h")

    ax.set_xlabel("iteration (×1000)", color=SOFT)
    ax.set_ylabel("PSNR [dB]", color=SOFT)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.0)
    ax.margins(x=0)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.95, bottom=0.2)
    fig.text(0.08, 0.04, "Solid: validation volume t = 36 (5-evaluation mean, band = ±1 std). "
             "Dashed: training crops.\nReduced subset: 10 training crops, 2 validation crops.",
             fontsize=10.5, color=SOFT)
    fig.savefig(a.out, dpi=200)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
