import matplotlib.pyplot as plt
import pandas as pd

EDSR_CSV = "outputs/logs/dual_train_log_l1_only_6927424.csv"
HAT_CSV = "outputs/logs/dualhat_train_log_l1_only_6927425.csv"

# Column names, exactly as in your CSVs.
COL_ITER = "iteration"
COL_TRAIN_LOSS = "train_loss_xyz"
COL_VAL_LOSS = "val_loss_xyz"
COL_TRAIN_PSNR = "train_psnr_xyz"
COL_VAL_PSNR = "val_psnr_xyz"
COL_TIME = "elapsed_s"
COL_MEM = "peak_mem_gb"

BG_SLIDE = "#0F172A"
GRID = "#2C3A52"
TEXT_MAIN = "#F8FAFC"
TEXT_SOFT = "#94A3B8"
TRAIN_COLOR = "#7FB2E5"   # light blue
VAL_COLOR = "#F2A65A"     # amber
TIME_COLOR = "#7FB2E5"    # light blue
MEM_COLOR = "#E8837F"     # coral

# Force the dark background everywhere, belt-and-suspenders style
plt.rcParams.update({
    "figure.facecolor": BG_SLIDE,
    "savefig.facecolor": BG_SLIDE,
    "savefig.edgecolor": BG_SLIDE,
    "axes.facecolor": BG_SLIDE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "text.color": TEXT_MAIN,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SOFT,
    "xtick.color": TEXT_SOFT,
    "ytick.color": TEXT_SOFT,
    "axes.titlecolor": TEXT_MAIN,
})

def style_axis(ax, title):
    ax.set_facecolor(BG_SLIDE)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color=TEXT_MAIN)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT_SOFT, labelsize=9)

def style_legend(leg):
    leg.get_frame().set_facecolor(BG_SLIDE)
    leg.get_frame().set_edgecolor(TEXT_SOFT)
    leg.get_frame().set_alpha(0.6)
    leg.get_frame().set_linewidth(0.6)
    for text in leg.get_texts():
        text.set_color(TEXT_MAIN)

def plot_loss(ax, df):
    ax.plot(df[COL_ITER], df[COL_TRAIN_LOSS], color=TRAIN_COLOR, linewidth=0.9, label="train")
    ax.plot(df[COL_ITER], df[COL_VAL_LOSS], color=VAL_COLOR, linewidth=1.1, label="val")
    style_axis(ax, "Training Loss")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    style_legend(ax.legend(fontsize=8))

def plot_psnr(ax, df):
    ax.plot(df[COL_ITER], df[COL_TRAIN_PSNR], color=TRAIN_COLOR, linewidth=0.9, label="train")
    ax.plot(df[COL_ITER], df[COL_VAL_PSNR], color=VAL_COLOR, linewidth=1.1, label="val")
    style_axis(ax, "PSNR")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("PSNR (dB)", fontsize=9)
    style_legend(ax.legend(fontsize=8, loc="lower right"))

def plot_time_memory(ax, df):
    ax.plot(df[COL_ITER], df[COL_TIME], color=TIME_COLOR, linewidth=1.2)
    style_axis(ax, "Time & Memory")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Elapsed time (s)", fontsize=9, color=TIME_COLOR)
    ax.tick_params(axis="y", colors=TIME_COLOR)
    ax2 = ax.twinx()
    ax2.plot(df[COL_ITER], df[COL_MEM], color=MEM_COLOR, linewidth=1.2)
    ax2.set_ylabel("Peak GPU memory (GB)", fontsize=9, color=MEM_COLOR)
    ax2.tick_params(axis="y", colors=MEM_COLOR, labelsize=9)
    ax2.set_facecolor("none")
    for spine in ax2.spines.values():
        spine.set_visible(False)

def main():
    edsr_df = pd.read_csv(EDSR_CSV)
    hat_df = pd.read_csv(HAT_CSV)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(BG_SLIDE)
    fig.set_facecolor(BG_SLIDE)

    row_specs = [
        ("DualEDSR", edsr_df, axes[0]),
        ("DualHAT", hat_df, axes[1]),
    ]

    for row_label, df, (ax_loss, ax_psnr, ax_time) in row_specs:
        plot_loss(ax_loss, df)
        plot_psnr(ax_psnr, df)
        plot_time_memory(ax_time, df)
        ax_loss.annotate(
            row_label, xy=(-0.32, 0.5), xycoords="axes fraction",
            fontsize=13, fontweight="bold", color=TEXT_MAIN,
            va="center", ha="left", rotation=90,
        )

    fig.tight_layout(rect=[0.06, 0.02, 0.98, 0.98], h_pad=4, w_pad=2.5)

    fig.savefig("dualedsr_vs_dualhat_training_curves.png", dpi=200,
                facecolor=BG_SLIDE, edgecolor=BG_SLIDE)
    print("Saved: dualedsr_vs_dualhat_training_curves.png")

if __name__ == "__main__":
    main()
