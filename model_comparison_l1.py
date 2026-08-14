import os
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DualEDSR_arch import DualEDSR
from HATDualSR_arch import DualHAT
from dataset import DeepRockDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 4
SAMPLE_INDICES = [0, 1, 2]
SLICE_AXIS = "z"
OUT_PATH = "outputs/figures/model_comparison_l1.png"

BG_SLIDE = "#0F172A"
TEXT_MAIN = "#F8FAFC"
TEXT_SOFT = "#94A3B8"

plt.rcParams.update({
    "figure.facecolor": BG_SLIDE,
    "savefig.facecolor": BG_SLIDE,
    "savefig.edgecolor": BG_SLIDE,
    "axes.facecolor": BG_SLIDE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "text.color": TEXT_MAIN,
})

MODELS = {
    "DualEDSR (L1)": dict(
        model_type="dualedsr",
        ckpt="outputs/checkpoints/l1_only_6927424/model_iter0100000.pt",
        base_filters=64, n_res_blocks=16, symmetric_z=False,
    ),
    "DualHAT (L1)": dict(
        model_type="dualhat",
        ckpt="outputs/checkpoints/dualhat_l1_only_6927425/model_iter0100000.pt",
        embed_dim=60, window_size=5, hat_depths=(6, 6), hat_num_heads=(6, 6),
    ),
}

def build_model(cfg):
    if cfg["model_type"] == "dualedsr":
        model = DualEDSR(scale=SCALE, base_filters=cfg["base_filters"],
                          n_res_blocks=cfg["n_res_blocks"], in_channels=1,
                          symmetric_z=cfg.get("symmetric_z", False))
    else:
        model = DualHAT(scale=SCALE, in_channels=1,
                         embed_dim=cfg["embed_dim"],
                         depths=cfg["hat_depths"],
                         num_heads=cfg["hat_num_heads"],
                         window_size=cfg["window_size"])
    state = torch.load(cfg["ckpt"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def get_slice(vol, axis):
    vol = vol.squeeze(0)
    if axis == "z":
        return vol[:, :, vol.shape[2] // 2]
    elif axis == "y":
        return vol[:, vol.shape[1] // 2, :]
    else:
        return vol[vol.shape[0] // 2, :, :]

def bicubic_slice_baseline(lr_vol, axis, scale):
    lr_slice = get_slice(lr_vol, axis).unsqueeze(0).unsqueeze(0)
    hr_slice = F.interpolate(lr_slice, scale_factor=scale, mode="bicubic", align_corners=False)
    return hr_slice.squeeze(0).squeeze(0)

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    base = "data/deeprock_sr/DeepRockSR-3D/shuffled3D"
    val_set = DeepRockDataset(
        hr_dir=f"{base}/shuffled3D_valid_HR",
        lr_dir=f"{base}/shuffled3D_valid_LR_default_X{SCALE}",
        scale=SCALE)

    print(f"Loading {len(MODELS)} model(s)...")
    models = {}
    for label, cfg in MODELS.items():
        if not os.path.exists(cfg["ckpt"]):
            print(f"  SKIPPING '{label}': checkpoint not found at {cfg['ckpt']}")
            continue
        models[label] = build_model(cfg)
        print(f"  loaded '{label}' from {cfg['ckpt']}")

    if not models:
        raise SystemExit("No checkpoints found -- fix the paths in the CONFIG section.")

    n_cols = 3 + len(models)
    n_rows = len(SAMPLE_INDICES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows))
    fig.patch.set_facecolor(BG_SLIDE)
    if n_rows == 1:
        axes = axes[None, :]

    col_labels = ["LR (input)", "Bicubic (baseline)"] + list(models.keys()) + ["HR (ground truth)"]

    for row, idx in enumerate(SAMPLE_INDICES):
        sample = val_set[idx]
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).to(device)

        bicubic_sl = bicubic_slice_baseline(lr[0].cpu(), SLICE_AXIS, SCALE)

        preds = {}
        with torch.no_grad():
            for label, model in models.items():
                _, pred_xyz = model(lr)
                preds[label] = pred_xyz

        lr_up_nn = F.interpolate(lr, scale_factor=SCALE, mode="nearest")
        lr_sl = get_slice(lr_up_nn[0].cpu(), SLICE_AXIS).numpy()
        hr_sl = get_slice(hr[0].cpu(), SLICE_AXIS).numpy()
        pred_slices = [get_slice(preds[l][0].cpu(), SLICE_AXIS).numpy() for l in models.keys()]

        images = [lr_sl, bicubic_sl.numpy()] + pred_slices + [hr_sl]

        for col, img in enumerate(images):
            ax = axes[row, col]
            ax.set_facecolor(BG_SLIDE)
            ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=10, fontweight="bold", color=TEXT_MAIN, pad=8)
            if col == 0:
                ax.set_ylabel(f"sample {idx}", fontsize=9, color=TEXT_SOFT)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, facecolor=BG_SLIDE, edgecolor=BG_SLIDE)
    print(f"\nSaved comparison figure to {OUT_PATH}")

if __name__ == "__main__":
    main()
