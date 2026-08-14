"""
visualize_predictions.py

Loads trained DualSR checkpoints, runs inference on the SAME DeepRock-SR-3D
validation sample(s), and saves a side-by-side comparison figure:
    LR (nearest-upsampled) | trilinear baseline | model 1 | model 2 | ... | HR

Slices shown: the middle XY slice (fixed Z) of each volume, by default.
Change SLICE_AXIS below to compare XZ or YZ instead.

EDIT THE CONFIG SECTION BELOW before running -- fill in the correct
checkpoint paths for your jobs (check with `ls outputs/checkpoints/`).
"""

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
from torch.utils.data import DataLoader


# =======================================================================
# CONFIG -- edit this section
# =======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 4
SAMPLE_INDICES = [0, 1, 2]     # which validation volumes to visualize (rows)
SLICE_AXIS = "z"               # "z" -> XY slice, "y" -> XZ slice, "x" -> YZ slice
OUT_PATH = "outputs/figures/model_comparison.png"

# label -> checkpoint config. Adjust ckpt paths to match your actual
# outputs/checkpoints/<run>/model_iter0100000.pt files (check with
# `ls outputs/checkpoints/` on Alvis if unsure of the exact folder names).
MODELS = {
    "DualEDSR (MSE)": dict(
        model_type="dualedsr",
        ckpt="outputs/checkpoints/6927030/model_iter0100000.pt",
        base_filters=64, n_res_blocks=16, symmetric_z=False,
    ),
    "DualEDSR (L1)": dict(
        model_type="dualedsr",
        ckpt="outputs/checkpoints/l1_only_6927424/model_iter0100000.pt",
        base_filters=64, n_res_blocks=16, symmetric_z=False,
    ),
    "DualEDSR (L1, batch4)": dict(
        model_type="dualedsr",
        ckpt="outputs/checkpoints/paper_match_6927420/model_iter0100000.pt",
        base_filters=64, n_res_blocks=16, symmetric_z=False,
    ),
    "DualEDSR (symZ)": dict(
        model_type="dualedsr",
        ckpt="outputs/checkpoints/symz_6927979/model_iter0100000.pt",
        base_filters=64, n_res_blocks=16, symmetric_z=True,
    ),
    "DualHAT (MSE)": dict(
        model_type="dualhat",
        ckpt="outputs/checkpoints/6927079/model_iter0100000.pt",
        embed_dim=60, window_size=5, hat_depths=(6, 6), hat_num_heads=(6, 6),
    ),
    "DualHAT (L1)": dict(
        model_type="dualhat",
        ckpt="outputs/checkpoints/dualhat_l1_only_6927425/model_iter0100000.pt",
        embed_dim=60, window_size=5, hat_depths=(6, 6), hat_num_heads=(6, 6),
    ),
}

# =======================================================================
# End of config
# =======================================================================


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
    """vol: [C, D, H, W] tensor (numpy-convertible). Returns a 2D slice."""
    vol = vol.squeeze(0)  # drop channel dim (C=1)
    if axis == "z":
        return vol[:, :, vol.shape[2] // 2]
    elif axis == "y":
        return vol[:, vol.shape[1] // 2, :]
    else:  # "x"
        return vol[vol.shape[0] // 2, :, :]


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

    n_cols = 3 + len(models)  # LR, trilinear, [models...], HR
    n_rows = len(SAMPLE_INDICES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_labels = ["LR (input)", "Trilinear (baseline)"] + list(models.keys()) + ["HR (ground truth)"]

    for row, idx in enumerate(SAMPLE_INDICES):
        sample = val_set[idx]
        lr = sample["lr"].unsqueeze(0).to(device)   # [1,1,d,h,w]
        hr = sample["hr"].unsqueeze(0).to(device)    # [1,1,D,H,W]

        trilinear = F.interpolate(lr, scale_factor=SCALE, mode="trilinear",
                                   align_corners=False)

        preds = {}
        with torch.no_grad():
            for label, model in models.items():
                _, pred_xyz = model(lr)
                preds[label] = pred_xyz

        lr_up_nn = F.interpolate(lr, scale_factor=SCALE, mode="nearest")

        images = [lr_up_nn[0].cpu(), trilinear[0].cpu()] + \
                 [preds[l][0].cpu() for l in models.keys()] + [hr[0].cpu()]

        for col, img in enumerate(images):
            sl = get_slice(img, SLICE_AXIS).numpy()
            ax = axes[row, col]
            ax.imshow(sl, cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"sample {idx}", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved comparison figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
