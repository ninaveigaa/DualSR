"""
src/train.py

Iteration-based training loop for axial-factorized super-resolution
models. Supports two 2D-solver backbones, selected via --model_type:
  - "dualedsr": EDSR2D-based (Wang et al. port)
  - "dualhat":  HAT-based (XPixelGroup official architecture)

Both models share the same forward-pass permutation logic (XY stage,
then Z stage) and the same dict interface for datasets
({"lr", "hr_z", "hr", "axis_mode"}).

Tracks the physics-informed loss (per stage: xy, xyz) and PSNR
(per stage), logged to CSV every `log_every` iterations.

--loss_type ("mse", default, or "l1") selects the base reconstruction
loss used inside dual_physics_loss(). "l1" matches Wang, Armstrong &
Mostaghimi (2019)'s SRCNN training recipe (their eq. 2).
"""

import os
import time
import csv
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from DualEDSR_arch import DualEDSR
from HATDualSR_arch import DualHAT
from dataset import WangCTDataset, DeepRockDataset
from losses import dual_physics_loss, DEFAULT_PHYSICS_WEIGHTS


# ---------------------------------------------------------------------
# Temporary synthetic dataset (for testing only -- no JHTDB required)
# ---------------------------------------------------------------------

class DummyTurbulenceDataset(Dataset):
    """
    Generates random tensors matching the shape/key convention of
    dataset.TurbulencePatchDataset's __getitem__ output. Used only to
    validate the training loop end-to-end while JHTDB access is down.
    """

    def __init__(self, n_samples, domain_size_lr, scale=4):
        self.n_samples = n_samples
        self.domain_size_lr = domain_size_lr
        self.scale = scale

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        n = self.domain_size_lr
        s = self.scale
        lr = torch.randn(3, n, n, n)
        hr_z = torch.randn(3, n * s, n * s, n)
        hr = torch.randn(3, n * s, n * s, n * s)
        return {"lr": lr, "hr_z": hr_z, "hr": hr, "axis_mode": 0}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_psnr(pred, target, eps=1e-8):
    """PSNR using the target's own data range (max - min)."""
    mse = F.mse_loss(pred, target).item()
    data_range = (target.max() - target.min()).item()
    data_range = max(data_range, eps)
    mse = max(mse, eps)
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def infinite_loader(loader):
    """Wraps a DataLoader so it can be iterated indefinitely, cycling
    through the dataset repeatedly."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------
# Single training / validation step
# ---------------------------------------------------------------------

def train_step(model, batch, optimizer, weights, device,
                dx_hr=1.0, dy_hr=1.0, dz_hr=1.0, dz_lr=None,
                loss_type="mse"):
    if dz_lr is None:
        dz_lr = dz_hr * model.scale

    model.train()
    lr = batch["lr"].to(device)
    hr_z = batch["hr_z"].to(device)
    hr = batch["hr"].to(device)

    optimizer.zero_grad()
    pred_xy, pred_xyz = model(lr)

    loss_total, loss_xy, loss_xyz = dual_physics_loss(
        pred_xy, pred_xyz, hr_z, hr,
        dx_hr=dx_hr, dy_hr=dy_hr, dz_hr=dz_hr, dz_lr=dz_lr,
        weights=weights, loss_type=loss_type,
    )
    loss_total.backward()
    optimizer.step()

    with torch.no_grad():
        psnr_xy = compute_psnr(pred_xy, hr_z)
        psnr_xyz = compute_psnr(pred_xyz, hr)

    return loss_xy.item(), loss_xyz.item(), psnr_xy, psnr_xyz


@torch.no_grad()
def validate(model, loader, weights, device, n_batches=None,
             dx_hr=1.0, dy_hr=1.0, dz_hr=1.0, dz_lr=None,
             loss_type="mse"):
    if dz_lr is None:
        dz_lr = dz_hr * model.scale

    model.eval()
    total_loss_xy, total_loss_xyz = 0.0, 0.0
    total_psnr_xy, total_psnr_xyz = 0.0, 0.0
    count = 0

    for batch in loader:
        lr = batch["lr"].to(device)
        hr_z = batch["hr_z"].to(device)
        hr = batch["hr"].to(device)

        pred_xy, pred_xyz = model(lr)
        _, loss_xy, loss_xyz = dual_physics_loss(
            pred_xy, pred_xyz, hr_z, hr,
            dx_hr=dx_hr, dy_hr=dy_hr, dz_hr=dz_hr, dz_lr=dz_lr,
            weights=weights, loss_type=loss_type,
        )

        total_loss_xy += loss_xy.item()
        total_loss_xyz += loss_xyz.item()
        total_psnr_xy += compute_psnr(pred_xy, hr_z)
        total_psnr_xyz += compute_psnr(pred_xyz, hr)
        count += 1

        if n_batches is not None and count >= n_batches:
            break

    return (total_loss_xy / count, total_loss_xyz / count,
            total_psnr_xy / count, total_psnr_xyz / count)


# ---------------------------------------------------------------------
# Full training run (iteration-based)
# ---------------------------------------------------------------------

def train(model, train_loader, val_loader, optimizer, total_iterations,
          weights, device, log_path, checkpoint_dir,
          val_every=1000, checkpoint_every=5000, log_every=100,
          val_batches=20, loss_type="mse"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Using loss_type='{loss_type}' for the base reconstruction term.")

    fieldnames = ["iteration",
                  "train_loss_xy", "train_loss_xyz",
                  "val_loss_xy", "val_loss_xyz",
                  "train_psnr_xy", "train_psnr_xyz",
                  "val_psnr_xy", "val_psnr_xyz",
                  "elapsed_s", "peak_mem_gb"]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    train_iter = infinite_loader(train_loader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()
    running_loss_xy, running_loss_xyz = 0.0, 0.0
    running_psnr_xy, running_psnr_xyz = 0.0, 0.0
    val_loss_xy = val_loss_xyz = val_psnr_xy = val_psnr_xyz = float("nan")

    for it in range(1, total_iterations + 1):
        batch = next(train_iter)
        loss_xy, loss_xyz, psnr_xy, psnr_xyz = train_step(
            model, batch, optimizer, weights, device, loss_type=loss_type)

        running_loss_xy += loss_xy
        running_loss_xyz += loss_xyz
        running_psnr_xy += psnr_xy
        running_psnr_xyz += psnr_xyz

        if it % val_every == 0 or it == total_iterations:
            val_loss_xy, val_loss_xyz, val_psnr_xy, val_psnr_xyz = validate(
                model, val_loader, weights, device, n_batches=val_batches,
                loss_type=loss_type)

        if it % log_every == 0 or it == total_iterations:
            n = log_every if it % log_every == 0 else (it % log_every)
            n = max(n, 1)
            elapsed = time.time() - t_start
            peak_mem = (torch.cuda.max_memory_allocated(device) / 1e9
                        if device.type == "cuda" else 0.0)

            row = {
                "iteration": it,
                "train_loss_xy": running_loss_xy / n,
                "train_loss_xyz": running_loss_xyz / n,
                "val_loss_xy": val_loss_xy, "val_loss_xyz": val_loss_xyz,
                "train_psnr_xy": running_psnr_xy / n,
                "train_psnr_xyz": running_psnr_xyz / n,
                "val_psnr_xy": val_psnr_xy, "val_psnr_xyz": val_psnr_xyz,
                "elapsed_s": elapsed, "peak_mem_gb": peak_mem,
            }
            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            print(f"[iter {it:06d}/{total_iterations}] "
                  f"loss_xyz: train={row['train_loss_xyz']:.4f} val={val_loss_xyz:.4f} | "
                  f"PSNR_xyz: train={row['train_psnr_xyz']:.2f}dB val={val_psnr_xyz:.2f}dB | "
                  f"elapsed={elapsed:.0f}s mem={peak_mem:.2f}GB")

            running_loss_xy, running_loss_xyz = 0.0, 0.0
            running_psnr_xy, running_psnr_xyz = 0.0, 0.0

        if it % checkpoint_every == 0 or it == total_iterations:
            ckpt_path = os.path.join(checkpoint_dir, f"model_iter{it:07d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def parse_int_tuple(s):
    """Parses a comma-separated string like '6,6' into (6, 6)."""
    return tuple(int(x) for x in s.split(","))


def main():
    parser = argparse.ArgumentParser()

    # data / general
    parser.add_argument("--domain_size_lr", type=int, default=32,
                         help="LR cube size. Only used by dummy/wang/jhtdb "
                              "data sources.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=200,
                         help="Only used by dummy/wang data sources "
                              "(deeprock uses its full available split).")
    parser.add_argument("--n_valid", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="mse",
                         choices=["mse", "l1"],
                         help="Base reconstruction loss. 'l1' matches "
                              "Wang, Armstrong & Mostaghimi (2019) eq. 2.")
    parser.add_argument("--data_source", type=str, default="dummy",
                         choices=["dummy", "wang", "deeprock", "jhtdb"])
    parser.add_argument("--log_path", type=str,
                         default="outputs/logs/train_log.csv")
    parser.add_argument("--checkpoint_dir", type=str,
                         default="outputs/checkpoints")

    # model selection
    parser.add_argument("--model_type", type=str, default="dualedsr",
                         choices=["dualedsr", "dualhat"])

    # DualEDSR-specific
    parser.add_argument("--base_filters", type=int, default=64,
                         help="[dualedsr] Matches Wang et al.'s original "
                              "ngsrf default (64).")
    parser.add_argument("--n_res_blocks", type=int, default=16,
                         help="[dualedsr] Matches Wang et al.'s original "
                              "numResBlocks default (16).")
    parser.add_argument("--symmetric_z", action="store_true",
                         help="[dualedsr] Give the Z stage the SAME "
                              "capacity as the XY stage (no //2 split). "
                              "Deviates from Wang et al.'s original "
                              "design; roughly doubles model size/cost. "
                              "Default: off (matches original behavior).")

    # DualHAT-specific
    parser.add_argument("--embed_dim", type=int, default=60,
                         help="[dualhat] Embedding dimension for the XY "
                              "stage (Z stage uses half).")
    parser.add_argument("--window_size", type=int, default=5,
                         help="[dualhat] Must evenly divide the LR input's "
                              "H/W and the resulting HR H/W.")
    parser.add_argument("--hat_depths", type=parse_int_tuple, default="6,6",
                         help="[dualhat] Comma-separated RHAG depths for "
                              "the XY stage, e.g. '6,6' or '6,6,6,6'.")
    parser.add_argument("--hat_num_heads", type=parse_int_tuple, default="6,6",
                         help="[dualhat] Comma-separated attention heads "
                              "per RHAG, matching --hat_depths length.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    in_channels = 1 if args.data_source in ("wang", "deeprock") else 3

    if args.model_type == "dualedsr":
        model = DualEDSR(scale=args.scale, base_filters=args.base_filters,
                          n_res_blocks=args.n_res_blocks,
                          in_channels=in_channels,
                          symmetric_z=args.symmetric_z).to(device)
    else:  # dualhat
        model = DualHAT(scale=args.scale, in_channels=in_channels,
                         embed_dim=args.embed_dim,
                         depths=args.hat_depths,
                         num_heads=args.hat_num_heads,
                         window_size=args.window_size).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} | parameters: {n_params:,}")

    if args.data_source == "dummy":
        print("WARNING: using synthetic dummy data.")
        train_set = DummyTurbulenceDataset(args.n_train, args.domain_size_lr, args.scale)
        val_set = DummyTurbulenceDataset(args.n_valid, args.domain_size_lr, args.scale)

    elif args.data_source == "wang":
        print("NOTE: using Wang et al. PEMFC micro-CT data (single volume, "
              "expect overfitting -- see README notes).")
        train_set = WangCTDataset(
            "data/wang_pemfc/training/HR/HR.npy",
            "data/wang_pemfc/training/LR/LR.npy",
            scale=args.scale, patch_size_lr=args.domain_size_lr, n_patches=args.n_train)
        val_set = WangCTDataset(
            "data/wang_pemfc/validation/HR/HR.npy",
            "data/wang_pemfc/validation/LR/LR.npy",
            scale=args.scale, patch_size_lr=args.domain_size_lr, n_patches=args.n_valid)

    elif args.data_source == "deeprock":
        print("NOTE: using DeepRock-SR-3D shuffled3D (sandstone+carbonate+coal, "
              "pre-combined) data.")
        base = "data/deeprock_sr/DeepRockSR-3D/shuffled3D"
        train_set = DeepRockDataset(
            hr_dir=f"{base}/shuffled3D_train_HR",
            lr_dir=f"{base}/shuffled3D_train_LR_default_X{args.scale}",
            scale=args.scale)
        val_set = DeepRockDataset(
            hr_dir=f"{base}/shuffled3D_valid_HR",
            lr_dir=f"{base}/shuffled3D_valid_LR_default_X{args.scale}",
            scale=args.scale)
        print(f"  train samples: {len(train_set)}, valid samples: {len(val_set)}")

    else:  # jhtdb
        from dataset import TurbulencePatchDataset
        train_set = TurbulencePatchDataset("./data", "train", args.n_train, args.scale)
        val_set = TurbulencePatchDataset("./data", "valid", args.n_valid, args.scale)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.data_source in ("wang", "deeprock"):
        weights = dict(w_U=1.0, w_dU=0.0, w_continuity=0.0,
                       w_omega=0.0, w_enstrophy=0.0, w_strain=0.0)
    else:
        weights = DEFAULT_PHYSICS_WEIGHTS

    train(model, train_loader, val_loader, optimizer, args.iterations,
          weights, device, args.log_path, args.checkpoint_dir,
          val_every=args.val_every, checkpoint_every=args.checkpoint_every,
          log_every=args.log_every, val_batches=args.val_batches,
          loss_type=args.loss_type)


if __name__ == "__main__":
    main()
