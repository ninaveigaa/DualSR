"""
src/dataset.py
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *


DATASET_TITLE = "isotropic1024coarse"
AUTH_TOKEN = "edu.jhu.pha.turbulence.testing-201311"  # public test token
FULL_GRID_SIZE = 1024  # full domain size (1024^3)
FULL_TIME_RANGE = [1, 1024]  # adjust to the actual available time range


def get_dataset(output_path="./giverny_output"):
    os.makedirs(output_path, exist_ok=True)
    return turb_dataset(dataset_title=DATASET_TITLE,
                         output_path=output_path,
                         auth_token=AUTH_TOKEN)


def download_random_cutout(dataset, domain_size=128, seed=None,
                            time_range=FULL_TIME_RANGE, grid_size=FULL_GRID_SIZE):
    """
    Downloads one random subcube (domain_size^3) at a random time instant.
    Returns: numpy array [domain_size, domain_size, domain_size, 3] (u, v, w)
    """
    rng = random.Random(seed)

    # random starting position, ensuring the cube fits inside the domain (1-indexed)
    x0 = rng.randint(1, grid_size - domain_size + 1)
    y0 = rng.randint(1, grid_size - domain_size + 1)
    z0 = rng.randint(1, grid_size - domain_size + 1)
    t0 = rng.randint(time_range[0], time_range[1])

    x_range = [x0, x0 + domain_size - 1]
    y_range = [y0, y0 + domain_size - 1]
    z_range = [z0, z0 + domain_size - 1]
    t_range = [t0, t0]

    axes_ranges = np.array([x_range, y_range, z_range, t_range])
    strides = np.array([1, 1, 1, 1])

    result = getCutout(dataset, "velocity", axes_ranges, strides)
    key = f"velocity_{str(t_range[0]).zfill(4)}"
    field = result[key]  # expected shape: [domain_size, domain_size, domain_size, 3]

    return np.asarray(field, dtype=np.float32)


def make_hr_lr_hrz(hr_field, scale=4, axis_mode=None):
    """
    hr_field: numpy [Nx, Ny, Nz, 3] -> converted to torch [3, Nx, Ny, Nz]
    axis_mode: 0 (XY->Z), 1 (YZ->X), 2 (XZ->Y), or None (randomly sampled)
    """
    hr = torch.from_numpy(hr_field).permute(3, 0, 1, 2).contiguous()  # [3, Nx, Ny, Nz]

    if axis_mode is None:
        axis_mode = random.choice([0, 1, 2])

    perms = {
        0: (0, 1, 2, 3),  # XY -> Z (identity)
        1: (0, 2, 3, 1),  # YZ -> X
        2: (0, 1, 3, 2),  # XZ -> Y
    }
    hr_canonical = hr.permute(*perms[axis_mode]).contiguous()

    hr_batched = hr_canonical.unsqueeze(0)

    lr = F.avg_pool3d(hr_batched, kernel_size=scale, stride=scale).squeeze(0)
    hr_z = F.avg_pool3d(hr_batched, kernel_size=(1, 1, scale),
                         stride=(1, 1, scale)).squeeze(0)

    return {
        "hr": hr_canonical,
        "hr_z": hr_z,
        "lr": lr,
        "axis_mode": axis_mode,
    }


def build_dataset_cache(n_samples, domain_size, scale, seed_offset=0,
                         cache_dir="./data", split="train"):
    """
    Downloads n_samples subcubes and caches them to disk (.pt), so both
    the baseline and the dual model can reuse the exact same samples
    without re-downloading.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = get_dataset()

    for i in range(n_samples):
        seed = seed_offset + i
        cache_file = os.path.join(cache_dir, f"{split}_{i:04d}.pt")
        if os.path.exists(cache_file):
            continue  # already downloaded, skip

        field = download_random_cutout(dataset, domain_size=domain_size, seed=seed)
        hr = torch.from_numpy(field).permute(3, 0, 1, 2).contiguous()
        torch.save(hr, cache_file)
        print(f"[{split}] sample {i+1}/{n_samples} saved to {cache_file}")


class TurbulencePatchDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, split, n_samples, scale=4):
        self.cache_dir = cache_dir
        self.split = split
        self.n_samples = n_samples
        self.scale = scale

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cache_file = os.path.join(self.cache_dir, f"{self.split}_{idx:04d}.pt")
        hr_field = torch.load(cache_file)  # [3, Nx, Ny, Nz]
        return make_hr_lr_hrz(hr_field.permute(1, 2, 3, 0).numpy(),
                               scale=self.scale, axis_mode=None)


if __name__ == "__main__":
    # example usage: build the training/validation cache
    build_dataset_cache(n_samples=512, domain_size=128, scale=4,
                         seed_offset=1000, cache_dir="./data", split="train")
    build_dataset_cache(n_samples=16, domain_size=128, scale=4,
                         seed_offset=2000, cache_dir="./data", split="valid")


class WangCTDataset(torch.utils.data.Dataset):
    """
    Loads Wang et al.'s PEMFC micro-CT HR/LR volumes (single-channel,
    uint8) as an out-of-domain smoke test for the DualEDSR pipeline.
    Crops random patches from the full volume on each __getitem__ call,
    following the same strategy as the original TensorFlow code
    (createTrainingCubes2). NOT used for final turbulence evaluation --
    physical loss terms should be disabled (w_U=1.0 only) since this
    field is porous-material density, not velocity.
    """

    def __init__(self, hr_path, lr_path, scale=4, patch_size_lr=16,
                 n_patches=200):
        hr_np = np.load(hr_path).astype(np.float32) / 255.0
        lr_np = np.load(lr_path).astype(np.float32) / 255.0

        self.hr = torch.from_numpy(hr_np).unsqueeze(0)  # [1, Nx, Ny, Nz]
        self.lr = torch.from_numpy(lr_np).unsqueeze(0)  # [1, Nx/scale, Ny/scale, Nz/scale]

        self.hr_z = F.avg_pool3d(
            self.hr.unsqueeze(0), kernel_size=(1, 1, scale), stride=(1, 1, scale)
        ).squeeze(0)  # [1, Nx, Ny, Nz/scale]

        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.n_patches = n_patches

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        p = self.patch_size_lr
        s = self.scale
        _, Nx, Ny, Nz = self.lr.shape

        x = torch.randint(0, Nx - p, (1,)).item()
        y = torch.randint(0, Ny - p, (1,)).item()
        z = torch.randint(0, Nz - p, (1,)).item()

        lr_patch = self.lr[:, x:x+p, y:y+p, z:z+p]
        hr_z_patch = self.hr_z[:, x*s:(x+p)*s, y*s:(y+p)*s, z:z+p]
        hr_patch = self.hr[:, x*s:(x+p)*s, y*s:(y+p)*s, z*s:(z+p)*s]

        return {"lr": lr_patch, "hr_z": hr_z_patch, "hr": hr_patch, "axis_mode": 0}
