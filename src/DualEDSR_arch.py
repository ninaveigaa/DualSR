"""
DualEDSR_arch.py
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block matching Wang et al.'s res_block_EDSR exactly:
    conv -> ReLU -> conv, plain (unscaled) skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual


class EDSR2D(nn.Module):
    """
    Generic 2D EDSR-style super-resolution sub-network. Used identically
    for both the XY stage (symmetric upsample) and the Z stage
    (asymmetric upsample) -- only the upsample_factor argument differs.

    upsample_factor: tuple (height_factor, width_factor)
        (scale, scale) -> symmetric upsampling (XY stage), via PixelShuffle.
        (scale, 1)     -> asymmetric upsampling (Z stage): only the
                          height axis is upsampled; the width axis is
                          kept as spatial context, not enlarged again.
    """

    def __init__(self, in_channels=3, base_filters=32, n_res_blocks=4,
                 upsample_factor=(2, 2)):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.conv_in = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_filters) for _ in range(n_res_blocks)]
        )
        self.conv_mid = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)

        # Matches Wang et al.'s upsampleEDSR / upsampleEDSR1D exactly:
        # for scale=4, apply TWO successive blocks of
        # [conv(F->F, 3x3) -> ReLU -> UpSampling(nearest, factor_per_step)],
        # rather than a single PixelShuffle-based step. Each step's factor
        # is (2,2) for the symmetric (XY) case or (2,1) for the asymmetric
        # (Z) case, matching upsample_factor exactly.
        h_factor, w_factor = upsample_factor
        n_steps = {2: 1, 4: 2, 8: 3}.get(h_factor, 1)
        step_factor = (2, 2) if h_factor == w_factor else (2, 1)

        steps = []
        for _ in range(n_steps):
            steps.append(nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1))
            steps.append(nn.ReLU(inplace=True))
            steps.append(nn.Upsample(scale_factor=step_factor, mode="nearest"))
        self.upsample = nn.Sequential(*steps)

        self.conv_out = nn.Conv2d(base_filters, in_channels, kernel_size=3, padding=1)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        feat = self.conv_in(x)
        res = self.res_blocks(feat)
        res = self.conv_mid(res) + feat  # long skip connection (EDSR-style)
        up = self.upsample(res)
        return self.output_activation(self.conv_out(up))


class DualEDSR(nn.Module):
    """
    Full axial-factorized super-resolution model.

    Input:  lr  [B, 3, Nx, Ny, Nz]  (low resolution, all axes downsampled)
    Output: pred_xy  [B, 3, Nx*scale, Ny*scale, Nz]
                (XY resolved, Z still at LR resolution -- matches the
                "hr_z" intermediate target produced by dataset.py)
            pred_xyz [B, 3, Nx*scale, Ny*scale, Nz*scale]
                (fully resolved volume -- matches the "hr" target)

    Axis convention: this model always assumes the LAST spatial axis
    (Z, in the tensor's canonical orientation) is the one resolved in
    Stage 2. dataset.py's axis-permutation logic (axis_mode) is
    responsible for mapping the physically sampled axis onto this
    canonical convention before the tensor reaches the model.
    """

    def __init__(self, scale=4, base_filters=64, n_res_blocks=16, in_channels=3,
                 symmetric_z=False):
        """
        Matches the original TensorFlow DualEDSR (Wang et al.) exactly
        when symmetric_z=False (the default): the XY stage uses the
        full base_filters/n_res_blocks, while the Z stage uses HALF of
        both (fewer filters, fewer residual blocks), since it only
        needs to resolve the remaining axis using the already-upsampled
        Y neighborhood as spatial context.

        Defaults (base_filters=64, n_res_blocks=16) match the original
        code's argument defaults (args.ngsrf=64, args.numResBlocks=16),
        giving SRz base_filters=32, n_res_blocks=8 when symmetric_z=False.

        symmetric_z=True: gives the Z stage the SAME capacity as the XY
        stage (base_filters, n_res_blocks unchanged, no //2). This is a
        deliberate deviation from Wang et al.'s original design, added
        to test whether the asymmetric-capacity split is responsible
        for the disproportionate PSNR drop observed between the XY-only
        stage and the final XYZ output (see project logbook, Milestone
        16/17 diagnostic: val_psnr_xy ~29.7-30.0 dB vs. val_psnr_xyz
        ~26.5-27.8 dB, a 2.2-3.1 dB drop attributed to the Z stage).
        Roughly doubles parameter count and compute cost relative to
        symmetric_z=False.
        """
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.symmetric_z = symmetric_z

        if symmetric_z:
            srz_filters = base_filters
            srz_blocks = n_res_blocks
        else:
            srz_filters = base_filters // 2
            srz_blocks = max(n_res_blocks // 2, 1)

        self.srxy = EDSR2D(in_channels=in_channels, base_filters=base_filters,
                            n_res_blocks=n_res_blocks,
                            upsample_factor=(scale, scale))

        self.srz = EDSR2D(in_channels=in_channels, base_filters=srz_filters,
                           n_res_blocks=srz_blocks,
                           upsample_factor=(scale, 1))

    def forward(self, lr):
        B, C, Nx, Ny, Nz = lr.shape

        # ---- Stage 1: resolve X and Y jointly, Z as batch ----
        # [B,C,Nx,Ny,Nz] -> [B,Nz,C,Nx,Ny] -> [(B*Nz),C,Nx,Ny]
        x = lr.permute(0, 4, 1, 2, 3).reshape(B * Nz, C, Nx, Ny)
        x = self.srxy(x)  # [(B*Nz), C, Nx*scale, Ny*scale]

        Nx2, Ny2 = x.shape[-2], x.shape[-1]
        # back to volume shape: [(B*Nz),C,Nx2,Ny2] -> [B,Nz,C,Nx2,Ny2] -> [B,C,Nx2,Ny2,Nz]
        pred_xy = x.reshape(B, Nz, C, Nx2, Ny2).permute(0, 2, 3, 4, 1)

        # ---- Stage 2: resolve Z, X as batch, Y as spatial context ----
        # [B,C,Nx2,Ny2,Nz] -> [B,Nx2,C,Nz,Ny2] -> [(B*Nx2),C,Nz,Ny2]
        y = pred_xy.permute(0, 2, 1, 4, 3).reshape(B * Nx2, C, Nz, Ny2)
        y = self.srz(y)  # [(B*Nx2), C, Nz*scale, Ny2]

        Nz2 = y.shape[-2]
        # back to volume shape: [(B*Nx2),C,Nz2,Ny2] -> [B,Nx2,C,Nz2,Ny2] -> [B,C,Nx2,Ny2,Nz2]
        pred_xyz = y.reshape(B, Nx2, C, Nz2, Ny2).permute(0, 2, 1, 4, 3)

        return pred_xy, pred_xyz


if __name__ == "__main__":
    # sanity check with dummy tensors -- no real data required
    scale = 4
    B, C, Nx, Ny, Nz = 1, 3, 32, 32, 32  # LR input size

    for symmetric_z in (False, True):
        model = DualEDSR(scale=scale, base_filters=64, n_res_blocks=16,
                          in_channels=C, symmetric_z=symmetric_z)
        lr_dummy = torch.randn(B, C, Nx, Ny, Nz)
        pred_xy, pred_xyz = model(lr_dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"symmetric_z={symmetric_z}: pred_xy={tuple(pred_xy.shape)}, "
              f"pred_xyz={tuple(pred_xyz.shape)}, params={n_params:,}")
