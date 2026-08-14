"""
src/HATDualSR_arch.py

DualHAT: axial-factorized super-resolution using the official HAT
(Hybrid Attention Transformer, Chen et al., CVPR 2023) architecture as
the 2D solver backbone, in place of EDSR2D.

The official HAT forward() only implements the 'pixelshuffle' upsample
path (symmetric powers of 2, or 3). Since our Z stage needs asymmetric
upsampling (scale, 1), we subclass HAT (HATSolver2D) to stop after deep
feature extraction and attach our own upsample tail -- symmetric
PixelShuffle steps for the XY stage, asymmetric nearest+conv steps
(matching EDSR2D's approach) for the Z stage.

Reference: Chen et al., "Activating More Pixels in Image
Super-Resolution Transformer", CVPR 2023.
Official code: https://github.com/XPixelGroup/HAT
"""

import math
import torch
import torch.nn as nn

from external.hat_arch import HAT


class HATSolver2D(nn.Module):
    """
    Wraps the official HAT backbone (shallow + deep feature extraction,
    RHAG groups) and replaces its upsample tail with an axis-aware one,
    matching the same (h_factor, w_factor) convention used in
    DualEDSR_arch.EDSR2D.

    window_size must evenly divide the LR input's H and W -- choose it
    accordingly for your patch/volume size.
    """

    def __init__(self, in_channels=1, embed_dim=60, depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6), window_size=5, upsample_factor=(2, 2)):
        super().__init__()

        # instantiate HAT with upsampler='pixelshuffle' so __init__ builds
        # all deep-feature-extraction submodules; we discard/replace its
        # upsample tail afterwards.
        self.hat = HAT(
            img_size=64,  # only affects patch embed bookkeeping, not used at inference
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            upscale=2,  # placeholder, ignored -- we override the tail below
            upsampler='pixelshuffle',
            resi_connection='1conv',
        )
        # drop HAT's own reconstruction tail; we replace it entirely
        del self.hat.conv_before_upsample
        del self.hat.upsample
        del self.hat.conv_last

        num_feat = embed_dim

        h_factor, w_factor = upsample_factor
        n_steps = {2: 1, 4: 2, 8: 3}.get(h_factor, 1)
        step_factor = (2, 2) if h_factor == w_factor else (2, 1)

        steps = []
        for _ in range(n_steps):
            steps.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1))
            steps.append(nn.ReLU(inplace=True))
            steps.append(nn.Upsample(scale_factor=step_factor, mode="nearest"))
        self.upsample_tail = nn.Sequential(*steps)

        self.conv_out = nn.Conv2d(num_feat, in_channels, kernel_size=3, padding=1)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        hat = self.hat
        hat.mean = hat.mean.type_as(x)
        x = (x - hat.mean) * hat.img_range

        feat = hat.conv_first(x)
        feat = hat.conv_after_body(hat.forward_features(feat)) + feat

        up = self.upsample_tail(feat)
        out = self.output_activation(self.conv_out(up))
        return out


class DualHAT(nn.Module):
    """
    Axial-factorized super-resolution model using HATSolver2D as the
    2D backbone for both stages (in place of DualEDSR's EDSR2D).
    Forward-pass permutation logic is identical to DualEDSR.
    """

    def __init__(self, scale=4, in_channels=1,
                 embed_dim=60, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=5):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels

        # Z stage uses a lighter config (fewer groups/heads), mirroring
        # DualEDSR's "half capacity" convention for the Z solver.
        z_depths = tuple(d // 2 if d // 2 > 0 else 1 for d in depths)
        z_heads = tuple(h // 2 if h // 2 > 0 else 1 for h in num_heads)
        z_embed_dim = max(embed_dim // 2, 16)

        self.srxy = HATSolver2D(
            in_channels=in_channels, embed_dim=embed_dim, depths=depths,
            num_heads=num_heads, window_size=window_size,
            upsample_factor=(scale, scale))

        self.srz = HATSolver2D(
            in_channels=in_channels, embed_dim=z_embed_dim, depths=z_depths,
            num_heads=z_heads, window_size=window_size,
            upsample_factor=(scale, 1))

    def forward(self, lr):
        B, C, Nx, Ny, Nz = lr.shape

        x = lr.permute(0, 4, 1, 2, 3).reshape(B * Nz, C, Nx, Ny)
        x = self.srxy(x)

        Nx2, Ny2 = x.shape[-2], x.shape[-1]
        pred_xy = x.reshape(B, Nz, C, Nx2, Ny2).permute(0, 2, 3, 4, 1)

        y = pred_xy.permute(0, 2, 1, 4, 3).reshape(B * Nx2, C, Nz, Ny2)
        y = self.srz(y)

        Nz2 = y.shape[-2]
        pred_xyz = y.reshape(B, Nx2, C, Nz2, Ny2).permute(0, 2, 1, 4, 3)

        return pred_xy, pred_xyz


if __name__ == "__main__":
    scale = 4
    B, C, Nx, Ny, Nz = 1, 1, 20, 20, 20  # must be divisible by window_size

    model = DualHAT(scale=scale, in_channels=C, window_size=5,
                     embed_dim=60, depths=(6, 6), num_heads=(6, 6))
    lr_dummy = torch.randn(B, C, Nx, Ny, Nz)

    pred_xy, pred_xyz = model(lr_dummy)

    print("Input (LR):      ", lr_dummy.shape)
    print("Stage 1 output:  ", pred_xy.shape,
          "-- expected:", (B, C, Nx * scale, Ny * scale, Nz))
    print("Stage 2 output:  ", pred_xyz.shape,
          "-- expected:", (B, C, Nx * scale, Ny * scale, Nz * scale))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
