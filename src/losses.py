"""
src/losses.py

Physics-informed loss functions for turbulence super-resolution.

Formulation follows NVIDIA PhysicsNeMo's Turbulence Super Resolution
example: a weighted sum of velocity error, velocity-gradient error,
continuity (divergence) residual error, vorticity error, enstrophy
error, and strain-rate error. Default weights (w_U=1.0, w_omega=0.1,
rest=0) match the reference example's configuration, kept here for
comparability with the baseline.

dual_physics_loss() applies this same physics_loss() to BOTH stages
of the DualEDSR model (XY stage and XYZ stage), using per-axis grid
spacing since the intermediate XY-resolved field has an anisotropic
grid (X, Y already at HR resolution; Z still at LR resolution).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Differential operators
# ---------------------------------------------------------------------

def velocity_gradient(u, dx, dy, dz):
    """
    u: [B, 3, Nx, Ny, Nz]  (velocity field, channels = u,v,w)
    dx, dy, dz: grid spacing along each spatial axis (can differ
                between axes, e.g. in the intermediate XY stage where
                Z is still at low resolution).

    Returns: du [B, 3(i), 3(j), Nx, Ny, Nz]  -- d(u_i)/d(x_j)
    """
    spacings = [dx, dy, dz]
    grads = []
    for i in range(3):  # velocity component u_i
        gi = []
        for j, spacing in enumerate(spacings):  # spatial axis x_j
            gi.append(torch.gradient(u[:, i], dim=1 + j, spacing=spacing)[0])
        grads.append(torch.stack(gi, dim=1))
    return torch.stack(grads, dim=1)


def continuity_residual(du):
    """div(u) = du/dx + dv/dy + dw/dz -- trace of the gradient tensor."""
    return du[:, 0, 0] + du[:, 1, 1] + du[:, 2, 2]


def vorticity(du):
    """omega = curl(u)"""
    omega_x = du[:, 2, 1] - du[:, 1, 2]  # dw/dy - dv/dz
    omega_y = du[:, 0, 2] - du[:, 2, 0]  # du/dz - dw/dx
    omega_z = du[:, 1, 0] - du[:, 0, 1]  # dv/dx - du/dy
    return torch.stack([omega_x, omega_y, omega_z], dim=1)


def enstrophy(omega):
    """epsilon = 0.5 * |omega|^2"""
    return 0.5 * (omega ** 2).sum(dim=1)


def strain_rate_magnitude(du):
    """|D| = magnitude of the symmetric strain-rate tensor."""
    D = 0.5 * (du + du.transpose(1, 2))
    return torch.sqrt((D ** 2).sum(dim=(1, 2)) + 1e-8)


# ---------------------------------------------------------------------
# Physics-informed loss (single-stage)
# ---------------------------------------------------------------------

def physics_loss(pred, target, dx=1.0, dy=1.0, dz=1.0,
                  w_U=1.0, w_dU=0.0, w_continuity=0.0,
                  w_omega=0.1, w_enstrophy=0.0, w_strain=0.0):
    """
    Weighted sum of velocity MSE plus optional physics-based terms.
    Default weights match the NVIDIA PhysicsNeMo turbulence
    super-resolution example (w_U=1.0, w_omega=0.1, rest=0).
    """
    loss = w_U * F.mse_loss(pred, target)

    need_grad = (w_dU + w_continuity + w_omega + w_enstrophy + w_strain) > 0
    if need_grad:
        du_pred = velocity_gradient(pred, dx, dy, dz)
        du_true = velocity_gradient(target, dx, dy, dz)

        if w_dU > 0:
            loss = loss + w_dU * F.mse_loss(du_pred, du_true)

        if w_continuity > 0:
            cont_pred = continuity_residual(du_pred)
            cont_true = continuity_residual(du_true)
            loss = loss + w_continuity * F.mse_loss(cont_pred, cont_true)

        need_vort = (w_omega + w_enstrophy) > 0
        if need_vort:
            omega_pred = vorticity(du_pred)
            omega_true = vorticity(du_true)
            if w_omega > 0:
                loss = loss + w_omega * F.mse_loss(omega_pred, omega_true)
            if w_enstrophy > 0:
                enst_pred = enstrophy(omega_pred)
                enst_true = enstrophy(omega_true)
                loss = loss + w_enstrophy * F.mse_loss(enst_pred, enst_true)

        if w_strain > 0:
            strain_pred = strain_rate_magnitude(du_pred)
            strain_true = strain_rate_magnitude(du_true)
            loss = loss + w_strain * F.mse_loss(strain_pred, strain_true)

    return loss


# ---------------------------------------------------------------------
# Dual-stage loss (DualEDSR-specific)
# ---------------------------------------------------------------------

DEFAULT_PHYSICS_WEIGHTS = dict(
    w_U=1.0, w_dU=0.0, w_continuity=0.0,
    w_omega=0.1, w_enstrophy=0.0, w_strain=0.0,
)


def dual_physics_loss(pred_xy, pred_xyz, hr_z, hr,
                       dx_hr=1.0, dy_hr=1.0, dz_hr=1.0, dz_lr=1.0,
                       weights=None):
    """
    pred_xy:  [B, 3, Nx*scale, Ny*scale, Nz]        -- XY resolved, Z still LR
    pred_xyz: [B, 3, Nx*scale, Ny*scale, Nz*scale]  -- fully resolved volume
    hr_z:     intermediate target (X, Y at HR resolution; Z at LR resolution)
    hr:       final target (fully HR)

    dz_lr: grid spacing along Z in the intermediate stage (Z not yet
           upsampled, so spacing is coarser than the final dz_hr).

    Returns: (loss_total, loss_xy, loss_xyz) -- kept separate so both
    training and validation loops can log each stage's contribution
    independently.
    """
    if weights is None:
        weights = DEFAULT_PHYSICS_WEIGHTS

    # Stage 1 (XY): X and Y already at HR spacing, Z still at LR spacing.
    loss_xy = physics_loss(
        pred_xy, hr_z,
        dx=dx_hr, dy=dy_hr, dz=dz_lr,
        **weights
    )

    # Stage 2 (XYZ): all axes at HR spacing.
    loss_xyz = physics_loss(
        pred_xyz, hr,
        dx=dx_hr, dy=dy_hr, dz=dz_hr,
        **weights
    )

    loss_total = loss_xy + loss_xyz
    return loss_total, loss_xy, loss_xyz


if __name__ == "__main__":
    # sanity check with dummy tensors -- no real data required
    B, C = 1, 3
    Nx, Ny, Nz = 128, 128, 32       # intermediate (XY-resolved) shape
    Nz2 = 128                        # fully resolved Z

    pred_xy = torch.randn(B, C, Nx, Ny, Nz, requires_grad=True)
    pred_xyz = torch.randn(B, C, Nx, Ny, Nz2, requires_grad=True)
    hr_z = torch.randn(B, C, Nx, Ny, Nz)
    hr = torch.randn(B, C, Nx, Ny, Nz2)

    loss_total, loss_xy, loss_xyz = dual_physics_loss(
        pred_xy, pred_xyz, hr_z, hr,
        dx_hr=1.0, dy_hr=1.0, dz_hr=1.0, dz_lr=4.0,
    )

    print("loss_xy:   ", loss_xy.item())
    print("loss_xyz:  ", loss_xyz.item())
    print("loss_total:", loss_total.item())

    loss_total.backward()
    print("Backward pass OK -- pred_xy.grad is not None:", pred_xy.grad is not None)
    print("Backward pass OK -- pred_xyz.grad is not None:", pred_xyz.grad is not None)
