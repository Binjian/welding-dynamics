# -*- coding: utf-8 -*-
"""Intermediate thermocapillary / Marangoni surface-flow model.

The model is deliberately separate from the conduction-only heat solvers.  It
captures the first-order coupling missing from the baseline Goldak model:
surface temperature gradients create shear, shear drives a recirculating molten
pool velocity field, and that velocity advects heat on the pool surface.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceMarangoniFlow2D:
    """Reduced 2D surface recirculation driven by thermocapillary shear."""

    dgamma_dT: float = -4.0e-4  # N/(m K)
    mu: float = 6.0e-3          # Pa s
    rho: float = 7000.0         # kg/m^3
    cp: float = 780.0           # J/(kg K)
    k: float = 30.0             # W/(m K)
    depth: float = 3.8e-3       # m, velocity scale length
    speed_limiter: float = 1.5  # m/s, keeps the reduced model conservative

    @property
    def alpha(self):
        return self.k / (self.rho * self.cp)

    def velocity(self, T, dx, dy, melt_mask=None):
        """Return ``u, v`` surface velocities from ``tau = dgamma/dT grad_s T``.

        A streamfunction projection keeps the reduced field divergence-free in
        the surface plane while preserving the thermocapillary flow direction.
        """
        dTdx, dTdy = np.gradient(T, dx, dy, edge_order=1)
        ux = self.dgamma_dT * dTdx * self.depth / self.mu
        uy = self.dgamma_dT * dTdy * self.depth / self.mu

        speed = np.hypot(ux, uy)
        scale = np.minimum(1.0, self.speed_limiter / np.maximum(speed, 1e-12))
        ux *= scale
        uy *= scale

        if melt_mask is not None:
            ux = np.where(melt_mask, ux, 0.0)
            uy = np.where(melt_mask, uy, 0.0)

        # Project to a simple recirculating field by subtracting the mean drift.
        if melt_mask is not None and np.any(melt_mask):
            ux = np.where(melt_mask, ux - np.mean(ux[melt_mask]), 0.0)
            uy = np.where(melt_mask, uy - np.mean(uy[melt_mask]), 0.0)
        else:
            ux -= np.mean(ux)
            uy -= np.mean(uy)
        return ux, uy

    def advect_diffuse_step(self, T, dx, dy, dt, melt_mask=None):
        """Advance a surface temperature slice by one explicit coupled step."""
        u, v = self.velocity(T, dx, dy, melt_mask=melt_mask)

        dTdx, dTdy = np.gradient(T, dx, dy, edge_order=1)
        Tp = np.pad(T, 1, mode="edge")
        lap = ((Tp[2:, 1:-1] - 2*T + Tp[:-2, 1:-1]) / dx**2
               + (Tp[1:-1, 2:] - 2*T + Tp[1:-1, :-2]) / dy**2)
        T_new = T + dt * (self.alpha * lap - u*dTdx - v*dTdy)

        if melt_mask is not None:
            T_new = np.where(melt_mask, T_new, T)
        return T_new, u, v

    def stable_dt(self, dx, dy):
        """Return a conservative explicit time-step estimate."""
        h = min(dx, dy)
        dt_diff = 0.2 * h**2 / max(self.alpha, 1e-12)
        dt_adv = 0.4 * h / max(self.speed_limiter, 1e-12)
        return min(dt_diff, dt_adv)

    def diagnostics(self, T, dx, dy, melt_mask=None):
        u, v = self.velocity(T, dx, dy, melt_mask=melt_mask)
        speed = np.hypot(u, v)
        length = max(T.shape[0] * dx, T.shape[1] * dy)
        return dict(
            max_speed=float(speed.max()),
            mean_speed=float(speed[melt_mask].mean() if melt_mask is not None
                             and np.any(melt_mask) else speed.mean()),
            Pe=float(speed.max() * length / max(self.alpha, 1e-12)),
            direction="outward" if self.dgamma_dT < 0.0 else "inward",
        )
