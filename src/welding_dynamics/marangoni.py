# -*- coding: utf-8 -*-
"""Fuller reduced Marangoni melt-pool flow model.

This branch adds a compact incompressible flow prototype rather than another
post-processing correction.  It solves a 2D vertical-section streamfunction /
vorticity system with a thermocapillary shear condition on the free surface and
uses the resulting velocity field to advect heat.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class IncompressibleMarangoniFlow2D:
    """2D incompressible melt-pool flow with Marangoni surface shear.

    Coordinates are ``x`` along the weld and ``z`` downward.  The top row
    ``z=0`` is the free surface.  This is not a production CFD solver; it is a
    transparent research prototype suitable for comparing against the current
    conduction-only Goldak temperature field.
    """

    dgamma_dT: float = -4.0e-4  # N/(m K)
    mu: float = 6.0e-3          # Pa s
    rho: float = 7000.0         # kg/m^3
    cp: float = 780.0           # J/(kg K)
    k: float = 30.0             # W/(m K)
    cfl: float = 0.25
    speed_limiter: float = 2.0  # m/s, reduced-model stability guard

    @property
    def nu(self):
        return self.mu / self.rho

    @property
    def alpha(self):
        return self.k / (self.rho * self.cp)

    def initial_state(self, shape, T0=298.0):
        return dict(
            T=np.full(shape, T0, dtype=float),
            omega=np.zeros(shape, dtype=float),
            psi=np.zeros(shape, dtype=float),
            u=np.zeros(shape, dtype=float),
            w=np.zeros(shape, dtype=float),
        )

    def apply_surface_shear(self, omega, T, dx, dz, melt_mask=None):
        """Apply free-surface vorticity from ``mu du/dz = dgamma/dT dT/dx``."""
        dTdx = np.gradient(T[:, 0], dx, edge_order=1)
        omega = omega.copy()
        surface_omega = -self.dgamma_dT * dTdx / (self.mu * max(dz, 1e-12))
        omega[:, 0] = surface_omega
        if omega.shape[1] > 2:
            omega[:, 1] = surface_omega
        if melt_mask is not None:
            omega = np.where(melt_mask, omega, 0.0)
        return omega

    def solve_streamfunction(self, omega, dx, dz, melt_mask=None,
                             iterations=250, tolerance=1e-7):
        """Solve ``laplacian(psi) = -omega`` with no-through-flow walls."""
        psi = np.zeros_like(omega)
        dx2, dz2 = dx**2, dz**2
        denom = 2.0 * (dx2 + dz2)
        active = melt_mask if melt_mask is not None else np.ones_like(omega,
                                                                      dtype=bool)
        for _ in range(iterations):
            old = psi.copy()
            psi[1:-1, 1:-1] = (
                (psi[2:, 1:-1] + psi[:-2, 1:-1]) * dz2
                + (psi[1:-1, 2:] + psi[1:-1, :-2]) * dx2
                + omega[1:-1, 1:-1] * dx2 * dz2
            ) / denom
            psi = np.where(active, psi, 0.0)
            psi[0, :] = psi[-1, :] = 0.0
            psi[:, 0] = psi[:, -1] = 0.0
            if np.max(np.abs(psi - old)) < tolerance:
                break
        return psi

    def velocity_from_streamfunction(self, psi, dx, dz, melt_mask=None):
        """Return velocities ``u = dpsi/dz`` and ``w = -dpsi/dx``."""
        dpsidx, dpsidz = np.gradient(psi, dx, dz, edge_order=1)
        u = dpsidz
        w = -dpsidx
        speed = np.hypot(u, w)
        scale = np.minimum(1.0, self.speed_limiter / np.maximum(speed, 1e-12))
        u *= scale
        w *= scale
        if melt_mask is not None:
            u = np.where(melt_mask, u, 0.0)
            w = np.where(melt_mask, w, 0.0)
        return u, w

    def step(self, state, dx, dz, dt, heat_source=None, melt_mask=None):
        """Advance vorticity, streamfunction, velocity and temperature one step."""
        T = state["T"]
        omega = self.apply_surface_shear(state["omega"], T, dx, dz, melt_mask)
        psi = self.solve_streamfunction(omega, dx, dz, melt_mask=melt_mask)
        u, w = self.velocity_from_streamfunction(psi, dx, dz, melt_mask)

        dTdx, dTdz = np.gradient(T, dx, dz, edge_order=1)
        domdx, domdz = np.gradient(omega, dx, dz, edge_order=1)
        Tp = np.pad(T, 1, mode="edge")
        Op = np.pad(omega, 1, mode="edge")
        lap_T = ((Tp[2:, 1:-1] - 2*T + Tp[:-2, 1:-1]) / dx**2
                 + (Tp[1:-1, 2:] - 2*T + Tp[1:-1, :-2]) / dz**2)
        lap_o = ((Op[2:, 1:-1] - 2*omega + Op[:-2, 1:-1]) / dx**2
                 + (Op[1:-1, 2:] - 2*omega + Op[1:-1, :-2]) / dz**2)

        q = 0.0 if heat_source is None else heat_source
        T_new = T + dt * (self.alpha * lap_T - u*dTdx - w*dTdz
                          + q / (self.rho * self.cp))
        omega_new = omega + dt * (self.nu * lap_o - u*domdx - w*domdz)
        omega_new = self.apply_surface_shear(omega_new, T_new, dx, dz,
                                             melt_mask)

        if melt_mask is not None:
            T_new = np.where(melt_mask, T_new, T)
            omega_new = np.where(melt_mask, omega_new, 0.0)

        return dict(T=T_new, omega=omega_new, psi=psi, u=u, w=w)

    def stable_dt(self, dx, dz, max_speed=1.0):
        h = min(dx, dz)
        dt_diff = 0.2 * h**2 / max(self.alpha, self.nu, 1e-12)
        dt_adv = self.cfl * h / max(max_speed, 1e-12)
        return min(dt_diff, dt_adv)

    def diagnostics(self, state, dx, dz, melt_mask=None):
        speed = np.hypot(state["u"], state["w"])
        active_speed = speed[melt_mask] if melt_mask is not None and np.any(
            melt_mask) else speed.ravel()
        length = max(state["T"].shape[0] * dx, state["T"].shape[1] * dz)
        return dict(
            max_speed=float(active_speed.max()) if active_speed.size else 0.0,
            mean_speed=float(active_speed.mean()) if active_speed.size else 0.0,
            Pe=float(speed.max() * length / max(self.alpha, 1e-12)),
            direction="outward" if self.dgamma_dT < 0.0 else "inward",
        )
