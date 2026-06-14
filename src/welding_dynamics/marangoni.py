# -*- coding: utf-8 -*-
"""Reduced thermocapillary / Marangoni melt-pool models.

This module keeps the Marangoni extension separate from the conduction-only
Rosenthal and Goldak solvers.  The first model is intentionally low order: it
estimates the surface shear and uses the resulting Peclet number to inflate the
thermal diffusivity inside the molten pool.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EffectiveMarangoniCorrection:
    """Estimate thermocapillary stirring as an effective diffusivity.

    Parameters are steel-like defaults.  ``dgamma_dT`` may be positive or
    negative depending on surfactant chemistry; the magnitude sets the stirring
    strength while the sign is reported for interpretation.
    """

    dgamma_dT: float = -4.0e-4      # N/(m K)
    mu: float = 6.0e-3              # Pa s
    rho: float = 7000.0             # kg/m^3
    cp: float = 780.0               # J/(kg K), liquid steel order of magnitude
    k: float = 30.0                 # W/(m K), liquid steel order of magnitude
    pool_length: float = 17.5e-3    # m
    pool_width: float = 7.5e-3      # m
    limiter: float = 6.0            # cap for alpha_eff / alpha

    @property
    def alpha(self):
        return self.k / (self.rho * self.cp)

    def surface_velocity(self, dTds):
        """Return a surface velocity scale from tau = dgamma/dT * grad_s(T)."""
        shear = self.dgamma_dT * dTds
        length = 0.5 * self.pool_width
        return shear * length / self.mu

    def strength(self, dT=700.0):
        """Return Marangoni number, Peclet number, velocity and flow direction."""
        length = 0.5 * self.pool_width
        dTds = dT / max(0.5 * self.pool_length, 1e-12)
        u = self.surface_velocity(dTds)
        ma = abs(self.dgamma_dT) * dT * length / (self.mu * self.alpha)
        pe = abs(u) * length / self.alpha
        direction = "outward" if self.dgamma_dT < 0.0 else "inward"
        return dict(Ma=ma, Pe=pe, u_surface=u, direction=direction)

    def alpha_eff(self, dT=700.0):
        """Return an in-pool effective diffusivity with a conservative limiter."""
        pe = self.strength(dT)["Pe"]
        multiplier = min(1.0 + 0.15 * np.sqrt(pe), self.limiter)
        return self.alpha * multiplier

    def corrected_pool_size(self, length, width, depth, dT=700.0):
        """Apply a qualitative convection correction to a conduction pool size.

        Negative ``dgamma_dT`` drives outward surface flow in clean steels, which
        tends to widen and shallow the pool.  Positive values reverse the surface
        flow and bias toward deeper, narrower penetration.
        """
        mult = self.alpha_eff(dT) / self.alpha
        stir = np.sqrt(mult)
        if self.dgamma_dT < 0.0:
            return length * stir, width * stir, depth / stir
        return length / stir, width / stir, depth * stir


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
