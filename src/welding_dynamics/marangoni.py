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
