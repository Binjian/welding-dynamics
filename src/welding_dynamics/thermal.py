# -*- coding: utf-8 -*-
import numpy as np
"""模块 2 & 4: Rosenthal 解析解 与 Goldak 双椭球 + 3D 瞬态 FDM"""


class RosenthalThermal:
    def __init__(self, Q=8200.0, eta=0.8, v=8e-3,
                 k=41.0, alpha=8.7e-6, T0=298.0, Tm=1773.0):
        self.Q, self.eta, self.v = Q, eta, v
        self.k, self.alpha, self.T0, self.Tm = k, alpha, T0, Tm

    def temperature(self, xi, y, z):
        R = np.maximum(np.sqrt(xi**2 + y**2 + z**2), 1e-5)
        return self.T0 + self.eta * self.Q / (2*np.pi*self.k*R) * np.exp(
            -self.v * (R + xi) / (2 * self.alpha))

    def surface_field(self, xlim=(-0.05, 0.015), ylim=(-0.015, 0.015), n=400):
        xi = np.linspace(*xlim, n); y = np.linspace(*ylim, n)
        XI, Y = np.meshgrid(xi, y)
        return XI, Y, self.temperature(XI, Y, 0.0)


class GoldakFDM:
    """rho*c*dT/dt = k * laplacian(T) + q_goldak(x,y,z,t)
    半对称模型 (y>=0, y=0 为对称面), 显式差分。
    """

    def __init__(self, Q=8200.0, eta=0.8, v=8e-3,
                 a=4e-3, b=4e-3, cf=4e-3, cr=9e-3, ff=0.6,
                 Lx=0.10, Ly=0.025, Lz=0.020, dx=1.25e-3,
                 rho=7850.0, cp=600.0, k=41.0, T0=298.0, Tm=1773.0):
        self.Q, self.eta, self.v = Q, eta, v
        self.a, self.b, self.cf, self.cr = a, b, cf, cr
        self.ff, self.fr = ff, 2.0 - ff
        self.rho, self.cp, self.k, self.T0, self.Tm = rho, cp, k, T0, Tm
        self.alpha = k / (rho * cp)
        self.dx = dx
        self.Nx, self.Ny, self.Nz = (int(Lx/dx), int(Ly/dx), int(Lz/dx))
        self.x = np.arange(self.Nx) * dx
        self.y = np.arange(self.Ny) * dx
        self.z = np.arange(self.Nz) * dx
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,
                                             indexing="ij")
        self.T = np.full((self.Nx, self.Ny, self.Nz), T0)

    def goldak_q(self, xs):
        """体积热源功率密度 [W/m^3], 热源中心位于 (xs, 0, 0)"""
        xi = self.X - xs
        c = np.where(xi >= 0, self.cf, self.cr)
        f = np.where(xi >= 0, self.ff, self.fr)
        coef = 6*np.sqrt(3)*f*self.eta*self.Q / (self.a*self.b*c*np.pi**1.5)
        return coef * np.exp(-3*(xi/c)**2 - 3*(self.Y/self.a)**2
                             - 3*(self.Z/self.b)**2)

    def run(self, t_end=5.0, x_start=0.015):
        dt = 0.4 * self.dx**2 / (6 * self.alpha)      # 显式稳定性
        n_steps = int(t_end / dt)
        T, dx2 = self.T, self.dx**2
        peak = np.full_like(T, self.T0)               # 记录峰值温度
        P_target = self.eta * self.Q / 2.0            # 半模型应吸收功率
        for n in range(n_steps):
            xs = x_start + self.v * n * dt
            q = self.goldak_q(xs)
            q *= P_target / max(q.sum() * self.dx**3, 1e-9)  # 数值重归一化
            # edge-pad => 所有边界零通量(Neumann), y=0 即对称面
            Tp = np.pad(T, 1, mode="edge")
            lap = (Tp[2:, 1:-1, 1:-1] + Tp[:-2, 1:-1, 1:-1]
                   + Tp[1:-1, 2:, 1:-1] + Tp[1:-1, :-2, 1:-1]
                   + Tp[1:-1, 1:-1, 2:] + Tp[1:-1, 1:-1, :-2] - 6*T)
            T = T + dt*(self.alpha*lap/dx2 + q/(self.rho*self.cp))
            # 远场边界 Dirichlet (大件散热)
            T[0] = T[-1] = self.T0
            T[:, -1] = self.T0
            T[:, :, -1] = self.T0
            peak = np.maximum(peak, T)
        self.T, self.peak, self.xs_end = T, peak, xs
        return T

    def pool_size(self):
        melt = self.T >= self.Tm
        if not melt.any():
            return 0, 0, 0
        ix, iy, iz = np.where(melt)
        L = (ix.max()-ix.min())*self.dx*1e3
        W = 2*(iy.max())*self.dx*1e3            # 半模型 -> 全宽
        D = (iz.max())*self.dx*1e3
        return L, W, D
