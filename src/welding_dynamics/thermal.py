# -*- coding: utf-8 -*-
import warnings

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
    """rho*c*dT/dt = k * laplacian(T) + q_goldak(x,y,z,t), 显式差分。

    网格随是否摆动切换:
    - ``weave=None`` (默认): 半对称模型 (y>=0, y=0 为对称面), Ny = Ly/dx。
    - ``weave`` 给定: 焊枪横向摆动破坏 y=0 镜像对称性, 改用全宽网格
      (y ∈ [-Ly, Ly], Ny = 2*Ly/dx - 1, y=0 落在单元中心), 计算量约翻倍。

    ``Ly`` 始终是**半宽** (自焊缝中心线到远场边界的距离)。
    """

    def __init__(self, Q=8200.0, eta=0.8, v=8e-3,
                 a=4e-3, b=4e-3, cf=4e-3, cr=9e-3, ff=0.6,
                 Lx=0.10, Ly=0.025, Lz=0.020, dx=1.25e-3,
                 rho=7850.0, cp=600.0, k=41.0, T0=298.0, Tm=1773.0,
                 weave=None):
        self.Q, self.eta, self.v = Q, eta, v
        self.a, self.b, self.cf, self.cr = a, b, cf, cr
        self.ff, self.fr = ff, 2.0 - ff
        self.rho, self.cp, self.k, self.T0, self.Tm = rho, cp, k, T0, Tm
        self.alpha = k / (rho * cp)
        self.dx = dx

        # 空 dict/None/零摆幅一律视为无摆动 (Hydra 的 weave=none 组合出空节点)
        self.weaving = bool(weave) and getattr(weave, "amplitude_m", 0.0) > 0
        self.weave = weave if self.weaving else None
        self.symmetric = not self.weaving

        self.Nx, self.Nz = int(Lx/dx), int(Lz/dx)
        ny_half = int(Ly/dx)
        if self.symmetric:
            self.Ny = ny_half
            self.y = np.arange(self.Ny) * dx          # 0 .. Ly-dx
            self.j_center = 0                         # y=0 即对称面
        else:
            self.Ny = 2*ny_half - 1                   # 关于 y=0 严格镜像
            self.y = (np.arange(self.Ny) - (ny_half - 1)) * dx
            self.j_center = ny_half - 1               # y=0 所在的 j 下标
            half_travel = 0.5*weave.amplitude_m
            if half_travel + 3*self.a > Ly:           # 摆动行程 + 热源尾迹撞上远场
                warnings.warn(
                    f"摆幅半行程 {half_travel*1e3:.1f} mm 加热源半宽 {3*a*1e3:.1f} mm "
                    f"已接近远场边界 Ly={Ly*1e3:.1f} mm; 请增大 solver.Ly。",
                    stacklevel=2)

        self.x = np.arange(self.Nx) * dx
        self.z = np.arange(self.Nz) * dx
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,
                                             indexing="ij")
        self.T = np.full((self.Nx, self.Ny, self.Nz), T0)

    def goldak_q(self, xs, ys=0.0):
        """体积热源功率密度 [W/m^3], 热源中心位于 (xs, ys, 0)"""
        xi = self.X - xs
        yi = self.Y - ys
        c = np.where(xi >= 0, self.cf, self.cr)
        f = np.where(xi >= 0, self.ff, self.fr)
        coef = 6*np.sqrt(3)*f*self.eta*self.Q / (self.a*self.b*c*np.pi**1.5)
        return coef * np.exp(-3*(xi/c)**2 - 3*(yi/self.a)**2
                             - 3*(self.Z/self.b)**2)

    def run(self, t_end=5.0, x_start=0.015):
        dt = 0.4 * self.dx**2 / (6 * self.alpha)      # 显式稳定性
        n_steps = int(t_end / dt)
        T, dx2 = self.T, self.dx**2
        peak = np.full_like(T, self.T0)               # 记录峰值温度
        # 半模型只含物理热源的一半; 全宽模型含全部
        P_target = self.eta * self.Q * (0.5 if self.symmetric else 1.0)
        xs = ys = 0.0
        for n in range(n_steps):
            t = n * dt
            dxo, dyo = self.weave.offset(t) if self.weaving else (0.0, 0.0)
            xs, ys = x_start + self.v*t + dxo, dyo
            q = self.goldak_q(xs, ys)
            q *= P_target / max(q.sum() * self.dx**3, 1e-9)  # 数值重归一化
            # edge-pad => 所有边界零通量(Neumann); 半模型下 y=0 即对称面
            Tp = np.pad(T, 1, mode="edge")
            lap = (Tp[2:, 1:-1, 1:-1] + Tp[:-2, 1:-1, 1:-1]
                   + Tp[1:-1, 2:, 1:-1] + Tp[1:-1, :-2, 1:-1]
                   + Tp[1:-1, 1:-1, 2:] + Tp[1:-1, 1:-1, :-2] - 6*T)
            T = T + dt*(self.alpha*lap/dx2 + q/(self.rho*self.cp))
            # 远场边界 Dirichlet (大件散热)
            T[0] = T[-1] = self.T0
            T[:, -1] = self.T0
            if not self.symmetric:                    # 全宽模型的 -y 远场
                T[:, 0] = self.T0
            T[:, :, -1] = self.T0
            peak = np.maximum(peak, T)
        self.T, self.peak, self.xs_end, self.ys_end = T, peak, xs, ys
        return T

    def pool_size(self):
        melt = self.T >= self.Tm
        if not melt.any():
            return 0, 0, 0
        ix, iy, iz = np.where(melt)
        L = (ix.max()-ix.min())*self.dx*1e3
        if self.symmetric:
            W = 2*(iy.max())*self.dx*1e3        # 半模型 -> 全宽
        else:
            W = (iy.max()-iy.min())*self.dx*1e3
        D = (iz.max())*self.dx*1e3
        return L, W, D
