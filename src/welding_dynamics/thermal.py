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

    熔池对流 (模块 10A 耦合, docs/melt_convection_assessment.md 方案 A):
    - ``convection=None`` (默认): 纯传导, 常系数扩散, 逐位复现 README 结果。
    - ``convection`` 给定 (``EffectiveMarangoniCorrection`` 对象): 池内导热率
      放大 ``alpha_eff()/alpha`` 倍 (糊状区线性过渡), 扩散项改为变系数通量
      形式 (面上调和平均, 保持能量守恒); dt 随最大扩散率缩小, 步数约 ×倍率。
    """

    MUSHY_DT = 100.0    # K  固相 k -> 池内 k_eff 的线性过渡温度宽度 (糊状区)

    def __init__(self, Q=8200.0, eta=0.8, v=8e-3,
                 a=4e-3, b=4e-3, cf=4e-3, cr=9e-3, ff=0.6,
                 Lx=0.10, Ly=0.025, Lz=0.020, dx=1.25e-3,
                 rho=7850.0, cp=600.0, k=41.0, T0=298.0, Tm=1773.0,
                 weave=None, convection=None):
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

        # 空 dict/None 一律视为无对流 (Hydra 的 convection=none 组合出空节点)。
        # 注意 EffectiveMarangoniCorrection 是 dataclass: 经 instantiate 的
        # kwarg 覆盖传入时会被 OmegaConf 重新结构化成 mapping, 这里还原。
        if bool(convection) and not hasattr(convection, "alpha_eff") \
                and hasattr(convection, "keys"):
            from .marangoni import EffectiveMarangoniCorrection
            convection = EffectiveMarangoniCorrection(
                **{k: v for k, v in dict(convection).items()
                   if not k.startswith("_")})
        if bool(convection) and hasattr(convection, "alpha_eff"):
            self.k_pool_mult = float(convection.alpha_eff() / convection.alpha)
        else:
            self.k_pool_mult = 1.0
        self.convecting = self.k_pool_mult > 1.0
        self.convection = convection if self.convecting else None

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

    def _kappa(self, T):
        """相对导热率场: 固相 1, 池内 k_pool_mult, 糊状区线性过渡。"""
        return 1.0 + (self.k_pool_mult - 1.0) * np.clip(
            (T - (self.Tm - self.MUSHY_DT)) / self.MUSHY_DT, 0.0, 1.0)

    def _var_k_div(self, T):
        """变系数扩散: Σ_faces κ_face·(T_nb - T), 面上取调和平均 (通量守恒)。

        κ=1 处处成立时退化为标准 6 点 Laplacian。edge-pad 与常系数路径一致,
        即所有外边界零通量; 远场 Dirichlet 仍由 run() 事后覆盖。
        """
        K = self._kappa(T)
        Tp = np.pad(T, 1, mode="edge")
        Kp = np.pad(K, 1, mode="edge")
        div = np.zeros_like(T)
        core = (slice(1, -1),) * 3
        for ax in range(3):
            for lo in (False, True):
                idx = list(core)
                idx[ax] = slice(None, -2) if lo else slice(2, None)
                Tn, Kn = Tp[tuple(idx)], Kp[tuple(idx)]
                div += 2.0*Kn*K/(Kn + K) * (Tn - T)
        return div

    def _solve(
            self, t_end, x_start, *, frame_times=None,
            snapshot_dtype=None):
        # 显式稳定性; 对流增强时按池内最大扩散率缩小 dt
        dt = 0.4 * self.dx**2 / (6 * self.alpha * self.k_pool_mult)
        n_steps = int(t_end / dt)
        T, dx2 = self.T, self.dx**2
        peak = np.full_like(T, self.T0)               # 记录峰值温度
        snapshots = None
        snapshot_steps = {}
        targets = None
        if frame_times is not None:
            targets = np.asarray(frame_times, dtype=float)
            if targets.ndim != 1:
                raise ValueError("frame_times must be a one-dimensional sequence")
            if not np.all(np.isfinite(targets)):
                raise ValueError("frame_times must contain only finite values")
            if np.any(np.diff(targets) < 0):
                raise ValueError("frame_times must be sorted")
            if np.any(targets < 0) or np.any(targets > float(t_end)):
                raise ValueError("frame_times must lie within [0, t_end]")
            if n_steps <= 0 and np.any(targets > 0):
                raise ValueError("t_end is too short to capture a frame")
            dtype = None if snapshot_dtype is None else np.dtype(
                snapshot_dtype)
            snapshots = [None] * len(targets)
            # T is advanced from t=n*dt to (n+1)*dt. Capture the first
            # post-step state at or after each requested time, except t=0
            # which intentionally records the unadvanced initial state.
            steps = np.ceil(targets / dt).astype(int) - 1
            steps = np.clip(steps, -1, max(n_steps - 1, -1))
            for slot, step in enumerate(steps):
                snapshot_steps.setdefault(int(step), []).append(slot)

            def capture(step):
                for slot in snapshot_steps.get(step, ()):
                    if dtype is None:
                        T_frame, peak_frame = T.copy(), peak.copy()
                    else:
                        T_frame = np.array(T, dtype=dtype, copy=True)
                        peak_frame = np.array(
                            peak, dtype=dtype, copy=True)
                    snapshots[slot] = (
                        float(targets[slot]), T_frame, peak_frame)

            capture(-1)

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
            if self.convecting:
                T = T + dt*(self.alpha*self._var_k_div(T)/dx2
                            + q/(self.rho*self.cp))
            else:
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
            if snapshots is not None:
                capture(n)
        self.T, self.peak, self.xs_end, self.ys_end = T, peak, xs, ys
        return T, snapshots

    def run(self, t_end=5.0, x_start=0.015):
        """Advance to ``t_end`` and retain the final and peak fields."""

        T, _ = self._solve(t_end, x_start)
        return T

    def run_with_snapshots(
            self, t_end=5.0, x_start=0.015, *, frame_times,
            snapshot_dtype=None):
        """Advance once and retain fields at requested animation times.

        Each returned tuple is ``(requested_time, T, peak)``. ``t=0``
        records the unadvanced initial state; later frames use the first
        completed integration step at or after the requested time. A target
        beyond the truncated integration horizon is mapped to the final
        completed step. The final full-precision ``T`` and ``peak`` remain
        available on this instance, exactly as after :meth:`run`.

        ``snapshot_dtype=np.float32`` is useful for visualization caches while
        keeping the numerical integration and retained final fields in their
        original precision.
        """

        _, snapshots = self._solve(
            t_end,
            x_start,
            frame_times=frame_times,
            snapshot_dtype=snapshot_dtype,
        )
        return snapshots

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
