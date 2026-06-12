# -*- coding: utf-8 -*-
"""
工业焊接动力学模型 — 完整版 (GMAW / MIG)
==========================================
模块 1: 电弧-熔化自调节动力学 (集中参数 ODE)
模块 2: Rosenthal 三维移动点热源解析解
模块 3: 熔滴过渡动力学 — 静力平衡 + 电磁收缩(pinch)不稳定性
         预测熔滴尺寸/过渡频率, 复现 滴状->喷射 过渡电流
模块 4: Goldak 双椭球热源 + 三维瞬态有限差分(FDM)数值解
模块 5: 短路过渡 与 CMT(冷金属过渡) 混杂(hybrid)动力学模型
         电弧相/短路相状态机 + 液桥缩颈 + 送丝回抽

运行: python welding_dynamics_full.py
依赖: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
MU0 = 4e-7 * np.pi


# =====================================================================
# 模块 1: GMAW 电弧-熔化自调节动力学
# =====================================================================
class GMAWDynamics:
    """状态 x=[s, I]: 干伸长 s [m], 电流 I [A]"""

    def __init__(self):
        self.Voc, self.Rs, self.Rl, self.Ls = 32.0, 0.004, 0.004, 3.0e-4
        self.V0, self.Ea, self.Ra = 15.5, 800.0, 0.022
        self.rw = 0.6e-3
        self.k1, self.k2, self.rho_r = 3.0e-4, 5.0e-5, 0.25

    def melting_rate(self, s, I):
        return self.k1 * I + self.k2 * s * I ** 2

    def arc_voltage(self, la, I):
        return self.V0 + self.Ea * la + self.Ra * I

    def rhs(self, t, x, WFS_fun, CTWD_fun):
        s, I = x
        la = max(CTWD_fun(t) - s, 1e-4)
        ds = WFS_fun(t) - self.melting_rate(s, I)
        dI = (self.Voc - (self.Rs + self.Rl + self.rho_r * s) * I
              - self.arc_voltage(la, I)) / self.Ls
        return [ds, dI]

    def simulate(self, t_end=1.0, x0=(6e-3, 150.0),
                 WFS_fun=None, CTWD_fun=None):
        WFS_fun = WFS_fun or (lambda t: 0.12)
        CTWD_fun = CTWD_fun or (lambda t: 0.018 + (0.003 if t >= 0.5 else 0))
        sol = solve_ivp(self.rhs, (0, t_end), x0, args=(WFS_fun, CTWD_fun),
                        method="LSODA", max_step=1e-3, dense_output=True)
        t = np.linspace(0, t_end, 2000)
        s, I = sol.sol(t)
        CTWD = np.array([CTWD_fun(ti) for ti in t])
        la = CTWD - s
        Va = self.arc_voltage(la, I)
        return dict(t=t, s=s, I=I, la=la, Va=Va, P=Va * I)


# =====================================================================
# 模块 2: Rosenthal 三维准稳态解析解
# =====================================================================
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


# =====================================================================
# 模块 3: 熔滴过渡动力学 (静力平衡 SFBT + pinch 不稳定性)
# =====================================================================
class DropletDynamics:
    """熔滴在焊丝端部生长, 受力平衡破坏时脱落。

    保持力:  表面张力 F_gamma = 2*pi*rw*gamma
    脱落力:  重力 F_g + 电磁力(Lorentz/pinch) F_em + 等离子拖拽 F_d
    F_em 采用 Amson 简化式: (mu0 I^2 / 4pi) * ln(r_d / r_w)
    高电流下 F_em ~ I^2 主导 -> pinch 不稳定 -> 小熔滴高频喷射过渡。
    """

    def __init__(self, rw=0.6e-3, gamma=1.2, rho=7000.0,
                 k1=3.0e-4, k2=5.0e-5, s=6e-3,
                 rho_p=0.06, v_p=100.0, Cd=0.44):
        self.rw, self.gamma, self.rho = rw, gamma, rho
        self.k1, self.k2, self.s = k1, k2, s
        self.rho_p, self.v_p, self.Cd = rho_p, v_p, Cd   # 等离子流参数

    # ---- 各项作用力 ----
    def F_gamma(self):
        return 2 * np.pi * self.rw * self.gamma

    def F_em(self, I, rd):
        """Lorentz/pinch 力, Amson 简化式 + 高电流锥化(taper)增强项:
        电流增大时电弧爬升包络熔滴、焊丝端部锥化, 几何因子增大。"""
        ln = np.log(max(rd / self.rw, 1.0))
        geom = ln + 0.5 * (I / 250.0) ** 2
        return MU0 * I**2 / (4*np.pi) * geom

    def F_drag(self, rd):
        A = np.pi * max(rd**2 - self.rw**2, 0.0)
        return 0.5 * self.Cd * self.rho_p * self.v_p**2 * A

    # ---- 单电流工况仿真: 生长-脱落循环 ----
    def simulate(self, I, t_end=0.3, dt=2e-6):
        MR = self.k1*I + self.k2*self.s*I**2          # 熔化速率 [m/s]
        dVdt = MR * np.pi * self.rw**2                # 体积增长率
        rd = self.rw * 1.05
        t, events, hist_t, hist_r = 0.0, [], [], []
        while t < t_end:
            V = 4/3*np.pi*rd**3 + dVdt*dt
            rd = (3*V/(4*np.pi))**(1/3)
            m = self.rho * 4/3*np.pi*rd**3
            Fdet = m*9.81 + self.F_em(I, rd) + self.F_drag(rd)
            if Fdet >= self.F_gamma():
                events.append((t, rd))
                rd = self.rw * 1.05                   # 脱落后重新生长
            t += dt
            if len(hist_t) < 4000:
                hist_t.append(t); hist_r.append(rd)
        if len(events) > 1:
            freq = (len(events)-1) / (events[-1][0] - events[0][0])
            d_mean = 2*np.mean([r for _, r in events])
        else:
            freq, d_mean = 0.0, 2*rd
        return dict(freq=freq, d=d_mean, t=np.array(hist_t),
                    rd=np.array(hist_r))

    # ---- 电流扫描: 过渡模式图 ----
    def current_sweep(self, I_arr):
        f, d = [], []
        for I in I_arr:
            r = self.simulate(I)
            f.append(r["freq"]); d.append(r["d"])
        return np.array(f), np.array(d)


# =====================================================================
# 模块 4: Goldak 双椭球热源 + 三维瞬态 FDM
# =====================================================================
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


# =====================================================================
# 模块 5: 短路过渡 / CMT 混杂动力学 (电弧相 <-> 短路相 状态机)
# =====================================================================
class ShortCircuitGMAW:
    """低电压短路过渡。状态: [s, I, r_b, phase]
    电弧相: 熔化 < 送丝 -> 弧长缩短 -> 熔滴接触熔池 -> 短路
    短路相: 电弧熄灭, 电流上升, 液桥受 pinch+表面张力 缩颈 -> 断裂重燃
    CMT 模式: 短路时电流降至背景值并机械回抽焊丝, 靠表面张力过渡。
    """

    def __init__(self, cmt=False):
        self.cmt = cmt
        self.Voc, self.Rtot, self.Ls = 19.0, 0.012, 1.5e-4
        self.V0, self.Ea, self.Ra = 14.0, 800.0, 0.02
        self.k1, self.k2 = 3.0e-4, 5.0e-5
        self.WFS, self.CTWD = 0.080, 0.012
        self.rw = 0.6e-3
        self.la_short = 0.4e-3          # 弧长低于此 -> 短路
        self.rb0 = 0.75 * self.rw       # 液桥初始颈缩半径
        self.rb_min = 0.12 * self.rw    # 断桥半径
        self.A_gam, self.B_pinch = 0.045, 1.6e-6   # 缩颈速率系数
        self.drop_len = 0.8e-3          # 每次过渡转移的焊丝长度
        # CMT 控制参数
        self.I_bg, self.I_boost = 45.0, 160.0
        self.WFS_retract = -0.10

    def simulate(self, t_end=0.12, dt=2e-6):
        n = int(t_end/dt)
        s, I, rb = 10.6e-3, 120.0, self.rb0
        phase = 0                                    # 0=arc, 1=short
        out = np.zeros((n, 4))                       # t, I, V, phase
        for i in range(n):
            t = i*dt
            la = self.CTWD - s
            if phase == 0:                           # ---- 电弧相 ----
                Va = self.V0 + self.Ea*max(la, 0) + self.Ra*I
                if self.cmt:                         # CMT: 电流分段控制
                    I_ref = self.I_boost if la > 0.8e-3 else self.I_bg
                    I += (I_ref - I)/2e-4 * dt       # 快速电流环
                else:
                    I += (self.Voc - self.Rtot*I - Va)/self.Ls * dt
                s += (self.WFS - (self.k1*I + self.k2*s*I**2)) * dt
                if la <= self.la_short:              # 熔滴触池 -> 短路
                    phase, rb = 1, self.rb0
            else:                                    # ---- 短路相 ----
                Rb = 0.004 * (self.rw / max(rb, 1e-5)) ** 2   # 液桥电阻
                Va = Rb * I
                if self.cmt:
                    I += (self.I_bg - I)/2e-4 * dt   # CMT: 压低短路电流
                    wfs = self.WFS_retract           # 机械回抽
                    neck = self.A_gam + 0.06         # 回抽加速缩颈
                else:
                    I += (self.Voc - self.Rtot*I - Va)/self.Ls * dt
                    wfs = self.WFS
                    neck = self.A_gam + self.B_pinch*I**2
                s += wfs * dt
                rb -= neck * dt
                if rb <= self.rb_min:                # 断桥 -> 电弧重燃
                    phase = 0
                    s -= self.drop_len               # 熔滴并入熔池
                    I = min(I, 250.0)
            out[i] = (t, I, Va, phase)
        return out


# =====================================================================
# 主程序
# =====================================================================
def main():
    # ---------- 模块 1 ----------
    res = GMAWDynamics().simulate()
    P_ss = float(np.mean(res["P"][1800:]))
    print(f"[1 自调节] 稳态 I={res['I'][-1]:.0f} A, "
          f"Va={res['Va'][-1]:.1f} V, P={P_ss:.0f} W")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.suptitle("Module 1: GMAW self-regulation (CTWD step @0.5 s)")
    for ax, key, lab, c in zip(axes.flat,
                               ["I", "la", "s", "Va"],
                               ["I [A]", "arc length [mm]",
                                "stick-out [mm]", "V_arc [V]"],
                               "brgm"):
        ysc = 1e3 if key in ("la", "s") else 1
        ax.plot(res["t"], res[key]*ysc, c); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.axvline(0.5, color="k", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig("m1_self_regulation.png", dpi=140)

    # ---------- 模块 3: 熔滴过渡 ----------
    dd = DropletDynamics()
    I_arr = np.linspace(120, 340, 23)
    freq, dmean = dd.current_sweep(I_arr)
    # 过渡电流: 熔滴直径降到约 1.3 倍焊丝直径以下 -> 喷射过渡
    below = np.where(dmean <= 1.3 * 2 * dd.rw)[0]
    i_tr = below[0] if len(below) else len(I_arr) - 1
    print(f"[3 熔滴] 滴状->喷射 过渡电流 ~ {I_arr[i_tr]:.0f} A")

    r_glob = dd.simulate(170, t_end=0.25)
    r_spray = dd.simulate(300, t_end=0.05)

    fig3, ax3 = plt.subplots(1, 3, figsize=(13, 4))
    ax3[0].plot(I_arr, dmean*1e3, "o-")
    ax3[0].axhline(2*dd.rw*1e3, color="gray", ls=":", label="wire dia.")
    ax3[0].axvline(I_arr[i_tr], color="r", ls="--", label="transition I")
    ax3[0].set_xlabel("I [A]"); ax3[0].set_ylabel("droplet dia. [mm]")
    ax3[0].legend(); ax3[0].set_title("Droplet size vs current")
    ax3[1].plot(I_arr, freq, "s-", color="purple")
    ax3[1].set_xlabel("I [A]"); ax3[1].set_ylabel("detach freq [Hz]")
    ax3[1].set_title("Transfer frequency")
    ax3[2].plot(r_glob["t"]*1e3, r_glob["rd"]*1e3, label="170 A (globular)")
    ax3[2].plot(r_spray["t"]*1e3, r_spray["rd"]*1e3,
                label="300 A (spray)")
    ax3[2].set_xlabel("t [ms]"); ax3[2].set_ylabel("droplet radius [mm]")
    ax3[2].legend(); ax3[2].set_title("Growth-detach cycles")
    for a in ax3: a.grid(alpha=0.3)
    fig3.tight_layout(); fig3.savefig("m3_droplet.png", dpi=140)

    # ---------- 模块 4: Goldak FDM ----------
    g = GoldakFDM(Q=P_ss)
    g.run(t_end=5.0)
    L, W, D = g.pool_size()
    print(f"[4 Goldak-FDM] 熔池 长 {L:.1f} / 宽 {W:.1f} / 深 {D:.1f} mm")

    ros = RosenthalThermal(Q=P_ss)
    XI, Y, Tr = ros.surface_field()
    w_ros = 0.0
    melt = Tr >= ros.Tm
    if melt.any():
        w_ros = (Y[melt].max()-Y[melt].min())*1e3
    print(f"[4 对比] Rosenthal 熔池宽 {w_ros:.1f} mm (点源, 偏宽属正常)")

    fig4, a4 = plt.subplots(1, 3, figsize=(14, 4))
    lv = [298, 600, 900, 1273, 1773, 2600]
    c0 = a4[0].contourf(g.x*1e3, g.y*1e3, g.T[:, :, 0].T, levels=lv,
                        cmap="hot")
    a4[0].contour(g.x*1e3, g.y*1e3, g.T[:, :, 0].T, levels=[1773],
                  colors="cyan")
    a4[0].set_title("Top view T (half model)")
    a4[0].set_xlabel("x [mm]"); a4[0].set_ylabel("y [mm]")
    fig4.colorbar(c0, ax=a4[0])
    c1 = a4[1].contourf(g.x*1e3, -g.z*1e3, g.T[:, 0, :].T, levels=lv,
                        cmap="hot")
    a4[1].contour(g.x*1e3, -g.z*1e3, g.T[:, 0, :].T, levels=[1773],
                  colors="cyan")
    a4[1].set_title("Longitudinal section")
    a4[1].set_xlabel("x [mm]"); a4[1].set_ylabel("z [mm]")
    fig4.colorbar(c1, ax=a4[1])
    c2 = a4[2].contourf(g.x*1e3, -g.z*1e3, g.peak[:, 0, :].T,
                        levels=[298, 773, 1073, 1273, 1773, 3000],
                        cmap="inferno")
    a4[2].contour(g.x*1e3, -g.z*1e3, g.peak[:, 0, :].T,
                  levels=[1273, 1773], colors=["lime", "cyan"])
    a4[2].set_title("Peak T: fusion zone & HAZ")
    a4[2].set_xlabel("x [mm]")
    fig4.colorbar(c2, ax=a4[2])
    fig4.suptitle("Module 4: Goldak double-ellipsoid + 3D transient FDM")
    fig4.tight_layout(); fig4.savefig("m4_goldak_fdm.png", dpi=140)

    # ---------- 模块 5: 短路 / CMT ----------
    std = ShortCircuitGMAW(cmt=False).simulate()
    cmt = ShortCircuitGMAW(cmt=True).simulate()
    for name, o in [("标准短路", std), ("CMT", cmt)]:
        ncyc = int(np.sum(np.diff(o[:, 3]) > 0))
        f = ncyc / o[-1, 0]
        Ipk = o[:, 1].max()
        print(f"[5 {name}] 过渡频率 ~ {f:.0f} Hz, 峰值电流 {Ipk:.0f} A")

    fig5, a5 = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
    for j, (o, name) in enumerate([(std, "Standard short-circuit"),
                                   (cmt, "CMT (current control + retract)")]):
        a5[0, j].plot(o[:, 0]*1e3, o[:, 1], "b", lw=0.9)
        a5[0, j].set_title(name); a5[0, j].set_ylabel("I [A]")
        a5[1, j].plot(o[:, 0]*1e3, o[:, 2], "r", lw=0.9)
        a5[1, j].fill_between(o[:, 0]*1e3, 0, 30, where=o[:, 3] > 0.5,
                              color="gray", alpha=0.25,
                              label="short phase")
        a5[1, j].set_ylabel("V [V]"); a5[1, j].set_xlabel("t [ms]")
        a5[1, j].legend(loc="upper right", fontsize=8)
        for a in (a5[0, j], a5[1, j]):
            a.grid(alpha=0.3)
    fig5.suptitle("Module 5: short-circuit transfer vs CMT waveforms")
    fig5.tight_layout(); fig5.savefig("m5_short_cmt.png", dpi=140)


if __name__ == "__main__":
    main()
