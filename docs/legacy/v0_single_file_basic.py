# -*- coding: utf-8 -*-
"""
工业焊接动力学模型 (GMAW / MIG 焊)
====================================
模块 1: 电弧-熔化自调节动力学 (集中参数 ODE 模型)
    状态变量: 焊丝伸出长度 s(t), 焊接电流 I(t)
    - 电源外特性 (恒压源 + 内阻 + 回路电感)
    - 电弧电压模型: V_arc = V0 + Ea*la + Ra*I
    - 焊丝熔化速率 (burn-off): MR = k1*I + k2*s*I^2
    - 自调节机理: ds/dt = WFS - MR

模块 2: 移动热源温度场 (Rosenthal 三维准稳态解析解)
    T(x,y,z) = T0 + (eta*Q / (2*pi*k*R)) * exp(-v*(R + xi) / (2*alpha))
    用于预测熔池尺寸与热影响区 (HAZ)

参考: Tuchinskii/Quinn 燃弧模型, Rosenthal (1946)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"


# =====================================================================
# 模块 1: GMAW 电弧-熔化动力学
# =====================================================================
class GMAWDynamics:
    """GMAW 过程集中参数动力学模型。

    状态向量 x = [s, I]
        s : 焊丝干伸长 (electrode stick-out) [m]
        I : 焊接电流 [A]
    输入:
        WFS  : 送丝速度 [m/s]
        Voc  : 电源空载电压 [V]
        CTWD : 导电嘴到工件距离 [m] (可随时间变化, 模拟扰动)
    """

    def __init__(self):
        # ---- 电源与回路参数 ----
        self.Voc = 32.0        # 空载电压 [V]
        self.Rs = 0.004        # 电源内阻/外特性斜率 [Ohm]
        self.Rl = 0.004        # 回路电阻 [Ohm]
        self.Ls = 3.0e-4       # 回路电感 [H]

        # ---- 电弧参数 ----
        self.V0 = 15.5         # 阴阳极压降之和 [V]
        self.Ea = 800.0        # 弧柱电场强度 [V/m]
        self.Ra = 0.022        # 电弧等效电阻 [Ohm]

        # ---- 焊丝与熔化参数 (1.2 mm 钢焊丝典型值) ----
        self.rw = 0.6e-3                       # 焊丝半径 [m]
        self.k1 = 3.0e-4                       # 电弧熔化系数 [m/(s·A)]
        self.k2 = 5.0e-5                       # 电阻热熔化系数 [1/(s·A^2)]
        self.rho_r = 0.25                      # 干伸长单位电阻 [Ohm/m] (近似)

    # -------------------------------------------------------------
    def melting_rate(self, s, I):
        """焊丝熔化速率 MR [m/s] (burn-off 方程)"""
        return self.k1 * I + self.k2 * s * I ** 2

    def arc_voltage(self, la, I):
        """电弧电压 [V]"""
        return self.V0 + self.Ea * la + self.Ra * I

    # -------------------------------------------------------------
    def rhs(self, t, x, WFS_fun, CTWD_fun):
        """状态方程 dx/dt"""
        s, I = x
        WFS = WFS_fun(t)
        CTWD = CTWD_fun(t)

        la = max(CTWD - s, 1e-4)          # 弧长 = CTWD - 干伸长
        Rstick = self.rho_r * s           # 干伸长电阻

        # 干伸长动力学: 送丝 - 熔化
        ds = WFS - self.melting_rate(s, I)

        # 电路动力学: L dI/dt = Voc - (Rs+Rl+Rstick)*I - V_arc
        dI = (self.Voc - (self.Rs + self.Rl + Rstick) * I
              - self.arc_voltage(la, I)) / self.Ls
        return [ds, dI]

    # -------------------------------------------------------------
    def simulate(self, t_end=1.0, x0=(6e-3, 150.0),
                 WFS_fun=None, CTWD_fun=None):
        if WFS_fun is None:
            WFS_fun = lambda t: 0.12                      # 送丝 0.12 m/s
        if CTWD_fun is None:
            # 0.5 s 时 CTWD 阶跃 +3 mm, 模拟工件表面起伏扰动
            CTWD_fun = lambda t: 0.018 + (0.003 if t >= 0.5 else 0.0)

        sol = solve_ivp(self.rhs, (0, t_end), x0,
                        args=(WFS_fun, CTWD_fun),
                        method="LSODA", max_step=1e-3, dense_output=True)
        t = np.linspace(0, t_end, 2000)
        s, I = sol.sol(t)
        CTWD = np.array([CTWD_fun(ti) for ti in t])
        la = CTWD - s
        Va = self.arc_voltage(la, I)
        return dict(t=t, s=s, I=I, la=la, Va=Va, CTWD=CTWD,
                    P=Va * I)


# =====================================================================
# 模块 2: Rosenthal 三维移动点热源温度场
# =====================================================================
class RosenthalThermal:
    """厚板三维准稳态移动热源解 (低碳钢默认参数)。

    坐标系随热源移动: xi = x - v*t (热源位于原点)
    """

    def __init__(self, Q=4500.0, eta=0.8, v=8e-3,
                 k=41.0, alpha=1.0e-5, T0=298.0, Tm=1773.0):
        self.Q = Q          # 电弧功率 [W] (= V*I)
        self.eta = eta      # 热效率
        self.v = v          # 焊接速度 [m/s]
        self.k = k          # 导热系数 [W/(m·K)]
        self.alpha = alpha  # 热扩散率 [m^2/s]
        self.T0 = T0        # 初始/环境温度 [K]
        self.Tm = Tm        # 熔点 [K]

    def temperature(self, xi, y, z):
        """准稳态温度场 T(xi, y, z) [K]"""
        R = np.sqrt(xi ** 2 + y ** 2 + z ** 2)
        R = np.maximum(R, 1e-5)
        q = self.eta * self.Q
        return self.T0 + q / (2 * np.pi * self.k * R) * np.exp(
            -self.v * (R + xi) / (2 * self.alpha))

    def surface_field(self, xlim=(-0.04, 0.012), ylim=(-0.012, 0.012), n=400):
        xi = np.linspace(*xlim, n)
        y = np.linspace(*ylim, n)
        XI, Y = np.meshgrid(xi, y)
        T = self.temperature(XI, Y, 0.0)
        return XI, Y, T

    def thermal_cycle(self, y_off, z=0.0, t=np.linspace(-1, 8, 1500)):
        """固定点 (距焊缝中心 y_off) 经历的热循环 T(t)"""
        xi = -self.v * t
        return t, self.temperature(xi, y_off, z)


# =====================================================================
# 主程序: 仿真 + 可视化
# =====================================================================
def main():
    # ---------------- 模块 1 仿真 ----------------
    gmaw = GMAWDynamics()
    res = gmaw.simulate()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("GMAW Arc Self-Regulation Dynamics (CTWD step +3 mm @ t=0.5 s)",
                 fontsize=12)

    axes[0, 0].plot(res["t"], res["I"], "b")
    axes[0, 0].set_ylabel("Current I [A]")
    axes[0, 1].plot(res["t"], res["la"] * 1e3, "r")
    axes[0, 1].set_ylabel("Arc length [mm]")
    axes[1, 0].plot(res["t"], res["s"] * 1e3, "g")
    axes[1, 0].set_ylabel("Stick-out s [mm]")
    axes[1, 0].set_xlabel("t [s]")
    axes[1, 1].plot(res["t"], res["Va"], "m")
    axes[1, 1].set_ylabel("Arc voltage [V]")
    axes[1, 1].set_xlabel("t [s]")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig("gmaw_dynamics.png", dpi=150)

    # 稳态工作点 (用其功率驱动热模型)
    P_ss = float(np.mean(res["P"][res["t"] > 0.45][:50]))
    print(f"[GMAW] 稳态: I = {res['I'][900]:.0f} A, "
          f"Va = {res['Va'][900]:.1f} V, P = {P_ss:.0f} W")

    # ---------------- 模块 2 仿真 ----------------
    th = RosenthalThermal(Q=P_ss, v=8e-3)
    XI, Y, T = th.surface_field()

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    cs = a1.contourf(XI * 1e3, Y * 1e3, T,
                     levels=[298, 600, 900, 1100, 1273, 1773, 3000],
                     cmap="hot")
    a1.contour(XI * 1e3, Y * 1e3, T, levels=[1773], colors="cyan", linewidths=2)
    a1.set_xlabel("xi = x - v t [mm]")
    a1.set_ylabel("y [mm]")
    a1.set_title("Surface temperature field (cyan = fusion line)")
    fig2.colorbar(cs, ax=a1, label="T [K]")

    for y_off, c in zip([0.002, 0.004, 0.006], ["r", "g", "b"]):
        t, Tc = th.thermal_cycle(y_off)
        a2.plot(t, Tc, c, label=f"y = {y_off*1e3:.0f} mm")
    a2.axhline(1773, color="k", ls="--", lw=0.8, label="Melting point")
    a2.axhline(996 + 273, color="gray", ls=":", lw=0.8, label="Ac3 (~1269 K)")
    a2.set_xlabel("t [s]")
    a2.set_ylabel("T [K]")
    a2.set_title("Thermal cycles at fixed points")
    a2.legend()
    a2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("thermal_field.png", dpi=150)

    # 熔池尺寸估计
    melt = T >= th.Tm
    if melt.any():
        width = (Y[melt].max() - Y[melt].min()) * 1e3
        length = (XI[melt].max() - XI[melt].min()) * 1e3
        print(f"[Thermal] 熔池宽度 ~ {width:.1f} mm, 长度 ~ {length:.1f} mm")

    # 冷却速率 t8/5 (焊缝边缘点)
    t, Tc = th.thermal_cycle(0.0028)
    above800 = t[Tc >= 1073]
    above500 = t[Tc >= 773]
    if len(above800) and len(above500):
        t85 = above500.max() - above800.max()
        print(f"[Thermal] t8/5 冷却时间 ~ {t85:.2f} s (决定 HAZ 组织)")


if __name__ == "__main__":
    main()
