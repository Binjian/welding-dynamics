# -*- coding: utf-8 -*-
import numpy as np
MU0 = 4e-7 * np.pi

"""模块 3: 熔滴过渡动力学 (静力平衡 SFBT + pinch 不稳定性)"""


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
