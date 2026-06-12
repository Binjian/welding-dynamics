# -*- coding: utf-8 -*-
import numpy as np
"""模块 5: 短路过渡 / CMT 混杂动力学"""


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
