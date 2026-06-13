# -*- coding: utf-8 -*-
"""模块 6: 悬垂熔滴轴向振荡与脉冲 MIG 共振 (变分积分器)

模型:  m x'' + c x' + k x = F_em(t)        (脉冲电磁力方波激励)
刚度:  k = 32*pi*gamma/3   (Rayleigh l=2 模态等效, 与熔滴尺寸无关)
固有频率:  w0^2 = 8*gamma/(rho*rd^3)       (Rayleigh 频率)

脉冲 MIG 的"一脉一滴"控制依赖脉冲频率与熔滴固有频率的共振匹配。
数值要点: 普通格式 (如隐式 Euler) 的人工数值阻尼会压低并展宽共振峰,
导致预测的最优脉冲频率失真; 辛/变分积分器无人工耗散, 用粗步长
也能保真共振曲线 —— 这是变分积分器在本场景的核心优势。
"""
import numpy as np
from .variational import ForcedVerlet


class DropletOscillatorVI:

    def __init__(self, rd=0.5e-3, gamma=1.2, rho=7000.0, zeta=0.02,
                 F0_frac=0.15, duty=0.35):
        self.rd = rd
        self.m = rho * 4/3*np.pi*rd**3
        self.k = 32*np.pi*gamma/3
        self.w0 = np.sqrt(self.k/self.m)
        self.f0 = self.w0/(2*np.pi)
        self.c = 2*zeta*np.sqrt(self.k*self.m)
        self.F0 = F0_frac * self.k * rd          # 脉冲力幅 (静挠度 F0_frac*rd)
        self.duty = duty

    # ---------------- 激励与导数 ----------------
    def pulse(self, t, fp):
        return self.F0 if (t*fp) % 1.0 < self.duty else 0.0

    def _force(self, fp):
        return lambda q, v, t: -self.c*v + self.pulse(t, fp)

    def _gradV(self, q):
        return self.k*q

    # ---------------- 积分器 ----------------
    def run_vi(self, fp, t_end, h, x0=0.0, v0=0.0):
        """变分积分器: 辛 Verlet + DLA 强迫"""
        vi = ForcedVerlet(self.m, self._gradV, self._force(fp))
        return vi.run(np.array([x0]), np.array([v0]), t_end, h)

    def run_implicit_euler(self, fp, t_end, h, x0=0.0, v0=0.0):
        """非辛基线: 隐式 Euler (一阶, 强人工数值阻尼)"""
        n = int(round(t_end/h)); x, v = x0, v0
        X = np.zeros(n+1); X[0] = x0
        a11, a12 = 1.0, -h
        a21, a22 = h*self.k/self.m, 1.0 + h*self.c/self.m
        det = a11*a22 - a12*a21
        for i in range(n):
            F = self.pulse((i+1)*h, fp)/self.m
            b1, b2 = x, v + h*F
            x = (b1*a22 - a12*b2)/det
            v = (a11*b2 - b1*a21)/det
            X[i+1] = x
        return h*np.arange(n+1), X

    def run_explicit_euler(self, fp, t_end, h, x0=0.0, v0=0.0):
        """非辛基线: 显式 Euler (能量持续注入, 发散)"""
        n = int(round(t_end/h)); x, v = x0, v0
        X = np.zeros(n+1); X[0] = x0
        for i in range(n):
            ax = (-self.k*x - self.c*v + self.pulse(i*h, fp))/self.m
            x, v = x + h*v, v + h*ax
            X[i+1] = x
        return h*np.arange(n+1), X

    def energy(self, x, v):
        return 0.5*self.m*v**2 + 0.5*self.k*x**2

    # ---------------- 共振曲线 ----------------
    def resonance_sweep(self, fp_arr, method="vi", periods=60,
                        steps_per_period=22):
        amp = []
        for fp in fp_arr:
            T0 = 1.0/self.f0
            h = T0/steps_per_period
            t_end = periods*T0
            if method == "vi":
                t, X, _ = self.run_vi(fp, t_end, h)
            else:
                t, X = self.run_implicit_euler(fp, t_end, h)
            tail = X[int(0.7*len(X)):]
            amp.append(0.5*(tail.max() - tail.min()))
        return np.array(amp)

    def analytic_fundamental(self, fp_arr):
        """方波激励基波分量的解析稳态幅值 (参照)"""
        a1 = 2*self.F0/np.pi*np.abs(np.sin(np.pi*self.duty))
        w = 2*np.pi*fp_arr
        return a1/np.sqrt((self.k - self.m*w**2)**2 + (self.c*w)**2)
