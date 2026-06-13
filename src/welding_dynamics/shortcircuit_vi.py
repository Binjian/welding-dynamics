# -*- coding: utf-8 -*-
"""模块 8: 短路接触事件的非光滑变分模型 (CMT 机械振荡简化)

物理图像 (一维, 沿焊丝轴向, x 向下为正):
  自由相 : 悬垂熔滴振荡  m x'' = -k (x - x_eq(t)) - c x'
           x_eq(t) 随送丝前进 (v_f) 将熔滴推向熔池
  触池   : x 达到熔池面 gap -> 湿接触, 变分碰撞映射 v+ = -e v- (e=0 全塑性)
  附着相 : 熔滴钉扎于熔池 (x = gap), 焊丝回抽 (v_r) 使悬垂弹簧张力增大
  断桥   : 张力 k*(gap - x_eq) 超过 F_rup -> 释放, 回到自由相 (周期循环)

数值要点: 接触时刻用二分精确定位 + 变分碰撞映射, 相内用辛 Verlet ——
能量只在物理事件 (湿接触/断桥) 处变化, 不会像罚函数法那样在固定步长
下因穿透-反弹产生虚假能量注入或抖振。这里同时给出弹性反弹基准实验
(e=0.85) 对比 非光滑VI vs 罚函数法 的能量保真度。
"""
import numpy as np


class ContactCycleVI:

    def __init__(self, rd=0.5e-3, gamma=1.2, rho=7000.0, zeta=0.03,
                 gap=0.4e-3, v_feed=0.08, v_retract=0.10,
                 F_rup_frac=1.4):
        self.m = rho*4/3*np.pi*rd**3
        self.k = 32*np.pi*gamma/3
        self.c = 2*zeta*np.sqrt(self.k*self.m)
        self.gap = gap                      # 熔池表面位置
        self.v_f, self.v_r = v_feed, v_retract
        self.F_rup = F_rup_frac*self.k*gap  # 断桥张力阈值

    # ---------------- CMT 接触循环 (非光滑变分混杂仿真) ----------------
    def simulate_cycle(self, t_end=0.05, h=2e-6):
        n = int(t_end/h)
        x, v = 0.0, 0.0
        x_eq, phase = 0.0, 0            # 0=自由(送丝), 1=附着(回抽)
        out = np.zeros((n, 4))          # t, x, x_eq, phase
        events = []
        for i in range(n):
            t = i*h
            if phase == 0:
                # 辛 Verlet (相内保守+弱阻尼)
                a0 = (-self.k*(x - x_eq) - self.c*v)/self.m
                v_half = v + 0.5*h*a0
                x_new = x + h*v_half
                if x_new >= self.gap:                      # 触池事件
                    # 二分精确定位撞击时刻
                    lo, hi = 0.0, h
                    for _ in range(40):
                        mid = 0.5*(lo+hi)
                        if x + mid*v_half >= self.gap:
                            hi = mid
                        else:
                            lo = mid
                    x = self.gap
                    v = 0.0                                # e=0 全塑性湿接触
                    phase = 1
                    events.append((t, "dip"))
                else:
                    a1 = (-self.k*(x_new - x_eq) - self.c*v_half)/self.m
                    x, v = x_new, v_half + 0.5*h*a1
                x_eq += self.v_f*h
            else:
                x_eq -= self.v_r*h                         # 机械回抽
                if self.k*(self.gap - x_eq) >= self.F_rup:  # 断桥事件
                    phase = 0
                    x, v = x_eq, 0.0       # 熔滴并入熔池, 新滴自焊丝端开始
                    events.append((t, "rupture"))
            out[i] = (t, x, x_eq, phase)
        return out, events

    # ---------------- 能量保真基准: 弹性反冲 e=0.85 ----------------
    # 自由质点以 -F_const 推向壁面 (x=0), 撞壁恢复系数 e。
    # 解析: 每次反弹动能 *e^2 (能量只在碰撞处按物理规律减少)。
    def bounce_nonsmooth_vi(self, t_end=0.03, h=4e-6, e=0.85,
                            x0=-0.4e-3, Fc=None):
        Fc = Fc if Fc is not None else 0.5*self.k*self.gap
        n = int(t_end/h)
        x, v = x0, 0.0
        T = np.zeros(n); X = np.zeros(n); E = np.zeros(n)
        for i in range(n):
            a = Fc/self.m
            v_half = v + 0.5*h*a
            x_new = x + h*v_half
            if x_new >= 0.0:
                # 二分定位 + 变分碰撞映射
                lo, hi = 0.0, h
                for _ in range(40):
                    mid = 0.5*(lo+hi)
                    if x + mid*v_half >= 0.0:
                        hi = mid
                    else:
                        lo = mid
                dt1 = hi
                v_imp = v + dt1*a                   # 撞击时刻速度
                v = -e*v_imp                        # 碰撞映射
                x = 0.0
                dt2 = h - dt1                        # 剩余子步
                vh2 = v + 0.5*dt2*a
                x, v = x + dt2*vh2, vh2 + 0.5*dt2*a
            else:
                x, v = x_new, v_half + 0.5*h*a
            T[i], X[i] = i*h, x
            E[i] = 0.5*self.m*v**2 - Fc*x           # 总机械能
        return T, X, E

    def bounce_penalty(self, t_end=0.03, h=4e-6, k_pen_frac=2.5e4,
                       x0=-0.4e-3, Fc=None):
        """罚函数法基线: 穿透时加刚性罚弹簧, 同步长无事件定位。
        刚罚弹簧在固定步长下欠解析 -> 虚假能量注入。"""
        Fc = Fc if Fc is not None else 0.5*self.k*self.gap
        k_pen = k_pen_frac*self.k
        n = int(t_end/h)
        x, v = x0, 0.0
        T = np.zeros(n); X = np.zeros(n); E = np.zeros(n)
        for i in range(n):
            def force(xx):
                return Fc - (k_pen*xx if xx > 0 else 0.0)
            a = force(x)/self.m
            v_half = v + 0.5*h*a
            x = x + h*v_half
            v = v_half + 0.5*h*force(x)/self.m
            T[i], X[i] = i*h, np.clip(x, -2e-3, 2e-3)
            E[i] = (0.5*self.m*min(v, 1e3)**2 - Fc*min(x, 0.0)
                    + 0.5*k_pen*max(x, 0.0)**2)
        return T, X, E
