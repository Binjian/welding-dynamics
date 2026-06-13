# -*- coding: utf-8 -*-
"""模块 7: 焊接机器人 (二连杆机械臂携带焊枪) — DEL 变分积分器

竖直平面二连杆 (点质量), 构型相关质量矩阵 M(q):
  L(q, qd) = T(q, qd) - V(q)
变分积分器: 中点离散 Euler-Lagrange (MidpointDEL), 隐式 Newton 求解。

演示:
A) 无驱动长时间摆动 — 能量保持性: VI 能量误差有界振荡,
   RK4 同步长能量单调漂移 (长焊缝/长轨迹仿真不失真)。
B) 焊缝跟踪 — PD + 重力补偿力矩, 强迫 DEL 积分, 焊枪尖端沿直线焊缝。
"""
import numpy as np
from .variational import MidpointDEL, rk4_run


class TwoLinkArm:

    def __init__(self, m1=4.0, m2=2.5, l1=0.40, l2=0.35, g=9.81):
        self.m1, self.m2, self.l1, self.l2, self.g = m1, m2, l1, l2, g

    # ---------------- 拉格朗日量 / 能量 ----------------
    def lagrangian(self, q, qd):
        q1, q2 = q; w1, w2 = qd
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        v1sq = (l1*w1)**2
        v2sq = (l1*w1)**2 + (l2*(w1+w2))**2 \
            + 2*l1*l2*w1*(w1+w2)*np.cos(q2)
        T = 0.5*m1*v1sq + 0.5*m2*v2sq
        V = m1*g*l1*np.sin(q1) + m2*g*(l1*np.sin(q1) + l2*np.sin(q1+q2))
        return T - V

    def energy(self, q, qd):
        return self.lagrangian(q, qd) \
            + 2*(self.m1*self.g*self.l1*np.sin(q[0])
                 + self.m2*self.g*(self.l1*np.sin(q[0])
                                   + self.l2*np.sin(q[0]+q[1])))

    # ---------------- 动力学 (RK4 基线 / 启动用) ----------------
    def _MCG(self, q, qd):
        q1, q2 = q; w1, w2 = qd
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        c2 = np.cos(q2)
        M = np.array([[(m1+m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*c2,
                       m2*l2**2 + m2*l1*l2*c2],
                      [m2*l2**2 + m2*l1*l2*c2, m2*l2**2]])
        hh = m2*l1*l2*np.sin(q2)
        C = np.array([-hh*(2*w1*w2 + w2**2), hh*w1**2])
        G = np.array([(m1+m2)*g*l1*np.cos(q1) + m2*g*l2*np.cos(q1+q2),
                      m2*g*l2*np.cos(q1+q2)])
        return M, C, G

    def accel(self, q, qd, t, tau=None):
        M, C, G = self._MCG(q, qd)
        f = (tau if tau is not None else 0.0) - C - G
        return np.linalg.solve(M, f)

    def gravity_comp(self, q):
        return self._MCG(q, np.zeros(2))[2]

    # ---------------- 运动学 ----------------
    def fk_tip(self, q):
        q1, q2 = q
        x = self.l1*np.cos(q1) + self.l2*np.cos(q1+q2)
        y = self.l1*np.sin(q1) + self.l2*np.sin(q1+q2)
        return np.array([x, y])

    def ik(self, x, y):
        l1, l2 = self.l1, self.l2
        c2 = np.clip((x*x + y*y - l1*l1 - l2*l2)/(2*l1*l2), -1, 1)
        q2 = -np.arccos(c2)                       # 肘下解
        q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1 + l2*np.cos(q2))
        return np.array([q1, q2])

    # ---------------- 演示 A: 无驱动能量保持 ----------------
    def passive_compare(self, q0=(1.05, 0.30), t_end=200.0, h=0.02):
        q0 = np.array(q0, float); v0 = np.zeros(2)
        acc = lambda q, v, t: self.accel(q, v, t)
        # VI
        vi = MidpointDEL(self.lagrangian, 2)
        t_vi, Q = vi.run(q0, v0, t_end, h, rhs_for_bootstrap=acc)
        Vc = (Q[2:] - Q[:-2])/(2*h)               # 中心差分速度 (诊断用)
        E_vi = np.array([self.energy(Q[k+1], Vc[k]) for k in range(len(Vc))])
        # RK4
        t_rk, Qr, Vr = rk4_run(acc, q0, v0, t_end, h)
        E_rk = np.array([self.energy(Qr[k], Vr[k]) for k in range(len(t_rk))])
        E0 = self.energy(q0, v0)
        return (t_vi[1:-1], np.abs(E_vi/E0 - 1),
                t_rk, np.abs(E_rk/E0 - 1))

    # ---------------- 演示 B: 焊缝跟踪 (强迫 DEL) ----------------
    def seam_tracking(self, p_start=(0.30, 0.10), p_end=(0.60, 0.25),
                      t_weld=4.0, h=0.01, Kp=120.0, Kd=24.0):
        p0, p1 = np.array(p_start), np.array(p_end)

        def q_ref(t):
            sgm = np.clip(t/t_weld, 0, 1)
            sgm = 3*sgm**2 - 2*sgm**3             # 平滑 S 曲线
            return self.ik(*(p0 + sgm*(p1 - p0)))

        def qd_ref(t, d=1e-4):
            return (q_ref(t+d) - q_ref(t-d))/(2*d)

        def tau(q, v, t):
            return (Kp*(q_ref(t) - q) + Kd*(qd_ref(t) - v)
                    + self.gravity_comp(q))

        acc = lambda q, v, t: self.accel(q, v, t, tau(q, v, t))
        vi = MidpointDEL(self.lagrangian, 2, force=tau)
        t, Q = vi.run(q_ref(0.0), qd_ref(0.0), t_weld + 1.0, h,
                      rhs_for_bootstrap=acc)
        tip = np.array([self.fk_tip(qk) for qk in Q])
        ref = np.array([self.fk_tip(q_ref(tk)) for tk in t])
        err = np.linalg.norm(tip - ref, axis=1)
        return t, tip, ref, err
