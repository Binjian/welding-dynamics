# -*- coding: utf-8 -*-
"""变分积分器核心 (Variational Integrators)

- ForcedVerlet : 常质量矩阵 + 离散 Lagrange-d'Alembert (DLA) 强迫项的
                 辛 Stormer-Verlet 格式 (二阶, 显式)
- MidpointDEL  : 中点离散 Euler-Lagrange (DEL) 积分器, 适用于一般
                 L(q, qd) (构型相关质量矩阵, 如机械臂), Newton 隐式求解
- impact_map / locate_event : 非光滑变分力学的碰撞映射与事件精确定位

参考: Marsden & West (2001), Discrete mechanics and variational integrators.
"""
import numpy as np
from scipy.optimize import fsolve


class ForcedVerlet:
    """m q'' = -grad_V(q) + F(q, v, t) 的辛 Verlet + DLA 梯形强迫离散。"""

    def __init__(self, m, grad_V, force=None):
        self.m = m
        self.grad_V = grad_V
        self.force = force if force is not None else (lambda q, v, t: 0.0)

    def step(self, q, p, t, h):
        v = p / self.m
        F0 = -self.grad_V(q) + self.force(q, v, t)
        p_half = p + 0.5 * h * F0
        q_new = q + h * p_half / self.m
        v_pred = p_half / self.m
        F1 = -self.grad_V(q_new) + self.force(q_new, v_pred, t + h)
        p_new = p_half + 0.5 * h * F1
        return q_new, p_new

    def run(self, q0, v0, t_end, h, t0=0.0):
        n = int(round((t_end - t0) / h))
        Q = np.zeros((n + 1, np.size(q0))); P = np.zeros_like(Q)
        Q[0], P[0] = q0, self.m * np.asarray(v0, float)
        t = t0
        for k in range(n):
            Q[k+1], P[k+1] = self.step(Q[k], P[k], t, h)
            t += h
        return t0 + h*np.arange(n+1), Q.squeeze(), (P / self.m).squeeze()


class MidpointDEL:
    """中点离散 Euler-Lagrange 变分积分器 (隐式, 二阶, 辛)。

    Ld(q0, q1) = h * L((q0+q1)/2, (q1-q0)/h)
    强迫 DEL:  D2Ld(q_{k-1}, q_k) + D1Ld(q_k, q_{k+1})
               + (h/2) F(mid_{k-1}) + (h/2) F(mid_k) = 0
    L 的斜率用中心差分数值计算 -> 适用于任意光滑拉格朗日量。
    """

    def __init__(self, lagrangian, n_dof, force=None, eps=1e-6):
        self.L, self.n, self.force, self.eps = lagrangian, n_dof, force, eps

    def _Ld(self, q0, q1, h):
        return h * self.L(0.5*(q0 + q1), (q1 - q0)/h)

    def _D(self, q0, q1, h, slot):
        g = np.zeros(self.n)
        for i in range(self.n):
            e = np.zeros(self.n); e[i] = self.eps
            if slot == 1:
                g[i] = self._Ld(q0+e, q1, h) - self._Ld(q0-e, q1, h)
            else:
                g[i] = self._Ld(q0, q1+e, h) - self._Ld(q0, q1-e, h)
        return g / (2*self.eps)

    def step(self, q_prev, q, t, h):
        base = self._D(q_prev, q, h, slot=2)
        if self.force is not None:
            v_prev = (q - q_prev)/h
            base = base + 0.5*h*self.force(0.5*(q_prev+q), v_prev, t - 0.5*h)

        def residual(q_next):
            r = base + self._D(q, q_next, h, slot=1)
            if self.force is not None:
                v = (q_next - q)/h
                r = r + 0.5*h*self.force(0.5*(q+q_next), v, t + 0.5*h)
            return r

        return np.asarray(fsolve(residual, 2*q - q_prev, xtol=1e-10))

    def run(self, q0, v0, t_end, h, rhs_for_bootstrap=None):
        """rhs_for_bootstrap(q, v, t)->qdd 用 RK4 细步生成 q1 启动。"""
        n = int(round(t_end / h))
        Q = np.zeros((n + 1, self.n)); Q[0] = q0
        # 启动: 细分 RK4 得到 q1
        q, v = np.asarray(q0, float), np.asarray(v0, float)
        sub = 20; hs = h/sub
        for _ in range(sub):
            q, v = _rk4_qv(rhs_for_bootstrap, q, v, 0.0, hs)
        Q[1] = q
        t = h
        for k in range(1, n):
            Q[k+1] = self.step(Q[k-1], Q[k], t, h)
            t += h
        return h*np.arange(n+1), Q


def _rk4_qv(acc, q, v, t, h):
    def f(state, t):
        qq, vv = state[:len(q)], state[len(q):]
        return np.concatenate([vv, acc(qq, vv, t)])
    y = np.concatenate([q, v])
    k1 = f(y, t); k2 = f(y + h/2*k1, t + h/2)
    k3 = f(y + h/2*k2, t + h/2); k4 = f(y + h*k3, t + h)
    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return y[:len(q)], y[len(q):]


def rk4_run(acc, q0, v0, t_end, h):
    """非辛基线: 经典 RK4 (四阶)。"""
    n = int(round(t_end / h))
    Q = np.zeros((n+1, np.size(q0))); V = np.zeros_like(Q)
    Q[0], V[0] = q0, v0
    t = 0.0
    for k in range(n):
        q, v = _rk4_qv(acc, Q[k], V[k], t, h)
        Q[k+1], V[k+1] = q, v
        t += h
    return h*np.arange(n+1), Q.squeeze(), V.squeeze()


# ---------------- 非光滑变分力学工具 ----------------
def locate_event(g0, g1, q0, p0, stepper, t, h, tol=1e-12, it=60):
    """事件函数 g 在 [t, t+h] 内变号: 二分定位精确撞击时刻 (变分相容)。
    返回 (h_star, q_star, p_star)。stepper(q,p,t,h)->(q,p); g(q)->标量。"""
    lo, hi = 0.0, h
    for _ in range(it):
        mid = 0.5*(lo + hi)
        qm, pm = stepper(q0, p0, t, mid)
        if g0(qm) * g0_sign_cache(g1) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    q_star, p_star = stepper(q0, p0, t, hi)
    return hi, q_star, p_star


def g0_sign_cache(sign):
    return sign


def impact_map(p, e):
    """变分碰撞映射 (Newton 恢复系数): 法向动量反转 p+ = -e p-。
    e=0 为完全塑性 (湿接触/熔合), e=1 为完全弹性。"""
    return -e * p
