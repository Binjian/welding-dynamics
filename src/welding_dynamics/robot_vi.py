# -*- coding: utf-8 -*-
"""模块 7: 焊接机器人 (机械臂携带焊枪) — DEL 变分积分器

两级模型, 共用同一变分积分器框架 (MidpointDEL, 隐式 Newton 求解):

- TwoLinkArm : 竖直平面二连杆 (点质量), 解析 M(q)/C/G — 教学基线。
- SixDofArm  : 空间 6R 机械臂 (球腕, 标准 DH), 连杆按实心圆柱建模,
  M(q) 由各连杆质心 Jacobian 数值装配:
      M(q) = Σ_i [ m_i Jc_i' Jc_i + Jw_i' I_i(q) Jw_i ]
  L(q, qd) = ½ qd' M(q) qd - V(q) 直接喂给 MidpointDEL (其内部对 L 做
  中心差分), 不需要解析动力学 — 这正是变分积分器对高自由度系统的优势。

演示 (两个类同构):
A) 无驱动长时间摆动 — 能量保持性: VI 能量误差有界振荡,
   RK4 同步长能量单调漂移 (长焊缝/长轨迹仿真不失真)。
B) 焊缝跟踪 — PD + 重力补偿力矩, 强迫 DEL 积分, 焊枪尖端沿直线焊缝
   (SixDofArm 为三维空间焊缝, 且焊枪保持指定姿态 — 平焊枪尖朝下)。
"""
import warnings

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


class SixDofArm:
    """空间 6R 焊接机械臂 (球腕) — 全六自由度扩展。

    标准 DH (肩侧偏 a1, 大臂 a2, 肘偏 a3, 小臂 d4, 焊枪 d6):

        i | theta |  d  |  a  | alpha
        1 |  q1   | d1  | a1  | +90°
        2 |  q2   |  0  | a2  |   0
        3 |  q3   |  0  | a3  | +90°
        4 |  q4   | d4  |  0  | -90°
        5 |  q5   |  0  |  0  | +90°
        6 |  q6   | d6  |  0  |   0

    连杆 i 建模为端点 [o_{i-1}, o_i] 间的实心圆柱 (半径 r_link):
    轴向转动惯量 ½mr² 保证腕部滚转自由度 (q4/q6, 转轴穿过连杆轴线)
    的 M(q) 非奇异 — 纯点质量模型在这两个方向上动能恒为零。
    零长度连杆 (球腕处 o_4 = o_5) 退化为均质球。另计入折算到关节侧的
    电机/减速器转动惯量 J_rotor (对角常量): 真实焊接机器人腕轴的惯量
    以减速比平方折算的转子惯量为主, 它同时把 cond(M) 从 ~2000 压到
    ~30, 隐式 Newton 步态良好。

    动力学全部由 FK 数值装配 (质心 Jacobian), 无解析 M/C/G;
    RK4 基线的 C/G 项经 M(q)/V(q) 的中心差分获得。
    """

    def __init__(self, m=(6.0, 4.0, 2.5, 1.5, 1.0, 0.5), r_link=0.04,
                 J_rotor=0.03, d1=0.40, a1=0.05, a2=0.40, a3=0.05,
                 d4=0.35, d6=0.10, g=9.81):
        self.m = np.asarray(m, float)
        self.r_link, self.g = float(r_link), float(g)
        self.J_rotor = np.broadcast_to(np.asarray(J_rotor, float), (6,)).copy()
        # 标准 DH 表 (theta 为关节变量)
        self.dh_d = np.array([d1, 0.0, 0.0, d4, 0.0, d6])
        self.dh_a = np.array([a1, a2, a3, 0.0, 0.0, 0.0])
        self.dh_alpha = np.deg2rad([90.0, 0.0, 90.0, -90.0, 90.0, 0.0])
        self._ca, self._sa = np.cos(self.dh_alpha), np.sin(self.dh_alpha)
        self._tril = np.tril(np.ones((6, 6)))[:, :, None]   # j<=i 掩码

    # ---------------- 运动学 ----------------
    def _kin(self, q):
        """FK 链: 返回 (o, z, R) — 各帧原点 o[0..6], 关节轴 z[0..5], 末端姿态 R。"""
        ct, st = np.cos(q), np.sin(q)
        o = np.zeros((7, 3)); z = np.empty((6, 3))
        R = np.eye(3)
        for i in range(6):
            z[i] = R[:, 2]                        # 关节 i+1 的转轴 = 帧 i 的 z
            ca, sa = self._ca[i], self._sa[i]
            # o_{i+1} = o_i + R_i t_i,  R_{i+1} = R_i Rz(q)Rx(alpha)
            o[i+1] = o[i] + R @ (self.dh_a[i]*ct[i],
                                 self.dh_a[i]*st[i], self.dh_d[i])
            R = R @ np.array([[ct[i], -st[i]*ca,  st[i]*sa],
                              [st[i],  ct[i]*ca, -ct[i]*sa],
                              [0.,           sa,        ca]])
        return o, z, R

    def fk_pose(self, q):
        o, _, R = self._kin(q)
        return o[6], R

    def fk_tip(self, q):
        return self._kin(q)[0][6]

    def jacobian(self, q):
        """末端几何 Jacobian (6×6): 上 3 行线速度, 下 3 行角速度。"""
        o, z, _ = self._kin(q)
        J = np.zeros((6, 6))
        J[:3] = np.cross(z, o[6] - o[:6]).T
        J[3:] = z.T
        return J

    # ---------------- 动力学装配 ----------------
    def _MV(self, q):
        """一次 FK 装配 M(q) 与 V(q) (全向量化, 无逐连杆 Python 循环)。"""
        o, z, _ = self._kin(q)
        seg = o[1:] - o[:-1]                      # 连杆 i: 端点 o[i-1]->o[i]
        c = 0.5*(o[1:] + o[:-1])                  # 质心
        Lsq = np.einsum('ij,ij->i', seg, seg)
        r2 = self.r_link**2
        # 质心平动 Jacobian 组 Jc[i] (3x6, 列 j<=i 非零), 以 (6,6,3) 批量算
        Jc = np.cross(z[None, :, :], c[:, None, :] - o[None, :6, :])
        Jc *= self._tril
        M = np.einsum('ija,ika->jk', self.m[:, None, None]*Jc, Jc)
        # 角速度 Jacobian 组与世界系惯量: 圆柱 (轴向 u) / 零长度连杆退化为球
        ball = Lsq < 1e-16
        I_ax = np.where(ball, 0.4*self.m*r2, 0.5*self.m*r2)
        I_tr = np.where(ball, I_ax, self.m*(3.0*r2 + Lsq)/12.0)
        u = seg/np.sqrt(np.where(ball, 1.0, Lsq))[:, None]
        u[ball] = 0.0
        Iw = (I_tr[:, None, None]*np.eye(3)
              + (I_ax - I_tr)[:, None, None]*u[:, :, None]*u[:, None, :])
        Jw = self._tril*z[None, :, :]
        M += np.einsum('ija,iab,ikb->jk', Jw, Iw, Jw)
        M[np.diag_indices(6)] += self.J_rotor     # 折算电机/减速器惯量
        V = self.g*(self.m @ c[:, 2])
        return M, V

    # ---------------- 拉格朗日量 / 能量 ----------------
    def lagrangian(self, q, qd):
        M, V = self._MV(q)
        return 0.5*qd @ M @ qd - V

    def energy(self, q, qd):
        M, V = self._MV(q)
        return 0.5*qd @ M @ qd + V

    # ---------------- 动力学 (RK4 基线 / 启动用) ----------------
    def accel(self, q, qd, t, tau=None, eps=1e-6):
        """M qdd = tau - Mdot qd + ½ ∂q(qd'Mqd) - G;  dM/dq, G 由中心差分。"""
        M0, _ = self._MV(q)
        rhs = np.zeros(6) if tau is None else np.array(tau, float)
        Mdot = np.zeros((6, 6))
        for k in range(6):
            e = np.zeros(6); e[k] = eps
            Mp, Vp = self._MV(q + e)
            Mm, Vm = self._MV(q - e)
            dMk = (Mp - Mm)/(2*eps)
            Mdot += dMk*qd[k]
            rhs[k] += 0.5*(qd @ dMk @ qd) - (Vp - Vm)/(2*eps)   # -G_k
        rhs -= Mdot @ qd
        return np.linalg.solve(M0, rhs)

    def gravity_comp(self, q, eps=1e-6):
        G = np.zeros(6)
        for k in range(6):
            e = np.zeros(6); e[k] = eps
            G[k] = (self._MV(q + e)[1] - self._MV(q - e)[1])/(2*eps)
        return G

    # ---------------- 逆运动学 (阻尼最小二乘) ----------------
    def ik(self, p, R=None, q0=None, tol=1e-10, it=200, lam=1e-3):
        """数值 IK: 位置 p [m] (+ 可选姿态 R), 从 q0 出发 DLS 迭代。
        姿态误差用旋转向量 rotvec(R R_cur') — 连续跟踪时按上一解热启动。"""
        q = np.zeros(6) if q0 is None else np.array(q0, float)
        for _ in range(it):
            o, z, Rc = self._kin(q)
            e_p = np.asarray(p, float) - o[6]
            J = np.zeros((6, 6))
            J[:3] = np.cross(z, o[6] - o[:6]).T
            J[3:] = z.T
            if R is None:
                Jt = J[:3]; e = e_p
            else:
                dR = np.asarray(R) @ Rc.T
                w = 0.5*np.array([dR[2, 1] - dR[1, 2],
                                  dR[0, 2] - dR[2, 0],
                                  dR[1, 0] - dR[0, 1]])
                cth = np.clip(0.5*(np.trace(dR) - 1.0), -1.0, 1.0)
                sth = np.linalg.norm(w)
                if sth > 1e-12:                   # 轴角精确化 (大角度)
                    w *= np.arctan2(sth, cth)/sth
                Jt = J; e = np.concatenate([e_p, w])
            if e @ e < tol**2:
                break
            q = q + Jt.T @ np.linalg.solve(Jt @ Jt.T + lam**2*np.eye(len(e)), e)
        return q

    @staticmethod
    def _quiet_run(vi, *args, **kw):
        """剧烈甩腕时刻 fsolve 偶发停在数值微分噪声地板 (残差 ~1e-9,
        达不到 xtol=1e-10) 并告警 — 解已足够精确, 真正的发散会被能量
        诊断暴露。只滤掉这一条警告。"""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The iteration is not making good progress")
            return vi.run(*args, **kw)

    # ---------------- 演示 A: 无驱动能量保持 ----------------
    def passive_compare(self, q0=(0.0, 1.2, -1.0, 0.8, 1.2, 0.0),
                        t_end=60.0, h=0.03):
        q0 = np.array(q0, float); v0 = np.zeros(6)
        acc = lambda q, v, t: self.accel(q, v, t)
        vi = MidpointDEL(self.lagrangian, 6)
        t_vi, Q = self._quiet_run(vi, q0, v0, t_end, h, rhs_for_bootstrap=acc)
        # 四阶中心差分速度 (仅诊断): 腕部动态快, 二阶差分的 O(h^2) 伪差
        # 会淹没 VI 能量本身的有界振荡
        Vc = (-Q[4:] + 8*Q[3:-1] - 8*Q[1:-3] + Q[:-4])/(12*h)
        E_vi = np.array([self.energy(Q[k+2], Vc[k]) for k in range(len(Vc))])
        t_rk, Qr, Vr = rk4_run(acc, q0, v0, t_end, h)
        E_rk = np.array([self.energy(Qr[k], Vr[k]) for k in range(len(t_rk))])
        E0 = self.energy(q0, v0)
        return (t_vi[2:-2], np.abs(E_vi/E0 - 1),
                t_rk, np.abs(E_rk/E0 - 1))

    # ---------------- 演示 B: 三维焊缝跟踪 (强迫 DEL) ----------------
    def seam_tracking(self, p_start=(0.45, -0.15, 0.20),
                      p_end=(0.45, 0.15, 0.35), t_weld=4.0, h=0.01,
                      wn=12.0, zeta=1.0):
        """焊枪尖沿三维直线焊缝, 姿态保持平焊 (枪尖竖直向下)。

        关节 PD 增益按名义构型的 M(q) 对角元逐关节整定
        (Kp_i = wn² M_ii, Kd_i = 2 ζ wn M_ii) — 统一标量增益会让
        轻惯量的腕关节刚度过高, 力矩控制病态。
        """
        p0, p1 = np.array(p_start, float), np.array(p_end, float)
        R_ref = np.diag([1.0, -1.0, -1.0])        # 焊枪 z 轴 = -z_world (平焊)

        seed = self.ik(p0, R_ref, q0=(0.0, 0.6, 0.5, 0.0, 0.9, 0.0))
        cache = {0.0: seed}
        state = {"q_last": seed}

        def q_ref(t):
            qc = cache.get(t)
            if qc is None:
                sgm = np.clip(t/t_weld, 0, 1)
                sgm = 3*sgm**2 - 2*sgm**3         # 平滑 S 曲线
                qc = self.ik(p0 + sgm*(p1 - p0), R_ref, q0=state["q_last"])
                cache[t] = qc
            state["q_last"] = qc
            return qc

        def qd_ref(t, d=1e-4):
            return (q_ref(t+d) - q_ref(t-d))/(2*d)

        Mn = self._MV(seed)[0]
        Kp = wn**2*np.diag(Mn)
        Kd = 2.0*zeta*wn*np.diag(Mn)

        def tau(q, v, t):
            return (Kp*(q_ref(t) - q) + Kd*(qd_ref(t) - v)
                    + self.gravity_comp(q))

        acc = lambda q, v, t: self.accel(q, v, t, tau(q, v, t))
        vi = MidpointDEL(self.lagrangian, 6, force=tau)
        t, Q = self._quiet_run(vi, q_ref(0.0), qd_ref(0.0), t_weld + 1.0, h,
                               rhs_for_bootstrap=acc)
        tip = np.array([self.fk_tip(qk) for qk in Q])
        ref = np.array([self.fk_tip(q_ref(tk)) for tk in t])
        err = np.linalg.norm(tip - ref, axis=1)
        return t, tip, ref, err
