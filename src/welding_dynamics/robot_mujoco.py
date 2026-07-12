# -*- coding: utf-8 -*-
"""模块 7b 扩展: SixDofArm 的 MuJoCo 多体动力学后端 (可选依赖)。

同一份参数, 两个动力学引擎:

- ``build_mjcf(arm)`` 把 SixDofArm 的标准 DH 表 + 圆柱连杆几何/质量 +
  折算电机惯量 (映射为 MuJoCo joint ``armature``) 程序化生成 MJCF —
  不存在第二份参数来源, 两个后端的 M(q) 应当一致到机器精度
  (validation notebook 里核对到 ~1e-12)。
- ``MujocoArm`` 包装 MjModel/MjData: 质量矩阵 / FK 交叉验证、无驱动
  ``passive_rollout`` (Euler / implicitfast / RK4), 以及与
  ``SixDofArm.track_path`` **同一 PD 力矩律** (共享 _pd_tracking_law)
  的轨迹跟踪。跟踪可叠加 MuJoCo 才有的关节粘滞阻尼与库仑摩擦
  (frictionloss) — 评估理想变分模型之外的执行器损耗对摆动执行的影响。

MuJoCo 仅在 ``MujocoArm`` 构造时延迟导入 (装法: ``uv sync --extra
mujoco``); 本模块顶层只依赖 numpy, 缺 mujoco 时包的其余功能不受影响。

SixDofArm -> MJCF 的对应关系:
    DH: A_i = Rz(q_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)
        -> body_i 相对 body_{i-1} 固定位姿 = 平移 (a_{i-1},0,d_{i-1}) +
           Rx(α_{i-1}); hinge 绕 body_i 局部 z (即帧 i-1 的 z)
    连杆 i 圆柱 [o_{i-1}, o_i] -> cylinder geom fromto (0,0,0)-(a_i,0,d_i)
    零长度连杆 (球腕 o4=o5)    -> sphere geom (均质球, 同 _MV 退化分支)
    J_rotor                    -> joint armature
    TCP (帧 6)                 -> body_6 内 site "tcp"
接触全局关闭 (相邻圆柱在关节处必然重叠), 重力 (0, 0, -g)。
"""
import numpy as np


def _fmt(*vals):
    return " ".join(f"{v:.12g}" for v in vals)


def build_mjcf(arm, timestep=1e-3, integrator="Euler",
               damping=0.0, frictionloss=0.0):
    """由 SixDofArm 生成 MJCF XML 字符串。

    damping [N·m·s/rad] / frictionloss [N·m] 逐关节同值 (标量),
    默认 0 = 与理想变分模型同设; integrator 可选 Euler (半隐式,
    对保守系统即辛 Euler) / implicitfast / RK4。
    """
    d, a, al = arm.dh_d, arm.dh_a, arm.dh_alpha
    lines = [
        '<mujoco model="sixdofarm">',
        '  <compiler angle="radian"/>',
        f'  <option timestep="{timestep:g}" gravity="0 0 {-arm.g:g}" '
        f'integrator="{integrator}">',
        '    <flag energy="enable" contact="disable"/>',
        '  </option>',
        '  <worldbody>',
    ]
    indent = '    '
    for i in range(6):
        if i == 0:
            pose = 'pos="0 0 0"'
        else:                                     # 帧 i-1 的固定部分
            half = 0.5*al[i-1]
            pose = (f'pos="{_fmt(a[i-1], 0.0, d[i-1])}" '
                    f'quat="{_fmt(np.cos(half), np.sin(half), 0.0, 0.0)}"')
        lines.append(f'{indent}<body name="link{i+1}" {pose}>')
        indent += '  '
        lines.append(
            f'{indent}<joint name="j{i+1}" type="hinge" axis="0 0 1" '
            f'armature="{arm.J_rotor[i]:.12g}" damping="{damping:g}" '
            f'frictionloss="{frictionloss:g}"/>')
        if np.hypot(a[i], d[i]) > 1e-9:
            lines.append(
                f'{indent}<geom type="cylinder" '
                f'fromto="0 0 0 {_fmt(a[i], 0.0, d[i])}" '
                f'size="{arm.r_link:.12g}" mass="{arm.m[i]:.12g}"/>')
        else:
            lines.append(
                f'{indent}<geom type="sphere" size="{arm.r_link:.12g}" '
                f'mass="{arm.m[i]:.12g}"/>')
    half = 0.5*al[5]                              # TCP = 帧 6
    lines.append(
        f'{indent}<site name="tcp" pos="{_fmt(a[5], 0.0, d[5])}" '
        f'quat="{_fmt(np.cos(half), np.sin(half), 0.0, 0.0)}"/>')
    for _ in range(6):
        indent = indent[:-2]
        lines.append(f'{indent}</body>')
    lines += ['  </worldbody>', '</mujoco>']
    return "\n".join(lines)


class MujocoArm:
    """SixDofArm 的 MuJoCo 后端 (构造时才导入 mujoco)。"""

    def __init__(self, arm, timestep=1e-3, integrator="Euler",
                 damping=0.0, frictionloss=0.0):
        import mujoco                             # 可选依赖, 延迟导入
        self._mj = mujoco
        self.arm = arm
        self.xml = build_mjcf(arm, timestep=timestep, integrator=integrator,
                              damping=damping, frictionloss=frictionloss)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._tcp = self.model.site("tcp").id

    # ---------------- 交叉验证接口 ----------------
    def mass_matrix(self, q):
        """M(q) (含 armature) — 与 SixDofArm._MV 的 M 交叉核对。"""
        m, d = self.model, self.data
        self._mj.mj_resetData(m, d)
        d.qpos[:] = q
        self._mj.mj_forward(m, d)
        M = np.zeros((6, 6))
        self._mj.mj_fullM(m, d, M)
        return M

    def fk_tip(self, q):
        m, d = self.model, self.data
        d.qpos[:] = q
        self._mj.mj_kinematics(m, d)
        return d.site_xpos[self._tcp].copy()

    # ---------------- 无驱动滚转 ----------------
    def passive_rollout(self, q0, t_end, h=None):
        """无驱动摆动: 返回 (t, Q, V, E)。

        h=None 沿用模型 timestep, 否则临时覆盖 (与 VI 同步长对比用)。
        能量用 SixDofArm.energy 统一度量 — 两个后端同一参数、同一定义,
        E 的差异即积分器的差异。
        """
        mj, m, d = self._mj, self.model, self.data
        if h is not None:
            m.opt.timestep = float(h)
        hh = m.opt.timestep
        n = int(round(t_end/hh))
        mj.mj_resetData(m, d)
        d.qpos[:] = np.asarray(q0, float)
        Q = np.zeros((n + 1, 6)); V = np.zeros((n + 1, 6))
        Q[0] = d.qpos
        for k in range(n):
            mj.mj_step(m, d)
            Q[k+1] = d.qpos; V[k+1] = d.qvel
        E = np.array([self.arm.energy(Q[k], V[k]) for k in range(n + 1)])
        return hh*np.arange(n + 1), Q, V, E

    # ---------------- 轨迹跟踪 (与 SixDofArm.track_path 同一力矩律) ----------------
    def track_path(self, p_ref_fun, t_end, h_ctrl=0.01, wn=12.0, zeta=1.0,
                   R_ref=None, q_seed=(0.0, 0.6, 0.5, 0.0, 0.9, 0.0)):
        """MuJoCo 执行的轨迹跟踪, 返回 (t, tip, ref, err) — 与
        SixDofArm.track_path 同构。

        力矩律取自 SixDofArm._pd_tracking_law (完全同一闭包); 控制周期
        h_ctrl 内力矩零阶保持, 物理积分用模型 timestep (默认 1 ms) 细步。
        关节阻尼/摩擦由构造参数决定, 不在力矩律里 — 即控制器对执行器
        损耗"无感", 正如真实部署。
        """
        mj, m, d = self._mj, self.model, self.data
        q_ref, qd_ref, tau = self.arm._pd_tracking_law(
            p_ref_fun, wn, zeta, R_ref, q_seed)
        sub = max(1, int(round(h_ctrl/m.opt.timestep)))
        n = int(round(t_end/h_ctrl))
        mj.mj_resetData(m, d)
        d.qpos[:] = q_ref(0.0); d.qvel[:] = qd_ref(0.0)
        mj.mj_forward(m, d)
        t = h_ctrl*np.arange(n + 1)
        tip = np.zeros((n + 1, 3))
        tip[0] = d.site_xpos[self._tcp]
        for k in range(n):
            u = tau(d.qpos.copy(), d.qvel.copy(), k*h_ctrl)
            d.qfrc_applied[:] = u
            for _ in range(sub):
                mj.mj_step(m, d)
            tip[k+1] = d.site_xpos[self._tcp]
        ref = np.array([np.asarray(p_ref_fun(tk), float) for tk in t])
        err = np.linalg.norm(tip - ref, axis=1)
        return t, tip, ref, err

    def launch_viewer(self):
        """交互式查看模型 (阻塞, 需本地显示): mujoco.viewer.launch。"""
        from mujoco import viewer
        viewer.launch(self.model, self.data)
