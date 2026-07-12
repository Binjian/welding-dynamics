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


class MujocoWarpArm:
    """SixDofArm 的 MuJoCo Warp 批量后端 (可选依赖 mujoco-warp)。

    与 MujocoArm 完全同一 MJCF (build_mjcf 单参数源), 但由 NVIDIA Warp
    内核对 ``nworld`` 个世界**同步批量**积分: 有 CUDA GPU 时可上千世界
    并行 (参数扫描 / 蒙特卡洛 / 强化学习式 rollout), 无 GPU 时自动落到
    Warp 的 CPU 设备 — 功能与数值等价, 只是失去大规模并行吞吐。
    构造后 ``self.device`` 报告实际设备 ("cuda"/"cpu")。

    注意事项:

    - mjwarp 以 **float32** 计算 (C-MuJoCo 为 float64): 短程轨迹差
      ~1e-6, 混沌轨迹长程会指数放大 — 参数扫描/统计用途不受影响,
      但不要拿它做长时能量保真结论 (那是变分积分器的地盘);
    - 物理步长在构造时经 MJCF 固定 (timestep), 不像 MujocoArm 可在
      rollout 时覆盖;
    - 力矩律仍在 CPU 上逐世界求值 (track_batch 的 tau_funs), 物理步
      批量执行 — 控制器求值是 CPU⇄设备同步点, GPU 上想要满吞吐需把
      控制也写成 Warp 内核 (超出本包范围)。
    """

    def __init__(self, arm, nworld=1, timestep=1e-3, integrator="Euler",
                 damping=0.0, frictionloss=0.0):
        import mujoco
        import warp as wp                          # 可选依赖, 延迟导入
        import mujoco_warp as mjw
        wp.init()
        self._wp, self._mjw = wp, mjw
        self.arm, self.nworld = arm, int(nworld)
        self.timestep = float(timestep)
        self.xml = build_mjcf(arm, timestep=timestep, integrator=integrator,
                              damping=damping, frictionloss=frictionloss)
        self.mjm = mujoco.MjModel.from_xml_string(self.xml)
        self.model = mjw.put_model(self.mjm)
        self.data = mjw.put_data(self.mjm, mujoco.MjData(self.mjm),
                                 nworld=self.nworld)
        self._tcp = self.mjm.site("tcp").id
        self.device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"

    # ---------------- 批量状态读写 ----------------
    def _write(self, dst, src):
        wp = self._wp
        wp.copy(dst, wp.array(np.asarray(src), dtype=dst.dtype,
                              device=dst.device))

    def set_state(self, q0, v0=None):
        """写入各世界状态 (可广播: 单个 q0 -> 所有世界) 并 forward。"""
        q0 = np.broadcast_to(np.asarray(q0, float), (self.nworld, 6))
        v0 = (np.zeros((self.nworld, 6)) if v0 is None
              else np.broadcast_to(np.asarray(v0, float), (self.nworld, 6)))
        self._write(self.data.qpos, q0)
        self._write(self.data.qvel, v0)
        self._mjw.forward(self.model, self.data)

    def tip(self):
        """各世界当前 TCP 位置 (nworld, 3)。"""
        return self.data.site_xpos.numpy()[:, self._tcp].astype(float)

    # ---------------- 批量无驱动滚转 ----------------
    def passive_rollout_batch(self, q0_batch, t_end):
        """无驱动批量摆动: 返回 (t, Q, V, E), 形状 (nworld, n+1, ...)。

        能量用 SixDofArm.energy 统一度量 (float64 上转)。
        """
        mjw, m, d = self._mjw, self.model, self.data
        n = int(round(t_end/self.timestep))
        self.set_state(q0_batch)
        Q = np.zeros((self.nworld, n + 1, 6))
        V = np.zeros_like(Q)
        Q[:, 0] = d.qpos.numpy()
        for k in range(n):
            mjw.step(m, d)
            Q[:, k+1] = d.qpos.numpy()
            V[:, k+1] = d.qvel.numpy()
        E = np.array([[self.arm.energy(Q[w, k], V[w, k])
                       for k in range(n + 1)] for w in range(self.nworld)])
        return self.timestep*np.arange(n + 1), Q, V, E

    # ---------------- 批量轨迹跟踪 ----------------
    def track_batch(self, tau_funs, q0, v0, t_end, h_ctrl=0.01):
        """批量跟踪: 每个世界一个力矩律 ``tau_funs[i](q, v, t)``。

        q0/v0 (nworld, 6) 为各世界初始状态; 力矩在控制周期 h_ctrl 内
        零阶保持 (CPU 逐世界求值), 物理步批量执行。
        返回 (t, tip): tip 形状 (nworld, n+1, 3)。
        """
        if len(tau_funs) != self.nworld:
            raise ValueError(f"需要 {self.nworld} 个力矩律, 得到 {len(tau_funs)}")
        mjw, m, d = self._mjw, self.model, self.data
        sub = max(1, int(round(h_ctrl/self.timestep)))
        n = int(round(t_end/h_ctrl))
        self.set_state(q0, v0)
        tip = np.zeros((self.nworld, n + 1, 3))
        tip[:, 0] = self.tip()
        for k in range(n):
            qs = d.qpos.numpy().astype(float)
            vs = d.qvel.numpy().astype(float)
            tau = np.stack([f(qs[w], vs[w], k*h_ctrl)
                            for w, f in enumerate(tau_funs)])
            self._write(d.qfrc_applied, tau)
            for _ in range(sub):
                mjw.step(m, d)
            tip[:, k+1] = self.tip()
        return h_ctrl*np.arange(n + 1), tip
