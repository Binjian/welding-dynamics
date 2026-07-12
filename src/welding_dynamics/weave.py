# -*- coding: utf-8 -*-
"""焊枪摆动 (weaving): 给 Goldak 热源中心一个随时间变化的横向偏移 y_s(t)。

约定 (与工艺数据库 `passes.weave.*` 对齐):

- ``amplitude_m`` 是**峰-峰摆幅** (数据库 `amplitude_mm` 那一列, 典型 4 mm),
  因此横向偏移 y_s(t) ∈ [-A/2, +A/2], 焊道总宽度约为 A + 熔池宽。
- ``frequency_Hz`` 是摆动频率 (数据库典型 2 Hz)。注意它比熔滴 Rayleigh 固有
  频率 (~527 Hz, 见模块 6) 低两个数量级 —— 摆动只搬运热源, 不激发熔滴模态。
- ``offset(t) -> (dx, dy)`` 返回相对于无摆动轨迹 (x = x_start + v·t, y = 0)
  的偏移 [m]; dx 为纵向分量 (摆动库中部分波形带前后分量), dy 为横向分量。

摆动库波形的 ``z_pct`` (垂直抬枪) **不建模**: 热源始终贴在工件上表面 z=0。

摆动破坏了 y=0 镜像对称性, 因此 ``GoldakFDM(weave=...)`` 会自动改用全宽网格
(见 thermal.py)。计算量约为半对称模型的两倍。
"""
import numpy as np


class _Weave:
    """摆动运动基类: 子类实现 offset(t)。"""

    amplitude_m = 0.0
    frequency_Hz = 0.0

    def offset(self, t):
        raise NotImplementedError

    def describe(self):
        return (f"{self.__class__.__name__} "
                f"{self.frequency_Hz:g} Hz × {self.amplitude_m*1e3:g} mm")


class HarmonicWeave(_Weave):
    """解析波形摆动: 正弦或三角 (直线摆)。

    shape="sine"     y_s = A/2 · sin(2π f t + φ)
    shape="triangle" 等速往复 (数据库备注 8 的"直线摆"), 同幅同频。
    两者都从 y_s(0)=0 出发, 即热源起步于焊缝中心线。
    """

    def __init__(self, amplitude_m=4.0e-3, frequency_Hz=2.0,
                 shape="triangle", phase=0.0):
        if shape not in ("sine", "triangle"):
            raise ValueError(f"shape 必须是 'sine' 或 'triangle', 得到 {shape!r}")
        self.amplitude_m = float(amplitude_m)
        self.frequency_Hz = float(frequency_Hz)
        self.shape = shape
        self.phase = float(phase)

    def offset(self, t):
        if self.shape == "sine":
            w = np.sin(2*np.pi*self.frequency_Hz*t + self.phase)
        else:
            # 归一化相位 u ∈ [0,1); 平移 1/4 周期使 u=0 时 w=0 (从中心线出发)
            u = (self.frequency_Hz*t + self.phase/(2*np.pi) + 0.25) % 1.0
            w = 4.0*abs(u - 0.5) - 1.0        # 三角波: 0 -> -1 -> 0 -> +1 -> 0
        return 0.0, 0.5*self.amplitude_m*w

    def describe(self):
        name = {"sine": "正弦", "triangle": "三角(直线摆)"}[self.shape]
        return f"{name} {self.frequency_Hz:g} Hz × {self.amplitude_m*1e3:g} mm"


class WaypointWeave(_Weave):
    """摆动库路点波形 (MongoDB `weave_pattern` 文档)。

    一个周期内以 (time_pct, x_pct, y_pct) 路点定义波形, 线性插值并周期延拓。
    百分比的基准在原始工作簿中未定义, 这里按**该波形自身的最大 |y_pct|** 归一,
    使波形的横向行程恰好等于 amplitude_m (峰-峰)。x_pct 用同一比例缩放,
    以保持波形的纵横比。
    """

    def __init__(self, time_pct, y_pct, x_pct=None,
                 amplitude_m=4.0e-3, frequency_Hz=2.0, pattern_id=None):
        tp = np.asarray(time_pct, dtype=float)
        yp = np.asarray(y_pct, dtype=float)
        xp = np.zeros_like(yp) if x_pct is None else np.asarray(x_pct, dtype=float)
        if not (tp.size == yp.size == xp.size) or tp.size < 2:
            raise ValueError("time_pct / y_pct / x_pct 长度必须一致且 >= 2")

        order = np.argsort(tp)               # np.interp 要求 xp 递增
        self.time_pct, self.y_pct, self.x_pct = tp[order], yp[order], xp[order]
        self.amplitude_m = float(amplitude_m)
        self.frequency_Hz = float(frequency_Hz)
        self.pattern_id = pattern_id

        peak = np.abs(self.y_pct).max()
        # 纯纵向波形 (y 全零) 时退化为按 100% 缩放, 避免除零
        self.scale = 0.5*self.amplitude_m / (peak if peak > 0 else 100.0)

    def offset(self, t):
        ph = (self.frequency_Hz*t*100.0) % 100.0    # 周期内相位 [0,100)
        dy = self.scale*np.interp(ph, self.time_pct, self.y_pct, period=100.0)
        dx = self.scale*np.interp(ph, self.time_pct, self.x_pct, period=100.0)
        return float(dx), float(dy)

    def describe(self):
        pid = f"#{self.pattern_id}" if self.pattern_id is not None else "自定义"
        return (f"路点波形 {pid} ({len(self.time_pct)} 点) "
                f"{self.frequency_Hz:g} Hz × {self.amplitude_m*1e3:g} mm")


class RobotExecutedWeave(_Weave):
    """机器人实际执行的摆动: 由 6DOF 机械臂跟踪仿真的 TCP 轨迹采样构成。

    与解析摆动 (指令波形) 不同, 这里的 (dx, dy) 是**实际枪尖**相对
    无摆动匀速中心线的偏差采样序列 — 含指令摆动加上跟踪滞后、幅值
    衰减与换向超调。把机器人执行结果注入 GoldakFDM 即可比较
    "理想摆动 vs 机器人执行摆动" 的熔池差异。

    - ``amplitude_m`` 取实际横向行程的峰-峰值 (>0 时 GoldakFDM 自动
      切换全宽网格), 不再是指令幅值;
    - 采样范围之外按端点保持 (np.interp 的钳位行为);
    - ``from_tracking`` 假定焊缝沿机器人基座 x 轴、恒速 v:
      dx = tip_x - (p0_x + v t), dy = tip_y - p0_y。
    """

    def __init__(self, t, dx, dy, frequency_Hz=0.0, label="机器人执行"):
        t = np.asarray(t, float)
        dx = np.asarray(dx, float)
        dy = np.asarray(dy, float)
        if not (t.size == dx.size == dy.size) or t.size < 2:
            raise ValueError("t / dx / dy 长度必须一致且 >= 2")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t 必须严格递增")
        self._t, self._dx, self._dy = t, dx, dy
        self.amplitude_m = float(np.ptp(dy))
        self.frequency_Hz = float(frequency_Hz)
        self.label = label

    @classmethod
    def from_tracking(cls, t, tip, p0, v, frequency_Hz=0.0, label="机器人执行"):
        """由 SixDofArm.track_path 的 (t, tip) 构造 (焊缝沿基座 x 轴)。

        p0: 焊缝起点 [m]; v: 焊接速度 [m/s] (与 GoldakFDM 的 v 一致,
        参考中心线为 p0 + v t x̂)。"""
        t = np.asarray(t, float)
        tip = np.asarray(tip, float)
        p0 = np.asarray(p0, float)
        return cls(t, tip[:, 0] - (p0[0] + v*t), tip[:, 1] - p0[1],
                   frequency_Hz=frequency_Hz, label=label)

    def offset(self, t):
        return (float(np.interp(t, self._t, self._dx)),
                float(np.interp(t, self._t, self._dy)))

    def describe(self):
        return (f"{self.label} ({self._t.size} 采样) "
                f"{self.frequency_Hz:g} Hz × 实际 {self.amplitude_m*1e3:.2f} mm")
