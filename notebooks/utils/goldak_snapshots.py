# -*- coding: utf-8 -*-
"""GoldakFDM 带快照的求解 + GIF 写出 (源自 robot6 notebook §5b).

``GoldakFDM.run()`` 只保留末时刻场; :func:`solve_with_snapshots` 复刻其
**常系数路径**步进循环 (thermal.py 的显式差分 + 逐步热源重归一化 + 远场
Dirichlet), 在给定帧时刻抓拍 ``(t, T, peak)`` — 单次通过, 无 O(N²) 重解。
改动 thermal.py 的步进逻辑后需同步本文件 (对拍: 末帧 T 应与
``run()`` 结果逐位一致, 见 tests 提示)。

GIF 用 Pillow 写出 (matplotlib 既有依赖, 不为 imageio 加包)。
"""
import numpy as np


def solve_with_snapshots(g, t_end, x_start, frame_times):
    """步进求解 ``g`` (GoldakFDM) 并在 ``frame_times`` 时刻抓拍。

    返回 ``[(t, T, peak), ...]`` (数组为拷贝)。仅支持常系数传导路径 —
    对流增强 (``convecting``) 的变系数步进未复刻。
    注意: 会消耗 ``g`` 的初始温度场; 求解后 ``g.T``/``g.peak`` **不**更新
    (与 ``run()`` 不同), 需要时用返回的末帧。
    """
    if getattr(g, 'convecting', False):
        raise NotImplementedError("仅复刻常系数传导路径 (convection=None)")
    dt = 0.4 * g.dx**2 / (6 * g.alpha)            # 显式稳定性 (同 run())
    n_steps = int(t_end / dt)
    T, dx2 = g.T, g.dx**2
    peak = np.full_like(T, g.T0)
    P_target = g.eta * g.Q * (0.5 if g.symmetric else 1.0)
    snap_steps = set(np.clip((np.asarray(frame_times) / dt).astype(int),
                             0, n_steps - 1))
    snaps = []
    for n in range(n_steps):
        t = n * dt
        dxo, dyo = g.weave.offset(t) if g.weaving else (0.0, 0.0)
        xs, ys = x_start + g.v*t + dxo, dyo
        q = g.goldak_q(xs, ys)
        q *= P_target / max(q.sum() * g.dx**3, 1e-9)   # 数值重归一化
        Tp = np.pad(T, 1, mode="edge")                 # 边界零通量
        lap = (Tp[2:, 1:-1, 1:-1] + Tp[:-2, 1:-1, 1:-1]
               + Tp[1:-1, 2:, 1:-1] + Tp[1:-1, :-2, 1:-1]
               + Tp[1:-1, 1:-1, 2:] + Tp[1:-1, 1:-1, :-2] - 6*T)
        T = T + dt*(g.alpha*lap/dx2 + q/(g.rho*g.cp))
        T[0] = T[-1] = g.T0                            # 远场 Dirichlet
        T[:, -1] = g.T0
        if not g.symmetric:
            T[:, 0] = g.T0
        T[:, :, -1] = g.T0
        peak = np.maximum(peak, T)
        if n in snap_steps:
            snaps.append((t, T.copy(), peak.copy()))
    return snaps


def save_gif(pil_frames, path, fps=8, hold_last_ms=1500):
    """Pillow 写 GIF: ``fps`` 帧率, 尾帧停留 ``hold_last_ms``, 无限循环。"""
    ms = int(round(1000.0 / fps))
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:],
                       loop=0, optimize=True,
                       duration=[ms]*(len(pil_frames) - 1) + [hold_last_ms])
