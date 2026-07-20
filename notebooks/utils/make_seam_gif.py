# -*- coding: utf-8 -*-
"""焊缝成形 GIF 生成脚本 (robot6 notebook §5b 的独立版).

UR5e 沿 2 Hz × 4 mm 三角摆轨迹施焊 (数据库中位工况), 逐帧渲染:
枪尖处亮红瞬时熔池 + 身后暗红凝固焊缝 (peak 包络生长, 摆动扇贝纹) +
顶面 ≥400 K 热晕。默认写到仓库 results/robot6_weave_seam.gif。

用法 (仓库根目录)::

    uv run python notebooks/utils/make_seam_gif.py [输出.gif] [帧数]
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))   # notebooks/ -> utils 可导入
from utils.goldak_snapshots import solve_with_snapshots, save_gif

from welding_dynamics import ensure_display, RobotExecutedWeave
from welding_dynamics.config import arc_power
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/robot6_weave_seam.gif"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40

# 渲染几何常量 (同 notebook §2)
R_LINK = [45.0, 38.0, 32.0, 22.0, 20.0, 16.0]    # mm 连杆显示半径
R_JOINT = [50.0, 42.0, 36.0, 26.0, 24.0]         # mm 关节球


def compose_cfg(config_name, *overrides):
    with initialize_config_module(config_module="welding_dynamics.conf",
                                  version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def add_arm(p, arm, q):
    """按 _kin(q) 画机械臂 (单位 mm): 连杆圆柱 / 关节球 / 焊枪锥 / 底座。"""
    o, z, R = arm._kin(q)
    o = o*1e3
    tz = R[:, 2]
    ends = o.copy()
    ends[6] = o[6] - tz*50.0
    for i in range(1, 7):
        seg = ends[i] - o[i-1]
        L = np.linalg.norm(seg)
        if L > 1e-3:
            p.add_mesh(pv.Cylinder(center=0.5*(o[i-1] + ends[i]), direction=seg,
                                   radius=R_LINK[i-1], height=L),
                       color="#a7a9ac", smooth_shading=True)
    for i in range(1, 6):
        p.add_mesh(pv.Sphere(radius=R_JOINT[i-1], center=o[i]),
                   color="#57a7c6", smooth_shading=True)
    p.add_mesh(pv.Cone(center=o[6] - tz*27.5, direction=tz,
                       height=55.0, radius=11.0),
               color="#c0392b", smooth_shading=True)
    p.add_mesh(pv.Cylinder(center=(0, 0, -15.0), direction=(0, 0, 1),
                           radius=90.0, height=30.0), color="#6b7078")


def main():
    ensure_display()

    # ---- §1 复刻: UR5e 跟踪 + IK 姿态帧 ----
    arm = instantiate(compose_cfg("sim_vi", "model@robot6=robot6_ur5e").robot6)
    cfg = compose_cfg("sim_3d", "process=db_median", "solver=fine")
    weave = instantiate(compose_cfg("sim_3d", "weave=triangle").weave)
    v_weld = float(cfg.process.travel_speed_m_s)
    p0 = np.array([0.45, 0.0, 0.25])
    t_track = float(cfg.solver.t_end)
    r_ref = np.diag([1.0, -1.0, -1.0])
    q_seed0 = (0.3, -2.0, -1.6, 2.1, -1.57, 1.9)

    def p_ref(t):
        dx, dy = weave.offset(t)
        return p0 + np.array([v_weld*t + dx, dy, 0.0])

    print("跟踪仿真...", flush=True)
    t_tr, tip, _, _ = arm.track_path(p_ref, t_track, q_seed=q_seed0)
    dtq = t_track/250
    q_grid, q_seed = [], q_seed0
    for tk in np.arange(251)*dtq:
        q_seed = arm.ik(p_ref(tk), r_ref, q0=q_seed)
        q_grid.append(q_seed)
    q_grid = np.array(q_grid)

    def q_at(t):
        return q_grid[min(250, max(0, int(round(t/dtq))))]

    rw = RobotExecutedWeave.from_tracking(t_tr, tip, p0, v_weld,
                                          frequency_Hz=weave.frequency_Hz)

    # ---- 带快照的求解 ----
    g = instantiate(cfg.goldak, Q=arc_power(cfg), weave=rw)
    x_start = float(cfg.run.goldak.x_start)
    frame_t = np.linspace(0.0, t_track, N_FRAMES + 1)[1:]
    print(f"求解 (fine, {N_FRAMES} 快照)...", flush=True)
    snaps = solve_with_snapshots(g, t_track, x_start, frame_t)

    # ---- 逐帧渲染 ----
    ggx = (g.x - x_start + p0[0])*1e3
    ggy = (g.y + p0[1])*1e3
    ggz = (p0[2] - g.z)*1e3
    X, Y, Z = np.meshgrid(ggx, ggy, ggz, indexing='ij')
    tm = float(g.Tm)
    # 取景对准焊缝中段, 拉近到 4 mm 摆动清晰可辨 (腕部/焊枪保持入画)
    focal = np.array([p0[0]*1e3 + 0.5*v_weld*t_track*1e3, 0.0, ggz.max()])
    p = pv.Plotter(off_screen=True, window_size=(880, 540))
    frames = []
    print("渲染...", flush=True)
    for t, Tk, peakk in snaps:
        p.clear()
        p.set_background("white")
        add_arm(p, arm, q_at(t))
        p.add_mesh(pv.Box(bounds=(p0[0]*1e3 - 90, p0[0]*1e3 + 110, -70, 70,
                                  p0[2]*1e3 - 20, p0[2]*1e3)),
                   color="#d8d2c4", opacity=0.4)
        seam = np.array([p0*1e3, (p0 + [v_weld*t_track, 0, 0])*1e3]) + [0, 0, 0.3]
        p.add_mesh(pv.lines_from_points(seam), color="black", line_width=2)
        m = t_tr <= t
        if m.sum() > 2:
            p.add_mesh(pv.lines_from_points(tip[m]*1e3 + [0, 0, 0.3])
                       .tube(radius=0.8), color="#d81b1b")
        sg = pv.StructuredGrid(X, Y, Z)
        sg['peak'] = peakk.ravel(order='F')
        sg['inst'] = Tk.ravel(order='F')
        hist = sg.contour([tm], scalars='peak')       # 凝固焊缝 (生长的包络)
        if hist.n_points:
            p.add_mesh(hist, color=(0.62, 0.14, 0.10), opacity=0.9)
        pool = sg.contour([tm], scalars='inst')       # 瞬时熔池 (枪尖亮红)
        if pool.n_points:
            p.add_mesh(pool, color=(1.0, 0.35, 0.05))
        halo = sg.slice(normal='z', origin=(ggx.mean(), 0.0, ggz.max() - 1e-3)) \
                 .threshold(400.0, scalars='inst')    # 热晕, 冷板不遮挡
        if halo.n_points:
            p.add_mesh(halo, scalars='inst', cmap='inferno', opacity=0.85,
                       clim=(g.T0, 2600.0), show_scalar_bar=False)
        p.add_text(f"t = {t:.2f} s", font_size=12, color="black")
        p.add_text("bright: molten pool | dark: solidified seam",
                   position='upper_right', font_size=8, color="#666666")
        p.camera.focal_point = tuple(focal)
        p.camera.position = tuple(focal + [95, -160, 120])
        p.camera.up = (0.0, 0.0, 1.0)
        frames.append(Image.fromarray(p.screenshot(return_img=True)))
    p.close()

    OUT.parent.mkdir(exist_ok=True)
    save_gif(frames, OUT, fps=8)
    print(f"{OUT}: {len(frames)} 帧, {OUT.stat().st_size/1024:.0f} kB")


if __name__ == "__main__":
    main()
