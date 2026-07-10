# -*- coding: utf-8 -*-
"""仿真主程序: 运行模块 1/3/4/5, 图片输出到 cfg.output.dir (默认 ./results/)。

配置由 Hydra 组合, 见 `conf/sim.yaml`:

    uv run welding-sim                                   # 默认工况
    uv run welding-sim process=db_median                 # 生产数据库中位工况
    uv run welding-sim material=stainless_steel solver=fine
    uv run welding-sim --multirun process=db_p10,db_median,db_p90 output=per_run
"""
from pathlib import Path

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import DictConfig

from .config import arc_power  # noqa: F401  (导入即注册 wd.* 解析器)

plt.rcParams["font.family"] = "DejaVu Sans"


@hydra.main(version_base="1.3", config_path="conf", config_name="sim")
def main(cfg: DictConfig):
    OUT = Path(cfg.output.dir); OUT.mkdir(parents=True, exist_ok=True)
    dpi = cfg.output.dpi
    print(f"[配置] process={cfg.process.name} material={cfg.material.name} "
          f"solver={cfg.solver.name} -> {OUT}/")

    # ---------- 模块 1 ----------
    rg = cfg.run.gmaw
    ctwd_fun = (lambda t: rg.ctwd + (rg.ctwd_step if t >= rg.step_time else 0))
    res = instantiate(cfg.gmaw).simulate(
        t_end=rg.t_end, x0=tuple(rg.x0),
        WFS_fun=lambda t: rg.wfs, CTWD_fun=ctwd_fun)
    P_ss = float(np.mean(res["P"][-rg.steady_tail:]))
    print(f"[1 自调节] 稳态 I={res['I'][-1]:.0f} A, "
          f"Va={res['Va'][-1]:.1f} V, P={P_ss:.0f} W")

    # 模块 1 的工作点由 (WFS, CTWD) 决定, 而工艺数据库不记录送丝速度 ->
    # 稳态电流不一定等于 process.current_A。热源功率另由 process.arc_power_W 给定,
    # 不受此偏差影响; 这里显式提示, 免得把 db_* 预设的标称电流误当成模型输出。
    I_db = cfg.process.current_A
    if I_db is not None:
        dev = abs(float(res["I"][-1]) - I_db) / I_db
        if dev > 0.1:
            print(f"[1 提示] 与 {cfg.process.name} 标称的 {I_db:.0f} A 相差 {dev:.0%}: "
                  f"数据库无 WFS, 模块 1 仍以 wfs={rg.wfs} m/s 驱动。")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.suptitle("Module 1: GMAW self-regulation "
                 f"(CTWD step @{rg.step_time} s)")
    for ax, key, lab, c in zip(axes.flat,
                               ["I", "la", "s", "Va"],
                               ["I [A]", "arc length [mm]",
                                "stick-out [mm]", "V_arc [V]"],
                               "brgm"):
        ysc = 1e3 if key in ("la", "s") else 1
        ax.plot(res["t"], res[key]*ysc, c); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.axvline(rg.step_time, color="k", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(OUT / "m1_self_regulation.png", dpi=dpi)

    # ---------- 模块 3: 熔滴过渡 ----------
    rd_ = cfg.run.droplet
    dd = instantiate(cfg.droplet)
    I_arr = np.linspace(rd_.I_min, rd_.I_max, rd_.n_I)
    freq, dmean = dd.current_sweep(I_arr)
    # 过渡电流: 熔滴直径降到约 1.3 倍焊丝直径以下 -> 喷射过渡
    below = np.where(dmean <= rd_.transition_dia_factor * 2 * dd.rw)[0]
    i_tr = below[0] if len(below) else len(I_arr) - 1
    print(f"[3 熔滴] 滴状->喷射 过渡电流 ~ {I_arr[i_tr]:.0f} A")

    r_glob = dd.simulate(rd_.globular_I, t_end=rd_.globular_t_end)
    r_spray = dd.simulate(rd_.spray_I, t_end=rd_.spray_t_end)

    fig3, ax3 = plt.subplots(1, 3, figsize=(13, 4))
    ax3[0].plot(I_arr, dmean*1e3, "o-")
    ax3[0].axhline(2*dd.rw*1e3, color="gray", ls=":", label="wire dia.")
    ax3[0].axvline(I_arr[i_tr], color="r", ls="--", label="transition I")
    ax3[0].set_xlabel("I [A]"); ax3[0].set_ylabel("droplet dia. [mm]")
    ax3[0].legend(); ax3[0].set_title("Droplet size vs current")
    ax3[1].plot(I_arr, freq, "s-", color="purple")
    ax3[1].set_xlabel("I [A]"); ax3[1].set_ylabel("detach freq [Hz]")
    ax3[1].set_title("Transfer frequency")
    ax3[2].plot(r_glob["t"]*1e3, r_glob["rd"]*1e3,
                label=f"{rd_.globular_I:.0f} A (globular)")
    ax3[2].plot(r_spray["t"]*1e3, r_spray["rd"]*1e3,
                label=f"{rd_.spray_I:.0f} A (spray)")
    ax3[2].set_xlabel("t [ms]"); ax3[2].set_ylabel("droplet radius [mm]")
    ax3[2].legend(); ax3[2].set_title("Growth-detach cycles")
    for a in ax3: a.grid(alpha=0.3)
    fig3.tight_layout(); fig3.savefig(OUT / "m3_droplet.png", dpi=dpi)

    # ---------- 模块 4: Goldak FDM ----------
    # process.arc_power_W = null 时, 热源功率取模块 1 的自调节稳态功率
    Q = arc_power(cfg, fallback=P_ss)
    src = "process.arc_power_W" if cfg.process.arc_power_W is not None \
        else "模块 1 稳态 P_ss"
    print(f"[4 热源] Q = {Q:.0f} W (来自 {src}), v = {cfg.process.travel_speed_m_s*1e3:.2f} mm/s")

    g = instantiate(cfg.goldak, Q=Q)
    if g.weaving:
        print(f"[4 摆动] {g.weave.describe()} -> 全宽网格 "
              f"{g.Nx}x{g.Ny}x{g.Nz} (半对称失效, 计算量约 ×2)")
    g.run(t_end=cfg.run.goldak.t_end, x_start=cfg.run.goldak.x_start)
    L, W, D = g.pool_size()
    print(f"[4 Goldak-FDM] 熔池 长 {L:.1f} / 宽 {W:.1f} / 深 {D:.1f} mm")

    ros = instantiate(cfg.rosenthal, Q=Q)
    XI, Y, Tr = ros.surface_field()
    w_ros = 0.0
    melt = Tr >= ros.Tm
    if melt.any():
        w_ros = (Y[melt].max()-Y[melt].min())*1e3
    print(f"[4 对比] Rosenthal 熔池宽 {w_ros:.1f} mm (点源, 偏宽属正常)")

    fig4, a4 = plt.subplots(1, 3, figsize=(14, 4))
    Tm = cfg.material.Tm

    def levels(*vals):
        """等值线电平必须严格递增; 低熔点材料 (如铝 Tm=913 K) 会与固定电平冲突。"""
        return sorted(set(float(v) for v in vals))

    lv = levels(cfg.material.T0, 600, 900, 1273, Tm, 2600)
    jc = g.j_center                       # 半模型=0 (对称面); 全宽摆动模型=中心线
    c0 = a4[0].contourf(g.x*1e3, g.y*1e3, g.T[:, :, 0].T, levels=lv,
                        cmap="hot")
    a4[0].contour(g.x*1e3, g.y*1e3, g.T[:, :, 0].T, levels=[Tm],
                  colors="cyan")
    a4[0].set_title("Top view T (half model)" if g.symmetric
                    else "Top view T (full width, weaving)")
    a4[0].set_xlabel("x [mm]"); a4[0].set_ylabel("y [mm]")
    fig4.colorbar(c0, ax=a4[0])
    c1 = a4[1].contourf(g.x*1e3, -g.z*1e3, g.T[:, jc, :].T, levels=lv,
                        cmap="hot")
    a4[1].contour(g.x*1e3, -g.z*1e3, g.T[:, jc, :].T, levels=[Tm],
                  colors="cyan")
    a4[1].set_title("Longitudinal section")
    a4[1].set_xlabel("x [mm]"); a4[1].set_ylabel("z [mm]")
    fig4.colorbar(c1, ax=a4[1])
    c2 = a4[2].contourf(g.x*1e3, -g.z*1e3, g.peak[:, jc, :].T,
                        levels=levels(cfg.material.T0, 773, 1073, 1273, Tm, 3000),
                        cmap="inferno")
    a4[2].contour(g.x*1e3, -g.z*1e3, g.peak[:, jc, :].T,
                  levels=levels(1273, Tm), colors=["lime", "cyan"][:len(levels(1273, Tm))])
    a4[2].set_title("Peak T: fusion zone & HAZ")
    a4[2].set_xlabel("x [mm]")
    fig4.colorbar(c2, ax=a4[2])
    fig4.suptitle("Module 4: Goldak double-ellipsoid + 3D transient FDM")
    fig4.tight_layout(); fig4.savefig(OUT / "m4_goldak_fdm.png", dpi=dpi)

    # ---------- 模块 5: 短路 / CMT ----------
    rs = cfg.run.short_circuit
    std = instantiate(cfg.short_circuit, cmt=False).simulate(t_end=rs.t_end, dt=rs.dt)
    cmt = instantiate(cfg.short_circuit, cmt=True).simulate(t_end=rs.t_end, dt=rs.dt)
    for name, o in [("标准短路", std), ("CMT", cmt)]:
        ncyc = int(np.sum(np.diff(o[:, 3]) > 0))
        f = ncyc / o[-1, 0]
        Ipk = o[:, 1].max()
        print(f"[5 {name}] 过渡频率 ~ {f:.0f} Hz, 峰值电流 {Ipk:.0f} A")

    fig5, a5 = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
    for j, (o, name) in enumerate([(std, "Standard short-circuit"),
                                   (cmt, "CMT (current control + retract)")]):
        a5[0, j].plot(o[:, 0]*1e3, o[:, 1], "b", lw=0.9)
        a5[0, j].set_title(name); a5[0, j].set_ylabel("I [A]")
        a5[1, j].plot(o[:, 0]*1e3, o[:, 2], "r", lw=0.9)
        a5[1, j].fill_between(o[:, 0]*1e3, 0, 30, where=o[:, 3] > 0.5,
                              color="gray", alpha=0.25,
                              label="short phase")
        a5[1, j].set_ylabel("V [V]"); a5[1, j].set_xlabel("t [ms]")
        a5[1, j].legend(loc="upper right", fontsize=8)
        for a in (a5[0, j], a5[1, j]):
            a.grid(alpha=0.3)
    fig5.suptitle("Module 5: short-circuit transfer vs CMT waveforms")
    fig5.tight_layout(); fig5.savefig(OUT / "m5_short_cmt.png", dpi=dpi)


if __name__ == "__main__":
    main()
