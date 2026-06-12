# -*- coding: utf-8 -*-
"""仿真主程序: 运行全部 5 个模块并将图片输出到 ./results/"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .gmaw import GMAWDynamics
from .thermal import RosenthalThermal, GoldakFDM
from .droplet import DropletDynamics
from .short_circuit import ShortCircuitGMAW

plt.rcParams["font.family"] = "DejaVu Sans"
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def main():
    # ---------- 模块 1 ----------
    res = GMAWDynamics().simulate()
    P_ss = float(np.mean(res["P"][1800:]))
    print(f"[1 自调节] 稳态 I={res['I'][-1]:.0f} A, "
          f"Va={res['Va'][-1]:.1f} V, P={P_ss:.0f} W")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.suptitle("Module 1: GMAW self-regulation (CTWD step @0.5 s)")
    for ax, key, lab, c in zip(axes.flat,
                               ["I", "la", "s", "Va"],
                               ["I [A]", "arc length [mm]",
                                "stick-out [mm]", "V_arc [V]"],
                               "brgm"):
        ysc = 1e3 if key in ("la", "s") else 1
        ax.plot(res["t"], res[key]*ysc, c); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.axvline(0.5, color="k", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(OUT / "m1_self_regulation.png", dpi=140)

    # ---------- 模块 3: 熔滴过渡 ----------
    dd = DropletDynamics()
    I_arr = np.linspace(120, 340, 23)
    freq, dmean = dd.current_sweep(I_arr)
    # 过渡电流: 熔滴直径降到约 1.3 倍焊丝直径以下 -> 喷射过渡
    below = np.where(dmean <= 1.3 * 2 * dd.rw)[0]
    i_tr = below[0] if len(below) else len(I_arr) - 1
    print(f"[3 熔滴] 滴状->喷射 过渡电流 ~ {I_arr[i_tr]:.0f} A")

    r_glob = dd.simulate(170, t_end=0.25)
    r_spray = dd.simulate(300, t_end=0.05)

    fig3, ax3 = plt.subplots(1, 3, figsize=(13, 4))
    ax3[0].plot(I_arr, dmean*1e3, "o-")
    ax3[0].axhline(2*dd.rw*1e3, color="gray", ls=":", label="wire dia.")
    ax3[0].axvline(I_arr[i_tr], color="r", ls="--", label="transition I")
    ax3[0].set_xlabel("I [A]"); ax3[0].set_ylabel("droplet dia. [mm]")
    ax3[0].legend(); ax3[0].set_title("Droplet size vs current")
    ax3[1].plot(I_arr, freq, "s-", color="purple")
    ax3[1].set_xlabel("I [A]"); ax3[1].set_ylabel("detach freq [Hz]")
    ax3[1].set_title("Transfer frequency")
    ax3[2].plot(r_glob["t"]*1e3, r_glob["rd"]*1e3, label="170 A (globular)")
    ax3[2].plot(r_spray["t"]*1e3, r_spray["rd"]*1e3,
                label="300 A (spray)")
    ax3[2].set_xlabel("t [ms]"); ax3[2].set_ylabel("droplet radius [mm]")
    ax3[2].legend(); ax3[2].set_title("Growth-detach cycles")
    for a in ax3: a.grid(alpha=0.3)
    fig3.tight_layout(); fig3.savefig(OUT / "m3_droplet.png", dpi=140)

    # ---------- 模块 4: Goldak FDM ----------
    g = GoldakFDM(Q=P_ss)
    g.run(t_end=5.0)
    L, W, D = g.pool_size()
    print(f"[4 Goldak-FDM] 熔池 长 {L:.1f} / 宽 {W:.1f} / 深 {D:.1f} mm")

    ros = RosenthalThermal(Q=P_ss)
    XI, Y, Tr = ros.surface_field()
    w_ros = 0.0
    melt = Tr >= ros.Tm
    if melt.any():
        w_ros = (Y[melt].max()-Y[melt].min())*1e3
    print(f"[4 对比] Rosenthal 熔池宽 {w_ros:.1f} mm (点源, 偏宽属正常)")

    fig4, a4 = plt.subplots(1, 3, figsize=(14, 4))
    lv = [298, 600, 900, 1273, 1773, 2600]
    c0 = a4[0].contourf(g.x*1e3, g.y*1e3, g.T[:, :, 0].T, levels=lv,
                        cmap="hot")
    a4[0].contour(g.x*1e3, g.y*1e3, g.T[:, :, 0].T, levels=[1773],
                  colors="cyan")
    a4[0].set_title("Top view T (half model)")
    a4[0].set_xlabel("x [mm]"); a4[0].set_ylabel("y [mm]")
    fig4.colorbar(c0, ax=a4[0])
    c1 = a4[1].contourf(g.x*1e3, -g.z*1e3, g.T[:, 0, :].T, levels=lv,
                        cmap="hot")
    a4[1].contour(g.x*1e3, -g.z*1e3, g.T[:, 0, :].T, levels=[1773],
                  colors="cyan")
    a4[1].set_title("Longitudinal section")
    a4[1].set_xlabel("x [mm]"); a4[1].set_ylabel("z [mm]")
    fig4.colorbar(c1, ax=a4[1])
    c2 = a4[2].contourf(g.x*1e3, -g.z*1e3, g.peak[:, 0, :].T,
                        levels=[298, 773, 1073, 1273, 1773, 3000],
                        cmap="inferno")
    a4[2].contour(g.x*1e3, -g.z*1e3, g.peak[:, 0, :].T,
                  levels=[1273, 1773], colors=["lime", "cyan"])
    a4[2].set_title("Peak T: fusion zone & HAZ")
    a4[2].set_xlabel("x [mm]")
    fig4.colorbar(c2, ax=a4[2])
    fig4.suptitle("Module 4: Goldak double-ellipsoid + 3D transient FDM")
    fig4.tight_layout(); fig4.savefig(OUT / "m4_goldak_fdm.png", dpi=140)

    # ---------- 模块 5: 短路 / CMT ----------
    std = ShortCircuitGMAW(cmt=False).simulate()
    cmt = ShortCircuitGMAW(cmt=True).simulate()
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
    fig5.tight_layout(); fig5.savefig(OUT / "m5_short_cmt.png", dpi=140)


if __name__ == "__main__":
    main()
