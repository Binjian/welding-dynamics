# -*- coding: utf-8 -*-
"""变分积分器扩展主程序: 模块 6-8, 图片输出至 cfg.output.dir (默认 ./results/)。

配置见 `conf/sim_vi.yaml`:

    uv run welding-sim-vi
    uv run welding-sim-vi material=aluminum          # gamma/rho 改变熔滴共振频率
    uv run welding-sim-vi run.robot.passive_h=0.04   # 加大步长, 检验能量误差仍有界
"""
from pathlib import Path

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import DictConfig

from . import config  # noqa: F401  (导入即注册 wd.* 解析器)

plt.rcParams["font.family"] = "DejaVu Sans"


@hydra.main(version_base="1.3", config_path="conf", config_name="sim_vi")
def main(cfg: DictConfig):
    OUT = Path(cfg.output.dir); OUT.mkdir(parents=True, exist_ok=True)
    dpi = cfg.output.dpi
    print(f"[配置] material={cfg.material.name} -> {OUT}/")

    # ============ 模块 6: 熔滴振荡 / 脉冲共振 ============
    rd_ = cfg.run.droplet_vi
    d = instantiate(cfg.droplet_vi)
    print(f"[6 熔滴VI] Rayleigh 固有频率 f0 = {d.f0:.0f} Hz, "
          f"k = {d.k:.1f} N/m, m = {d.m*1e6:.2f} mg")

    # (a) 无阻尼自由振荡能量 (c=0): 三种积分器
    d_free = instantiate(cfg.droplet_vi, zeta=0.0)
    T0 = 1/d_free.f0
    h = T0/rd_.free_steps_per_period
    t_free = rd_.free_periods*T0
    x0 = rd_.x0
    t, X, V = d_free.run_vi(0.0, t_free, h, x0=x0)
    E_vi = d_free.energy(X, V)
    tE, XE = d_free.run_explicit_euler(0.0, t_free, h, x0=x0)
    # 显式 Euler 能量重建 (速度差分)
    VE = np.gradient(XE, h)
    E_ee = d_free.energy(XE, VE)
    tI, XI = d_free.run_implicit_euler(0.0, t_free, h, x0=x0)
    VI_ = np.gradient(XI, h)
    E_ie = d_free.energy(XI, VI_)
    E0 = E_vi[0]

    # (b) 共振曲线
    fp = np.linspace(rd_.fp_lo_frac*d.f0, rd_.fp_hi_frac*d.f0, rd_.n_fp)
    sweep = dict(periods=rd_.sweep_periods,
                 steps_per_period=rd_.sweep_steps_per_period)
    A_vi = d.resonance_sweep(fp, "vi", **sweep)
    A_ie = d.resonance_sweep(fp, "ie", **sweep)
    A_an = d.analytic_fundamental(fp)
    pk_vi, pk_ie = fp[np.argmax(A_vi)], fp[np.argmax(A_ie)]
    print(f"[6 熔滴VI] 共振峰: 解析 {fp[np.argmax(A_an)]:.0f} Hz | "
          f"VI {pk_vi:.0f} Hz (峰值 {A_vi.max()*1e3:.3f} mm) | "
          f"隐式Euler {pk_ie:.0f} Hz (峰值 {A_ie.max()*1e3:.3f} mm, "
          f"被人工阻尼压低 {100*(1-A_ie.max()/A_vi.max()):.0f}%)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogy(t/T0, np.maximum(E_vi/E0, 1e-12), label="Variational (Verlet)")
    ax[0].semilogy(tE/T0, np.clip(E_ee/E0, 1e-12, 1e6), label="Explicit Euler")
    ax[0].semilogy(tI/T0, np.maximum(E_ie/E0, 1e-12), label="Implicit Euler")
    ax[0].set_xlabel("t / T0"); ax[0].set_ylabel("E / E0")
    ax[0].set_title(f"Free oscillation energy (h = T0/{rd_.free_steps_per_period})")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(fp, A_an*1e3, "k--", label="analytic (fundamental)")
    ax[1].plot(fp, A_vi*1e3, "o-", ms=3, label="Variational")
    ax[1].plot(fp, A_ie*1e3, "s-", ms=3, label="Implicit Euler")
    ax[1].axvline(d.f0, color="gray", ls=":", lw=0.8)
    ax[1].set_xlabel("pulse frequency [Hz]")
    ax[1].set_ylabel("steady amplitude [mm]")
    ax[1].set_title("Pulsed-MIG droplet resonance curve")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("Module 6: pendant-droplet oscillation, variational vs classic")
    fig.tight_layout(); fig.savefig(OUT/"m6_droplet_vi.png", dpi=dpi)

    # ============ 模块 7: 焊接机器人 ============
    rr = cfg.run.robot
    arm = instantiate(cfg.robot)
    tv, ev, tr, er = arm.passive_compare(q0=tuple(rr.q0),
                                         t_end=rr.passive_t_end, h=rr.passive_h)
    print(f"[7 机器人VI] {rr.passive_t_end:.0f} s 无驱动摆动 (h={rr.passive_h*1e3:.0f} ms): "
          f"VI 能量误差有界 max {ev.max():.2e}, 末段 {ev[-1]:.2e} | "
          f"RK4 漂移至 {er[-1]:.2e} (单调增长)")

    sm = rr.seam
    ts, tip, ref, err = arm.seam_tracking(
        p_start=tuple(sm.p_start), p_end=tuple(sm.p_end),
        t_weld=sm.t_weld, h=sm.h, Kp=sm.Kp, Kd=sm.Kd)
    print(f"[7 机器人VI] 焊缝跟踪 RMS 误差 = {1e3*np.sqrt((err**2).mean()):.2f} mm")

    fig7, a7 = plt.subplots(1, 2, figsize=(12, 4.2))
    a7[0].semilogy(tv, np.maximum(ev, 1e-12), label="Variational (midpoint DEL)")
    a7[0].semilogy(tr, np.maximum(er, 1e-12), label="RK4 (same h)")
    a7[0].set_xlabel("t [s]"); a7[0].set_ylabel("|E/E0 - 1|")
    a7[0].set_title(f"Passive swing energy error, {rr.passive_t_end:.0f} s, "
                    f"h = {rr.passive_h*1e3:.0f} ms")
    a7[0].legend(); a7[0].grid(alpha=0.3)
    a7[1].plot(ref[:, 0]*1e3, ref[:, 1]*1e3, "k--", lw=1.5, label="weld seam")
    a7[1].plot(tip[:, 0]*1e3, tip[:, 1]*1e3, "r", lw=1, label="torch tip (forced DEL)")
    a7[1].set_xlabel("x [mm]"); a7[1].set_ylabel("y [mm]")
    a7[1].set_title("Seam tracking (PD + gravity comp.)")
    a7[1].legend(); a7[1].grid(alpha=0.3); a7[1].axis("equal")
    fig7.suptitle("Module 7: welding-robot 2-link arm, variational integrator")
    fig7.tight_layout(); fig7.savefig(OUT/"m7_robot_vi.png", dpi=dpi)

    # ============ 模块 7b: 六自由度机械臂 ============
    r6 = cfg.run.robot6
    arm6 = instantiate(cfg.robot6)
    tv6, ev6, tr6, er6 = arm6.passive_compare(
        q0=tuple(r6.q0), t_end=r6.passive_t_end, h=r6.passive_h)
    print(f"[7b 6DOF-VI] {r6.passive_t_end:.0f} s 无驱动摆动 "
          f"(h={r6.passive_h*1e3:.0f} ms, 数值装配 M(q), 无解析动力学): "
          f"VI 能量误差有界 max {ev6.max():.2e} | RK4 漂移至 {er6[-1]:.2e}")

    s6 = r6.seam
    ts6, tip6, ref6, err6 = arm6.seam_tracking(
        p_start=tuple(s6.p_start), p_end=tuple(s6.p_end),
        t_weld=s6.t_weld, h=s6.h, wn=s6.wn, zeta=s6.zeta)
    print(f"[7b 6DOF-VI] 三维焊缝跟踪 (位姿 IK + 逐关节 PD): "
          f"RMS 误差 = {1e3*np.sqrt((err6**2).mean()):.2f} mm, "
          f"max = {1e3*err6.max():.2f} mm")

    fig7b = plt.figure(figsize=(12, 4.6))
    b0 = fig7b.add_subplot(1, 2, 1)
    b0.semilogy(tv6, np.maximum(ev6, 1e-12), label="Variational (midpoint DEL)")
    b0.semilogy(tr6, np.maximum(er6, 1e-12), label="RK4 (same h)")
    b0.set_xlabel("t [s]"); b0.set_ylabel("|E/E0 - 1|")
    b0.set_title(f"Passive swing energy error, {r6.passive_t_end:.0f} s, "
                 f"h = {r6.passive_h*1e3:.0f} ms")
    b0.legend(); b0.grid(alpha=0.3)
    b1 = fig7b.add_subplot(1, 2, 2, projection="3d")
    b1.plot(ref6[:, 0]*1e3, ref6[:, 1]*1e3, ref6[:, 2]*1e3,
            "k--", lw=1.5, label="weld seam")
    b1.plot(tip6[:, 0]*1e3, tip6[:, 1]*1e3, tip6[:, 2]*1e3,
            "r", lw=1, label="torch tip (forced DEL)")
    b1.set_xlabel("x [mm]"); b1.set_ylabel("y [mm]"); b1.set_zlabel("z [mm]")
    b1.set_title("3D seam tracking (pose IK + per-joint PD)")
    b1.legend()
    # 三轴等比例: 否则自动缩放会把 ~0.4 mm 的 x 向偏差拉满整轴, 视觉失真
    mid = 0.5*(ref6.min(axis=0) + ref6.max(axis=0))*1e3
    half = 0.55*(ref6.max(axis=0) - ref6.min(axis=0)).max()*1e3
    b1.set_xlim(mid[0]-half, mid[0]+half)
    b1.set_ylim(mid[1]-half, mid[1]+half)
    b1.set_zlim(mid[2]-half, mid[2]+half)
    fig7b.suptitle("Module 7b: 6-DOF welding robot (spherical wrist), "
                   "variational integrator")
    fig7b.tight_layout(); fig7b.savefig(OUT/"m7b_robot6_vi.png", dpi=dpi)

    # ============ 模块 8: 非光滑接触 ============
    rc = cfg.run.contact
    cc = instantiate(cfg.contact)
    out, events = cc.simulate_cycle(t_end=rc.cycle_t_end)
    dips = [t for t, kind in events if kind == "dip"]
    f_dip = (len(dips)-1)/(dips[-1]-dips[0]) if len(dips) > 1 else 0
    print(f"[8 接触VI] CMT 机械振荡循环频率 ~ {f_dip:.0f} Hz "
          f"({len(dips)} 次触池)")

    bounce = dict(t_end=rc.bounce_t_end, h=rc.bounce_h)
    Tn, Xn, En = cc.bounce_nonsmooth_vi(e=rc.bounce_e, **bounce)
    Tp, Xp, Ep = cc.bounce_penalty(k_pen_frac=rc.penalty_k_frac, **bounce)
    inj = (Ep.max()/Ep[0] - 1)*100
    print(f"[8 接触VI] 弹性反冲基准 (e={rc.bounce_e}): 非光滑VI 能量单调阶梯下降 "
          f"(物理); 罚函数法虚假能量注入 +{inj:.0f}%")

    fig8, a8 = plt.subplots(1, 3, figsize=(14, 4.2))
    a8[0].plot(out[:, 0]*1e3, out[:, 1]*1e3, label="droplet x")
    a8[0].plot(out[:, 0]*1e3, out[:, 2]*1e3, "--", label="wire x_eq (feed/retract)")
    a8[0].axhline(cc.gap*1e3, color="gray", lw=0.8, label="pool surface")
    a8[0].fill_between(out[:, 0]*1e3, -0.2, cc.gap*1e3*1.3,
                       where=out[:, 3] > 0.5, color="orange", alpha=0.2)
    a8[0].set_xlabel("t [ms]"); a8[0].set_ylabel("x [mm]")
    a8[0].set_title("CMT dip-transfer cycle (nonsmooth VI)")
    a8[0].legend(fontsize=8); a8[0].grid(alpha=0.3)
    a8[1].plot(Tn*1e3, Xn*1e3, label="nonsmooth VI")
    a8[1].plot(Tp*1e3, Xp*1e3, alpha=0.7, label="penalty method")
    a8[1].axhline(0, color="gray", lw=0.8)
    a8[1].set_xlabel("t [ms]"); a8[1].set_ylabel("x [mm]")
    a8[1].set_title(f"Bounce benchmark trajectory (e={rc.bounce_e})")
    a8[1].legend(fontsize=8); a8[1].grid(alpha=0.3)
    a8[2].plot(Tn*1e3, En/En[0], label="nonsmooth VI")
    a8[2].plot(Tp*1e3, Ep/Ep[0], alpha=0.7, label="penalty method")
    a8[2].set_xlabel("t [ms]"); a8[2].set_ylabel("E / E0")
    a8[2].set_title("Energy fidelity at contact events")
    a8[2].legend(fontsize=8); a8[2].grid(alpha=0.3)
    fig8.suptitle("Module 8: nonsmooth variational contact (short-circuit / CMT)")
    fig8.tight_layout(); fig8.savefig(OUT/"m8_contact_vi.png", dpi=dpi)


if __name__ == "__main__":
    main()
