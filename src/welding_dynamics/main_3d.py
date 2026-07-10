# -*- coding: utf-8 -*-
"""三维扩展主程序 (模块 9):
求解 GoldakFDM 三维温度场 -> 导出 OpenFOAM 算例 -> PyVista 体渲染。

配置见 `conf/sim_3d.yaml`:

    uv run welding-sim-3d
    uv run welding-sim-3d process=db_p90 solver=fine
    uv run welding-sim-3d render.enabled=false        # 无头环境只导出算例

OpenFOAM 算例输出至 <output.dir>/openfoam_case/ (可在 ParaView 直接打开 case.foam)。
PyVista 截图输出至 <output.dir>/m9_goldak_3d.png (需可选依赖: uv sync --extra viz)。
"""
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from .config import arc_power
from .thermal3d import OpenFOAMExporter, render


@hydra.main(version_base="1.3", config_path="conf", config_name="sim_3d")
def main(cfg: DictConfig):
    OUT = Path(cfg.output.dir); OUT.mkdir(parents=True, exist_ok=True)
    t_end = cfg.run.goldak.t_end
    print(f"[配置] process={cfg.process.name} material={cfg.material.name} "
          f"solver={cfg.solver.name} -> {OUT}/")

    # 求解三维瞬态温度场。process.arc_power_W = null 时沿用 GoldakFDM 的类默认功率。
    Q = arc_power(cfg)
    g = instantiate(cfg.goldak, **({} if Q is None else {"Q": Q}))
    if g.weaving:
        print(f"[9 摆动] {g.weave.describe()} -> 全宽网格 (半对称失效)")
    g.run(t_end=t_end, x_start=cfg.run.goldak.x_start)
    L, W, D = g.pool_size()
    print(f"[9 三维场] 网格 {g.Nx}x{g.Ny}x{g.Nz} "
          f"({'半模型' if g.symmetric else '全宽模型'}), "
          f"熔池 L×W×D = {L:.1f}×{W:.1f}×{D:.1f} mm")
    print(f"[9 三维场] 末时刻 T_max = {g.T.max():.0f} K, "
          f"峰值场 T_max = {g.peak.max():.0f} K")

    # 导出 OpenFOAM 算例
    if cfg.export.enabled:
        case = OpenFOAMExporter(g).export(OUT / cfg.export.case_dir, t_end=t_end)
        print(f"[9 三维场] OpenFOAM 算例已导出: {case}/  "
              f"(ParaView 打开 {case.name}/case.foam)")

    # PyVista 体渲染 (可选; 离屏截图, 无显示环境则跳过)
    if cfg.render.enabled:
        png = OUT / cfg.render.outfile
        try:
            render(g, field=cfg.render.field, outfile=png,
                   offscreen=cfg.render.offscreen)
            print(f"[9 三维场] PyVista 截图: {png}")
        except ImportError as e:
            print(f"[9 三维场] 跳过 PyVista 渲染 ({e}).")
        except Exception as e:                   # 离屏渲染在无头环境可能失败
            print(f"[9 三维场] PyVista 离屏渲染失败 ({type(e).__name__}: {e}); "
                  f"OpenFOAM 算例已正常导出。交互式渲染请用 "
                  f"render(GoldakFDM().run() 后的实例).")


if __name__ == "__main__":
    main()
