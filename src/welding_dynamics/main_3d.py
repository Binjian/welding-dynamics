# -*- coding: utf-8 -*-
"""三维扩展主程序 (模块 9):
求解 GoldakFDM 三维温度场 -> 导出 OpenFOAM 算例 -> Mayavi 体渲染。

    uv run welding-sim-3d

OpenFOAM 算例输出至 results/openfoam_case/ (可在 ParaView 直接打开 case.foam)。
Mayavi 截图输出至 results/m9_goldak_3d.png (需可选依赖: uv sync --extra viz)。
"""
from pathlib import Path

from .thermal import GoldakFDM
from .thermal3d import OpenFOAMExporter, render

OUT = Path("results")
T_END = 5.0


def main():
    OUT.mkdir(exist_ok=True)

    # 求解三维瞬态温度场
    g = GoldakFDM()
    g.run(t_end=T_END)
    L, W, D = g.pool_size()
    print(f"[9 三维场] 网格 {g.Nx}x{g.Ny}x{g.Nz} (半模型), "
          f"熔池 L×W×D = {L:.1f}×{W:.1f}×{D:.1f} mm")
    print(f"[9 三维场] 末时刻 T_max = {g.T.max():.0f} K, "
          f"峰值场 T_max = {g.peak.max():.0f} K")

    # 导出 OpenFOAM 算例
    case = OpenFOAMExporter(g).export(OUT / "openfoam_case", t_end=T_END)
    print(f"[9 三维场] OpenFOAM 算例已导出: {case}/  "
          f"(ParaView 打开 {case.name}/case.foam)")

    # Mayavi 体渲染 (可选; 离屏截图, 无显示环境则跳过)
    png = OUT / "m9_goldak_3d.png"
    try:
        render(g, field="peak", outfile=png, offscreen=True)
        print(f"[9 三维场] Mayavi 截图: {png}")
    except ImportError as e:
        print(f"[9 三维场] 跳过 Mayavi 渲染 ({e}).")
    except Exception as e:                       # 离屏渲染在无头环境可能失败
        print(f"[9 三维场] Mayavi 离屏渲染失败 ({type(e).__name__}: {e}); "
              f"OpenFOAM 算例已正常导出。交互式渲染请用 "
              f"render(GoldakFDM().run() 后的实例).")


if __name__ == "__main__":
    main()
