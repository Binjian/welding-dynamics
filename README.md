# welding-dynamics — 工业焊接 (GMAW/MIG) 动力学模型

基于 Python 的熔化极气体保护焊 (GMAW) 多物理动力学仿真包，包含五个相互耦合的模块，覆盖从电弧电路、熔滴过渡到工件传热的全过程。

## 快速开始 (uv)

```bash
uv sync          # 创建虚拟环境并安装依赖
uv run welding-sim   # 运行全部 5 个模块, 图片输出至 ./results/
```

或在代码中调用：

```python
from welding_dynamics import GMAWDynamics, GoldakFDM

res = GMAWDynamics().simulate()          # 模块 1: 自调节动力学
g = GoldakFDM(Q=float(res["P"][-1]))     # 用稳态功率驱动热模型
g.run(t_end=5.0)
print(g.pool_size())                     # 熔池 长/宽/深 [mm]
```

## 模块说明

### 模块 1 — 电弧自调节动力学 (`gmaw.py`)
集中参数 ODE 模型，状态为干伸长 s 与电流 I：
- 电源外特性：V = Voc − Rs·I，回路电感 L dI/dt
- 电弧电压：V_arc = V0 + Ea·la + Ra·I
- 熔化方程：MR = k1·I + k2·s·I²
- ds/dt = WFS − MR（恒压 GMAW 自调节机理）

仿真含 CTWD 阶跃扰动，展示电流/弧长的自恢复过程。

### 模块 2 — Rosenthal 解析解 (`thermal.py`)
厚板三维准稳态移动点热源解，用于快速估计温度场、
熔池尺寸、固定点热循环与 t8/5 冷却时间。

### 模块 3 — 熔滴过渡动力学 (`droplet.py`)
静力平衡理论 (SFBT)：表面张力保持 vs 重力 + Lorentz/pinch
电磁力（含高电流锥化增强项）+ 等离子拖拽。
复现 1.2 mm 钢丝约 250 A 的滴状→喷射过渡电流，
输出熔滴直径与过渡频率随电流的变化。

### 模块 4 — Goldak 双椭球 + 3D 瞬态 FDM (`thermal.py`)
求解 ρc ∂T/∂t = k∇²T + q_goldak(x,y,z,t)：
- 半对称模型（y=0 对称面），edge-pad 实现守恒 Neumann 边界
- 热源逐步数值重归一化，保证能量精确守恒
- 输出瞬态温度场、峰值温度场（熔合区 / HAZ 划分）与熔池尺寸
- 与模块 2 的 Rosenthal 解交叉验证

### 模块 5 — 短路过渡 / CMT (`short_circuit.py`)
电弧相 ⇄ 短路相混杂 (hybrid) 状态机：
- 短路相液桥按 表面张力 + I² pinch 项缩颈直至断裂重燃
- 标准短路过渡：断桥峰值电流约 300 A（飞溅来源）
- CMT 模式：电流分段控制 + 焊丝机械回抽，断桥发生于
  低电流（≤160 A），体现低飞溅、低热输入机理

## 典型结果 (默认参数: 1.2 mm 钢丝, WFS 7.2 m/min, 8 mm/s)

| 量 | 数值 |
|---|---|
| 稳态工作点 | ~266 A / 29 V / 7.8 kW |
| 滴状→喷射过渡电流 | ~250 A |
| Goldak-FDM 熔池 (长×宽×深) | 17.5 × 7.5 × 3.8 mm |
| Rosenthal 熔池宽 (点源对照) | 9.4 mm |
| 短路过渡峰值电流 / CMT | ~300 A / ≤160 A |

结果图位于 `results/`：
`m1_self_regulation.png`, `m3_droplet.png`,
`m4_goldak_fdm.png`, `m5_short_cmt.png`
（早期单文件版本图 `gmaw_dynamics.png`, `thermal_field.png` 一并保留）。

## 项目结构

```
welding-dynamics/
├── pyproject.toml
├── README.md
├── uv.lock
├── src/welding_dynamics/
│   ├── __init__.py
│   ├── gmaw.py           # 模块 1
│   ├── thermal.py        # 模块 2 & 4
│   ├── droplet.py        # 模块 3
│   ├── short_circuit.py  # 模块 5
│   └── main.py           # 入口 (welding-sim)
├── docs/legacy/          # 早期单文件版本
└── results/              # 仿真结果图
```

## 参数修改
所有物理/工艺参数集中于各类的 `__init__`（焊丝直径、材料热物性、
电源参数、Goldak 椭球尺寸等），便于参数研究。

## 参考
- Rosenthal, D. (1946). The theory of moving sources of heat.
- Goldak, J. et al. (1984). A new finite element model for welding heat sources.
- 静力平衡理论 (SFBT) 与燃弧 (burn-off) 模型经典文献 (Lesnewich; Amson; Quinn et al.)

## 变分积分器扩展 (模块 6–8)

```bash
uv run welding-sim-vi   # 运行变分扩展, 图片输出至 ./results/
```

核心库 `variational.py`：`ForcedVerlet`（辛 Verlet + 离散
Lagrange–d'Alembert 强迫项）、`MidpointDEL`（中点离散 Euler–Lagrange，
支持构型相关质量矩阵，Newton 隐式求解）、非光滑碰撞映射工具。

### 模块 6 — 熔滴振荡 / 脉冲 MIG 共振 (`droplet_vi.py`)
悬垂熔滴 Rayleigh l=2 模态 (k = 32πγ/3, f0 ≈ 527 Hz)，方波脉冲电磁力
激励。结果：变分积分器以粗步长 (T0/22) 精确复现解析共振峰；
隐式 Euler 的人工数值阻尼把共振峰压低 87% 并使峰频偏移 —— 用于
脉冲参数整定时会严重误导"一脉一滴"频率匹配。

### 模块 7 — 焊接机器人二连杆 (`robot_vi.py`)
竖直平面二连杆 (构型相关 M(q))，MidpointDEL 积分。
200 s 无驱动摆动 (h=20 ms)：VI 能量误差有界振荡（末段 ~0.8%），
RK4 同步长单调漂移至 ~35% —— 长轨迹仿真不失真是变分积分器的
标志性优势。焊缝跟踪演示（PD+重力补偿、强迫 DEL）RMS 误差 0.24 mm。

### 模块 8 — 短路接触的非光滑变分模型 (`shortcircuit_vi.py`)
CMT 机械振荡循环：自由相辛 Verlet + 触池事件二分精确定位 +
变分碰撞映射 (湿接触 e=0) + 附着/回抽/断桥状态机，复现 ~80 Hz
熔滴过渡节律。弹性反冲基准 (e=0.85)：非光滑 VI 能量只在物理事件处
阶梯下降；罚函数法在同步长下因接触刚度欠解析产生巨量虚假能量注入。

## 三维体渲染 + OpenFOAM 导出 (模块 9)

```bash
uv sync --extra viz   # 安装可选依赖 PyVista/VTK
uv run welding-sim-3d  # 求解 GoldakFDM -> 导出 OpenFOAM 算例 -> PyVista 截图
```

`thermal3d.py` 在模块 4 `GoldakFDM` 三维温度场基础上提供两件事：

### OpenFOAM 算例导出 (`OpenFOAMExporter`, 纯 numpy)
把 FDM 结构化网格手工写成完整 `polyMesh`（points / faces / owner /
neighbour / boundary，内部面按上三角排序），并将 **末时刻温度场 `T`** 与
**峰值温度场 `Tpeak`** 写成 `volScalarField` 时间目录。半对称面 (y=0) 输出为
`symmetryPlane` patch，另含可直接 `laplacianFoam` 复算的 `system/`、`constant/`。
目录内放置 `case.foam` 占位文件，在 **ParaView 中可直接打开**：

```
results/openfoam_case/
├── case.foam                 # ParaView 入口
├── constant/polyMesh/        # points faces owner neighbour boundary
├── constant/transportProperties
├── 0/T                       # 初始场 (均匀 T0)
├── 5/{T,Tpeak}               # 末时刻温度场 / 峰值温度场
└── system/{controlDict,fvSchemes,fvSolution}
```

```python
from welding_dynamics import GoldakFDM, export_openfoam
g = GoldakFDM(); g.run(t_end=5.0)
export_openfoam(g, "results/openfoam_case", t_end=5.0)
```

导出网格经封闭性校验（每个单元各面面积矢量之和 ≈ 0），保证面定向与
owner/neighbour 关系正确（无需安装 OpenFOAM 即可验证）。

### PyVista 体渲染 (`render`)
将半模型沿 y=0 镜像为全熔池，绘制熔合区 (熔点) 与 HAZ 等温面 + 对称面温度切片。
`pyvista` 为**可选依赖**，在 `render()` 内延迟导入；未安装时导出功能不受影响，
CLI 自动跳过渲染。

```python
from welding_dynamics import GoldakFDM, render
g = GoldakFDM(); g.run(t_end=5.0)
render(g, field="peak")          # 交互式窗口; offscreen=True 可离屏存图
```

#### Jupyter 交互式演示
`notebooks/pyvista_interactive_demo.ipynb` 演示在 Jupyter 中内联交互旋转/缩放
三维温度场 (`render(g, notebook=True)`)：

```bash
uv sync --extra notebook   # pyvista + jupyter + 交互后端依赖
uv run jupyter lab notebooks/pyvista_interactive_demo.ipynb
```
