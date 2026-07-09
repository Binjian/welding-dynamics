# welding-dynamics — 工业焊接 (GMAW/MIG) 动力学模型

基于 Python 的熔化极气体保护焊 (GMAW) 多物理动力学仿真包，包含五个相互耦合的模块，覆盖从电弧电路、熔滴过渡到工件传热的全过程。

## 快速开始 (uv)

```bash
uv sync          # 创建虚拟环境并安装依赖
uv run welding-sim   # 运行全部 5 个模块, 图片输出至 ./results/
uv run welding-sim process=db_median material=aluminum   # 换工况/材料 (见"参数修改")
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

## 典型结果 (默认配置: 1.2 mm 钢丝, WFS 7.2 m/min, 8 mm/s)

由默认配置组合 `process=code_default material=carbon_steel solver=default` 精确复现。

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
│   ├── config.py         # Hydra 自定义解析器 (wd.half / wd.alpha)
│   ├── conf/             # Hydra 配置树
│   │   ├── sim.yaml      #   welding-sim 根配置
│   │   ├── sim_vi.yaml   #   welding-sim-vi 根配置
│   │   ├── sim_3d.yaml   #   welding-sim-3d 根配置
│   │   ├── process/      #   A 类工况 (code_default, db_p10/median/p90)
│   │   ├── material/     #   B 类物性 (carbon_steel, stainless_steel, ...)
│   │   ├── solver/       #   C 类数值配置 (coarse, default, fine)
│   │   ├── output/       #   输出目录 (results, per_run)
│   │   └── model/        #   各类的 _target_ 节点
│   ├── main.py           # 入口 (welding-sim)
│   ├── main_vi.py        # 入口 (welding-sim-vi)
│   └── main_3d.py        # 入口 (welding-sim-3d)
├── project_data/
│   ├── data/             # 原始工艺参数工作簿 (xlsx)
│   ├── ingest_mongo.py       # xlsx  -> MongoDB welding_parameters
│   └── ingest_config_mongo.py# conf/ -> MongoDB welding_config
├── notebooks/            # 数据库探索、PyVista 交互演示
├── docs/legacy/          # 早期单文件版本
└── results/              # 仿真结果图 (Hydra 的 runs/ multirun/ 已 gitignore)
```

## 参数修改 (Hydra 配置)

所有物理/工艺参数集中于各类的 `__init__`（焊丝直径、材料热物性、
电源参数、Goldak 椭球尺寸等）。三个 CLI 入口用 [Hydra](https://hydra.cc)
从 `src/welding_dynamics/conf/` 组合这些参数，便于成组切换与批量扫描：

```bash
uv run welding-sim --cfg job         # 只打印合成后的完整配置, 不运行
uv run welding-sim process=db_median # 换一组工况 (生产数据库中位值)
uv run welding-sim material=aluminum solver=fine   # 换材料 + 加密网格
uv run welding-sim gmaw.Voc=30.0 run.goldak.t_end=8.0   # 覆盖单个叶子参数

# 批量扫描: 三组工况各跑一次, 图片分别写入各自的输出目录
uv run welding-sim --multirun process=db_p10,db_median,db_p90 output=per_run
```

配置分组对应三类参数：

| 分组 | 含义 | 可选值 |
|---|---|---|
| `process/` | **A 类工况** — 电弧功率、焊接速度、干伸长、丝径 (工艺数据库可确定) | `code_default`, `db_p10`, `db_median`, `db_p90` |
| `material/` | **B 类材料物性** — ρ, cp, k, Tm, γ (手册值) | `carbon_steel`, `stainless_steel`, `cast_iron`, `aluminum` |
| `solver/` | **C 类数值配置** — 网格 dx、域尺寸、积分终点 (与工艺无关) | `coarse`, `default`, `fine` |
| `output/` | 图片输出目录与 dpi | `results`, `per_run` |

要点：

- 默认组合 (`process=code_default material=carbon_steel solver=default`) **精确复现**上面的"典型结果"表。
- 物理常数在 YAML 中只出现一次：`conf/model/*.yaml` 用 `${material.k}` 之类的插值引用分组；
  派生量由解析器现算 (`${wd.half:}` 直径→半径，`${wd.alpha:}` 热扩散率 `k/(ρ·cp)`)，不会与 `k, ρ, cp` 漂移。
- `process.arc_power_W: null` 表示"用上游功率"：`welding-sim` 取模块 1 的自调节稳态功率 `P_ss`；
  `db_*` 预设则直接给出实测 `U·I`，热源不再依赖模块 1 的估计。
- **数据库不记录送丝速度 (WFS)**，而模块 1 的工作点由 `(WFS, CTWD)` 决定，
  因此其稳态电流不一定等于 `db_*` 预设标称的 `current_A`（相差 >10% 时 `main.py` 会打印 `[1 提示]`）。
  `current_A` / `voltage_V` 仅作记录，真正驱动仿真的是 `arc_power_W`、`travel_speed_m_s`、`ctwd_m`、`wire_diameter_m`。
- 配置层不侵入库：各模块类均为普通关键字参数，`GoldakFDM(Q=9000)` 可脱离 Hydra 直接使用。
  数据库工况到各模块入参的完整对照见
  [`notebooks/welding_parameter_database_exploration.ipynb`](notebooks/welding_parameter_database_exploration.ipynb) 第 9 节。

## MongoDB 存储 (可选)

`project_data/` 下两个导入脚本把**生产工艺数据**与**仿真配置**分别落库到
`welding_dynamics` 库的两个集合。**仿真本身不依赖 MongoDB**——这两个集合服务于
参数溯源与工况分析（notebook 探索、批量扫描的配置留档）。

```bash
uv run python project_data/ingest_mongo.py                     # -> welding_parameters
uv run python project_data/ingest_config_mongo.py              # -> welding_config
uv run python project_data/ingest_config_mongo.py --dry-run    # 只打印统计, 不写库
```

两个脚本均**幂等**（重复运行先 `drop()` 再重建），均以 `doc_type` 区分文档类型。

### `welding_parameters` — 生产工艺参数数据库 (152 文档)

由 `Welding Process Parameter Database 2022_rev.2022.03.24.xlsx` 解析而来，
"焊接记录 → 焊道"两级组织；导入时已把"送丝设定(实际电流)"、双丝主/从、
`一元化` 电压等格式清洗为数值字段（原值保留在 `*_raw`）。

| `doc_type` | 数量 | 内容 |
|---|---|---|
| `procedure` | 130 | 一条工艺记录（设备、母材、坡口、位置），`passes` 内嵌 417 条焊道 |
| `weave_pattern` | 21 | 摆动库路点波形 |
| `source_meta` | 1 | 数据来源与录入规则 |

探索与可视化见
[`notebooks/welding_parameter_database_exploration.ipynb`](notebooks/welding_parameter_database_exploration.ipynb)；
`conf/process/db_*.yaml` 三个工况预设即由该集合的 P10 / 中位 / P90 统计得出。

### `welding_config` — Hydra 配置树留档 (34 文档)

| `doc_type` | 数量 | 内容 |
|---|---|---|
| `config_root` | 3 | `sim` / `sim_vi` / `sim_3d` 根配置：原文、`defaults`、入口点 |
| `config_group` | 21 | 各分组选项（`process/db_median`、`model/gmaw` …）：原文 + 解析后 dict |
| `config_composed` | 9 | **组合并求值后**的最终配置，按 `(root, process)` 切面 |
| `config_meta` | 1 | 源目录、git 提交、hydra 版本、逐文件 sha256 |

`config_composed` 是这个集合的价值所在：`${material.k}`、`${wd.alpha:...}` 等插值
**已全部求值**，存的就是仿真真正拿到的那份配置，可按分组直接查询，也能回灌重建对象：

```python
from pymongo import MongoClient
from omegaconf import OmegaConf
from hydra.utils import instantiate
import welding_dynamics.config          # 导入即注册 wd.* 解析器

cc = MongoClient("mongodb://localhost:27017")["welding_dynamics"]["welding_config"]
doc = cc.find_one({"doc_type": "config_composed", "root": "sim",
                   "groups.process": "db_median"})
cfg = OmegaConf.create(doc["resolved"])
g = instantiate(cfg.goldak, Q=cfg.process.arc_power_W)   # Q=8120 W, v=5.15 mm/s
```

注：`output=per_run` 含 `${hydra:runtime.output_dir}`，脱离 Hydra 运行期无法求值，
故不参与组合（仍以 `config_group` 保留原文）。

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

若使用 `pv.set_jupyter_backend('trame')`，需确保已安装 `notebook` extra，
其中显式包含 `trame-vtk` 与 `trame-vuetify`，否则 PyVista 会在导入
`trame.widgets.vtk` 时失败。

## Marangoni 扩展候选 (模块 10)

### 模块 10A — 有效热扩散修正 (`marangoni.py`)
`EffectiveMarangoniCorrection` 用表面张力温度系数 `dγ/dT` 与表面温度梯度
估计 Marangoni 数、Peclet 数和表面速度尺度，并将熔池内热扩散率修正为
`α_eff`。该模型是轻量级后处理/参数化修正，不求解速度场；`dγ/dT < 0`
时给出外向表面流导致的加宽/变浅趋势，`dγ/dT > 0` 时给出内向表面流导致的
变窄/加深趋势。

### 模块 10B — 2D 表面热毛细回流 (`marangoni.py`)
`SurfaceMarangoniFlow2D` 从表面温度梯度计算 `τ_s = (dγ/dT)∇_sT`，
生成受限速的二维表面回流速度场，并提供显式
`advect_diffuse_step()` 将速度场耦合到表面温度片的对流-扩散更新。
该模型比有效扩散修正更接近 Marangoni 流机理，但仍避免完整自由表面
Navier-Stokes 求解，适合作为 Goldak 顶面温度场的中等复杂度后处理或弱耦合项。

### 模块 10C — 不可压热毛细熔池流 (`marangoni.py`)
`IncompressibleMarangoniFlow2D` 在焊缝纵向-深度截面上求解
流函数-涡量形式的不可压流原型：自由表面施加
`μ ∂u/∂z = (dγ/dT)∂T/∂x`，内部进行粘性涡量扩散，并用速度场对温度执行
对流-扩散更新。它比 10A/10B 更接近真实 Marangoni 熔池对流，但仍是透明的
二维研究原型，而不是完整三维自由表面 CFD。
