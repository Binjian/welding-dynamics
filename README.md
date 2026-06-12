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
