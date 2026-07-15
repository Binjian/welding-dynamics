# 熔池对流耦合可行性评估 (模块 10 → 模块 4)

*评估日期: 2026-07-15。结论: 可行; 方案 A (有效导热率耦合) 已实现, 见文末实现说明。*

基线模型 (`GoldakFDM`, 模块 4) 是**纯热传导**: 熔池内部没有任何流动,
热量只靠扩散输运。真实 GMAW 熔池内的对流搅拌 (表面速度量级 0.1–1 m/s,
Pe ≫ 1) 会显著改变熔池形貌与峰值温度。本文评估把熔池对流耦合进三维瞬态
求解的几条路线。

## 1. 现状: 模块 10 的三个降阶原型

`marangoni.py` 已含三个 Marangoni 降阶模型, 但都是对**已算完的传导场**做
后处理, 均未反馈进 `GoldakFDM` 的时间推进:

| 模型 | 方法 | 与模块 4 的关系 |
|---|---|---|
| 10A `EffectiveMarangoniCorrection` | 由 Pe 数折算池内有效热扩散率 `α_eff/α = min(1+0.15√Pe, limiter)` | 事后对 L/W/D 做定性缩放 |
| 10B `SurfaceMarangoniFlow2D` | 顶面 (x-y) 热毛细回流的运动学速度场 | 对顶面温度切片做可视化 |
| 10C `IncompressibleMarangoniFlow2D` | 纵截面 (x-z) 流函数-涡量不可压流 + 对流换热 | 在冻结的 Goldak 截面上推进 |

配置侧 `material/*.yaml` 已提供驱动参数 `dgamma_dT` (表面张力温度系数)。

## 2. GMAW 熔池对流的物理驱动

- **Marangoni (热毛细)**: 表面剪切 `τ = (dγ/dT)∇T`。洁净钢 `dγ/dT < 0`
  → 外向表面流 → 熔池**变宽变浅**; 含活性元素 (S/O) 时符号反转 → 变窄变深。
- **Lorentz (电磁)**: GMAW 电流 ~280 A 时与 Marangoni 同量级, 驱动向内向下的
  环流 (区别于 GTAW 中 Marangoni 单独主导)。
- **熔滴冲击动量**: 喷射过渡熔滴以数 m/s 撞入熔池, 是 GMAW 特有的动量源
  (模块 3 `DropletDynamics` 已给出熔滴速度/频率, 可作跨模块标量耦合)。
- **浮力**: 相对较弱。

完整保真 = 三维不可压 NS + VOF 自由表面 + 相变, 属 Flow-3D / OpenFOAM
量级的工程, 不在纯 Python 库的合理范围内。

## 3. 耦合路线对比 (按成本递增)

### 方案 A — 池内有效导热率 (已选定, 已实现)

焊接热模拟文献中的标准工程处理: 液相区导热率乘以搅拌因子
(即 10A 的 `alpha_eff()`, 由 `dγ/dT` 出发, 带保守限幅 ~6)。

- **改动**: `GoldakFDM.run()` 的扩散项改为**变系数通量形式** (面上取相邻单元
  调和平均, 保持能量守恒); `dt` 由最大扩散率决定 → 步数 ×limiter (~6×)。
- **成本**: 数天工作量; `solver=fine` 5 s 瞬态从 ~1 min 增至 ~6 min。
- **收益**: 峰值温度从非物理的 ~5700 K (超过沸点) 回落; 池内混合抹平
  峰值场记录中的摆动"鱼鳞"棱脊; L/W/D 形貌响应 `dγ/dT` 符号
  (各向异性变体可进一步区分横向/深度增强)。
- **局限**: 不产生速度场; 各向同性增强无法区分宽浅/窄深, 只能整体加速热输运。

### 方案 B — 三维运动学对流 (中等成本, 性价比低)

把 10B/10C 推广为三维无散速度场模型, 每步对 T 做迎风对流。
CFL 限制 `dt ≈ 0.25·dx/u ≈ 0.4 ms` (u~0.5 m/s, dx=0.8 mm), 步数 ~12×,
且需要迎风格式 (中心差分对流不稳定, 10B 的 docstring 已自证)。
结论: 付出接近真 CFD 的代价, 得不到真 CFD 的保真度 — 不推荐。

### 方案 C — 三维不可压 NS (2–4 周)

投影法 + Boussinesq 浮力 + 平自由面 Marangoni 剪切边界 + 糊状区
Carman–Kozeny (Darcy) 阻尼 — 标准连续介质做法, 固相以大阻尼"多孔介质"
处理, 可在整个箱域上求解而无需掩膜。网格仅 ~19 万单元, 压力泊松可
迭代或 FFT 辅助; NumPy 下估计每次 5 s 运行 10–30 min。
本仓库特定的两点:

1. Lorentz 力与熔滴动量在 GMAW 下不可忽略 — 模块 3 可提供动量源,
   符合本包"模块间只传标量"的架构;
2. `warp` 已是可选依赖 (mujoco-warp), 泊松/对流核可 GPU 化。

潜热 (表观热容法) 应随本方案一并加入。**难点在标定**: 仓库无熔池流场
实验数据, 只能对照已发表的 GMAW 模拟与工艺数据库的焊道几何。

### 方案 D — OpenFOAM 外移

`OpenFOAMExporter` 已能写出可运行算例, 开源熔池求解器 (laserbeamFoam 等)
存在; 但本环境无 OpenFOAM 安装, 且背离纯 Python 定位 — 只作离线选项。

## 4. 约束: 默认结果钉死

CLAUDE.md 规定默认配置必须精确复现 README "典型结果"表。因此对流必须是
**opt-in 配置组** (`convection/none|effective`, 仿照 `weave/` 的加法),
`GoldakFDM()` 默认行为不变。

## 5. 推荐与实施

**推荐**: 先做方案 A 作为耦合基线; 仅当"焊透对表面活性元素/电流的响应
预测"成为研究目标时再投入方案 C (届时用 warp 加速)。

**方案 A 实现说明** (本仓库):

- `GoldakFDM(convection=...)` 接受一个 `EffectiveMarangoniCorrection`
  对象 (与 `weave=` 同风格); 构造时由 `alpha_eff()/alpha` 得到池内导热
  增强倍率, `run()` 切换到变系数通量形式的显式步。
- `convection=None` (默认) 走原常系数代码路径, 逐位复现钉死结果。
- 配置组: `conf/convection/{none,effective}.yaml`,
  `effective.yaml` 的 `dgamma_dT` 插值自 `${material.dgamma_dT}`;
  `model/goldak.yaml` 增加 `convection: ${convection}`。
- 演示: `notebooks/robot6_weave_interactive_demo.ipynb` §5 合成场景
  增加复选框, 可叠加显示对流修正后的熔合区等值面 (与仅传导池对比)。
