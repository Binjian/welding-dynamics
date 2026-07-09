# -*- coding: utf-8 -*-
"""Hydra/OmegaConf 支持: 自定义解析器与配置辅助函数。

配置树位于 `src/welding_dynamics/conf/`, 由三个 CLI 入口 (`main.py`,
`main_vi.py`, `main_3d.py`) 通过 `@hydra.main` 组合。分组语义:

    process/   A 类工况参数 (电流、功率、速度、干伸长、丝径) — 可由工艺数据库确定
    material/  B 类材料物性 (rho, cp, k, Tm, gamma) — 手册值
    solver/    C 类数值配置 (网格 dx、域尺寸、积分终点)
    output/    绘图输出目录与 dpi

物理常数只在 YAML 里出现一次: 各模块的 `model/*.yaml` 用 `${material.k}`
之类的插值引用分组, 而不是复制数值。派生量 (半径、热扩散率) 用下面的
自定义解析器现算, 避免 `alpha` 与 `k/(rho*cp)` 各写一份而漂移。
"""
from omegaconf import OmegaConf


def _thermal_diffusivity(k, rho, cp):
    """alpha = k / (rho * cp)  [m^2/s]"""
    return float(k) / (float(rho) * float(cp))


def register_resolvers():
    """注册 `wd.*` 解析器 (幂等, 可重复调用)。"""
    OmegaConf.register_new_resolver(
        "wd.half", lambda x: float(x) / 2.0, replace=True)
    OmegaConf.register_new_resolver(
        "wd.alpha", _thermal_diffusivity, replace=True)


def arc_power(cfg, fallback=None):
    """解析热源功率 Q [W]。

    `process.arc_power_W` 为 null 时表示"用上游给的功率": 在 `welding-sim`
    里是模块 1 的自调节稳态功率, 在 `welding-sim-3d` 里是 `GoldakFDM` 的类默认值。
    返回 None 表示调用方不应传 Q (即沿用类默认)。
    """
    Q = cfg.process.arc_power_W
    return fallback if Q is None else float(Q)


register_resolvers()
