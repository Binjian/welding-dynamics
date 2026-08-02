# notebooks/utils — notebook 工具箱

`robot6_weave_interactive_demo.ipynb` 开发过程中打磨出的可复用件。
notebook 内核 cwd 即 `notebooks/`, 可直接 `from utils import ...`。旧 html
fallback 仍有教学用内联副本；live PyVista frontend 直接复用这里的持久
scene/controller 实现，避免动画、交互和资源生命周期逻辑分叉。

| 模块 | 内容 |
|---|---|
| `pv_inline.py` | `html_view` — 'html' 后端内联渲染 + **内核卡死修复** (VTK add_text 把 sys.std* 换成只读对象, IPython≥9 的 _tee 随之炸掉 execute_request, 下一格永远"运行中"; 渲染后复原原始流)。`inject_layer_toggle` — 场景内客户端图层切换 (透明度万分位指纹 `TAG_ON`/`TAG_OFF` 标记两组 actor, 浏览器端翻转可见性, **切换时视角不变**)。`add_mouse_hint` — 鼠标操作角标。 |
| `view_controls.py` | `WidgetStore` — ipywidget 值 JSON 持久化 (重跑/重启内核自动复原; 状态文件记得 gitignore)。`view_widgets` — 平移/旋转/前景透明七滑块 (顺序 `VIEW_KEYS`)。`apply_view` — 从默认取景起算的可复现相机变换。 |
| `goldak_snapshots.py` | `solve_with_snapshots` — 复刻 `GoldakFDM.run()` 常系数路径、按帧时刻抓拍 `(t, T, peak)` (单次通过; `run()` 只保留末时刻场)。`save_gif` — Pillow 写 GIF (不为 imageio 加包)。**改 thermal.py 步进逻辑需同步。** |
| `robot6_pv_widget.py` | PyVista `server` widget 的持久 VTK 场景：逐帧拓扑变化、图层/瞬时场切换、摄影棚环境、世界坐标轴、机械臂/工艺对象鼠标选择与变换，以及 GIF/Composite/Seam tab 输出。 |
| `make_seam_gif.py` | 焊缝成形 GIF 独立脚本 (notebook §5b 场景): `uv run python notebooks/utils/make_seam_gif.py [输出.gif] [帧数]`, 默认写 `results/robot6_weave_seam.gif`。 |
