"""持久化视图控件 (源自 robot6_weave_interactive_demo).

- :class:`WidgetStore` — 把任意 ipywidget 的值持久化到 JSON 文件,
  重跑 notebook / 重启内核自动复原。
- :func:`view_widgets` — 七个视图滑块: 平移 x/y/z [mm], 方位/俯仰/滚转
  [°], 前景透明 [%] (如机械臂; 100% = 移出场景, 零遮挡)。
- :func:`apply_view` — 在场景默认取景基础上施加视图参数。

典型用法 (cwd = notebooks/)::

    from utils import WidgetStore, view_widgets, apply_view, VIEW_KEYS
    store = WidgetStore('.my_view_state.json')       # 记得 gitignore
    v = view_widgets(store, 'scene1')
    out = widgets.interactive_output(
        lambda **kw: display(render(**kw)), dict(zip(VIEW_KEYS, v)))
"""
import json
from pathlib import Path

import ipywidgets as widgets
import numpy as np

#: :func:`view_widgets` 返回滑块的参数名顺序
VIEW_KEYS = ('px', 'py', 'pz', 'az', 'el', 'roll', 'fg_t')


class WidgetStore:
    """控件值 JSON 持久化: ``tracked(scene, name, w)`` 先恢复上次的值,
    再挂 observer — 恢复本身不落盘; 之后任何变化都同步写盘, 重跑时
    控件初始化回上次的值, ``interactive_output`` 的首帧即按存储状态渲染。
    """

    def __init__(self, path):
        self.path = Path(path)
        try:
            self._state = json.loads(self.path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            self._state = {}

    def get(self, scene, name, default=None):
        """Return one persisted value without creating a widget."""
        return self._state.get(scene, {}).get(name, default)

    def set(self, scene, name, value):
        """Persist one value immediately (used by live animation frames)."""
        self._state.setdefault(scene, {})[name] = value
        self.path.write_text(json.dumps(self._state, indent=1,
                                        ensure_ascii=False))

    def tracked(self, scene, name, w):
        val = self._state.get(scene, {}).get(name)
        if val is not None:
            w.value = val

        def _save(change):
            self.set(scene, name, change['new'])

        w.observe(_save, 'value')
        return w


def view_widgets(store, scene, lim=300.0, step=10.0, alpha_label='臂透明 [%]'):
    """七个视图滑块 (顺序同 :data:`VIEW_KEYS`), 松开才重绘, 全部持久化。

    ``alpha_label`` 是前景透明滑块的标签 (默认机械臂); 100% 应由调用方
    实现为**整组不加入场景** (半透明 actor 仍参与深度混合, 只有不画才
    保证零遮挡)。
    """
    kw = {
        'continuous_update': False,
        'readout_format': '.0f',
        'layout': widgets.Layout(width='230px'),
    }
    pans = [store.tracked(scene, f'p{ax}',
                          widgets.FloatSlider(min=-lim, max=lim, step=step,
                                              value=0.0,
                                              description=f'平移 {ax} [mm]',
                                              **kw))
            for ax in 'xyz']
    rots = [store.tracked(scene, key,
                          widgets.FloatSlider(min=mn, max=mx, step=2.0,
                                              value=0.0,
                                              description=f'{lab} [°]', **kw))
            for key, lab, mn, mx in (('az', '方位角', -180.0, 180.0),
                                     ('el', '俯仰角', -80.0, 80.0),
                                     ('roll', '滚转', -180.0, 180.0))]
    alpha = store.tracked(scene, 'fg_t',
                          widgets.FloatSlider(min=0.0, max=100.0, step=5.0,
                                              value=0.0,
                                              description=alpha_label, **kw))
    return pans + rots + [alpha]


def apply_view(p, px, py, pz, az, el, roll, fg_t=None):
    """在场景默认取景基础上施加视图参数 (绝对量, 每次重绘从默认相机起算,
    故取景可复现): 先世界坐标平移 (相机+焦点同移, 视线方向不变), 再绕
    焦点做方位角/俯仰角/滚转 (vtkCamera.Azimuth/Elevation/Roll; 每步
    OrthogonalizeViewUp 防上方向漂移, 俯仰限 ±80° 避开极点翻转)。

    ``fg_t`` 仅为便于 ``dict(zip(VIEW_KEYS, sliders))`` 整包传参而接受,
    此处忽略 — 前景透明度应在建景时用 ``1 - fg_t/100`` 自行处理。
    """
    pan = np.array([px, py, pz], dtype=float)
    cam = p.camera
    cam.focal_point = tuple(np.asarray(cam.focal_point) + pan)
    cam.position = tuple(np.asarray(cam.position) + pan)
    if az:
        cam.Azimuth(az)
        cam.OrthogonalizeViewUp()
    if el:
        cam.Elevation(el)
        cam.OrthogonalizeViewUp()
    if roll:
        cam.Roll(roll)
