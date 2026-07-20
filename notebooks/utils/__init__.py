# -*- coding: utf-8 -*-
"""notebook 工具箱: robot6_weave_interactive_demo 中打磨出的可复用件。

从 notebooks/ 目录下的 notebook 直接 ``from utils import ...`` (内核 cwd
即 notebooks/)。各模块职责见 README.md。
"""
from .pv_inline import (TAG_OFF, TAG_ON, add_mouse_hint, html_view,
                        inject_layer_toggle)
from .view_controls import VIEW_KEYS, WidgetStore, apply_view, view_widgets
from .goldak_snapshots import save_gif, solve_with_snapshots

__all__ = [
    "TAG_ON", "TAG_OFF", "html_view", "inject_layer_toggle", "add_mouse_hint",
    "VIEW_KEYS", "WidgetStore", "view_widgets", "apply_view",
    "solve_with_snapshots", "save_gif",
]
