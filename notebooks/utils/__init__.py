"""notebook 工具箱: robot6_weave_interactive_demo 中打磨出的可复用件。

从 notebooks/ 目录下的 notebook 直接 ``from utils import ...`` (内核 cwd
即 notebooks/)。各模块职责见 README.md。
"""
from .goldak_snapshots import save_gif, solve_with_snapshots
from .pv_inline import TAG_OFF, TAG_ON, add_mouse_hint, html_view, inject_layer_toggle
from .robot6_pv_widget import (
    CompositePVScene,
    LayerSpec,
    PyVistaWidgetApp,
    RobotPVScene,
    RobotWeaveContext,
    SeamPVScene,
    build_output_tabs,
    close_app,
    close_output_tabs,
    initial_frame,
)
from .view_controls import VIEW_KEYS, WidgetStore, apply_view, view_widgets

__all__ = [
    "TAG_OFF",
    "TAG_ON",
    "VIEW_KEYS",
    "CompositePVScene",
    "LayerSpec",
    "PyVistaWidgetApp",
    "RobotPVScene",
    "RobotWeaveContext",
    "SeamPVScene",
    "WidgetStore",
    "add_mouse_hint",
    "apply_view",
    "build_output_tabs",
    "close_app",
    "close_output_tabs",
    "html_view",
    "initial_frame",
    "inject_layer_toggle",
    "save_gif",
    "solve_with_snapshots",
    "view_widgets",
]
