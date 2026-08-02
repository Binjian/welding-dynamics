import ast
import json
import os
import tempfile
import unittest
from html import escape
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "welding-dynamics-mpl")
)

import ipywidgets as widgets
import numpy as np
import vtk
from IPython.display import display
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from welding_dynamics.robot_ik import MinkArmIK
from welding_dynamics.robot_vi import SixDofArm

REPO_ROOT = Path(__file__).resolve().parents[1]
RTX_NOTEBOOK = REPO_ROOT / "notebooks" / "robot6_weave_interactive_rtx_demo.ipynb"


class FakeEGLContext:
    """Small offscreen VTK context for native-scene interaction tests."""

    def __init__(self, *, size, **_kwargs):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(True)
        self.render_window.SetSize(*map(int, size))
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

    def verify(self):
        return SimpleNamespace(label="fake offscreen VTK")

    def close(self):
        self.render_window.Finalize()


class DummyRTXLiveWidget:
    async def _toggle_play(self, **_kwargs):
        return None

    async def close(self):
        return None


def load_native_scene_definitions():
    document = json.loads(RTX_NOTEBOOK.read_text())
    cell = next(cell for cell in document["cells"] if cell["id"] == "arm-scene")
    source = "".join(cell["source"])
    definitions = source[source.index("R_LINK =") : source.index("arm_layers =")]
    namespace = {
        "REPO_ROOT": REPO_ROOT,
        "MinkArmIK": MinkArmIK,
        "NvidiaEGLContext": FakeEGLContext,
        "RTXLiveWidget": DummyRTXLiveWidget,
        "colormaps": colormaps,
        "display": display,
        "escape": escape,
        "json": json,
        "np": np,
        "numpy_to_vtk": numpy_to_vtk,
        "to_rgb": to_rgb,
        "vtk": vtk,
        "vtk_to_numpy": vtk_to_numpy,
        "widgets": widgets,
    }
    exec(  # noqa: S102 - execute the checked-in notebook definitions under test
        compile(definitions, str(RTX_NOTEBOOK), "exec"), namespace
    )
    return namespace


NATIVE = load_native_scene_definitions()
LayerTransformInteractor = NATIVE["LayerTransformInteractor"]
RobotRig = NATIVE["RobotRig"]
WeldingRTXSceneBase = NATIVE["WeldingRTXSceneBase"]
WeldingRTXLiveWidget = NATIVE["WeldingRTXLiveWidget"]


def make_ur5e():
    return SixDofArm(
        m=(3.761, 8.058, 2.846, 1.37, 1.3, 0.365),
        r_link=0.045,
        J_rotor=0.03,
        g=9.81,
        dh_d=(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
        dh_a=(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
        dh_alpha_deg=(90.0, 0.0, 0.0, 90.0, -90.0, 0.0),
    )


class NativeRobotScene(WeldingRTXSceneBase):
    def __init__(self, arm, configurations, layers):
        self.arm = arm
        self.configurations = tuple(np.asarray(q, dtype=float) for q in configurations)
        super().__init__(
            layers=layers,
            frame_count=len(self.configurations),
            current_frame=0,
            camera_home=(
                (550.0, -850.0, 650.0),
                (0.0, 0.0, 250.0),
                (0.0, 0.0, 1.0),
            ),
            size=(500, 380),
        )

    def _build_environment(self, _floor):
        self.renderer.SetBackground(0.8, 0.85, 0.9)

    def _build_scene(self):
        self.robot = RobotRig(self, self.arm)

    def _update_frame(self, index):
        self._update_robot_pose(self.configurations[index])
        self.caption = f"frame {index}"


class NativeRTXTipIKTest(unittest.TestCase):
    def setUp(self):
        self.arm = make_ur5e()
        self.q0 = np.array((0.3, -2.0, -1.6, 2.1, -1.57, 1.9))
        self.q1 = self.q0 + np.array((0.02, -0.03, 0.01, 0.02, -0.01, 0.01))
        self.layers = tuple(
            SimpleNamespace(key=key, label=key, visible=True)
            for key in ("robot", "ik_handle")
        )
        self.scene = NativeRobotScene(self.arm, (self.q0, self.q1), self.layers)

    def tearDown(self):
        self.scene.close()

    def test_live_handle_ik_transform_frame_reset_and_cleanup(self):
        handle = self.scene.robot.tip_handle_record["prop"]
        address = handle.GetAddressAsString("")
        self.assertFalse(self.scene.robot.tip_handle_enabled)
        self.assertFalse(handle.GetVisibility())

        statuses = []
        self.scene.set_tip_ik_status_callback(
            lambda level, message: statuses.append((level, message))
        )
        self.scene.enable_tip_ik(True, render=False)
        self.assertTrue(handle.GetVisibility())
        self.assertTrue(handle.GetPickable())
        self.assertIn(address, self.scene._prop_interaction_target)

        angle = np.deg2rad(25.0)
        robot_pose = np.eye(4)
        robot_pose[:3, :3] = (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
        robot_pose[:3, 3] = (40.0, -20.0, 10.0)
        self.scene._set_transform_pose("robot", robot_pose)
        local_target = self.scene.robot.tip_position_mm + (2.0, -1.0, 1.0)
        world_target = (robot_pose @ np.append(local_target, 1.0))[:3]
        rotation = self.scene.robot.tip_rotation.copy()

        self.assertTrue(self.scene.begin_tip_drag())
        self.assertTrue(self.scene.drag_tip(world_target))
        self.scene.end_tip_drag()
        np.testing.assert_allclose(
            self.scene.robot.tip_position_mm, local_target, rtol=0.0, atol=0.02
        )
        np.testing.assert_allclose(
            self.scene.robot.tip_rotation, rotation, rtol=0.0, atol=2.0e-4
        )
        self.assertEqual(statuses[-1][0], "ok")

        self.scene.set_frame(1, render=False)
        np.testing.assert_allclose(self.scene.robot.current_q, self.q1)
        self.assertIsNone(self.scene._tip_override_q)
        np.testing.assert_array_equal(self.scene._transform_poses["robot"], robot_pose)

        self.scene.reset_object_transforms(render=False)
        np.testing.assert_array_equal(self.scene._transform_poses["robot"], np.eye(4))

        self.scene.set_layer_visible("ik_handle", False, render=False)
        self.assertFalse(handle.GetVisibility())
        self.assertFalse(handle.GetPickable())
        self.scene.set_layer_visible("ik_handle", True, render=False)
        self.assertTrue(handle.GetVisibility())
        self.assertTrue(handle.GetPickable())

        self.scene.close()
        self.assertFalse(self.scene.robot.tip_handle_enabled)
        self.assertFalse(self.scene._prop_interaction_target)
        self.assertFalse(self.scene._prop_transform_group)

    def test_priority_tip_drag_and_modifier_drag_preserve_roles(self):
        self.assertIsInstance(self.scene._interaction_style, LayerTransformInteractor)
        self.scene.enable_tip_ik(True, render=False)
        style = self.scene._interaction_style
        # A detached interactor supplies synthetic mouse state without asking
        # macOS VTK to create an onscreen Cocoa window in the test process.
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(style)
        camera_before = self.scene._capture_camera()
        q_before = self.scene.robot.current_q.copy()

        display_position = self.scene.world_to_display(self.scene.tip_world_position())
        x, y = map(round, display_position[:2])
        for selected_group in ("robot", None):
            self.scene.select_transform_group(selected_group, render=False)
            target, group, _ = self.scene.pick_interaction_target(x + 20, y)
            self.assertEqual(target, "tip")
            self.assertEqual(group, "robot")

        interactor.SetEventInformation(x + 20, y)
        style._on_left_press(None, None)
        self.assertEqual(style.drag_mode, "tip")
        interactor.SetEventInformation(x + 27, y + 3)
        style._on_mouse_move(None, None)
        style._on_left_release(None, None)
        self.assertGreater(
            np.linalg.norm(self.scene.robot.current_q - q_before), 1.0e-5
        )
        np.testing.assert_array_equal(
            self.scene._capture_camera()["position"], camera_before["position"]
        )
        np.testing.assert_array_equal(
            self.scene._capture_camera()["focal_point"], camera_before["focal_point"]
        )

        q_after_ik = self.scene.robot.current_q.copy()
        pose_before = self.scene._transform_poses["robot"].copy()
        display_position = self.scene.world_to_display(self.scene.tip_world_position())
        x, y = map(round, display_position[:2])
        interactor.SetEventInformation(x, y, 0, 1)
        style._on_left_press(None, None)
        self.assertEqual(style.drag_mode, "translate")
        interactor.SetEventInformation(x + 6, y, 0, 1)
        style._on_mouse_move(None, None)
        style._on_left_release(None, None)
        np.testing.assert_array_equal(self.scene.robot.current_q, q_after_ik)
        self.assertFalse(
            np.array_equal(self.scene._transform_poses["robot"], pose_before)
        )


class NativeRTXNotebookStructureTest(unittest.TestCase):
    def test_native_cells_parse_and_preserve_fallback_mapping(self):
        document = json.loads(RTX_NOTEBOOK.read_text())
        cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(cells), 8)
        for index, cell in enumerate(cells):
            source = "".join(cell["source"])
            if index:
                magic, source = source.split("\n", 1)
                self.assertEqual(magic, f"%%rtx_only {index}")
            ast.parse(source, filename=f"{RTX_NOTEBOOK}:{cell['id']}")

        composite_source = "".join(cells[6]["source"])
        seam_source = "".join(cells[7]["source"])
        for source in (composite_source, seam_source):
            self.assertIn('"ik_handle", "TCP IK handle"', source)
            self.assertIn("enable_tip_ik=True", source)
        self.assertLess(
            seam_source.index("for index in range(seam_scene.frame_count):"),
            seam_source.index("seam_rtx = await launch_live("),
        )


class NativeRTXWidgetLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_starting_playback_clears_tip_override_and_refreshes(self):
        calls = []
        app = object.__new__(WeldingRTXLiveWidget)
        app._closed = False
        app._tip_ik_enabled = True
        app.state = SimpleNamespace(playing=False)
        app.scene = SimpleNamespace(
            clear_tip_ik_override=lambda **kwargs: calls.append(("clear", kwargs))
        )
        app.ctrl = SimpleNamespace(view_update=lambda: calls.append(("refresh", {})))

        await app._toggle_play()

        self.assertEqual(
            calls,
            [("clear", {"render": False}), ("refresh", {})],
        )


if __name__ == "__main__":
    unittest.main()
