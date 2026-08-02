import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ipywidgets as widgets
import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "welding-dynamics-mpl")
)
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "welding-dynamics-cache")
)

from notebooks.utils.robot6_pv_widget import (
    LayerSpec,
    PyVistaWidgetApp,
    RobotPVScene,
    RobotWeaveContext,
)
from notebooks.utils.view_controls import WidgetStore
from welding_dynamics.robot_vi import SixDofArm


def make_ur5e() -> SixDofArm:
    return SixDofArm(
        m=(3.761, 8.058, 2.846, 1.37, 1.3, 0.365),
        r_link=0.045,
        J_rotor=0.03,
        g=9.81,
        dh_d=(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
        dh_a=(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
        dh_alpha_deg=(90.0, 0.0, 0.0, 90.0, -90.0, 0.0),
    )


class RobotPVTipIKTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = make_ur5e()
        self.q0 = np.array((0.3, -2.0, -1.6, 2.1, -1.57, 1.9))
        self.q1 = self.q0 + np.array((0.02, -0.03, 0.01, 0.02, -0.01, 0.01))
        frame_times = np.array((0.0, 1.0))
        context = RobotWeaveContext(
            arm=self.arm,
            q_at=lambda t: self.q0.copy() if t < 0.5 else self.q1.copy(),
            t_track=1.0,
            t_trace=frame_times,
            tip=np.array((self.arm.fk_tip(self.q0), self.arm.fk_tip(self.q1))),
            p0=np.array((0.4, 0.0, 0.2)),
            travel_speed=0.01,
        )
        layers = tuple(
            LayerSpec(key, key)
            for key in ("robot", "workpiece", "seam", "executed_path", "axes")
        )
        self.scene = RobotPVScene(
            context,
            frame_times,
            layers=layers,
            size=(400, 300),
        )

    def tearDown(self) -> None:
        self.scene.close()

    def test_live_only_handle_drag_transform_and_frame_reset(self) -> None:
        handle = self.scene.robot.tip_handle_record["prop"]
        handle_address = handle.GetAddressAsString("")
        self.assertFalse(self.scene.robot.tip_handle_enabled)
        self.assertFalse(handle.GetVisibility())

        statuses = []
        self.scene.set_tip_ik_status_callback(
            lambda level, message: statuses.append((level, message))
        )
        self.scene.enable_tip_ik(True, render=False)
        self.assertTrue(handle.GetVisibility())
        self.assertTrue(handle.GetPickable())
        self.assertIn(handle_address, self.scene._prop_interaction_target)

        # The interaction handle must remain available even when the robot
        # actors are fully hidden by the existing transparency control.
        self.scene.set_layer_opacity("robot", 0.0, render=False)
        self.assertTrue(handle.GetVisibility())

        angle = np.deg2rad(25.0)
        robot_pose = np.eye(4)
        robot_pose[:3, :3] = (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
        robot_pose[:3, 3] = (40.0, -20.0, 10.0)
        self.scene._set_transform_pose("robot", robot_pose)
        process_pose = self.scene._transform_poses["process"].copy()

        start_q = self.scene.robot.current_q.copy()
        start_rotation = self.scene.robot.tip_rotation.copy()
        local_target_mm = self.scene.robot.tip_position_mm + (2.0, -1.0, 1.0)
        world_target_mm = (robot_pose @ np.append(local_target_mm, 1.0))[:3]

        self.assertTrue(self.scene.begin_tip_drag())
        self.assertTrue(self.scene.drag_tip(world_target_mm))
        self.scene.end_tip_drag()

        self.assertGreater(np.linalg.norm(self.scene.robot.current_q - start_q), 1.0e-5)
        np.testing.assert_allclose(
            self.scene.robot.tip_position_mm,
            local_target_mm,
            rtol=0.0,
            atol=0.02,
        )
        np.testing.assert_allclose(
            self.scene.robot.tip_rotation,
            start_rotation,
            rtol=0.0,
            atol=2.0e-4,
        )
        np.testing.assert_array_equal(
            self.scene._transform_poses["process"], process_pose
        )
        self.assertEqual(statuses[-1][0], "ok")

        self.assertTrue(self.scene.begin_tip_drag())
        self.assertFalse(
            self.scene.drag_tip(self.scene.tip_world_position() + (5000.0, 0.0, 0.0))
        )
        self.scene.end_tip_drag()
        np.testing.assert_allclose(
            handle.GetProperty().GetColor(),
            (0xD1 / 255.0, 0x24 / 255.0, 0x2F / 255.0),
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertEqual(statuses[-1][0], "error")

        # Advancing the timeline drops the current-frame override but leaves
        # ordinary object transforms alone until the reset control is used.
        self.scene.set_frame(1, render=False)
        np.testing.assert_allclose(self.scene.robot.current_q, self.q1)
        self.assertIsNone(self.scene._tip_override_q)
        np.testing.assert_array_equal(self.scene._transform_poses["robot"], robot_pose)

        self.scene.reset_object_transforms(render=False)
        np.testing.assert_array_equal(self.scene._transform_poses["robot"], np.eye(4))
        np.testing.assert_array_equal(
            self.scene._transform_poses["process"], np.eye(4)
        )

        self.scene.enable_tip_ik(False, render=False)
        self.assertFalse(handle.GetVisibility())
        self.assertFalse(handle.GetPickable())
        self.assertNotIn(handle_address, self.scene._prop_interaction_target)

    def test_vtk_tip_and_modifier_dispatch_preserve_camera_and_joint_roles(self) -> None:
        self.scene.enable_tip_ik(True, render=False)
        self.scene.render()
        style = self.scene._interaction_style
        interactor = self.scene.plotter.iren.interactor
        camera_before = self.scene._capture_camera()
        q_before = self.scene.robot.current_q.copy()

        display = self.scene.world_to_display(self.scene.tip_world_position())
        x, y = (round(display[0]), round(display[1]))
        target, group, _ = self.scene.pick_interaction_target(x, y)
        self.assertEqual(target, "tip")
        self.assertEqual(group, "robot")

        interactor.SetEventInformation(x, y)
        style._on_left_press(None, None)
        self.assertEqual(style.drag_mode, "tip")
        interactor.SetEventInformation(x + 7, y + 3)
        style._on_mouse_move(None, None)
        style._on_left_release(None, None)

        self.assertGreater(
            np.linalg.norm(self.scene.robot.current_q - q_before), 1.0e-5
        )
        np.testing.assert_array_equal(
            self.scene._capture_camera()["position"], camera_before["position"]
        )
        np.testing.assert_array_equal(
            self.scene._capture_camera()["focal_point"],
            camera_before["focal_point"],
        )

        # Shift-dragging the same handle keeps its existing rigid-group role.
        q_after_ik = self.scene.robot.current_q.copy()
        pose_before = self.scene._transform_poses["robot"].copy()
        display = self.scene.world_to_display(self.scene.tip_world_position())
        x, y = (round(display[0]), round(display[1]))
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

    def test_widget_opt_in_playback_and_close_lifecycle(self) -> None:
        viewer = widgets.HTML()
        layers = self.scene.layers
        with tempfile.TemporaryDirectory() as directory:
            store = WidgetStore(Path(directory) / "state.json")
            with mock.patch.object(self.scene.plotter, "show", return_value=viewer):
                app = PyVistaWidgetApp(
                    self.scene,
                    layers=layers,
                    store=store,
                    scene_name="test",
                    title="test scene",
                    enable_tip_ik=True,
                )

            self.assertTrue(self.scene.robot.tip_handle_enabled)
            self.assertIn("orange handle", app.title.value)
            self.assertIn("Mink IK ready", app.tip_ik_status.value)

            self.assertTrue(self.scene.begin_tip_drag())
            self.assertTrue(
                self.scene.drag_tip(self.scene.tip_world_position() + (1.0, 0.0, 0.0))
            )
            self.assertIsNotNone(self.scene._tip_override_q)
            app.play.playing = True
            self.assertIsNone(self.scene._tip_override_q)

            app.close()
            self.assertTrue(app.closed)
            self.assertTrue(self.scene.closed)
            self.assertFalse(self.scene.robot.tip_handle_enabled)
            self.assertFalse(self.scene._prop_interaction_target)


if __name__ == "__main__":
    unittest.main()
