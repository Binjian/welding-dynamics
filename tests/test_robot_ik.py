import unittest
from unittest import mock

import mink
import numpy as np

from welding_dynamics.robot_ik import MinkArmIK
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


class MinkArmIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arm = make_ur5e()

    def setUp(self) -> None:
        self.ik = MinkArmIK(self.arm)
        self.q0 = np.array((0.3, -2.0, -1.6, 2.1, -1.57, 1.9))

    def test_mink_fk_matches_six_dof_arm(self) -> None:
        for q in (
            np.zeros(6),
            self.q0,
            np.array((-1.2, -0.8, 1.4, -2.2, 0.75, 2.6)),
        ):
            self.ik._configuration.update(q)
            pose = self.ik._configuration.get_transform_frame_to_world("tcp", "site")
            expected_position, expected_rotation = self.arm.fk_pose(q)
            np.testing.assert_allclose(
                pose.translation(), expected_position, rtol=0.0, atol=2.0e-12
            )
            np.testing.assert_allclose(
                pose.rotation().as_matrix(), expected_rotation, rtol=0.0, atol=2.0e-12
            )

    def test_reachable_target_converges_with_full_orientation(self) -> None:
        target_q = self.q0 + np.array((0.025, -0.035, 0.03, 0.02, -0.018, 0.015))
        target_position, target_rotation = self.arm.fk_pose(target_q)

        result = self.ik.solve(self.q0, target_position, target_rotation)

        self.assertTrue(result.converged)
        self.assertLessEqual(result.position_error_m, self.ik.POSITION_TOLERANCE_M)
        self.assertLessEqual(
            result.orientation_error_rad, self.ik.ORIENTATION_TOLERANCE_RAD
        )
        _, achieved_rotation = self.arm.fk_pose(result.q)
        orientation_delta = achieved_rotation.T @ target_rotation
        orientation_angle = np.arccos(
            np.clip((np.trace(orientation_delta) - 1.0) / 2.0, -1.0, 1.0)
        )
        self.assertLessEqual(orientation_angle, self.ik.ORIENTATION_TOLERANCE_RAD)

    def test_successive_targets_warm_start_continuously(self) -> None:
        start_position, fixed_rotation = self.arm.fk_pose(self.q0)
        first = self.ik.solve(
            self.q0,
            start_position + np.array((0.010, -0.005, 0.008)),
            fixed_rotation,
            posture_q=self.q0,
        )
        second = self.ik.solve(
            first.q,
            start_position + np.array((0.012, -0.004, 0.009)),
            fixed_rotation,
            posture_q=self.q0,
        )

        self.assertTrue(first.converged)
        self.assertTrue(second.converged)
        self.assertLess(np.linalg.norm(second.q - first.q), 0.1)

    def test_seed_and_result_stay_inside_margin_limits(self) -> None:
        outside = np.array((10.0, -10.0, 8.0, -8.0, 7.0, -7.0))
        bounded = np.clip(outside, self.ik._lower_q, self.ik._upper_q)
        target_position, target_rotation = self.arm.fk_pose(bounded)

        result = self.ik.solve(outside, target_position, target_rotation)

        self.assertTrue(np.all(result.q >= self.ik._lower_q))
        self.assertTrue(np.all(result.q <= self.ik._upper_q))

    def test_unreachable_target_returns_best_finite_bounded_pose(self) -> None:
        _, rotation = self.arm.fk_pose(self.q0)

        result = self.ik.solve(self.q0, np.array((4.0, -3.0, 5.0)), rotation)

        self.assertFalse(result.converged)
        self.assertTrue(np.all(np.isfinite(result.q)))
        self.assertTrue(np.all(np.isfinite(result.achieved_position_m)))
        self.assertTrue(np.all(result.q >= self.ik._lower_q))
        self.assertTrue(np.all(result.q <= self.ik._upper_q))
        self.assertGreater(result.position_error_m, self.ik.POSITION_TOLERANCE_M)

    def test_solver_exception_retains_seed(self) -> None:
        position, rotation = self.arm.fk_pose(self.q0)
        target = position + np.array((0.01, 0.0, 0.0))

        with mock.patch.object(
            self.ik, "_solve_velocity", side_effect=mink.NoSolutionFound("daqp")
        ):
            result = self.ik.solve(self.q0, target, rotation)

        self.assertFalse(result.converged)
        np.testing.assert_array_equal(result.q, self.q0)

    def test_nonfinite_solver_output_retains_seed(self) -> None:
        position, rotation = self.arm.fk_pose(self.q0)
        target = position + np.array((0.01, 0.0, 0.0))

        with mock.patch.object(
            self.ik, "_solve_velocity", return_value=np.full(6, np.nan)
        ):
            result = self.ik.solve(self.q0, target, rotation)

        self.assertFalse(result.converged)
        np.testing.assert_array_equal(result.q, self.q0)

    def test_result_arrays_are_read_only(self) -> None:
        position, rotation = self.arm.fk_pose(self.q0)
        result = self.ik.solve(self.q0, position, rotation)

        with self.assertRaises(ValueError):
            result.q[0] = 0.0
        with self.assertRaises(ValueError):
            result.achieved_position_m[0] = 0.0


if __name__ == "__main__":
    unittest.main()
