"""Mink inverse kinematics for :class:`~robot_vi.SixDofArm`.

The adapter builds its MuJoCo model from :func:`robot_mujoco.build_mjcf`, so
the differential IK problem and the visualization use the same DH parameters.
It is intentionally independent of PyVista and can be reused by other live
frontends.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import numpy.typing as npt

from .robot_mujoco import build_mjcf


def _immutable_vector(value: npt.ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MinkIKResult:
    """Result of one bounded IK request.

    Array fields are defensive, read-only copies, making instances immutable in
    practice as well as through the frozen dataclass interface.
    """

    q: np.ndarray
    achieved_position_m: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    converged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _immutable_vector(self.q, (6,)))
        object.__setattr__(
            self,
            "achieved_position_m",
            _immutable_vector(self.achieved_position_m, (3,)),
        )
        object.__setattr__(self, "position_error_m", float(self.position_error_m))
        object.__setattr__(
            self, "orientation_error_rad", float(self.orientation_error_rad)
        )
        object.__setattr__(self, "converged", bool(self.converged))

    @property
    def achieved_tcp_position_m(self) -> np.ndarray:
        """Alias spelling out that ``achieved_position_m`` is the TCP position."""

        return self.achieved_position_m

    @property
    def position_residual_m(self) -> float:
        """Alias for :attr:`position_error_m`."""

        return self.position_error_m

    @property
    def orientation_residual_rad(self) -> float:
        """Alias for :attr:`orientation_error_rad`."""

        return self.orientation_error_rad


class MinkArmIK:
    """Bounded full-pose differential IK for a six-joint arm.

    Each :meth:`solve` starts at the displayed joint configuration supplied by
    the caller. Successive QP iterations, and successive drag calls seeded with
    the prior result, therefore remain on a continuous IK branch.
    """

    JOINT_LIMIT_RAD = 2.0 * np.pi
    JOINT_LIMIT_MARGIN_RAD = np.deg2rad(1.0)
    VELOCITY_LIMIT_RAD_S = np.pi
    POSITION_TOLERANCE_M = 1.0e-5
    ORIENTATION_TOLERANCE_RAD = 1.0e-4
    MAX_STEPS = 30

    def __init__(
        self,
        arm,
        *,
        dt: float = 0.05,
        max_steps: int = MAX_STEPS,
        position_tolerance_m: float = POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = ORIENTATION_TOLERANCE_RAD,
    ) -> None:
        self.arm = arm
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if self.orientation_tolerance_rad <= 0.0:
            raise ValueError("orientation_tolerance_rad must be positive")

        self.model = mujoco.MjModel.from_xml_string(build_mjcf(arm))
        if self.model.nq != 6 or self.model.nv != 6 or self.model.njnt != 6:
            raise ValueError("MinkArmIK requires a six-joint, six-DoF arm")

        # build_mjcf deliberately leaves joints unlimited for dynamics. The IK
        # frontend uses a broad periodic range, with an extra one-degree safety
        # margin applied by ConfigurationLimit below.
        self.model.jnt_limited[:] = True
        self.model.jnt_range[:, 0] = -self.JOINT_LIMIT_RAD
        self.model.jnt_range[:, 1] = self.JOINT_LIMIT_RAD
        self._lower_q = np.full(
            6, -self.JOINT_LIMIT_RAD + self.JOINT_LIMIT_MARGIN_RAD
        )
        self._upper_q = np.full(
            6, self.JOINT_LIMIT_RAD - self.JOINT_LIMIT_MARGIN_RAD
        )

        self._configuration = mink.Configuration(self.model)
        self._frame_task = mink.FrameTask(
            "tcp",
            "site",
            position_cost=1.0,
            orientation_cost=0.5,
            gain=0.7,
            lm_damping=1.0e-3,
        )
        self._posture_task = mink.PostureTask(self.model, cost=1.0e-3)
        configuration_limit = mink.ConfigurationLimit(
            self.model,
            gain=0.95,
            min_distance_from_limits=self.JOINT_LIMIT_MARGIN_RAD,
        )
        velocity_limit = mink.VelocityLimit(
            self.model,
            {f"j{joint}": self.VELOCITY_LIMIT_RAD_S for joint in range(1, 7)},
        )
        self._limits = (configuration_limit, velocity_limit)
        self._tasks = (self._frame_task, self._posture_task)

    def _bounded_q(self, q: npt.ArrayLike, name: str) -> np.ndarray:
        value = np.asarray(q, dtype=float)
        if value.shape != (6,):
            raise ValueError(f"{name} must have shape (6,), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return np.clip(value, self._lower_q, self._upper_q).copy()

    @staticmethod
    def _target_position(value: npt.ArrayLike) -> np.ndarray:
        target = np.asarray(value, dtype=float)
        if target.shape != (3,):
            raise ValueError(
                f"target_position_m must have shape (3,), got {target.shape}"
            )
        return target.copy()

    @staticmethod
    def _target_rotation(value: npt.ArrayLike) -> np.ndarray:
        rotation = np.asarray(value, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(
                f"target_rotation must have shape (3, 3), got {rotation.shape}"
            )
        return rotation.copy()

    def _pose_errors(
        self,
        target_position_m: np.ndarray,
        target_rotation: mink.SO3,
    ) -> tuple[np.ndarray, float, float]:
        pose = self._configuration.get_transform_frame_to_world("tcp", "site")
        position = pose.translation().copy()
        position_error = float(np.linalg.norm(position - target_position_m))
        orientation_error = float(
            np.linalg.norm(target_rotation.minus(pose.rotation()))
        )
        return position, position_error, orientation_error

    def _result_at_current_configuration(
        self,
        target_position_m: np.ndarray,
        target_rotation: mink.SO3 | None,
        *,
        converged: bool,
    ) -> MinkIKResult:
        pose = self._configuration.get_transform_frame_to_world("tcp", "site")
        position = pose.translation().copy()
        if target_rotation is None or not np.all(np.isfinite(target_position_m)):
            position_error = np.inf
            orientation_error = np.inf
        else:
            position_error = float(np.linalg.norm(position - target_position_m))
            orientation_error = float(
                np.linalg.norm(target_rotation.minus(pose.rotation()))
            )
        return MinkIKResult(
            q=self._configuration.q.copy(),
            achieved_position_m=position,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            converged=converged,
        )

    def _solve_velocity(self) -> np.ndarray:
        return mink.solve_ik(
            self._configuration,
            self._tasks,
            self.dt,
            solver="daqp",
            damping=1.0e-6,
            limits=self._limits,
        )

    def solve(
        self,
        q_seed: npt.ArrayLike,
        target_position_m: npt.ArrayLike,
        target_rotation: npt.ArrayLike,
        posture_q: npt.ArrayLike | None = None,
    ) -> MinkIKResult:
        """Solve for a bounded joint configuration matching a TCP pose.

        Args:
            q_seed: Current/displayed six-joint configuration in radians.
            target_position_m: Desired TCP position in metres.
            target_rotation: Desired world-from-TCP 3x3 rotation matrix.
            posture_q: Optional regularization posture. Defaults to ``q_seed``.

        Solver failures and non-finite solver output return the best finite
        configuration reached so far with ``converged=False``. Non-finite target
        values retain the bounded seed in the same way; malformed array shapes
        and non-finite seeds raise :class:`ValueError`.
        """

        seed = self._bounded_q(q_seed, "q_seed")
        posture = seed if posture_q is None else self._bounded_q(posture_q, "posture_q")
        target_position = self._target_position(target_position_m)
        target_matrix = self._target_rotation(target_rotation)
        self._configuration.update(seed)

        if not np.all(np.isfinite(target_position)) or not np.all(
            np.isfinite(target_matrix)
        ):
            return self._result_at_current_configuration(
                target_position, None, converged=False
            )

        gram = target_matrix.T @ target_matrix
        if not np.allclose(gram, np.eye(3), atol=1.0e-4, rtol=1.0e-4) or not np.isclose(
            np.linalg.det(target_matrix), 1.0, atol=1.0e-4, rtol=1.0e-4
        ):
            raise ValueError("target_rotation must be a proper rotation matrix")

        target_so3 = mink.SO3.from_matrix(target_matrix)
        target = mink.SE3.from_rotation_and_translation(
            rotation=target_so3,
            translation=target_position,
        )
        self._frame_task.set_target(target)
        self._posture_task.set_target(posture)

        best_q = seed.copy()
        best_score = np.inf
        for _ in range(self.max_steps + 1):
            position, position_error, orientation_error = self._pose_errors(
                target_position, target_so3
            )
            if not (
                np.all(np.isfinite(position))
                and np.isfinite(position_error)
                and np.isfinite(orientation_error)
            ):
                self._configuration.update(best_q)
                break

            score = (
                position_error / self.position_tolerance_m
            ) ** 2 + (
                orientation_error / self.orientation_tolerance_rad
            ) ** 2
            if score < best_score:
                best_score = score
                best_q = self._configuration.q.copy()
            if (
                position_error <= self.position_tolerance_m
                and orientation_error <= self.orientation_tolerance_rad
            ):
                return MinkIKResult(
                    q=self._configuration.q.copy(),
                    achieved_position_m=position,
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                    converged=True,
                )

            # The final pass above is only for measuring a 30th integration.
            if _ == self.max_steps:
                break
            previous_q = self._configuration.q.copy()
            try:
                velocity = np.asarray(self._solve_velocity(), dtype=float)
                if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
                    self._configuration.update(previous_q)
                    break
                self._configuration.integrate_inplace(velocity, self.dt)
                integrated_q = self._configuration.q.copy()
                if not np.all(np.isfinite(integrated_q)):
                    self._configuration.update(previous_q)
                    break
                bounded_q = np.clip(integrated_q, self._lower_q, self._upper_q)
                if not np.array_equal(bounded_q, integrated_q):
                    self._configuration.update(bounded_q)
            # QP backends do not share one exception base. A live drag should
            # retain its last usable pose rather than tear down the widget.
            except Exception:  # noqa: BLE001
                self._configuration.update(previous_q)
                break

        self._configuration.update(best_q)
        return self._result_at_current_configuration(
            target_position, target_so3, converged=False
        )


__all__ = ["MinkArmIK", "MinkIKResult"]
