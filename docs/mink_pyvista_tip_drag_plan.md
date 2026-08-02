# Mink-powered UR5e tip dragging

## Summary

Add live Mink inverse kinematics to the **Composite PyVista** and **Seam
PyVista** tabs produced by cell 8. Dragging a visible TCP handle pauses
playback and updates the UR5e joint configuration in place while preserving
the existing camera and rigid-object controls.

The native RTX renderer remains unchanged. This feature applies to the
PyVista frontend and the RTX notebook's non-EGL fallback.

## Implementation changes

- Add a reusable `MinkArmIK` adapter backed by the existing
  `build_mjcf(arm)` model.
  - Use the `tcp` site with a full-pose `FrameTask` and a low-weight
    `PostureTask`.
  - Preserve the tool orientation captured at drag start and warm-start each
    solve from the last valid configuration.
  - Use `daqp`, at most 30 iterations, a 0.01 mm position tolerance, and a
    0.0001 rad orientation tolerance.
  - Apply +/-2 pi joint limits with a 1 degree margin and pi rad/s velocity
    limits.
  - Return the best finite, bounded configuration for unreachable targets;
    retain the previous configuration for solver errors or non-finite output.
  - Do not enable collision avoidance because the generated MJCF uses
    intentionally overlapping link geometry.

- Extend the shared PyVista interaction layer.
  - Add an orange, separately pickable TCP handle that follows
    `RobotRig.update(q)`.
  - Use an unmodified left drag on the handle for IK. Dragging elsewhere keeps
    orbiting the camera, while Shift/Ctrl dragging retains rigid
    translate/rotate behavior.
  - Capture display depth and cursor offset at drag start, convert world
    millimetres into robot-local metres, and inverse-apply any rigid robot
    transform before solving.
  - Update links, joints, gun, and handle through persistent VTK objects.
  - Report convergence/residual feedback and colour the handle red for an
    unreachable request.

- Integrate scene and widget lifecycle behavior.
  - Enable the feature independently in the Composite and Seam apps.
  - Keep the handle disabled during Seam GIF capture, then enable it for the
    live widget.
  - Pause playback during dragging. Keep the manual pose until playback or the
    frame slider advances, at which point the recorded pose is restored.
  - Reset both rigid object transforms and the IK override from the existing
    reset control.
  - Remove IK callbacks and picker registrations when an app closes.
  - Preserve the portable notebook's eight-code-cell order and RTX fallback
    mapping.

## Interfaces

- `welding_dynamics.robot_ik.MinkArmIK.solve(q_seed, target_position_m,
  target_rotation, posture_q=...)` returns `MinkIKResult`.
- `MinkIKResult` exposes the solved configuration, achieved TCP position,
  position and orientation residuals, and convergence state.
- `PyVistaWidgetApp(..., enable_tip_ik=False)` opts a live scene into TCP
  manipulation.

## Test plan

- Verify MuJoCo/Mink and `SixDofArm` TCP poses agree for several
  configurations.
- Test reachable fixed-orientation targets, warm-start continuity, joint
  limits, unreachable targets, and solver-failure retention.
- Exercise synthetic TCP drags and verify that robot geometry changes while
  process geometry and rigid group transforms remain fixed.
- Test non-identity robot transforms, playback pause, frame/reset behavior,
  and independent state in both tabs.
- Confirm the handle is absent from generated GIF frames and visible in both
  live widgets.
- Run lint and static notebook parsing, retain the strict eight-cell fallback
  mapping, and run the RTX notebook through its non-EGL PyVista fallback.

## Assumptions

- Tool orientation is fixed during each drag.
- IK overrides apply only to the current frame.
- An always-visible TCP handle is used in live widgets.
- Composite and Seam tabs maintain independent IK poses.
- Native RTX widgets are outside this PyVista-specific change.
