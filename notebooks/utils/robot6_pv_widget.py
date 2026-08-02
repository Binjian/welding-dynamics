"""Live PyVista/Trame scenes for the UR5e weave notebook.

The ordinary PyVista notebook uses the ``server`` Jupyter backend so that the
browser forwards interaction events to the same VTK render window that owns
the scene.  This module supplies the application-side pieces which are shared
by sections 2, 3, 5 and 5b:

* persistent, topology-changing VTK pipelines for animation;
* actor layers and instantaneous/peak field modes;
* studio lighting, a generated sky and a finite grid floor;
* world-space axes whose labels scale naturally when the camera zooms;
* selection and mouse transforms for the robot and process-object groups;
* ipywidget playback/layer/view controls suitable for placing in tabs.

It intentionally does not depend on ``trame-rtx-widget``.  Rendering and the
notebook iframe are owned by :class:`pyvista.Plotter`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pyvista as pv
import vtk
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from vtk.util.numpy_support import numpy_to_vtk

from .view_controls import VIEW_KEYS, WidgetStore, view_widgets

_STD_STREAMS = (sys.stdout, sys.stderr)


def restore_std_streams():
    """Undo VTK's read-only Python stream capture (IPython 9 compatibility)."""

    for name, original in zip(("stdout", "stderr"), _STD_STREAMS):
        if type(getattr(sys, name)).__name__ == "vtkPythonStdStreamCaptureHelper":
            setattr(sys, name, original)


@dataclass(frozen=True)
class LayerSpec:
    """Description and initial visibility of one logical actor layer."""

    key: str
    label: str
    visible: bool = True


@dataclass(frozen=True)
class RobotWeaveContext:
    """Geometry and trajectory data shared by all robot-weave scenes."""

    arm: object
    q_at: Callable[[float], np.ndarray]
    t_track: float
    t_trace: np.ndarray
    tip: np.ndarray
    p0: np.ndarray
    travel_speed: float


MOVABLE_LAYER_GROUPS = {
    "robot": "robot",
    "workpiece": "process",
    "seam": "process",
    "executed_path": "process",
    "thermal": "process",
    "domain": "process",
    "convection": "process",
    "history": "process",
    "pool": "process",
    "halo": "process",
}

R_LINK = (45.0, 38.0, 32.0, 22.0, 20.0, 16.0)
R_JOINT = (50.0, 42.0, 36.0, 26.0, 24.0)


def _rgb(color):
    return tuple(float(channel) for channel in to_rgb(color))


def vtk_rgb_image(rgb):
    """Convert an H×W×3 uint8 image to self-contained VTK image data."""

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("RGB image must have exactly three channels")
    image = vtk.vtkImageData()
    image.SetDimensions(width, height, 1)
    image.GetPointData().SetScalars(
        numpy_to_vtk(
            rgb.reshape(-1, 3),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
    )
    return image


def make_sky_texture(width=256, height=128):
    """Generate a dependency-free equirectangular sky with soft sunlight."""

    longitude = np.linspace(-np.pi, np.pi, int(width))[None, :]
    latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(height))[:, None]
    dx = np.cos(latitude) * np.cos(longitude)
    dy = np.cos(latitude) * np.sin(longitude)
    dz = np.broadcast_to(np.sin(latitude), dx.shape)

    up = np.clip((dz + 0.08) / 1.08, 0.0, 1.0)[..., None]
    down = np.clip((-dz - 0.08) / 0.92, 0.0, 1.0)[..., None]
    horizon = np.array((0.82, 0.90, 0.97))
    zenith = np.array((0.32, 0.52, 0.78))
    nadir = np.array((0.48, 0.55, 0.62))
    rgb = horizon * (1.0 - up) + zenith * up
    rgb = rgb * (1.0 - down) + nadir * down

    sun = np.array((-0.35, -0.45, 0.82))
    sun /= np.linalg.norm(sun)
    glow = np.clip(dx * sun[0] + dy * sun[1] + dz * sun[2], 0.0, 1.0)
    rgb = np.clip(
        rgb + glow[..., None] ** 72 * np.array((0.42, 0.33, 0.18)),
        0.0,
        1.0,
    )
    image = vtk_rgb_image(np.rint(rgb * 255.0))
    texture = vtk.vtkTexture()
    texture.SetInputData(image)
    texture.SetColorModeToDirectScalars()
    texture.InterpolateOn()
    texture.MipmapOn()
    texture.UseSRGBColorSpaceOff()
    return image, texture


def make_lut(cmap_name, value_range):
    """Build a VTK lookup table from a Matplotlib colour map."""

    lo, hi = map(float, value_range)
    table = vtk.vtkLookupTable()
    table.SetNumberOfTableValues(256)
    table.SetTableRange(lo, hi)
    cmap = colormaps[cmap_name]
    for index in range(256):
        table.SetTableValue(index, *cmap(index / 255.0))
    table.SetNanColor(0.5, 0.5, 0.5, 0.0)
    table.Build()
    return table


def structured_grid(x, y, z, arrays, *, dynamic=False):
    """Build a VTK structured grid with NumPy/VTK arrays in Fortran order."""

    x, y, z = map(np.asarray, (x, y, z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    xyz = np.ascontiguousarray(
        np.column_stack(
            (X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F"))
        ),
        dtype=np.float32,
    )
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(len(x), len(y), len(z))
    grid.SetPoints(points)

    buffers, vtk_arrays = {}, {}
    for name, values in arrays.items():
        buffer = np.ascontiguousarray(
            np.asarray(values).ravel(order="F"), dtype=np.float32
        )
        vtk_array = numpy_to_vtk(buffer, deep=not dynamic)
        vtk_array.SetName(name)
        grid.GetPointData().AddArray(vtk_array)
        buffers[name] = buffer
        vtk_arrays[name] = vtk_array
    return grid, buffers, vtk_arrays


def goldak_grid(model, arrays, p0, *, x_start, dynamic=False):
    """Map a Goldak grid into robot-base coordinates measured in mm."""

    gx = (model.x - float(x_start) + p0[0]) * 1e3
    gy = (model.y + p0[1]) * 1e3
    gz = (p0[2] - model.z) * 1e3
    grid, buffers, vtk_arrays = structured_grid(
        gx, gy, gz, arrays, dynamic=dynamic
    )
    return grid, gx, gy, gz, buffers, vtk_arrays


def contour_filter(grid, scalars, value):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalars
    )
    contour.SetValue(0, float(value))
    contour.ComputeNormalsOn()
    return contour


def slice_filter(grid, origin, normal=(0.0, 0.0, 1.0)):
    plane = vtk.vtkPlane()
    plane.SetOrigin(*map(float, origin))
    plane.SetNormal(*map(float, normal))
    cutter = vtk.vtkCutter()
    cutter.SetInputData(grid)
    cutter.SetCutFunction(plane)
    return cutter


def threshold_upper(source, scalars, value):
    threshold = vtk.vtkThreshold()
    threshold.SetInputConnection(source.GetOutputPort())
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalars
    )
    threshold.SetLowerThreshold(float(value))
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(threshold.GetOutputPort())
    return geometry


def axis_rotation_matrix(axis, angle_degrees):
    """Return a homogeneous rotation about a world-space axis."""

    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4)
    x, y, z = axis / norm
    angle = np.deg2rad(float(angle_degrees))
    c, s, one_c = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )
    return matrix


class LayerTransformInteractor(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera plus Shift-translate/Ctrl-rotate actor groups."""

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.drag_group = None
        self.drag_mode = None
        self.drag_depth = 0.0
        self.drag_pivot = None
        self.last_position = None
        self._observer_tags = (
            self.AddObserver("LeftButtonPressEvent", self._on_left_press),
            self.AddObserver("LeftButtonReleaseEvent", self._on_left_release),
            self.AddObserver("MouseMoveEvent", self._on_mouse_move),
        )

    def _on_left_press(self, _caller, _event):
        interactor, scene = self.GetInteractor(), self.scene
        if interactor is None or scene is None or scene.closed:
            return
        x, y = map(float, interactor.GetEventPosition())
        group, pick_position = scene.pick_transform_group(x, y)
        scene.select_transform_group(group, render=False)
        control, shift = bool(interactor.GetControlKey()), bool(interactor.GetShiftKey())
        if group is not None and (control or shift):
            self.drag_group = group
            self.drag_mode = "rotate" if control else "translate"
            self.drag_depth = scene.world_to_display(pick_position)[2]
            self.drag_pivot = scene.transform_group_center(group)
            self.last_position = scene.normalize_display_position((x, y))
            scene.notify_transform_started()
            interactor.Render()
            return
        self._clear_drag()
        self.OnLeftButtonDown()

    def _on_mouse_move(self, _caller, _event):
        interactor, scene = self.GetInteractor(), self.scene
        if interactor is None or scene is None or scene.closed:
            return
        if self.drag_group is None:
            self.OnMouseMove()
            return
        position = scene.normalize_display_position(interactor.GetEventPosition())
        scene.drag_transform_group(
            self.drag_group,
            self.drag_mode,
            self.last_position,
            position,
            self.drag_depth,
            self.drag_pivot,
        )
        self.last_position = position
        interactor.Render()

    def _on_left_release(self, _caller, _event):
        if self.drag_group is None:
            self.OnLeftButtonUp()
            return
        self._clear_drag()
        interactor = self.GetInteractor()
        if interactor is not None:
            interactor.Render()

    def _clear_drag(self):
        self.drag_group = self.drag_mode = self.drag_pivot = self.last_position = None

    def detach(self):
        self._clear_drag()
        for tag in self._observer_tags:
            self.RemoveObserver(tag)
        self._observer_tags = ()
        self.scene = None


class WeldingPVSceneBase:
    """Persistent VTK scene displayed by PyVista's live server widget."""

    def __init__(
        self,
        *,
        layers: Sequence[LayerSpec],
        frame_count: int,
        current_frame: int,
        camera_home,
        size=(960, 640),
        floor=(-650.0, 1150.0, -700.0, 700.0, -31.5, 100.0),
    ):
        if int(frame_count) <= 0:
            raise ValueError("frame_count must be positive")
        self.layers = tuple(layers)
        self.frame_count = int(frame_count)
        self.current_frame = max(0, min(int(current_frame), self.frame_count - 1))
        self.caption = ""
        self.closed = False
        self._records = []
        self._transform_records = {}
        self._transform_poses = {}
        self._transform_objects = {}
        self._prop_transform_group = {}
        self._selected_transform_group = None
        self._transform_start_callback = None
        self._interaction_style = None
        self._interaction_picker = vtk.vtkCellPicker()
        self._interaction_picker.SetTolerance(0.005)
        self._interaction_picker.PickFromListOn()
        self._layer_visible = {
            spec.key: bool(spec.visible)
            for spec in self.layers
            if spec.key != "instant_mode"
        }
        self._instant_mode = next(
            (bool(spec.visible) for spec in self.layers if spec.key == "instant_mode"),
            False,
        )
        self._layer_opacity = {"robot": 1.0}
        self.plotter = pv.Plotter(notebook=True, window_size=tuple(map(int, size)))
        self.renderer = self.plotter.renderer
        self.render_window = self.plotter.render_window
        try:
            self._build_environment(floor)
            self._build_scene()
            self._build_selection_outline()
            self.set_frame(self.current_frame, render=False)
            self._set_camera(camera_home)
            self._camera_base = self._capture_camera()
            self._camera_home = dict(self._camera_base)
            self._interaction_style = LayerTransformInteractor(self)
            self.plotter.iren.interactor.SetInteractorStyle(self._interaction_style)
        except Exception:
            self.close()
            raise
        finally:
            restore_std_streams()

    def _build_environment(self, floor):
        """Add sky, bounded grid floor and a neutral photographic light kit."""

        xmin, xmax, ymin, ymax, z_floor, spacing = map(float, floor)
        if not (xmin < xmax and ymin < ymax and spacing > 0.0):
            raise ValueError("invalid floor bounds or spacing")
        self.plotter.set_background("#c7dbe8", top="#4d7fae")
        self.renderer.AutomaticLightCreationOff()
        self.renderer.RemoveAllLights()
        self.renderer.SetAmbient(0.18, 0.20, 0.24)
        self._light_kit = vtk.vtkLightKit()
        self._light_kit.MaintainLuminanceOn()
        self._light_kit.SetKeyLightIntensity(0.82)
        self._light_kit.SetKeyToFillRatio(2.5)
        self._light_kit.SetKeyToHeadRatio(4.0)
        self._light_kit.SetKeyToBackRatio(3.0)
        self._light_kit.SetKeyLightWarmth(0.58)
        self._light_kit.SetFillLightWarmth(0.42)
        self._light_kit.SetHeadLightWarmth(0.50)
        self._light_kit.SetBackLightWarmth(0.54)
        self._light_kit.AddLightsToRenderer(self.renderer)

        self._sky_image, self._sky_texture = make_sky_texture()
        sky = vtk.vtkSkybox()
        sky.SetTexture(self._sky_texture)
        sky.SetProjectionToSphere()
        sky.UseBoundsOff()
        sky.PickableOff()
        sky.DragableOff()
        self._register(sky, "environment", add_to_renderer=True)

        source = vtk.vtkPlaneSource()
        source.SetOrigin(xmin, ymin, z_floor)
        source.SetPoint1(xmax, ymin, z_floor)
        source.SetPoint2(xmin, ymax, z_floor)
        source.SetXResolution(max(1, round((xmax - xmin) / spacing)))
        source.SetYResolution(max(1, round((ymax - ymin) / spacing)))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        floor_actor = vtk.vtkActor()
        floor_actor.SetMapper(mapper)
        floor_actor.UseBoundsOff()
        prop = floor_actor.GetProperty()
        prop.SetColor(*_rgb("#74808a"))
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(*_rgb("#46535e"))
        prop.SetLineWidth(0.8)
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.12)
        prop.SetSpecularPower(24.0)
        self._register(
            floor_actor,
            "environment",
            opacity=1.0,
            transform_group=None,
            pickable=False,
            add_to_renderer=True,
        )

    def _build_scene(self):
        raise NotImplementedError

    def _update_frame(self, index):
        raise NotImplementedError

    def _build_selection_outline(self):
        source = vtk.vtkOutlineSource()
        source.SetBounds(-1, 1, -1, 1, -1, 1)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.72, 0.08)
        actor.GetProperty().SetLineWidth(3.0)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        actor.DragableOff()
        actor.UseBoundsOff()
        actor.VisibilityOff()
        self._selection_outline_source = source
        self._selection_outline = actor
        self.renderer.AddActor(actor)

    def _ensure_transform_group(self, group):
        if group not in self._transform_objects:
            self._transform_objects[group] = vtk.vtkTransform()
            self._transform_poses[group] = np.eye(4)
            self._transform_records[group] = []
        return self._transform_objects[group]

    def _set_transform_pose(self, group, pose):
        pose = np.asarray(pose, dtype=float).reshape(4, 4)
        matrix = vtk.vtkMatrix4x4()
        for row in range(4):
            for column in range(4):
                matrix.SetElement(row, column, pose[row, column])
        self._transform_poses[group] = pose
        self._transform_objects[group].SetMatrix(matrix)
        self._transform_objects[group].Modified()

    def transform_group_bounds(self, group):
        box = vtk.vtkBoundingBox()
        for record in self._transform_records.get(group, ()):
            prop = record["prop"]
            if not prop.GetVisibility():
                continue
            bounds = prop.GetBounds()
            if bounds is not None and np.all(np.isfinite(bounds)):
                box.AddBounds(bounds)
        if not box.IsValid():
            return None
        bounds = [0.0] * 6
        box.GetBounds(bounds)
        return np.asarray(bounds, dtype=float)

    def transform_group_center(self, group):
        bounds = self.transform_group_bounds(group)
        if bounds is None:
            return np.zeros(3)
        return np.array(
            [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
        )

    def _update_selection_outline(self):
        group = self._selected_transform_group
        bounds = None if group is None else self.transform_group_bounds(group)
        if bounds is None:
            self._selection_outline.VisibilityOff()
            return
        diagonal = np.linalg.norm(bounds[[1, 3, 5]] - bounds[[0, 2, 4]])
        padding = max(1.0, 0.0125 * diagonal)
        padded = bounds + np.array(
            [-padding, padding, -padding, padding, -padding, padding]
        )
        self._selection_outline_source.SetBounds(tuple(padded))
        self._selection_outline_source.Modified()
        self._selection_outline.VisibilityOn()

    def select_transform_group(self, group, *, render=True):
        if group not in self._transform_records:
            group = None
        self._selected_transform_group = group
        self._update_selection_outline()
        if render:
            self.render()

    def pick_transform_group(self, x, y):
        if self.closed:
            return None, None
        picked = self._interaction_picker.Pick(float(x), float(y), 0.0, self.renderer)
        prop = self._interaction_picker.GetViewProp() if picked else None
        if prop is None:
            return None, None
        group = self._prop_transform_group.get(prop.GetAddressAsString(""))
        if group is None:
            return None, None
        return group, np.asarray(self._interaction_picker.GetPickPosition(), dtype=float)

    def world_to_display(self, point):
        self.renderer.SetWorldPoint(*map(float, point), 1.0)
        self.renderer.WorldToDisplay()
        return np.asarray(self.renderer.GetDisplayPoint(), dtype=float)

    def display_to_world(self, x, y, depth):
        self.renderer.SetDisplayPoint(float(x), float(y), float(depth))
        self.renderer.DisplayToWorld()
        point = np.asarray(self.renderer.GetWorldPoint(), dtype=float)
        return None if abs(point[3]) < 1e-12 else point[:3] / point[3]

    def normalize_display_position(self, position):
        width, height = self.render_window.GetSize()
        return (
            float(position[0]) / max(1.0, float(width)),
            float(position[1]) / max(1.0, float(height)),
        )

    def denormalize_display_position(self, position):
        width, height = self.render_window.GetSize()
        return (
            float(position[0]) * max(1.0, float(width)),
            float(position[1]) * max(1.0, float(height)),
        )

    def notify_transform_started(self):
        if self._transform_start_callback is not None:
            self._transform_start_callback()

    def set_transform_start_callback(self, callback):
        self._transform_start_callback = callback

    def _apply_transform_delta(self, group, delta):
        self._set_transform_pose(group, np.asarray(delta) @ self._transform_poses[group])
        self._update_selection_outline()
        self.renderer.ResetCameraClippingRange()

    def drag_transform_group(self, group, mode, previous, current, depth, pivot):
        if group not in self._transform_poses or previous == current:
            return
        previous = self.denormalize_display_position(previous)
        current = self.denormalize_display_position(current)
        if mode == "translate":
            before = self.display_to_world(*previous, depth)
            after = self.display_to_world(*current, depth)
            if before is None or after is None:
                return
            delta = np.eye(4)
            delta[:3, 3] = after - before
        elif mode == "rotate":
            dx, dy = float(current[0] - previous[0]), float(current[1] - previous[1])
            width, height = self.render_window.GetSize()
            sensitivity = 180.0 / max(1.0, min(width, height))
            camera = self.renderer.GetActiveCamera()
            view_up = np.asarray(camera.GetViewUp(), dtype=float)
            direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
            camera_right = np.cross(direction, view_up)
            to_origin, from_origin = np.eye(4), np.eye(4)
            to_origin[:3, 3] = -np.asarray(pivot, dtype=float)
            from_origin[:3, 3] = np.asarray(pivot, dtype=float)
            delta = (
                from_origin
                @ axis_rotation_matrix(view_up, -dx * sensitivity)
                @ axis_rotation_matrix(camera_right, dy * sensitivity)
                @ to_origin
            )
        else:
            raise ValueError(f"unknown drag mode: {mode}")
        self._apply_transform_delta(group, delta)

    def reset_object_transforms(self, *, render=True):
        for group in tuple(self._transform_poses):
            self._set_transform_pose(group, np.eye(4))
        self.select_transform_group(None, render=False)
        self.renderer.ResetCameraClippingRange()
        if render:
            self.render()

    def _register(
        self,
        prop,
        layer,
        *,
        mode=None,
        opacity=None,
        enabled=True,
        transform_group="auto",
        pickable=True,
        add_to_renderer=False,
    ):
        group = MOVABLE_LAYER_GROUPS.get(layer) if transform_group == "auto" else transform_group
        movable = group is not None and isinstance(prop, vtk.vtkProp3D)
        record = {
            "prop": prop,
            "layer": layer,
            "mode": mode,
            "opacity": opacity,
            "enabled": bool(enabled),
            "transform_group": group if movable else None,
        }
        if movable:
            transform = self._ensure_transform_group(group)
            prop.SetUserTransform(transform)
            self._transform_records[group].append(record)
            if pickable:
                prop.PickableOn()
                self._interaction_picker.AddPickList(prop)
                self._prop_transform_group[prop.GetAddressAsString("")] = group
            else:
                prop.PickableOff()
        else:
            prop.PickableOff()
        self._records.append(record)
        if add_to_renderer:
            self.renderer.AddViewProp(prop)
        self._apply_record(record)
        return record

    def _apply_record(self, record):
        mode = record["mode"]
        mode_visible = (
            mode is None
            or (mode == "instant" and self._instant_mode)
            or (mode == "peak" and not self._instant_mode)
        )
        factor = self._layer_opacity.get(record["layer"], 1.0)
        visible = (
            record["enabled"]
            and self._layer_visible.get(record["layer"], True)
            and mode_visible
            and factor > 0.02
        )
        record["prop"].SetVisibility(bool(visible))
        if record["opacity"] is not None:
            record["prop"].GetProperty().SetOpacity(float(record["opacity"]) * factor)

    def _apply_visibility(self):
        for record in self._records:
            self._apply_record(record)

    def set_record_enabled(self, record, enabled):
        record["enabled"] = bool(enabled)
        self._apply_record(record)

    def add_surface(
        self,
        source,
        layer,
        *,
        mode=None,
        color="#a7a9ac",
        opacity=1.0,
        scalars=None,
        cmap="viridis",
        clim=None,
        smooth=True,
        transform_group="auto",
        pickable=True,
    ):
        mapper = vtk.vtkDataSetMapper()
        if hasattr(source, "GetOutputPort"):
            mapper.SetInputConnection(source.GetOutputPort())
        else:
            mapper.SetInputData(source)
        lut = None
        if scalars is None:
            mapper.ScalarVisibilityOff()
        else:
            if clim is None:
                raise ValueError("clim is required for scalar mapping")
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(scalars)
            mapper.SetColorModeToMapScalars()
            lut = make_lut(cmap, clim)
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(*map(float, clim))
            mapper.UseLookupTableScalarRangeOn()
            mapper.InterpolateScalarsBeforeMappingOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*_rgb(color))
        prop.SetInterpolationToPhong()
        thermal = scalars is not None or layer in {
            "thermal",
            "convection",
            "history",
            "pool",
            "halo",
        }
        prop.SetAmbient(0.50 if thermal else 0.24)
        prop.SetDiffuse(0.50 if thermal else 0.76)
        prop.SetSpecular(0.0 if thermal or not smooth else 0.16)
        prop.SetSpecularPower(24.0)
        record = self._register(
            actor,
            layer,
            mode=mode,
            opacity=float(opacity),
            transform_group=transform_group,
            pickable=pickable,
            add_to_renderer=True,
        )
        return record, lut

    def add_scalar_bar(self, lut, title, layer, *, mode=None, position=(0.83, 0.08)):
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle(title)
        bar.SetNumberOfLabels(5)
        bar.SetPosition(*position)
        bar.SetWidth(0.14)
        bar.SetHeight(0.34)
        bar.DrawBackgroundOn()
        bar.GetBackgroundProperty().SetColor(0.08, 0.12, 0.18)
        bar.GetBackgroundProperty().SetOpacity(0.22)
        bar.GetTitleTextProperty().SetColor(0.96, 0.97, 0.99)
        bar.GetLabelTextProperty().SetColor(0.96, 0.97, 0.99)
        return self._register(bar, layer, mode=mode, add_to_renderer=True)

    def add_text(self, text, layer="annotation", *, position=(12, 12), font_size=18):
        actor = vtk.vtkTextActor()
        actor.SetInput(str(text))
        actor.SetDisplayPosition(*map(int, position))
        actor.GetTextProperty().SetFontSize(int(font_size))
        actor.GetTextProperty().SetColor(*_rgb("#f4f7fa"))
        actor.GetTextProperty().SetFontFamilyToArial()
        actor.GetTextProperty().ShadowOn()
        return self._register(actor, layer, add_to_renderer=True)

    def add_axes(self, layer="axes", origin=(0, 0, 0), length=100.0, *, transform_group=None):
        """Add world axes with camera-facing, perspective-scaled 3-D labels."""

        length = float(length)
        origin = np.asarray(origin, dtype=float)
        specs = (
            ("x [mm]", np.array((1.0, 0.0, 0.0)), "#a31621"),
            ("y [mm]", np.array((0.0, 1.0, 0.0)), "#16833a"),
            ("z [mm]", np.array((0.0, 0.0, 1.0)), "#1f4fb2"),
        )
        for _, direction, color in specs:
            helper = np.array((0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (0.0, 1.0, 0.0))
            axis_y = np.cross(helper, direction)
            axis_y /= np.linalg.norm(axis_y)
            axis_z = np.cross(direction, axis_y)
            basis = np.column_stack((direction, axis_y, axis_z))
            matrix = vtk.vtkMatrix4x4()
            for row in range(3):
                for column in range(3):
                    matrix.SetElement(row, column, length * basis[row, column])
            for row in range(3):
                matrix.SetElement(row, 3, origin[row])
            transform = vtk.vtkTransform()
            transform.SetMatrix(matrix)
            arrow = vtk.vtkArrowSource()
            arrow.SetShaftRadius(0.025)
            arrow.SetShaftResolution(24)
            arrow.SetTipRadius(0.09)
            arrow.SetTipLength(0.25)
            arrow.SetTipResolution(32)
            placed = vtk.vtkTransformPolyDataFilter()
            placed.SetInputConnection(arrow.GetOutputPort())
            placed.SetTransform(transform)
            self.add_surface(
                placed,
                layer,
                color=color,
                transform_group=transform_group,
                pickable=False,
            )

        for text, direction, color in specs:
            source = vtk.vtkVectorText()
            source.SetText(text)
            source.Update()
            xmin, xmax, ymin, ymax, _, _ = source.GetOutput().GetBounds()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            mapper.ScalarVisibilityOff()
            label = vtk.vtkFollower()
            label.SetMapper(mapper)
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            label_size = 0.10 * length / max(ymax - ymin, 1e-9)
            label.SetOrigin(cx, cy, 0.0)
            label.SetPosition(*(origin + direction * (1.25 * length) - (cx, cy, 0.0)))
            label.SetScale(label_size, label_size, label_size)
            label.SetCamera(self.renderer.GetActiveCamera())
            label.GetProperty().SetColor(*_rgb(color))
            label.GetProperty().LightingOff()
            self._register(
                label,
                layer,
                transform_group=transform_group,
                pickable=False,
                add_to_renderer=True,
            )

    def set_frame(self, index, *, render=True):
        if self.closed:
            return self.caption
        self.current_frame = max(0, min(int(index), self.frame_count - 1))
        self._update_frame(self.current_frame)
        self._apply_visibility()
        self._update_selection_outline()
        self.renderer.ResetCameraClippingRange()
        if render:
            self.render()
        return self.caption

    def set_layer_visible(self, key, visible, *, render=True):
        if self.closed:
            return
        if key == "instant_mode":
            self._instant_mode = bool(visible)
        else:
            self._layer_visible[key] = bool(visible)
        self._on_layer_changed(key, bool(visible))
        self._apply_visibility()
        self._update_selection_outline()
        self.renderer.ResetCameraClippingRange()
        if render:
            self.render()

    def _on_layer_changed(self, key, visible):
        pass

    def set_layer_opacity(self, key, opacity, *, render=True):
        self._layer_opacity[key] = max(0.0, min(float(opacity), 1.0))
        self._apply_visibility()
        if render:
            self.render()

    def _capture_camera(self):
        camera = self.renderer.GetActiveCamera()
        return {
            "position": camera.GetPosition(),
            "focal_point": camera.GetFocalPoint(),
            "view_up": camera.GetViewUp(),
            "view_angle": camera.GetViewAngle(),
            "parallel_projection": camera.GetParallelProjection(),
            "parallel_scale": camera.GetParallelScale(),
        }

    def _restore_camera(self, state):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*state["position"])
        camera.SetFocalPoint(*state["focal_point"])
        camera.SetViewUp(*state["view_up"])
        camera.SetViewAngle(float(state["view_angle"]))
        camera.SetParallelProjection(bool(state["parallel_projection"]))
        camera.SetParallelScale(float(state["parallel_scale"]))
        camera.OrthogonalizeViewUp()

    def _set_camera(self, camera_home):
        position, focal_point, view_up = camera_home
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*map(float, position))
        camera.SetFocalPoint(*map(float, focal_point))
        camera.SetViewUp(*map(float, view_up))
        camera.SetViewAngle(30.0)
        camera.ParallelProjectionOff()
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()

    def apply_view(self, px=0, py=0, pz=0, az=0, el=0, roll=0, fg_t=0, *, render=True):
        """Apply persisted camera offsets and robot transparency from home."""

        self._restore_camera(self._camera_base)
        camera = self.renderer.GetActiveCamera()
        pan = np.array([px, py, pz], dtype=float)
        camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint()) + pan))
        camera.SetPosition(*(np.asarray(camera.GetPosition()) + pan))
        if az:
            camera.Azimuth(float(az))
            camera.OrthogonalizeViewUp()
        if el:
            camera.Elevation(float(el))
            camera.OrthogonalizeViewUp()
        if roll:
            camera.Roll(float(roll))
            camera.OrthogonalizeViewUp()
        self.set_layer_opacity("robot", 1.0 - float(fg_t) / 100.0, render=False)
        self.renderer.ResetCameraClippingRange()
        self._camera_home = self._capture_camera()
        if render:
            self.render()

    def reset_camera(self, *, render=True):
        self._restore_camera(self._camera_home)
        self.renderer.ResetCameraClippingRange()
        if render:
            self.render()

    def render(self):
        if not self.closed:
            try:
                self.plotter.render()
            finally:
                restore_std_streams()

    def capture_rgb(self):
        """Capture RGB from this same PyVista render window for GIF output."""

        try:
            return np.asarray(self.plotter.screenshot(return_img=True))[..., :3].copy()
        finally:
            restore_std_streams()

    def close(self):
        if self.closed:
            return
        self.pause()
        self.closed = True
        self._transform_start_callback = None
        if self._interaction_style is not None:
            self._interaction_style.detach()
            self._interaction_style = None
        self._interaction_picker.InitializePickList()
        if getattr(self, "renderer", None) is not None:
            self.renderer.RemoveAllViewProps()
        if getattr(self, "plotter", None) is not None:
            self.plotter.close()


class RobotRig:
    """Persistent robot actors; frame changes update sources in place."""

    def __init__(self, scene, arm, layer="robot"):
        self.scene, self.arm = scene, arm
        self.links, self.joints = [], []
        base_line = vtk.vtkLineSource()
        base_line.SetPoint1(0.0, 0.0, -30.0)
        base_line.SetPoint2(0.0, 0.0, 0.0)
        base_tube = vtk.vtkTubeFilter()
        base_tube.SetInputConnection(base_line.GetOutputPort())
        base_tube.SetRadius(90.0)
        base_tube.SetNumberOfSides(40)
        base_tube.CappingOn()
        scene.add_surface(base_tube, layer, color="#6b7078")

        for radius in R_LINK:
            line = vtk.vtkLineSource()
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(radius)
            tube.SetNumberOfSides(28)
            tube.CappingOn()
            record, _ = scene.add_surface(tube, layer, color="#a7a9ac")
            self.links.append((line, record))
        for radius in R_JOINT:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(28)
            sphere.SetPhiResolution(20)
            record, _ = scene.add_surface(sphere, layer, color="#57a7c6")
            self.joints.append((sphere, record))
        self.gun = vtk.vtkConeSource()
        self.gun.SetHeight(55.0)
        self.gun.SetRadius(11.0)
        self.gun.SetResolution(32)
        self.gun_record, _ = scene.add_surface(self.gun, layer, color="#c0392b")

    def update(self, q):
        origins, _, rotation = self.arm._kin(q)
        origins = origins * 1e3
        tool_z = rotation[:, 2]
        ends = origins.copy()
        ends[6] = origins[6] - tool_z * 50.0
        for index, (line, record) in enumerate(self.links, start=1):
            segment = ends[index] - origins[index - 1]
            enabled = np.linalg.norm(segment) > 1e-3
            if enabled:
                line.SetPoint1(*origins[index - 1])
                line.SetPoint2(*ends[index])
            self.scene.set_record_enabled(record, enabled)
        for index, (_, record) in enumerate(self.joints, start=1):
            record["prop"].SetPosition(*origins[index])
        self.gun.SetCenter(*(origins[6] - tool_z * 27.5))
        self.gun.SetDirection(*tool_z)


class PolylineRig:
    """A growing polyline/tube backed by persistent VTK objects."""

    def __init__(self, scene, layer, color, radius, opacity=1.0):
        self.scene = scene
        self.poly = vtk.vtkPolyData()
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.poly.SetPoints(self.points)
        self.poly.SetLines(self.lines)
        self.tube = vtk.vtkTubeFilter()
        self.tube.SetInputData(self.poly)
        self.tube.SetRadius(float(radius))
        self.tube.SetNumberOfSides(20)
        self.tube.CappingOn()
        self.record, _ = scene.add_surface(
            self.tube, layer, color=color, opacity=opacity
        )

    def update(self, points):
        points = np.ascontiguousarray(points, dtype=np.float32)
        enabled = len(points) >= 2
        if enabled:
            self.points.SetData(numpy_to_vtk(points, deep=True))
            self.lines.Reset()
            self.lines.InsertNextCell(len(points))
            for index in range(len(points)):
                self.lines.InsertCellPoint(index)
            self.points.Modified()
            self.lines.Modified()
            self.poly.Modified()
        self.scene.set_record_enabled(self.record, enabled)


def seam_points(context):
    return np.array(
        [
            context.p0 * 1e3,
            (context.p0 + [context.travel_speed * context.t_track, 0, 0]) * 1e3,
        ],
        dtype=float,
    ) + [0, 0, 0.3]


def add_workpiece(scene, context, layer="workpiece", opacity=0.55):
    cube = vtk.vtkCubeSource()
    cube.SetBounds(
        context.p0[0] * 1e3 - 90,
        context.p0[0] * 1e3 + 110,
        -70,
        70,
        context.p0[2] * 1e3 - 20,
        context.p0[2] * 1e3,
    )
    return scene.add_surface(cube, layer, color="#d8d2c4", opacity=opacity)[0]


class RobotPVScene(WeldingPVSceneBase):
    """Static or animated robot pose with a growing trace and optional pool."""

    def __init__(
        self,
        context,
        frame_times,
        *,
        layers,
        current_frame=0,
        grow_trace=True,
        pool_model=None,
        pool_x_start=0.015,
        size=(950, 620),
    ):
        self.context = context
        self.frame_times = np.asarray(frame_times, dtype=float)
        self.grow_trace = bool(grow_trace)
        self.pool_model = pool_model
        self.pool_x_start = float(pool_x_start)
        self._pool_grid = None
        self._pool_buffers = {}
        self._pool_arrays = {}
        super().__init__(
            layers=layers,
            frame_count=len(self.frame_times),
            current_frame=current_frame,
            camera_home=((960, -780, 710), (250, 0, 145), (0, 0, 1)),
            size=size,
        )

    def _build_scene(self):
        self.robot = RobotRig(self, self.context.arm)
        add_workpiece(self, self.context)
        self.seam = PolylineRig(self, "seam", "#151515", radius=0.9)
        self.seam.update(seam_points(self.context))
        self.trace = PolylineRig(self, "executed_path", "#d81b1b", radius=0.6)
        self.add_axes(origin=(0, 0, 0), length=110.0)
        if self.pool_model is not None:
            self.attach_pool(self.pool_model, x_start=self.pool_x_start, render=False)

    def attach_pool(self, pool_model, *, x_start=None, render=True):
        required = ("peak", "T", "Tm", "x", "y", "z")
        if not all(hasattr(pool_model, key) for key in required):
            raise ValueError("pool_model must be a completed Goldak field")
        if x_start is None:
            x_start = self.pool_x_start
        values = {"peak": pool_model.peak, "inst": pool_model.T}
        if self._pool_grid is None:
            (
                self._pool_grid,
                _,
                _,
                _,
                self._pool_buffers,
                self._pool_arrays,
            ) = goldak_grid(
                pool_model,
                values,
                self.context.p0,
                x_start=x_start,
                dynamic=True,
            )
            for field, mode, color, opacity in (
                ("peak", "peak", "#9f2f24", 0.88),
                ("inst", "instant", "#e02f20", 0.92),
            ):
                self.add_surface(
                    contour_filter(self._pool_grid, field, float(pool_model.Tm)),
                    "thermal",
                    mode=mode,
                    color=color,
                    opacity=opacity,
                )
        else:
            for field, value in values.items():
                flat = np.asarray(value).ravel(order="F")
                if flat.size != self._pool_buffers[field].size:
                    raise ValueError("thermal grid changed; rebuild the scene")
                np.copyto(self._pool_buffers[field], flat, casting="unsafe")
                self._pool_arrays[field].Modified()
            self._pool_grid.GetPointData().Modified()
            self._pool_grid.Modified()
        self.pool_model = pool_model
        self._update_frame(self.current_frame)
        self._apply_visibility()
        if render:
            self.render()

    def _update_frame(self, index):
        t_now = float(self.frame_times[index])
        self.robot.update(self.context.q_at(t_now))
        if self.grow_trace:
            mask = self.context.t_trace <= t_now
            points = self.context.tip[mask] * 1e3 + [0, 0, 0.3]
        else:
            points = self.context.tip * 1e3 + [0, 0, 0.3]
        self.trace.update(points)
        suffix = "" if self._pool_grid is not None else " · run §4 for thermal field"
        self.caption = f"t = {t_now:.2f} s{suffix}"


class CompositePVScene(WeldingPVSceneBase):
    """Animated robot plus conductive/convection Goldak fields."""

    def __init__(
        self,
        context,
        conductive_model,
        convection_model,
        conductive_frames,
        convection_frames,
        *,
        x_start,
        layers,
        current_frame=0,
    ):
        self.context = context
        self.conductive_model = conductive_model
        self.convection_model = convection_model
        self.conductive_frames = list(conductive_frames)
        self.convection_frames = list(convection_frames)
        self.x_start = float(x_start)
        if len(self.conductive_frames) < 2:
            raise ValueError("composite playback requires at least two frames")
        self.frame_times = np.asarray([sample[0] for sample in self.conductive_frames])
        conv_times = np.asarray([sample[0] for sample in self.convection_frames])
        if not np.array_equal(self.frame_times, conv_times):
            raise ValueError("conductive/convection frame times differ")
        if any(
            not np.array_equal(getattr(conductive_model, axis), getattr(convection_model, axis))
            for axis in ("x", "y", "z")
        ):
            raise ValueError("animation grids must share coordinates")
        gx = (conductive_model.x - self.x_start + context.p0[0]) * 1e3
        gz = (context.p0[2] - conductive_model.z) * 1e3
        focal = np.array([gx.mean(), 0.0, gz.max()])
        self._conductive_melt = []
        super().__init__(
            layers=layers,
            frame_count=len(self.frame_times),
            current_frame=current_frame,
            camera_home=(tuple(focal + [210, -320, 230]), tuple(focal), (0, 0, 1)),
            size=(1000, 640),
            floor=(focal[0] - 260, focal[0] + 260, -210, 210, gz.min() - 12, 25),
        )

    def _build_scene(self):
        self.robot = RobotRig(self, self.context.arm)
        self.trace = PolylineRig(self, "executed_path", "#d81b1b", radius=0.6)
        self.add_axes(
            origin=(self.context.p0[0] * 1e3 - 80, -60, self.context.p0[2] * 1e3 - 20),
            length=35.0,
            transform_group="process",
        )
        _, cond_inst0, cond_peak0 = self.conductive_frames[0]
        _, conv_inst0, conv_peak0 = self.convection_frames[0]
        (
            self.grid,
            gx,
            _,
            gz,
            self.buffers,
            self.vtk_arrays,
        ) = goldak_grid(
            self.conductive_model,
            {
                "cond_peak": cond_peak0,
                "cond_inst": cond_inst0,
                "conv_peak": conv_peak0,
                "conv_inst": conv_inst0,
            },
            self.context.p0,
            x_start=self.x_start,
            dynamic=True,
        )
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(self.grid)
        self.add_surface(outline, "domain", color="#777777", opacity=0.8, smooth=False)
        clim = (float(self.conductive_model.T0), float(np.max(self.conductive_model.peak)))
        top = slice_filter(self.grid, (gx.mean(), 0.0, gz.max() - 1e-3))
        conv_visible = self._layer_visible.get("convection", False)
        for field, mode, label in (
            ("cond_peak", "peak", "peak"),
            ("cond_inst", "instant", "instant"),
        ):
            record, _ = self.add_surface(
                contour_filter(self.grid, field, float(self.conductive_model.Tm)),
                "thermal",
                mode=mode,
                color="#d7191c",
                opacity=0.35 if conv_visible else 0.90,
            )
            self._conductive_melt.append(record)
            self.add_surface(
                contour_filter(self.grid, field, 1073.0),
                "thermal",
                mode=mode,
                color="#ffd92f",
                opacity=0.25,
            )
            _, lut = self.add_surface(
                top,
                "thermal",
                mode=mode,
                scalars=field,
                cmap="jet",
                clim=clim,
                opacity=0.80,
            )
            self.add_scalar_bar(lut, f"T {label} [K]", "thermal", mode=mode)
        for field, mode in (("conv_peak", "peak"), ("conv_inst", "instant")):
            self.add_surface(
                contour_filter(self.grid, field, float(self.convection_model.Tm)),
                "convection",
                mode=mode,
                color="#2459b3",
                opacity=0.95,
            )

    def _update_frame(self, index):
        t_now, cond_inst, cond_peak = self.conductive_frames[index]
        _, conv_inst, conv_peak = self.convection_frames[index]
        for field, values in (
            ("cond_inst", cond_inst),
            ("cond_peak", cond_peak),
            ("conv_inst", conv_inst),
            ("conv_peak", conv_peak),
        ):
            np.copyto(self.buffers[field], np.asarray(values).ravel(order="F"), casting="unsafe")
            self.vtk_arrays[field].Modified()
        self.grid.GetPointData().Modified()
        self.grid.Modified()
        self.robot.update(self.context.q_at(t_now))
        mask = self.context.t_trace <= t_now
        self.trace.update(self.context.tip[mask] * 1e3 + [0, 0, 0.3])
        self._update_caption(t_now)

    def _update_caption(self, t_now):
        mode = "instantaneous T" if self._instant_mode else "peak history"
        suffix = " + 10A convection" if self._layer_visible.get("convection", False) else ""
        self.caption = f"t = {float(t_now):.2f} s · robot-executed pool · {mode}{suffix}"

    def _on_layer_changed(self, key, visible):
        if key == "convection":
            for record in self._conductive_melt:
                record["opacity"] = 0.35 if visible else 0.90
        if key in {"convection", "instant_mode"}:
            self._update_caption(self.frame_times[self.current_frame])


class SeamPVScene(WeldingPVSceneBase):
    """Animated seam-formation view used both live and for the GIF."""

    def __init__(self, context, model, snapshots, *, x_start, layers):
        self.context = context
        self.model = model
        self.snapshots = list(snapshots)
        self.x_start = float(x_start)
        self.frame_times = np.asarray([sample[0] for sample in snapshots])
        gx = (model.x - self.x_start + context.p0[0]) * 1e3
        gz = (context.p0[2] - model.z) * 1e3
        focal = np.array(
            [context.p0[0] * 1e3 + 0.5 * context.travel_speed * context.t_track * 1e3, 0, gz.max()]
        )
        self._grid_x, self._grid_z = gx, gz
        super().__init__(
            layers=layers,
            frame_count=len(self.snapshots),
            current_frame=0,
            camera_home=(tuple(focal + [95, -160, 120]), tuple(focal), (0, 0, 1)),
            size=(880, 540),
            floor=(focal[0] - 260, focal[0] + 260, -210, 210, gz.min() - 12, 25),
        )

    def _build_scene(self):
        self.robot = RobotRig(self, self.context.arm)
        add_workpiece(self, self.context, opacity=0.40)
        self.seam = PolylineRig(self, "seam", "#151515", radius=0.75)
        self.seam.update(seam_points(self.context))
        self.trace = PolylineRig(self, "executed_path", "#d81b1b", radius=0.8)
        self.time_text = self.add_text("t = 0.00 s", "annotation", position=(16, 505))
        self.add_text(
            "bright: molten pool | dark: solidified seam",
            "annotation",
            position=(495, 510),
            font_size=13,
        )
        _, inst0, peak0 = self.snapshots[0]
        (
            self.grid,
            _,
            _,
            _,
            self.buffers,
            self.vtk_arrays,
        ) = goldak_grid(
            self.model,
            {"inst": inst0, "peak": peak0},
            self.context.p0,
            x_start=self.x_start,
            dynamic=True,
        )
        self.add_surface(
            contour_filter(self.grid, "peak", float(self.model.Tm)),
            "history",
            color="#9e241a",
            opacity=0.90,
        )
        self.add_surface(
            contour_filter(self.grid, "inst", float(self.model.Tm)),
            "pool",
            color="#ff5916",
        )
        top = slice_filter(
            self.grid, (self._grid_x.mean(), 0.0, self._grid_z.max() - 1e-3)
        )
        self.add_surface(
            threshold_upper(top, "inst", 400.0),
            "halo",
            scalars="inst",
            cmap="inferno",
            clim=(float(self.model.T0), 2600.0),
            opacity=0.85,
        )
        self.add_axes(
            origin=(self.context.p0[0] * 1e3 - 75, -55, self.context.p0[2] * 1e3 - 18),
            length=28.0,
            transform_group="process",
        )

    def _update_frame(self, index):
        t_now, inst, peak = self.snapshots[index]
        np.copyto(self.buffers["inst"], np.asarray(inst).ravel(order="F"), casting="unsafe")
        np.copyto(self.buffers["peak"], np.asarray(peak).ravel(order="F"), casting="unsafe")
        for vtk_array in self.vtk_arrays.values():
            vtk_array.Modified()
        self.grid.GetPointData().Modified()
        self.grid.Modified()
        self.robot.update(self.context.q_at(t_now))
        mask = self.context.t_trace <= t_now
        self.trace.update(self.context.tip[mask] * 1e3 + [0, 0, 0.3])
        self.time_text["prop"].SetInput(f"t = {t_now:.2f} s")
        self.caption = f"t = {t_now:.2f} s · bright molten / dark history"


class PyVistaWidgetApp:
    """ipywidget controller and lifecycle owner for one live PyVista scene."""

    def __init__(
        self,
        scene,
        *,
        layers,
        store: WidgetStore,
        scene_name,
        title,
        height=680,
        frame_ms=125,
        lim=300.0,
        step=10.0,
    ):
        self.scene = scene
        self.layers = tuple(layers)
        self.store = store
        self.scene_name = scene_name
        self.closed = False
        self._callbacks = []
        try:
            self.viewer = scene.plotter.show(
                jupyter_backend="server",
                return_viewer=True,
                window_size=scene.render_window.GetSize(),
                jupyter_kwargs={"add_menu": True, "collapse_menu": True},
            )
        finally:
            restore_std_streams()
        self.viewer.layout = widgets.Layout(width="100%", height=f"{int(height)}px")
        self.title = widgets.HTML(
            f"<h4 style='margin:4px 0'>{title}</h4>"
            "<span style='color:#4b5563'>Camera: drag orbit · middle-drag pan · "
            "right-drag/wheel zoom. Objects: click select · Shift+left-drag "
            "translate · Ctrl+left-drag rotate.</span>"
        )
        self.caption = widgets.HTML()
        self._build_frame_controls(frame_ms)
        self._build_layer_controls()
        self._build_view_controls(lim, step)
        self.reset_objects = widgets.Button(
            description="重置对象位姿", icon="undo", layout=widgets.Layout(width="180px")
        )
        self.reset_objects.on_click(self._reset_objects)
        self.scene.set_transform_start_callback(self._pause_for_object_drag)
        rows = [self.title]
        if self.play is not None:
            rows.append(widgets.HBox((self.play, self.frame_slider, self.caption)))
        else:
            rows.append(self.caption)
        rows.extend(
            (
                widgets.HBox(tuple(self.layer_controls)),
                widgets.HBox(tuple(self.view_controls[:3])),
                widgets.HBox(tuple(self.view_controls[3:])),
                widgets.HBox((self.reset_objects,)),
                self.viewer,
            )
        )
        self.panel = widgets.VBox(tuple(rows), layout=widgets.Layout(width="100%"))
        self._apply_view()
        self._update_frame()

    def _build_frame_controls(self, frame_ms):
        if self.scene.frame_count <= 1:
            self.play = self.frame_slider = self.frame_link = None
            self.caption.value = self.scene.caption
            return
        initial = self.scene.current_frame
        self.play = widgets.Play(
            value=initial,
            min=0,
            max=self.scene.frame_count - 1,
            step=1,
            interval=int(frame_ms),
            description="Play",
        )
        self.frame_slider = widgets.IntSlider(
            value=initial,
            min=0,
            max=self.scene.frame_count - 1,
            step=1,
            description="frame",
            continuous_update=False,
            layout=widgets.Layout(width="520px"),
        )
        self.frame_link = widgets.jslink((self.play, "value"), (self.frame_slider, "value"))
        self.frame_slider.observe(self._update_frame, names="value")

    def _build_layer_controls(self):
        self.layer_controls = []
        for spec in self.layers:
            storage = {"instant_mode": "inst", "convection": "conv"}.get(
                spec.key, f"layer_{spec.key}"
            )
            control = self.store.tracked(
                self.scene_name,
                storage,
                widgets.Checkbox(
                    value=spec.visible,
                    description=spec.label,
                    indent=False,
                    layout=widgets.Layout(width="auto"),
                ),
            )

            def changed(change, key=spec.key):
                if self.closed:
                    return
                self.scene.set_layer_visible(key, bool(change["new"]))
                self.caption.value = self.scene.caption

            control.observe(changed, names="value")
            self._callbacks.append((control, changed))
            self.scene.set_layer_visible(spec.key, control.value, render=False)
            self.layer_controls.append(control)

    def _build_view_controls(self, lim, step):
        self.view_controls = view_widgets(
            self.store,
            self.scene_name,
            lim=lim,
            step=step,
            alpha_label="臂透明 [%]",
        )
        for control in self.view_controls:
            control.observe(self._apply_view, names="value")
            self._callbacks.append((control, self._apply_view))

    def _update_frame(self, _change=None):
        if self.closed:
            return
        index = self.scene.current_frame if self.frame_slider is None else self.frame_slider.value
        self.scene.set_frame(index)
        if hasattr(self.scene, "frame_times"):
            self.store.set(
                self.scene_name,
                "t",
                float(self.scene.frame_times[index]),
            )
        self.caption.value = self.scene.caption

    def _apply_view(self, _change=None):
        if self.closed:
            return
        values = dict(zip(VIEW_KEYS, (control.value for control in self.view_controls)))
        self.scene.apply_view(**values)

    def _reset_objects(self, _button=None):
        if self.closed:
            return
        self.pause()
        self.scene.reset_object_transforms()

    def _pause_for_object_drag(self):
        self.pause()

    def pause(self):
        if self.play is not None:
            self.play.playing = False

    def refresh(self):
        if not self.closed:
            self.scene.render()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.scene.set_transform_start_callback(None)
        if self.frame_slider is not None:
            self.frame_slider.unobserve(self._update_frame, names="value")
        for control, callback in self._callbacks:
            try:
                control.unobserve(callback, names="value")
            except ValueError:
                pass
        self.reset_objects.on_click(self._reset_objects, remove=True)
        self.panel.children = ()
        self.viewer.close()
        self.scene.close()


def initial_frame(store, scene_name, frame_times, default=0.0):
    """Find the frame nearest the scene's persisted time."""

    target = float(store.get(scene_name, "t", default))
    return int(np.argmin(np.abs(np.asarray(frame_times, dtype=float) - target)))


def close_app(namespace, name):
    """Close and clear a live app kept in a notebook globals dictionary."""

    app = namespace.get(name)
    if app is not None:
        app.close()
    namespace[name] = None


def close_output_tabs(namespace):
    """Detach the combined-output tab without closing reusable live apps."""

    for name in ("composite_pv", "seam_pv"):
        app = namespace.get(name)
        if app is not None and not app.closed:
            app.pause()
    tabs = namespace.get("weld_output_tabs")
    handler = namespace.get("_on_weld_output_tab")
    if tabs is not None:
        if handler is not None:
            try:
                tabs.unobserve(handler, names="selected_index")
            except ValueError:
                pass
        tabs.children = ()
        tabs.close()
    for name in ("weld_gif_image", "weld_gif_caption", "weld_gif_panel"):
        owned = namespace.get(name)
        if owned is not None:
            owned.close()
        namespace[name] = None
    namespace["_on_weld_output_tab"] = None
    namespace["weld_output_tabs"] = None


def build_output_tabs(namespace, gif_path, gif_summary, composite_app, seam_app):
    """Combine GIF, composite and seam live widgets into one tabbed output."""

    close_output_tabs(namespace)
    path = Path(gif_path)
    gif_image = widgets.Image(
        value=path.read_bytes(),
        format="gif",
        layout=widgets.Layout(width="880px", max_width="100%", height="auto"),
    )
    gif_caption = widgets.HTML(
        "<div style='text-align:center;color:#4b5563;padding-top:6px'>"
        f"{gif_summary}</div>"
    )
    gif_panel = widgets.VBox(
        (gif_image, gif_caption),
        layout=widgets.Layout(width="100%", align_items="center", padding="12px 0"),
    )
    tabs = widgets.Tab(
        children=(gif_panel, composite_app.panel, seam_app.panel),
        selected_index=0,
        layout=widgets.Layout(width="100%"),
    )
    for index, title in enumerate(
        ("Gif animation (default)", "Composite PyVista", "Seam PyVista")
    ):
        tabs.set_title(index, title)

    def on_tab(change):
        selected = change["new"]
        for index, app in enumerate((composite_app, seam_app), start=1):
            if index != selected:
                app.pause()
        if selected in (1, 2):
            (composite_app, seam_app)[selected - 1].refresh()

    tabs.observe(on_tab, names="selected_index")
    namespace.update(
        weld_gif_image=gif_image,
        weld_gif_caption=gif_caption,
        weld_gif_panel=gif_panel,
        weld_output_tabs=tabs,
        _on_weld_output_tab=on_tab,
    )
    return tabs


__all__ = [
    "CompositePVScene",
    "LayerSpec",
    "PyVistaWidgetApp",
    "RobotPVScene",
    "RobotWeaveContext",
    "SeamPVScene",
    "build_output_tabs",
    "close_app",
    "close_output_tabs",
    "initial_frame",
]
