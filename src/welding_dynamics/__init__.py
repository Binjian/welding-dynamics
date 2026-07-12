"""welding_dynamics — 工业焊接 (GMAW/MIG) 动力学模型包"""
from .gmaw import GMAWDynamics
from .thermal import RosenthalThermal, GoldakFDM
from .weave import HarmonicWeave, WaypointWeave, RobotExecutedWeave
from .droplet import DropletDynamics
from .short_circuit import ShortCircuitGMAW
from .marangoni import (EffectiveMarangoniCorrection, SurfaceMarangoniFlow2D,
                        IncompressibleMarangoniFlow2D)
from .variational import ForcedVerlet, MidpointDEL
from .droplet_vi import DropletOscillatorVI
from .robot_vi import TwoLinkArm, SixDofArm
from .robot_mujoco import MujocoArm, build_mjcf
from .shortcircuit_vi import ContactCycleVI
from .thermal3d import OpenFOAMExporter, export_openfoam, render, ensure_display

__version__ = "1.0.0"
__all__ = ["GMAWDynamics", "RosenthalThermal", "GoldakFDM",
           "HarmonicWeave", "WaypointWeave", "RobotExecutedWeave",
           "DropletDynamics", "ShortCircuitGMAW",
           "EffectiveMarangoniCorrection", "SurfaceMarangoniFlow2D",
           "IncompressibleMarangoniFlow2D",
           "ForcedVerlet", "MidpointDEL", "DropletOscillatorVI",
           "TwoLinkArm", "SixDofArm", "MujocoArm", "build_mjcf", "ContactCycleVI",
           "OpenFOAMExporter", "export_openfoam", "render", "ensure_display"]
