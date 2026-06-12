"""welding_dynamics — 工业焊接 (GMAW/MIG) 动力学模型包"""
from .gmaw import GMAWDynamics
from .thermal import RosenthalThermal, GoldakFDM
from .droplet import DropletDynamics
from .short_circuit import ShortCircuitGMAW

__version__ = "1.0.0"
__all__ = ["GMAWDynamics", "RosenthalThermal", "GoldakFDM",
           "DropletDynamics", "ShortCircuitGMAW"]
