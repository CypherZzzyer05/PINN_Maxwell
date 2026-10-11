"""Maxwell PINN: exp(-i omega t), propagation along +z, TE E along y."""

from .geometry import Waveguide
from .model import MaxwellTEPINN

__all__ = ["Waveguide", "MaxwellTEPINN"]
