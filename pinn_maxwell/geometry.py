from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Waveguide:
    """Lossless isotropic layers in x; outer layers extend to infinity.

    All lengths are micrometres. First/last boundaries specify export range,
    not physical walls. y and z are translationally invariant.
    """

    wavelength_um: float
    boundaries_um: tuple[float, ...]
    indices: tuple[float, ...]
    anchor_um: float = 0.0
    length_um: float = 10.0
    electric_amplitude_v_m: float = 1.0
    launch: str = "mode"
    gaussian_waist_um: float = 0.5

    def __post_init__(self):
        values = (*self.boundaries_um, *self.indices, self.wavelength_um,
                  self.anchor_um, self.length_um, self.electric_amplitude_v_m,
                  self.gaussian_waist_um)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("All physical parameters must be finite")
        if len(self.indices) < 3 or len(self.boundaries_um) != len(self.indices) + 1:
            raise ValueError("Need at least two claddings and one interior layer")
        if any(b <= a for a, b in zip(self.boundaries_um, self.boundaries_um[1:])):
            raise ValueError("Boundaries must strictly increase")
        if min(self.indices) <= 0 or self.wavelength_um <= 0:
            raise ValueError("Wavelength and real refractive indices must be positive")
        if self.n_upper <= self.n_lower:
            raise ValueError("Interior index must exceed both cladding indices")
        if not self.boundaries_um[1] < self.anchor_um < self.boundaries_um[-2]:
            raise ValueError("Normalization anchor must lie in the interior")
        if self.length_um <= 0 or self.gaussian_waist_um <= 0:
            raise ValueError("Length and Gaussian waist must be positive")
        if self.electric_amplitude_v_m <= 0:
            raise ValueError("Electric amplitude must be positive for relative-error validation")
        if self.launch not in ("mode", "gaussian"):
            raise ValueError("launch must be mode or gaussian")

    @property
    def k0(self):
        return 2 * math.pi / self.wavelength_um

    @property
    def n_lower(self):
        return max(self.indices[0], self.indices[-1])

    @property
    def n_upper(self):
        return max(self.indices[1:-1])

    def index_at(self, x):
        # Right-hand value at an interface; Ey and Hz are continuous.
        layer = np.searchsorted(self.boundaries_um[1:-1], x, side="right")
        return np.asarray(self.indices)[layer]

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["boundaries_um"] = tuple(data["boundaries_um"])
        data["indices"] = tuple(data["indices"])
        return cls(**data)
