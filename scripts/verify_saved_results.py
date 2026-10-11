"""Verify saved weights reconstruct ALL complex E/H samples, not just a plot."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

from pinn_maxwell import MaxwellTEPINN
from pinn_maxwell.fields import propagate


def main(names):
    for name in names:
        saved = Path("results")/name
        metrics = json.loads((saved/"metrics.json").read_text(encoding="utf-8"))
        data = np.load(Path("runs")/name/"fields.npz")
        model = MaxwellTEPINN.from_checkpoint(saved/"model.pt")
        with torch.no_grad():
            e, q = model(torch.as_tensor(data["x_um"], dtype=torch.float64)).numpy().T
            n = float(model.neff)
        electric, magnetic = propagate(e, q, n, model.geometry.wavelength_um,
                                        data["z_um"], model.geometry.electric_amplitude_v_m*
                                        metrics["launch_modal_coefficient"])
        np.testing.assert_allclose(electric, data["E_v_m"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(magnetic, data["H_a_m"], rtol=1e-12, atol=1e-12)
        print(f"{name}: checkpoint reproduces all complex E/H samples")


if __name__ == "__main__":
    main(sys.argv[1:] or ["slab", "coated_slab", "asymmetric_slab"])
