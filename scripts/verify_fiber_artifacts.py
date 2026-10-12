"""Verify exported interfaces and Cartesian cross sections using reloaded weights."""

from pathlib import Path

import numpy as np

from pinn_maxwell import FiberPINN
from pinn_maxwell.fiber_demo import interface_data
from pinn_maxwell.fiber_fields import fiber_fields


def main():
    model = FiberPINN.from_checkpoint("results/fiber_circular/model.pt")
    root = Path("runs/fiber_circular")
    with np.load(root/"cross_section.npz") as data:
        x, y = np.meshgrid(data["x_um"], data["y_um"])
        for position, z in (("input", 0.), ("output", model.geometry.length_um)):
            e, h = fiber_fields(model, x, y, z)
            for field, prediction, unit in (("E", e, "v_m"), ("H", h, "a_m")):
                np.testing.assert_allclose(prediction[data["inside"]], data[f"{field}_{position}_{unit}"][data["inside"]], rtol=1e-12, atol=1e-12)
    for i in range(len(model.geometry.interface_radii_um)):
        with np.load(root/f"interface_{i+1}.npz") as data:
            computed = interface_data(model, i, data["phi_rad"][0], data["z_um"][:, 0])
            for key in ("E_inner_v_m", "E_outer_v_m", "H_inner_a_m", "H_outer_a_m",
                        "E_inner_cyl_v_m", "E_outer_cyl_v_m", "H_inner_cyl_a_m", "H_outer_cyl_a_m"):
                np.testing.assert_allclose(computed[key], data[key], rtol=1e-12, atol=1e-12)
        print(f"Interface {i+1}: checkpoint reproduces all complex field traces")
    print("Input/output Cartesian cross sections also reproduce exactly")


if __name__ == "__main__":
    main()
