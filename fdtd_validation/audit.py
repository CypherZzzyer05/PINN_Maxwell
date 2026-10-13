"""Audit saved completed FSP projects and publish only sanitized solver evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import hash_file, sanitized_solver_evidence
from .lumerical import load_lumapi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", type=float, nargs="+", default=[80, 60, 40, 30, 25])
    args = parser.parse_args()
    target = Path("results/fdtd_validation")
    target.mkdir(parents=True, exist_ok=True)
    api = load_lumapi()
    f = api.FDTD(hide=True)
    rows = []
    try:
        for mesh in args.meshes:
            root = Path("runs/fdtd")/f"{mesh:g}nm"
            f.load(str((root/"fiber.fsp").resolve()))
            status = float(f.getresult("FDTD", "status"))
            freq = np.asarray(f.getdata("plane_00", "f")).ravel()
            if status != 2 or len(freq) != 1:
                raise ValueError("Saved FSP is not a completed single-frequency run")
            wavelength = float(299792458/freq[0]*1e6)
            if not np.isclose(wavelength, 1.55, rtol=1e-10):
                raise ValueError(f"Actual wavelength mismatch: {wavelength}")
            grid = {key+"_um": np.asarray(f.getresult("FDTD", key)).ravel()*1e6 for key in "xyz"}
            np.savez_compressed(root/"simulation_grid.npz", **grid)
            row = {"mesh_nm": mesh, "solver_version": str(f.version()), "status": status,
                   "actual_wavelength_um": wavelength,
                   "grid_shape": [len(grid[k+"_um"]) for k in "xyz"],
                   "source_to_lower_pml_um": float(-1-grid["z_um"][12]),
                   "output_to_upper_pml_um": float(grid["z_um"][-13]-10),
                   "source_window_to_side_pml_um": float(grid["x_um"][-13]-5),
                   "fsp_sha256": hash_file(root/"fiber.fsp")}
            for field in ("initialization time", "simulation run time", "total solver speed", "total threads"):
                row[field.replace(" ", "_")] = float(f.getresult("FDTD", field))
            if min(row["source_to_lower_pml_um"], row["output_to_upper_pml_um"], row["source_window_to_side_pml_um"]) <= 0:
                raise ValueError("Source or requested output enters the PML")
            rows.append(row)
            (target/f"solver_evidence_{mesh:g}nm.txt").write_text(sanitized_solver_evidence(root/"fiber_p0.log"), encoding="utf-8")
            print("Audited", mesh, "nm", row["grid_shape"], wavelength, flush=True)
    finally:
        f.close()
    (target/"solver_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
