from pathlib import Path
import argparse
import json

from pinn_maxwell.fiber import FiberGeometry
from .lumerical import FiberFDTD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-nm", type=float, default=80.)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(f"runs/fdtd/{args.mesh_nm:g}nm")
    root.mkdir(parents=True, exist_ok=True)
    g = FiberGeometry.from_json("configs/fiber_circular.json")
    with FiberFDTD(g, mesh_nm=args.mesh_nm) as simulation:
        print("FDTD version", simulation.version, flush=True)
        simulation.build(root/"fiber.fsp")
        info = simulation.prepare_sources()
        metadata = {"version": simulation.version, "mesh_nm": args.mesh_nm,
                    "dz_nm": 1.5*args.mesh_nm, "outer_mesh_nm": 120.,
                    "transverse_span_um": simulation.span, "axial_padding_um": 3.,
                    "time_limit_fs": simulation.time_fs, "shutoff_threshold": simulation.shutoff,
                    "threads": simulation.threads,
                    "geometry": __import__("dataclasses").asdict(g), "source": info}
        (root/"setup.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.run:
            status, elapsed = simulation.solve(root)
            metadata["status"] = np_to_json(status)
            metadata["elapsed_s"] = elapsed
            (root/"completed.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def np_to_json(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
