"""Train dielectric fiber PINN and export both sides of each cylindrical interface."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from .fields import integrate, poynting
from .fiber import FiberGeometry, FiberPINN
from .fiber_fields import fiber_fields, maxwell_diagnostics, stokes
from .fiber_reference import BesselReference
from .fiber_train import train_fiber


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def interface_data(solution, number, phi, z):
    g = solution.geometry
    radius = g.interface_radii_um[number]
    theta, zz = np.meshgrid(phi, z)
    x, y = radius*np.cos(theta), radius*np.sin(theta)
    ei, hi = fiber_fields(solution, x, y, zz, layer=number)
    eo, ho = fiber_fields(solution, x, y, zz, layer=number+1)
    eic, hic = fiber_fields(solution, x, y, zz, layer=number, cylindrical=True)
    eoc, hoc = fiber_fields(solution, x, y, zz, layer=number+1, cylindrical=True)
    return {"radius_um": radius, "phi_rad": theta, "z_um": zz, "x_um": x, "y_um": y,
            "E_inner_v_m": ei, "E_outer_v_m": eo, "H_inner_a_m": hi, "H_outer_a_m": ho,
            "E_inner_cyl_v_m": eic, "E_outer_cyl_v_m": eoc,
            "H_inner_cyl_a_m": hic, "H_outer_cyl_a_m": hoc}


def interface_scores(data, reference, geometry, number):
    ei, eo = data["E_inner_cyl_v_m"], data["E_outer_cyl_v_m"]
    hi, ho = data["H_inner_cyl_a_m"], data["H_outer_cyl_a_m"]
    eps_i, eps_o = geometry.indices[number]**2, geometry.indices[number+1]**2
    return {"radius_um": data["radius_um"], "inner_index": geometry.indices[number],
            "outer_index": geometry.indices[number+1],
            "e_tangential_jump_relative": relative(ei[..., 1:], eo[..., 1:]),
            "h_tangential_jump_relative": relative(hi[..., 1:], ho[..., 1:]),
            "d_normal_jump_relative": relative(eps_i*ei[..., 0], eps_o*eo[..., 0]),
            "b_normal_jump_relative": relative(hi[..., 0], ho[..., 0]),
            "e_inner_reference_error": relative(data["E_inner_v_m"], reference["E_inner_v_m"]),
            "e_outer_reference_error": relative(data["E_outer_v_m"], reference["E_outer_v_m"]),
            "h_inner_reference_error": relative(data["H_inner_a_m"], reference["H_inner_a_m"]),
            "h_outer_reference_error": relative(data["H_outer_a_m"], reference["H_outer_a_m"]),
            "e_normal_outer_inner_magnitude_ratio": float(np.linalg.norm(eo[..., 0])/np.linalg.norm(ei[..., 0])),
            "e_normal_expected_ratio": eps_i/eps_o,
            "inner_e_magnitude_v_m": float(np.linalg.norm(data["E_inner_v_m"][0, 0])),
            "outer_e_magnitude_v_m": float(np.linalg.norm(data["E_outer_v_m"][0, 0])),
            "inner_h_magnitude_a_m": float(np.linalg.norm(data["H_inner_a_m"][0, 0])),
            "outer_h_magnitude_a_m": float(np.linalg.norm(data["H_outer_a_m"][0, 0]))}


def interface_csv(data, path, indices):
    # Full complex traces at input and output planes, all angles, BOTH sides.
    selection = np.array([0, -1])
    cols = [data[key][selection].ravel() for key in ("x_um", "y_um", "z_um", "phi_rad")]
    headers = ["x_um", "y_um", "z_um", "phi_rad"]
    for side in ("inner", "outer"):
        for field, unit in (("E", "v_m"), ("H", "a_m")):
            vector = data[f"{field}_{side}_{unit}"][selection].reshape(-1, 3)
            for i, component in enumerate("xyz"):
                for name, values in (("real", vector[:, i].real), ("imag", vector[:, i].imag)):
                    headers.append(f"{field}{component}_{side}_{name}_{unit}")
                    cols.append(values)
            headers.append(f"{field}_{side}_magnitude_{unit}")
            cols.append(np.linalg.norm(vector, axis=-1))
        for field, unit in (("E", "v_m"), ("H", "a_m")):
            vector = data[f"{field}_{side}_cyl_{unit}"][selection].reshape(-1, 3)
            for i, component in enumerate(("r", "phi", "z")):
                for name, values in (("real", vector[:, i].real), ("imag", vector[:, i].imag)):
                    headers.append(f"{field}{component}_{side}_{name}_{unit}")
                    cols.append(values)
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(headers), comments="")


def interface_plot(data, path, indices):
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    extent = [0, 360, float(data["z_um"].min()), float(data["z_um"].max())]
    for row, (field, unit, pretty) in enumerate((("E", "v_m", "V/m"), ("H", "a_m", "A/m"))):
        for column, side in enumerate(("inner", "outer")):
            norm = np.linalg.norm(data[f"{field}_{side}_{unit}"], axis=-1)
            # Same color scale for BOTH sides: normal E may legitimately jump.
            other = np.linalg.norm(data[f"{field}_{'outer' if side == 'inner' else 'inner'}_{unit}"], axis=-1)
            vmax = max(float(norm.max()), float(other.max()))
            im = axes[row, column].imshow(norm, extent=extent, origin="lower", aspect="auto", vmin=0, vmax=vmax)
            figure.colorbar(im, ax=axes[row, column], label=pretty)
            axes[row, column].set(title=f"|{field}| {side} (n={indices[column]})", xlabel="Angle (deg)", ylabel="z (um)")
        for column, component in ((2, 0), (3, 2)):
            values = data[f"{field}_inner_cyl_{unit}"][..., component].real
            vmax = float(np.max(np.abs(values)))
            im = axes[row, column].imshow(values, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            figure.colorbar(im, ax=axes[row, column], label=pretty)
            axes[row, column].set(title=f"Re {field}{'r' if component == 0 else 'z'} (inner)", xlabel="Angle (deg)", ylabel="z (um)")
    figure.suptitle(f"Cylindrical material interface r={data['radius_um']} um: one-sided field traces")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def overview_plot(g, xy, electric, magnetic, outside, path):
    e, h = electric.copy(), magnetic.copy()
    e[outside], h[outside] = np.nan, np.nan
    pol = stokes(e)
    fields = [(np.linalg.norm(e, axis=-1), "|E|", "V/m", "viridis"),
              (np.linalg.norm(h, axis=-1), "|H|", "A/m", "viridis"),
              (poynting(e, h)[..., 2], "Sz", "W/m^2", "viridis"),
              (pol[..., 3]/pol[..., 0], "Transverse circularity S3/S0", "", "RdBu_r"),
              (e[..., 0].real, "Re Ex", "V/m", "RdBu_r"),
              (e[..., 2].real, "Re Ez", "V/m", "RdBu_r")]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (values, title, label, cmap) in zip(axes.flat, fields):
        options = {"vmin": -1, "vmax": 1} if title.startswith("Transverse") else {}
        im = axis.imshow(values, extent=[xy[0], xy[-1], xy[0], xy[-1]], origin="lower", cmap=cmap, **options)
        figure.colorbar(im, ax=axis, label=label)
        for radius in g.interface_radii_um:
            axis.add_patch(plt.Circle((0, 0), radius, fill=False, color="white", linestyle="--", linewidth=.8))
        axis.set(title=title, xlabel="x (um)", ylabel="y (um)")
    figure.suptitle("Circularly excited HE11 dielectric fiber: z=0; dashed circles are material interfaces")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args):
    if min(args.adam_steps, args.lbfgs_steps) < 0 or args.samples < 8 or args.grid < 21:
        raise ValueError("Nonnegative steps, >=8 samples and >=21 grid points required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    g = FiberGeometry.from_json(args.config)
    model = FiberPINN.from_checkpoint(args.checkpoint, args.device) if args.checkpoint else FiberPINN(g, args.width).to(args.device)
    args.width = model.layers[0].net[0].out_features
    if model.geometry != g:
        raise ValueError("Checkpoint geometry must match the requested configuration")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, loss_terms = train_fiber(model, args.adam_steps, args.lbfgs_steps, args.samples, args.seed)
    # Independent eigensolver used only after neural training has finished.
    reference = BesselReference(g)
    xy = np.linspace(-g.radius_um, g.radius_um, args.grid)
    xx, yy = np.meshgrid(xy, xy)
    outside = np.hypot(xx, yy) > g.radius_um
    e, h = fiber_fields(model, xx, yy)
    er, hr = fiber_fields(reference, xx, yy)
    e_out, h_out = fiber_fields(model, xx, yy, g.length_um)
    er_out, hr_out = fiber_fields(reference, xx, yy, g.length_um)
    phi = np.linspace(0, 2*np.pi, 129)
    z = np.linspace(0, g.length_um, 81)
    traces, scores = [], []
    for i, radius in enumerate(g.interface_radii_um):
        data = interface_data(model, i, phi, z)
        reference_data = interface_data(reference, i, phi, z)
        score = interface_scores(data, reference_data, g, i)
        scores.append(score)
        traces.append(data)
        np.savez_compressed(out/f"interface_{i+1}.npz", **data)
        interface_csv(data, out/f"interface_{i+1}.csv", g.indices[i:i+2])
        interface_plot(data, out/f"interface_{i+1}.png", g.indices[i:i+2])
    rng = np.random.default_rng(391)
    points = []
    for left, right in zip((0., *g.interface_radii_um), (*g.interface_radii_um, g.radius_um)):
        r = rng.uniform(left+0.05*(right-left), right-0.05*(right-left), 40)
        theta = rng.uniform(0, 2*np.pi, 40)
        points.extend(np.column_stack((r*np.cos(theta), r*np.sin(theta), rng.uniform(0, g.length_um, 40))))
    maxwell = maxwell_diagnostics(model, points)
    maxwell_reference = maxwell_diagnostics(reference, points)
    ei_axis, hi_axis = fiber_fields(model, 0., 0.)
    pol_axis = stokes(ei_axis)
    base_scores = {"neff_relative_error": abs(float(model.neff.detach())-reference.neff)/reference.neff,
                   "e_relative_l2": relative(e[~outside], er[~outside]),
                   "h_relative_l2": relative(h[~outside], hr[~outside]),
                   "output_e_relative_l2": relative(e_out[~outside], er_out[~outside]),
                   "output_h_relative_l2": relative(h_out[~outside], hr_out[~outside])}
    thresholds = {"neff_relative_error": 1e-4, "e_relative_l2": .02, "h_relative_l2": .02,
                  "output_e_relative_l2": .02, "output_h_relative_l2": .02}
    passed = all(base_scores[k] < v for k, v in thresholds.items())
    passed &= loss_terms["pde"] < 1e-5 and max(loss_terms["interface"], loss_terms["boundary"]) < 1e-7
    passed &= max(maxwell.values()) < 1e-2
    for score in scores:
        passed &= max(score[k] for k in ("e_tangential_jump_relative", "h_tangential_jump_relative",
                                       "d_normal_jump_relative", "b_normal_jump_relative")) < .002
        passed &= max(score[k] for k in ("e_inner_reference_error", "e_outer_reference_error",
                                       "h_inner_reference_error", "h_outer_reference_error")) < .02
    r = np.linspace(0, g.radius_um, 2001)
    ep, hp = fiber_fields(model, r, np.zeros_like(r))
    sz = poynting(ep, hp)[..., 2]
    power = 2*np.pi*integrate(sz*r, r)*1e-12
    metrics = {"validated": bool(passed), "neff_pinn": float(model.neff.detach()), "neff_reference": reference.neff,
               "beta_rad_per_um": g.k0*float(model.neff.detach()), "scores": base_scores,
               "thresholds": thresholds, "loss_terms": loss_terms, "interfaces": scores,
               "physics_thresholds": {"pde_loss": 1e-5, "interface_boundary_loss": 1e-7,
                                      "maxwell_relative": .01, "interface_jump_relative": .002,
                                      "interface_reference_relative": .02},
               "maxwell_cartesian": maxwell, "reference_maxwell_cartesian": maxwell_reference,
               "axis_circularity_s3_s0": float(pol_axis[3]/pol_axis[0]),
               "power_within_radius_w": float(power), "geometry": asdict(g), "training": vars(args),
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "scipy": scipy.__version__, "numpy": np.__version__, "device": args.device},
               "scope": "Full-vector core-guided m=+/-1 hybrid mode; radial isotropic layers, invariant along z; infinite outer cladding"}
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "geometry": asdict(g), "width": args.width}, out/"model.pt")
    for array in (e, h, e_out, h_out):
        array[outside] = np.nan
    np.savez_compressed(out/"cross_section.npz", x_um=xy, y_um=xy, inside=~outside,
                        E_input_v_m=e, H_input_a_m=h, E_output_v_m=e_out, H_output_a_m=h_out)
    overview_plot(g, xy, e, h, outside, out/"cross_section.png")
    print(json.dumps({"validated": bool(passed), "neff": metrics["neff_pinn"],
                      "scores": base_scores, "loss_terms": loss_terms, "maxwell": maxwell,
                      "interfaces": scores, "output": str(out.resolve())}, indent=2))
    if args.publish:
        if not passed:
            raise ValueError("Unvalidated fiber results cannot be published")
        target = Path(args.publish)
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("metrics.json", "history.json", "model.pt", "cross_section.png",
                         *[f"interface_{i+1}.{ext}" for i in range(len(traces)) for ext in ("csv", "png")]):
            shutil.copy2(out/filename, target/filename)
    return 0 if passed or args.allow_unvalidated else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fiber_circular.json")
    parser.add_argument("--output", default="runs/fiber_circular")
    parser.add_argument("--adam-steps", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid", type=int, default=121)
    parser.add_argument("--checkpoint")
    parser.add_argument("--publish", help="Copy only validated small artifacts to this folder")
    parser.add_argument("--allow-unvalidated", action="store_true")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
