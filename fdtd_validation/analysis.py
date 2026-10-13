"""Compare genuine FDTD fields against PINN; separate shape and phase errors."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_maxwell.fiber import FiberPINN
from pinn_maxwell.fiber_fields import fiber_fields, stokes
from pinn_maxwell.fiber_reference import BesselReference
from pinn_maxwell.fields import poynting
from .lumerical import interpolate_vector


def trapezoid_weights(x):
    x = np.asarray(x)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Strictly increasing grid with at least two points required")
    weights = np.empty(len(x))
    weights[0], weights[-1] = (x[1]-x[0])/2, (x[-1]-x[-2])/2
    weights[1:-1] = (x[2:]-x[:-2])/2
    return weights


def weighted_relative(prediction, reference, weights, mask):
    w = weights*mask
    diff = np.sum(np.abs(prediction-reference)**2, axis=-1)
    energy = np.sum(np.abs(reference)**2, axis=-1)
    return float(np.sqrt(np.sum(w*diff)/np.sum(w*energy)))


def magnitude_bias(prediction, reference, weights, mask):
    w = weights*mask
    a, b = np.linalg.norm(prediction, axis=-1), np.linalg.norm(reference, axis=-1)
    return float(np.sum(w*(a-b))/np.sum(w))


def vector_rmse(prediction, reference, weights, mask):
    w = weights*mask
    return float(np.sqrt(np.sum(w*np.sum(np.abs(prediction-reference)**2, axis=-1))/np.sum(w)))


def circumferential_components(field, angles):
    result = field.copy()
    result[..., 0] = field[..., 0]*np.cos(angles)+field[..., 1]*np.sin(angles)
    result[..., 1] = -field[..., 0]*np.sin(angles)+field[..., 1]*np.cos(angles)
    return result


def one_sided_trace(x, y, field, radius, angles, mesh_um, side, offset_cells=(2., 3., 4.)):
    offsets = side*mesh_um*np.asarray(offset_cells)
    radii = radius+offsets
    if np.any(radii <= 0):
        raise ValueError("One-sided samples cross the axis")
    sampled = []
    for r in radii:
        points = np.column_stack((r*np.cos(angles), r*np.sin(angles)))
        sampled.append(interpolate_vector(x, y, field, points))
    sampled = np.asarray(sampled)
    # Quadratic interpolation at zero offset, all samples in the same medium.
    weights = np.array([np.prod([-offsets[k]/(offsets[j]-offsets[k]) for k in range(3) if k != j]) for j in range(3)])
    return np.tensordot(weights, sampled, axes=(0, 0))


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized_solver_evidence(path):
    # Never publish raw engine command lines, license context, or host details.
    prefixes = ("Ansys Lumerical", "Running 3D Simulation", "Simulation size in gridpoints:",
                "Starting ", "Early termination", "Completed ", "Simulation completed successfully",
                "Overall wall time", "    time to ", "    FDTD solver speed", "    total FDTD solver speed")
    result = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if any(word in line.lower() for word in ("license", "-shared-", "host", "running fsp file")):
            continue
        if line.startswith(prefixes):
            result.append(line)
    return "\n".join(result)+"\n"


def load_case(directory, model, exclusion_um=.16):
    metadata = json.loads((directory/"completed.json").read_text(encoding="utf-8"))
    if float(metadata["status"]) != 2:
        raise ValueError(f"FDTD did not reach auto-shutoff: {directory}")
    if metadata.get("axial_padding_um", 0) < 3 or metadata.get("transverse_span_um", 0) < 14:
        raise ValueError("Source/monitor PML clearance does not match the accepted study")
    g = model.geometry
    expected_geometry = __import__("dataclasses").asdict(g)
    if json.dumps(metadata["geometry"], sort_keys=True) != json.dumps(expected_geometry, sort_keys=True):
        raise ValueError("FDTD and PINN geometries do not match")
    source = metadata["source"]
    source_matrix = np.array(source["axis_matrix_real"])+1j*np.array(source["axis_matrix_imag"])
    if np.linalg.cond(source_matrix) > 100:
        raise ValueError("FDTD circular-source basis is ill-conditioned")
    axis = np.load(directory/"axis.npz")
    z_axis, e_axis = axis["z_um"], axis["E_v_m"]
    circular = (e_axis[:, 0]-1j*g.handedness*e_axis[:, 1])/np.sqrt(2)
    fit_mask = (z_axis >= .25) & (z_axis <= g.length_um-.25)
    beta, intercept = np.polyfit(z_axis[fit_mask], np.unwrap(np.angle(circular[fit_mask])), 1)
    fitted = beta*z_axis[fit_mask]+intercept
    phase_residual = np.unwrap(np.angle(circular[fit_mask]))-fitted
    pol = stokes(e_axis)
    circularity = pol[:, 3]/pol[:, 0]
    if np.min(g.handedness*circularity[fit_mask]) < .999:
        raise ValueError(f"FDTD excitation is not circular: {directory}, min S3={circularity[fit_mask].min()}")
    data0 = np.load(directory/"plane_00.npz")
    origin = interpolate_vector(data0["x_um"], data0["y_um"], data0["E_v_m"], [[0., 0.]])[0]
    calibration = g.electric_amplitude_v_m/((origin[0]-1j*g.handedness*origin[1])/np.sqrt(2))
    planes, powers, phase_values = [], [], []
    for i in range(11):
        raw = np.load(directory/f"plane_{i:02d}.npz")
        x, y, z = raw["x_um"], raw["y_um"], float(raw["z_um"])
        e, h = calibration*raw["E_v_m"], calibration*raw["H_a_m"]
        xx, yy = np.meshgrid(x, y, indexing="ij")
        radius = np.hypot(xx, yy)
        mask = radius <= g.radius_um
        bulk = mask.copy()
        for interface in g.interface_radii_um:
            bulk &= abs(radius-interface) >= exclusion_um
        weights = trapezoid_weights(x)[:, None]*trapezoid_weights(y)[None, :]
        e_pinn, h_pinn = fiber_fields(model, xx, yy, z)
        e_origin = interpolate_vector(x, y, e, [[0., 0.]])[0]
        angle = float(np.angle((e_origin[0]-1j*g.handedness*e_origin[1])/np.sqrt(2)))
        phase_values.append(angle)
        e_shape, h_shape = e*np.exp(-1j*angle), h*np.exp(-1j*angle)
        e0_pinn, h0_pinn = fiber_fields(model, xx, yy)
        power = float(np.sum(weights*mask*poynting(e, h)[..., 2])*1e-12)
        powers.append(power)
        planes.append({"z_um": z, "x_um": x, "y_um": y, "E": e, "H": h,
                       "E_shape": e_shape, "H_shape": h_shape,
                       "scores": {"e_complex_full": weighted_relative(e_pinn, e, weights, mask),
                                  "h_complex_full": weighted_relative(h_pinn, h, weights, mask),
                                  "e_complex_bulk": weighted_relative(e_pinn, e, weights, bulk),
                                  "h_complex_bulk": weighted_relative(h_pinn, h, weights, bulk),
                                  "e_shape_full": weighted_relative(e0_pinn, e_shape, weights, mask),
                                  "h_shape_full": weighted_relative(h0_pinn, h_shape, weights, mask),
                                  "e_shape_bulk": weighted_relative(e0_pinn, e_shape, weights, bulk),
                                  "h_shape_bulk": weighted_relative(h0_pinn, h_shape, weights, bulk),
                                  "e_magnitude_bias_v_m": magnitude_bias(e0_pinn, e_shape, weights, bulk),
                                  "h_magnitude_bias_a_m": magnitude_bias(h0_pinn, h_shape, weights, bulk),
                                  "e_vector_rmse_v_m": vector_rmse(e0_pinn, e_shape, weights, bulk),
                                  "h_vector_rmse_a_m": vector_rmse(h0_pinn, h_shape, weights, bulk)}})
    return {"mesh_nm": metadata["mesh_nm"], "dz_nm": metadata["dz_nm"],
            "source_fde_neff": metadata["source"]["mode_neff"][0][0],
            "fdtd_phase_neff": beta/g.k0, "beta_rad_per_um": beta,
            "phase_fit_rms_rad": float(np.sqrt(np.mean(phase_residual**2))),
            "axis_circularity_min": float(np.min(g.handedness*circularity[fit_mask])),
            "calibration_real": float(calibration.real), "calibration_imag": float(calibration.imag),
            "power_relative_drift": float((max(powers)-min(powers))/powers[0]), "power_w": powers,
            "elapsed_s": metadata["elapsed_s"], "status": metadata["status"],
            "axis_z_um": z_axis, "axis_e_v_m": e_axis,
            "planes": planes, "metadata": metadata,
            "provenance": {"axis_sha256": hash_file(directory/"axis.npz"),
                           "input_plane_sha256": hash_file(directory/"plane_00.npz"),
                           "output_plane_sha256": hash_file(directory/"plane_10.npz"),
                           "source_sha256": hash_file(directory/"independent_source.npz")}}


def compare_interfaces(case, model):
    g = model.geometry
    plane = case["planes"][0]
    angles = np.linspace(0, 2*np.pi, 129)
    result, exports = [], []
    for i, radius in enumerate(g.interface_radii_um):
        row = {"radius_um": radius, "offset_cells": [2., 3., 4.], "offset_mesh_um": case["mesh_nm"]*1e-3}
        exported = {"phi_rad": angles, "radius_um": radius}
        for side, label, layer in ((-1, "inner", i), (1, "outer", i+1)):
            pinn_e, pinn_h = fiber_fields(model, radius*np.cos(angles), radius*np.sin(angles), layer=layer)
            for field, prediction in (("E", pinn_e), ("H", pinn_h)):
                fd = one_sided_trace(plane["x_um"], plane["y_um"], plane[field], radius, angles, case["mesh_nm"]*1e-3, side)
                alternate = one_sided_trace(plane["x_um"], plane["y_um"], plane[field], radius, angles, case["mesh_nm"]*1e-3, side, (3., 4., 5.))
                row[f"{field.lower()}_{label}_trace_error"] = float(np.linalg.norm(prediction-fd)/np.linalg.norm(fd))
                row[f"{field.lower()}_{label}_extrapolation_sensitivity"] = float(np.linalg.norm(alternate-fd)/np.linalg.norm(fd))
                exported[f"{field}_{label}_fdtd"] = fd
                exported[f"{field}_{label}_pinn"] = prediction
        e_in = circumferential_components(exported["E_inner_fdtd"], angles)
        e_out = circumferential_components(exported["E_outer_fdtd"], angles)
        h_in = circumferential_components(exported["H_inner_fdtd"], angles)
        h_out = circumferential_components(exported["H_outer_fdtd"], angles)
        row["e_normal_ratio_mean"] = float(np.mean(abs(e_out[:, 0]/e_in[:, 0])))
        row["e_normal_ratio_expected"] = g.indices[i]**2/g.indices[i+1]**2
        row["e_tangential_mismatch"] = float(np.linalg.norm(e_in[:, 1:]-e_out[:, 1:])/np.linalg.norm(e_out[:, 1:]))
        row["h_tangential_mismatch"] = float(np.linalg.norm(h_in[:, 1:]-h_out[:, 1:])/np.linalg.norm(h_out[:, 1:]))
        result.append(row)
        exports.append(exported)
    return result, exports


def zero_mesh_fit(meshes_nm, values, order=1):
    # Scale the x axis to avoid numerical conditioning from metre/nm powers.
    x = (np.asarray(meshes_nm)/100)**2
    design = np.vander(x, N=order+1, increasing=True)
    coefficients, _, _, _ = np.linalg.lstsq(design, np.asarray(values), rcond=None)
    return coefficients[0], coefficients


def summarize(cases, model, output):
    output.mkdir(parents=True, exist_ok=True)
    g = model.geometry
    meshes = np.array([case["mesh_nm"] for case in cases])
    neff = np.array([case["fdtd_phase_neff"] for case in cases])
    order = np.argsort(meshes)[::-1]
    cases = [cases[i] for i in order]
    meshes, neff = meshes[order], neff[order]
    neff_zero, _ = zero_mesh_fit(meshes[-3:], neff[-3:])
    neff_all, _ = zero_mesh_fit(meshes, neff)
    neff_quad, _ = zero_mesh_fit(meshes, neff, 2)
    if len(cases) < 4:
        raise ValueError("At least four converged meshes are required for this study")
    neff_primary, _ = zero_mesh_fit(meshes[-4:], neff[-4:], 2)
    neff_last3_quad, _ = zero_mesh_fit(meshes[-3:], neff[-3:], 2)
    pinn_neff = float(model.neff.detach())
    reference = BesselReference(g)
    result = {"date": "2026-10-13", "backend": "Ansys Lumerical 2024 R1 real 3D FDTD",
              "geometry": __import__("dataclasses").asdict(g), "neff_pinn": pinn_neff,
              "neff_bessel": reference.neff, "neff_fdtd_zero_mesh_last3_linear": float(neff_zero),
              "neff_fdtd_zero_mesh_all_linear": float(neff_all),
              "neff_fdtd_zero_mesh_all_quadratic": float(neff_quad),
              "neff_fdtd_zero_mesh_primary_last4_quadratic": float(neff_primary),
              "neff_fdtd_zero_mesh_last3_quadratic": float(neff_last3_quad),
              "neff_pinn_minus_primary_fdtd": float(pinn_neff-neff_primary),
              "neff_primary_fit_sensitivity": float(max(abs(neff_quad-neff_primary), abs(neff_last3_quad-neff_primary))),
              "neff_pinn_minus_fdtd_extrapolated": float(pinn_neff-neff_zero),
              "neff_extrapolation_fit_spread": float(max(abs(neff_all-neff_zero), abs(neff_quad-neff_zero))),
              "exclusion_band_um": .16,
              "checkpoint_sha256": hash_file("results/fiber_circular/model.pt"),
              "neff_primary_method": "Quadratic in h^2 (h^2+h^4) on the finest four meshes; linear fits retained as diagnostics",
              "normalization": "Single complex scalar from circular component of input axis; common to E/H and ALL planes. Shape metrics remove per-plane phase ONLY, never amplitude.",
              "cases": []}
    for case in cases:
        item = {key: value for key, value in case.items() if key not in ("planes", "metadata", "axis_z_um", "axis_e_v_m")}
        item["plane_scores"] = [p["scores"]|{"z_um": p["z_um"]} for p in case["planes"]]
        item["interfaces"], _ = compare_interfaces(case, model)
        result["cases"].append(item)
    # Same physical coordinates and same mask for numerical field extrapolation.
    xy = np.linspace(-g.radius_um, g.radius_um, 201)
    xx, yy = np.meshgrid(xy, xy, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    radius = np.hypot(xx, yy)
    mask = radius <= g.radius_um
    for r in g.interface_radii_um:
        mask &= abs(radius-r) >= .16
    weights = trapezoid_weights(xy)[:, None]*trapezoid_weights(xy)[None, :]
    e_pinn, h_pinn = fiber_fields(model, xx, yy)
    e_bessel, h_bessel = fiber_fields(reference, xx, yy)
    field_info, finest_sampled = {}, {}
    for field, prediction, bessel in (("E", e_pinn, e_bessel), ("H", h_pinn, h_bessel)):
        sample = np.stack([interpolate_vector(c["planes"][0]["x_um"], c["planes"][0]["y_um"], c["planes"][0][field], points).reshape(201, 201, 3) for c in cases])
        f0, _ = zero_mesh_fit(meshes[-3:], sample[-3:].reshape(3, -1))
        f_all, _ = zero_mesh_fit(meshes, sample.reshape(len(cases), -1))
        f_quad, _ = zero_mesh_fit(meshes, sample.reshape(len(cases), -1), 2)
        f0 = f0.reshape(201, 201, 3)
        f_all, f_quad = f_all.reshape(f0.shape), f_quad.reshape(f0.shape)
        field_info[field] = {"pinn_vs_fdtd_zero_mesh_bulk_l2": weighted_relative(prediction, f0, weights, mask),
                             "fdtd_zero_mesh_vs_bessel_bulk_l2": weighted_relative(f0, bessel, weights, mask),
                             "pinn_vs_bessel_bulk_l2": weighted_relative(prediction, bessel, weights, mask),
                             "fit_spread_bulk_l2": max(weighted_relative(f_all, f0, weights, mask), weighted_relative(f_quad, f0, weights, mask)),
                             "pinn_magnitude_bias": magnitude_bias(prediction, f0, weights, mask),
                             "pinn_vector_rmse": vector_rmse(prediction, f0, weights, mask)}
        finest_sampled[field] = sample[-1]
    result["field_extrapolation"] = field_info
    (output/"comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fine = cases[-1]
    # Small, unmodified axis exports let readers independently repeat beta fits.
    phase_archive = {}
    for case in cases:
        prefix = f"mesh_{case['mesh_nm']:g}nm_"
        phase_archive[prefix+"z_um"] = case["axis_z_um"]
        phase_archive[prefix+"E_v_m"] = case["axis_e_v_m"]
    np.savez_compressed(output/"axis_phase_traces.npz", **phase_archive)
    header = ["mesh_nm", "z_um"]+list(result["cases"][0]["plane_scores"][0].keys())[:-1]
    rows = [[item["mesh_nm"], score["z_um"]]+[score[key] for key in header[2:]]
            for item in result["cases"] for score in item["plane_scores"]]
    np.savetxt(output/"plane_metrics.csv", rows, delimiter=",", header=",".join(header), comments="")
    _, traces = compare_interfaces(fine, model)
    for i, trace in enumerate(traces, 1):
        header, columns = ["phi_rad"], [trace["phi_rad"]]
        for side in ("inner", "outer"):
            for field in ("E", "H"):
                for source in ("fdtd", "pinn"):
                    vector = trace[f"{field}_{side}_{source}"]
                    for j, component in enumerate("xyz"):
                        for part, value in (("real", vector[:, j].real), ("imag", vector[:, j].imag)):
                            header.append(f"{field}{component}_{side}_{source}_{part}")
                            columns.append(value)
        np.savetxt(output/f"interface_{i}_comparison.csv", np.column_stack(columns), delimiter=",", header=",".join(header), comments="")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(meshes**2, neff, "o-", label="Real FDTD phase fit")
    axes[0].axhline(pinn_neff, color="tab:red", linestyle="--", label="PINN")
    axes[0].axhline(reference.neff, color="k", linestyle=":", label="Bessel reference")
    axes[0].plot([0], [neff_primary], "x", label="Fine 4 h^2+h^4 extrapolation")
    axes[0].set(xlabel="Transverse mesh squared (nm^2)", ylabel="Effective index")
    axes[0].legend(fontsize=8)
    for c, item in zip(cases, result["cases"]):
        axes[1].plot(np.arange(11), [100*s["e_complex_bulk"] for s in item["plane_scores"]], label=f"{c['mesh_nm']:g} nm")
    axes[1].set(xlabel="z (um)", ylabel="Raw complex E difference (%)", title="Input calibration fixed; dispersion retained")
    axes[1].legend()
    axes[2].plot(meshes, [100*c["plane_scores"][0]["e_shape_bulk"] for c in result["cases"]], "o-", label="E shape")
    axes[2].plot(meshes, [100*c["plane_scores"][0]["h_shape_bulk"] for c in result["cases"]], "s-", label="H shape")
    axes[2].set(xlabel="Transverse mesh (nm)", ylabel="Bulk relative field error (%)")
    axes[2].legend()
    figure.savefig(output/"convergence.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    titles = ["PINN", f"Real FDTD {fine['mesh_nm']:g} nm", "Vector difference"]
    for row, (field, prediction) in enumerate((("E", e_pinn), ("H", h_pinn))):
        for column, values in enumerate((np.linalg.norm(prediction, axis=-1), np.linalg.norm(finest_sampled[field], axis=-1), np.linalg.norm(prediction-finest_sampled[field], axis=-1))):
            values = values.copy()
            values[radius > 2.2] = np.nan
            im = axes[row, column].imshow(values.T, extent=[xy[0], xy[-1], xy[0], xy[-1]], origin="lower")
            axes[row, column].set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), title=f"{field}: {titles[column]}", xlabel="x (um)", ylabel="y (um)")
            figure.colorbar(im, ax=axes[row, column], label="V/m" if field == "E" else "A/m")
    figure.savefig(output/"field_comparison.png", dpi=170)
    plt.close(figure)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "geometry")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=float, nargs="+", default=[80, 60, 40, 30, 25])
    parser.add_argument("--input", default="runs/fdtd")
    parser.add_argument("--output", default="results/fdtd_validation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = FiberPINN.from_checkpoint("results/fiber_circular/model.pt")
    cases = [load_case(Path(args.input)/f"{mesh:g}nm", model) for mesh in args.meshes]
    summarize(cases, model, Path(args.output))


if __name__ == "__main__":
    main()
