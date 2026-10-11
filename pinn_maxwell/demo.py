"""Train a single device, independently validate, and export complex E/H fields."""

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from .fields import Z0_OHM, gaussian_projection, integrate, poynting, propagate
from .geometry import Waveguide
from .model import MaxwellTEPINN
from .reference import symmetric_slab, transfer_reference
from .train import train


def rel_l2(prediction, reference):
    return float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))


def run(args):
    g = Waveguide.from_json(args.config)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.adam_steps < 0 or args.lbfgs_steps < 0 or args.samples < 8:
        raise ValueError("Steps must be nonnegative and samples >=8")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = MaxwellTEPINN(g, width=args.width).to(args.device)
    history, terms = train(model, args.adam_steps, args.lbfgs_steps, args.samples, args.seed)
    x = np.linspace(g.boundaries_um[0], g.boundaries_um[-1], 1601)
    z = np.linspace(0, g.length_um, 201)
    with torch.no_grad():
        eq = model(torch.as_tensor(x, dtype=torch.float64, device=args.device)).cpu().numpy()
        neff = float(model.neff.cpu())
    e, q = eq.T
    # Reference is called for the first time here, after optimizer has finished.
    n_ref, e_ref, q_ref = transfer_reference(g, x)
    analytic_consistency = None
    if len(g.indices) == 3 and np.isclose(g.indices[0], g.indices[-1]):
        n_analytic, e_analytic, q_analytic = symmetric_slab(g, x)
        analytic_consistency = {"neff_absolute": abs(n_ref-n_analytic),
                                "e_relative_l2": rel_l2(e_ref, e_analytic),
                                "q_relative_l2": rel_l2(q_ref, q_analytic)}
    coefficient, fraction, source = gaussian_projection(x, e, g.gaussian_waist_um)
    launch_factor = coefficient if g.launch == "gaussian" else 1.
    electric, magnetic = propagate(e, q, neff, g.wavelength_um, z,
                                    amplitude=g.electric_amplitude_v_m*launch_factor)
    ref_launch = gaussian_projection(x, e_ref, g.gaussian_waist_um)[0] if g.launch == "gaussian" else 1.
    electric_ref, magnetic_ref = propagate(e_ref, q_ref, n_ref, g.wavelength_um,
                                          np.array([g.length_um]),
                                          amplitude=g.electric_amplitude_v_m*ref_launch)
    power_density = poynting(electric, magnetic)[..., 2]
    power = integrate(power_density, x*1e-6, axis=1)  # W/m in invariant y.
    thresholds = {"neff_relative_error": 1e-3, "e_relative_l2": 1e-2,
                  "h_relative_l2": 2e-2, "heldout_pde_loss": 1e-5,
                  "interface_loss": 1e-5, "boundary_loss": 1e-5, "anchor_loss": 1e-5,
                  "output_e_relative_l2": 2e-2, "output_h_relative_l2": 2e-2}
    scores = {"neff_relative_error": abs(neff-n_ref)/n_ref,
              "e_relative_l2": rel_l2(e, e_ref),
              "h_relative_l2": rel_l2(np.stack((-neff*e, -q)),
                                      np.stack((-n_ref*e_ref, -q_ref))),
              "heldout_pde_loss": terms["pde"], "interface_loss": terms["interface"],
              "boundary_loss": terms["boundary"], "anchor_loss": terms["anchor"],
              "output_e_relative_l2": rel_l2(electric[-1], electric_ref[0]),
              "output_h_relative_l2": rel_l2(magnetic[-1], magnetic_ref[0])}
    passed = all(scores[k] < limit for k, limit in thresholds.items())
    metrics = {"validated": passed, "neff_pinn": neff, "neff_reference": n_ref,
               "beta_rad_per_um": neff*g.k0, "scores": scores, "thresholds": thresholds,
               "loss_terms": terms, "analytic_transfer_agreement": analytic_consistency,
               "power_w_per_m_y": float(power[0]),
               "power_relative_drift": float(np.ptp(power)/max(abs(power[0]), 1e-30)),
               "gaussian_l2_retained_fraction": float(fraction) if g.launch == "gaussian" else None,
               "launch_modal_coefficient": float(launch_factor),
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "scipy": scipy.__version__,
                           "device": args.device, "threads": args.threads},
               "training": vars(args), "geometry": asdict(g),
               "scope": "1D transverse, y/z invariant, lossless isotropic dielectric TE fundamental; single-mode propagation"}
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "geometry": asdict(g), "width": args.width},
               out/"model.pt")
    np.savez_compressed(out/"fields.npz", x_um=x, z_um=z, refractive_index=g.index_at(x),
                        E_v_m=electric, H_a_m=magnetic, Sz_w_m2=power_density,
                        neff=neff, reference_neff=n_ref, reference_e=e_ref, reference_q=q_ref,
                        input_e=source*g.electric_amplitude_v_m if g.launch == "gaussian" else e*g.electric_amplitude_v_m)
    np.savetxt(out/"output_cross_section.csv",
               np.column_stack([x, electric[-1, :, 1].real, electric[-1, :, 1].imag,
                                magnetic[-1, :, 0].real, magnetic[-1, :, 0].imag,
                                magnetic[-1, :, 2].real, magnetic[-1, :, 2].imag, power_density[-1]]),
               delimiter=",", header="x_um,Ey_real_V_m,Ey_imag_V_m,Hx_real_A_m,Hx_imag_A_m,Hz_real_A_m,Hz_imag_A_m,Sz_W_m2", comments="")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(x, e_ref, "k--", label="Independent reference")
    axes[0, 0].plot(x, e, label="PINN Ey / anchor")
    axes[0, 0].set(xlabel="x (um)", ylabel="Normalized electric field")
    axes[0, 0].legend()
    axes[0, 1].plot(x, -neff*e/Z0_OHM, label="Hx (real at z=0)")
    axes[0, 1].plot(x, -q/Z0_OHM, label="Hz (imag at z=0)")
    axes[0, 1].set(xlabel="x (um)", ylabel="H for unit anchor E (A/m)")
    axes[0, 1].legend()
    extent = [x[0], x[-1], z[0], z[-1]]
    im = axes[1, 0].imshow(electric[..., 1].real, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
    figure.colorbar(im, ax=axes[1, 0], label="Re Ey (V/m)")
    axes[1, 0].set(xlabel="x (um)", ylabel="z (um)", title="Single guided mode propagation")
    axes[1, 1].plot(x, np.abs(electric[-1, :, 1])**2, label="Output |Ey|^2")
    axes[1, 1].set(xlabel="x (um)", ylabel="Electric squared amplitude (V/m)^2")
    axes[1, 1].legend()
    figure.suptitle(f"TE Maxwell PINN: neff={neff:.7f}; E error={scores['e_relative_l2']:.2e}; validated={passed}")
    figure.savefig(out/"fields.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"validated": passed, "neff": neff, "scores": scores, "output": str(out.resolve())}, indent=2))
    return 0 if passed or args.allow_unvalidated else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/slab.json")
    parser.add_argument("--output", default="runs/slab")
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=400)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-unvalidated", action="store_true", help="Export diagnostic runs without failure exit code")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
