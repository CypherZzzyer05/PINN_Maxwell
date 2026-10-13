from collections import OrderedDict
import importlib.util
import os
import sys
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pinn_maxwell.fiber import FiberGeometry


def load_lumapi(path=None):
    path = Path(path or os.environ.get("LUMAPI_PATH", r"D:\Program Files\Lumerical\v241\api\python\lumapi.py")).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lumapi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector_dataset(data, field):
    x, y = np.asarray(data["x"]).ravel()*1e6, np.asarray(data["y"]).ravel()*1e6
    values = np.asarray(data[field])
    values = values.reshape(len(x), len(y), -1, 3)
    if values.shape[2] != 1:
        raise ValueError("Only single-frequency planar data is supported")
    return x, y, values[:, :, 0, :]


def interpolate_vector(x, y, field, points):
    return RegularGridInterpolator((x, y), field, bounds_error=True)(np.asarray(points))


def rotate_vector_90(x, y, field):
    """Rotate positions AND vector components; valid for the symmetric fiber."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((yy.ravel(), -xx.ravel()))
    tolerance = 1e-9  # um; roundoff only, never extend physical support.
    if (points[:, 0].min() < x[0]-tolerance or points[:, 0].max() > x[-1]+tolerance or
            points[:, 1].min() < y[0]-tolerance or points[:, 1].max() > y[-1]+tolerance):
        raise ValueError("Source grid does not support 90-degree symmetry")
    points[:, 0] = np.clip(points[:, 0], x[0], x[-1])
    points[:, 1] = np.clip(points[:, 1], y[0], y[-1])
    original = interpolate_vector(x, y, field, points).reshape(field.shape)
    return np.stack((-original[..., 1], original[..., 0], original[..., 2]), axis=-1)


class FiberFDTD:
    def __init__(self, geometry: FiberGeometry, mesh_nm=80., threads=8, time_fs=600.,
                 shutoff=1e-7, transverse_span_um=14., lumapi_path=None):
        self.g = geometry
        self.mesh_nm, self.threads, self.time_fs = mesh_nm, threads, time_fs
        self.shutoff, self.span = shutoff, transverse_span_um
        self.api = load_lumapi(lumapi_path) if lumapi_path else load_lumapi()
        self.f = None

    def __enter__(self):
        self.f = self.api.FDTD(hide=True)
        self.version = str(self.f.version())
        self.f.setresource("FDTD", 1, "processes", "1")
        self.f.setresource("FDTD", 1, "threads", str(self.threads))
        return self

    def __exit__(self, *args):
        if self.f is not None:
            self.f.close()

    def build(self, project):
        f, g = self.f, self.g
        um, nm = 1e-6, 1e-9
        f.switchtolayout()
        f.selectall()
        f.delete()
        f.addfdtd(properties=OrderedDict([
            ("dimension", "3D"), ("x", 0.), ("y", 0.),
            ("x span", self.span*um), ("y span", self.span*um),
            ("z min", -3*um), ("z max", (g.length_um+3)*um),
            ("index", g.indices[-1]), ("mesh accuracy", 1),
            ("x min bc", "PML"), ("x max bc", "PML"),
            ("y min bc", "PML"), ("y max bc", "PML"),
            ("z min bc", "PML"), ("z max bc", "PML"),
            ("pml layers", 12), ("simulation time", self.time_fs*1e-15),
            ("auto shutoff min", self.shutoff), ("dt stability factor", .9)
        ]))
        # Dielectric concentric disks, extruded through both axial PMLs.
        for i in reversed(range(len(g.interface_radii_um))):
            f.addcircle(properties=OrderedDict([
                ("name", f"layer_{i}"), ("x", 0.), ("y", 0.),
                ("radius", g.interface_radii_um[i]*um),
                ("z min", -4*um), ("z max", (g.length_um+4)*um),
                ("material", "<Object defined dielectric>"), ("index", g.indices[i]),
                ("override mesh order from material database", True), ("mesh order", i+1)
            ]))
        # Coarse outer grid limits total RAM; fixed aspect ratio in mesh study.
        f.addmesh(name="outer_grid", x=0., y=0., x_span=self.span*um, y_span=self.span*um,
                  z_min=-3*um, z_max=(g.length_um+3)*um,
                  dx=120*nm, dy=120*nm, dz=1.5*self.mesh_nm*nm)
        f.addmesh(name="fiber_grid", x=0., y=0., x_span=4.2*um, y_span=4.2*um,
                  z_min=-3*um, z_max=(g.length_um+3)*um,
                  dx=self.mesh_nm*nm, dy=self.mesh_nm*nm, dz=1.5*self.mesh_nm*nm)
        # Broad temporal pulse; harmonic center wavelength is exactly lambda0.
        start, stop = g.wavelength_um/1.12, g.wavelength_um/.88
        for name, selection in (("source_x", "fundamental TE mode"),):
            f.addmode(properties=OrderedDict([
                ("name", name), ("injection axis", "z-axis"), ("direction", "Forward"),
                ("x", 0.), ("y", 0.), ("x span", 10*um), ("y span", 10*um),
                ("z", -1*um), ("mode selection", selection), ("number of trial modes", 4),
                ("override global source settings", True),
                ("wavelength start", start*um), ("wavelength stop", stop*um)
            ]))
        f.setglobalmonitor("use source limits", False)
        f.setglobalmonitor("use wavelength spacing", True)
        f.setglobalmonitor("wavelength center", g.wavelength_um*um)
        f.setglobalmonitor("wavelength span", 0.)
        f.setglobalmonitor("frequency points", 1)
        self.planes = np.linspace(0., g.length_um, 11)
        for i, z in enumerate(self.planes):
            f.addprofile(name=f"plane_{i:02d}", monitor_type="2D Z-normal", x=0., y=0.,
                         x_span=2*g.radius_um*um, y_span=2*g.radius_um*um, z=float(z)*um)
            f.set("spatial interpolation", "specified position")
        f.addprofile(name="axis", monitor_type="Linear Z", x=0., y=0.,
                     z_min=0., z_max=g.length_um*um)
        f.set("spatial interpolation", "specified position")
        project = Path(project).resolve()
        project.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(project))
        self.project = project

    def prepare_sources(self):
        f = self.f
        profiles, neffs = [], []
        for mode_number, name in enumerate(("source_x",), start=1):
            f.select(name)
            # Keep the independently computed profiles fixed for the actual run.
            print("MODE", name, flush=True)
            f.set("mode selection", "user select")
            f.seteigensolver("override default boundaries", True)
            for axis_name in ("x min bc", "x max bc", "y min bc", "y max bc"):
                f.seteigensolver(axis_name, "PML")
            f.seteigensolver("wavelength", self.g.wavelength_um*1e-6)
            f.updatesourcemode(mode_number)
            available = str(f.getresult(name)).splitlines()
            result_name = "mode profiles" if "mode profiles" in available else "mode profile"
            data = f.getresult(name, result_name)
            x, y, e = vector_dataset(data, "E")
            axis = interpolate_vector(x, y, e, [[0., 0.]])[0]
            profiles.append(axis)
            n = f.getresult(name, "neff")
            neffs.append(np.asarray(n["neff"]).ravel()[0])
            print("SOURCE", name, "neff", neffs[-1], "axis E", axis, flush=True)
            f.set("mode selection", "user select")
            f.set("auto update", False)
        # Construct a partner by exact 90-degree symmetry. Two separate searches
        # can return the SAME vector in a degenerate eigenspace, making source
        # mixing ill-conditioned even when both neff values look correct.
        stored = [f.getresult("source_x", result_name)]
        all_e, all_h, profiles = [], [], []
        for data in stored:
            x, y, e = vector_dataset(data, "E")
            _, _, h = vector_dataset(data, "H")
            profiles.append(interpolate_vector(x, y, e, [[0., 0.]])[0])
            all_e.append(e)
            all_h.append(h)
        rotated_e, rotated_h = rotate_vector_90(x, y, all_e[0]), rotate_vector_90(x, y, all_h[0])
        all_e.append(rotated_e)
        all_h.append(rotated_h)
        profiles.append(interpolate_vector(x, y, rotated_e, [[0., 0.]])[0])
        matrix = np.asarray(profiles)[:, :2].T
        if np.linalg.cond(matrix) > 10:
            # A circular eigenvector is proportional to its 90-degree rotation.
            # Reciprocity gives the opposite circular forward mode: conjugate
            # transverse fields, reverse longitudinal signs (real dielectrics).
            opposite_e, opposite_h = np.conj(all_e[0]), np.conj(all_h[0])
            opposite_e[..., 2] *= -1
            opposite_h[..., 2] *= -1
            all_e[1], all_h[1] = opposite_e, opposite_h
            profiles[1] = interpolate_vector(x, y, opposite_e, [[0., 0.]])[0]
            matrix = np.asarray(profiles)[:, :2].T
        if np.linalg.cond(matrix) > 100:
            raise ValueError("Numerical mode basis cannot form a stable circular excitation")
        # Only fix input polarization, no reference/PINN field shapes or beta.
        target = np.array([1., 1j*self.g.handedness])/np.sqrt(2)
        coefficients = np.linalg.solve(matrix, target)
        combined_e = coefficients[0]*all_e[0]+coefficients[1]*all_e[1]
        combined_h = coefficients[0]*all_h[0]+coefficients[1]*all_h[1]
        for name in ("source_x",):
            f.select(name)
            f.delete()
        f.addimportedsource(properties=OrderedDict([
            ("name", "circular_source"), ("injection axis", "z-axis"), ("direction", "Forward"),
            ("x", 0.), ("y", 0.), ("z", -1e-6),
            ("override global source settings", True),
            ("wavelength start", self.g.wavelength_um/1.12*1e-6),
            ("wavelength stop", self.g.wavelength_um/.88*1e-6)
        ]))
        f.putv("sx", np.asarray(stored[0]["x"]))
        f.putv("sy", np.asarray(stored[0]["y"]))
        f.putv("sz", np.array([-1e-6]))
        f.putv("sf", np.array([299792458/(self.g.wavelength_um*1e-6)]))
        for prefix, vector in (("E", combined_e), ("H", combined_h)):
            for i, axis in enumerate("xyz"):
                f.putv(prefix+axis+"_src", vector[..., i].reshape(len(x), len(y), 1, 1))
        f.eval('source_fields=rectilineardataset("independent_FDE_circular",sx,sy,sz);'
               'source_fields.addparameter("lambda",c/sf,"f",sf);'
               'source_fields.addattribute("E",Ex_src,Ey_src,Ez_src);'
               'source_fields.addattribute("H",Hx_src,Hy_src,Hz_src);'
               'select("circular_source"); importdataset(source_fields);')
        self.source_info = {"mode_neff": [[float(n.real), float(n.imag)] for n in neffs],
                            "axis_matrix_real": matrix.real.tolist(), "axis_matrix_imag": matrix.imag.tolist(),
                            "coefficients_real": coefficients.real.tolist(), "coefficients_imag": coefficients.imag.tolist()}
        self.source_info["type"] = "Frozen numerical FDE superposition, imported E and H; no PINN source fields"
        self.source_info["basis_construction"] = "One independent FDE mode with a symmetry/reciprocity partner"
        self.source_info["axis_matrix_condition"] = float(np.linalg.cond(matrix))
        self.source_fields = {"x_um": x, "y_um": y, "E_v_m": combined_e, "H_a_m": combined_h}
        f.save(str(self.project))
        return self.source_info

    def solve(self, output):
        started = time.perf_counter()
        print("START real 3D FDTD", self.mesh_nm, "nm", flush=True)
        self.f.run()
        elapsed = time.perf_counter()-started
        self.f.save(str(self.project))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        status = self.f.getresult("FDTD", "status")
        print("STATUS", status, "elapsed", elapsed, flush=True)
        np.savez_compressed(output/"independent_source.npz", **self.source_fields)
        for i, z in enumerate(self.planes):
            e = self.f.getresult(f"plane_{i:02d}", "E")
            h = self.f.getresult(f"plane_{i:02d}", "H")
            frequencies = np.asarray(e["f"]).ravel()
            if len(frequencies) != 1:
                raise ValueError("Expected one actual monitor frequency")
            actual_wavelength = 299792458/frequencies[0]*1e6
            if not np.isclose(actual_wavelength, self.g.wavelength_um, rtol=1e-10):
                raise ValueError(f"Monitor wavelength mismatch: {actual_wavelength}")
            x, y, ev = vector_dataset(e, "E")
            hx, hy, hv = vector_dataset(h, "H")
            if not np.array_equal(x, hx) or not np.array_equal(y, hy):
                raise ValueError("E/H monitor coordinates differ")
            np.savez_compressed(output/f"plane_{i:02d}.npz", x_um=x, y_um=y, z_um=z,
                                wavelength_um=actual_wavelength, E_v_m=ev, H_a_m=hv)
        axis = self.f.getresult("axis", "E")
        np.savez_compressed(output/"axis.npz", z_um=np.asarray(axis["z"]).ravel()*1e6,
                            E_v_m=np.asarray(axis["E"]).reshape(-1, 3))
        np.savez_compressed(output/"simulation_grid.npz", **{
            key+"_um": np.asarray(self.f.getresult("FDTD", key)).ravel()*1e6 for key in "xyz"})
        return status, elapsed
