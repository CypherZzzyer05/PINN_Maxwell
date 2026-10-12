"""Reconstruct six complex fields in Cartesian/cylindrical coordinates."""

import numpy as np
import torch
from torch import nn

from .fields import Z0_OHM


def radial_values(solution, radius, layer=None):
    if isinstance(solution, nn.Module):
        with torch.no_grad():
            r = torch.as_tensor(radius, dtype=torch.float64, device=solution.raw_neff.device)
            return solution(r, layer).cpu().numpy()
    return solution.state(radius, layer)


def fiber_fields(solution, x_um, y_um, z_um=0., layer=None, handedness=None, cylindrical=False):
    """Complex phasors E[V/m], H[A/m]; last axis is xyz or r/phi/z.

    Ey=i*handedness*Ex on axis; exp(-i omega t). Interface traces can be
    evaluated at identical radius with explicit adjacent material `layer`.
    """
    g = solution.geometry
    x, y, z = np.broadcast_arrays(x_um, y_um, z_um)
    radius = np.hypot(x, y)
    phi = np.arctan2(y, x)
    unique, inverse = np.unique(radius.ravel(), return_inverse=True)
    state = radial_values(solution, unique, layer)[inverse].reshape((*radius.shape, 4))
    u, w, p, q = np.moveaxis(state, -1, 0)
    axis = [float(t.detach().cpu()) if torch.is_tensor(t) else float(t) for t in solution.axis_values()]
    u_over_r = np.divide(u, radius, out=np.full_like(u, axis[0]), where=radius > 0)
    w_over_r = np.divide(w, radius, out=np.full_like(w, axis[1]), where=radius > 0)
    labels = g.layer_at(radius) if layer is None else np.full(radius.shape, layer)
    eps = np.asarray(g.indices)[labels]**2
    n = float(solution.neff.detach().cpu()) if torch.is_tensor(solution.neff) else float(solution.neff)
    a = (n*q-w_over_r/g.k0)/eps
    b = -n*p+u_over_r/g.k0
    s = g.handedness if handedness is None else handedness
    if s not in (-1, 1):
        raise ValueError("Circular handedness must be +/-1")
    common = 1j*g.electric_amplitude_v_m*np.exp(1j*(s*phi+g.k0*n*z))
    electric_cyl = np.stack((1j*a, s*p, u), axis=-1)*common[..., None]
    magnetic_cyl = np.stack((s*b, 1j*q, 1j*s*w), axis=-1)*common[..., None]/Z0_OHM
    if cylindrical:
        return electric_cyl, magnetic_cyl
    electric = electric_cyl.copy()
    magnetic = magnetic_cyl.copy()
    for cart, cyl in ((electric, electric_cyl), (magnetic, magnetic_cyl)):
        cart[..., 0] = cyl[..., 0]*np.cos(phi)-cyl[..., 1]*np.sin(phi)
        cart[..., 1] = cyl[..., 0]*np.sin(phi)+cyl[..., 1]*np.cos(phi)
    return electric, magnetic


def linear_basis(solution, x, y, z=0.):
    plus_e, plus_h = fiber_fields(solution, x, y, z, handedness=1)
    minus_e, minus_h = fiber_fields(solution, x, y, z, handedness=-1)
    return ((plus_e+minus_e)/np.sqrt(2), (plus_h+minus_h)/np.sqrt(2),
            (plus_e-minus_e)/(1j*np.sqrt(2)), (plus_h-minus_h)/(1j*np.sqrt(2)))


def stokes(electric):
    ex, ey = electric[..., 0], electric[..., 1]
    s0 = np.abs(ex)**2+np.abs(ey)**2
    s1 = np.abs(ex)**2-np.abs(ey)**2
    s2 = 2*np.real(np.conj(ex)*ey)
    s3 = 2*np.imag(np.conj(ex)*ey)
    return np.stack((s0, s1, s2, s3), axis=-1)


def maxwell_diagnostics(solution, points, delta=1e-5):
    """Independent Cartesian finite-difference curls and divergences in um units.

    Points must be inside homogeneous layers, away from material interfaces.
    Reports global relative norms, avoiding division by local field nodes.
    """
    points = np.asarray(points, dtype=float)
    e, h_si = fiber_fields(solution, *points.T)
    h = Z0_OHM*h_si
    de, dh = [], []
    for direction in range(3):
        offset = np.zeros(3)
        offset[direction] = delta
        e_plus, h_plus = fiber_fields(solution, *(points+offset).T)
        e_minus, h_minus = fiber_fields(solution, *(points-offset).T)
        de.append((e_plus-e_minus)/(2*delta))
        dh.append(Z0_OHM*(h_plus-h_minus)/(2*delta))
    curl_e = np.stack((de[1][:, 2]-de[2][:, 1], de[2][:, 0]-de[0][:, 2], de[0][:, 1]-de[1][:, 0]), axis=-1)
    curl_h = np.stack((dh[1][:, 2]-dh[2][:, 1], dh[2][:, 0]-dh[0][:, 2], dh[0][:, 1]-dh[1][:, 0]), axis=-1)
    g = solution.geometry
    eps = np.asarray(g.indices)[g.layer_at(np.hypot(points[:, 0], points[:, 1]))]**2
    div_e = sum(de[i][:, i] for i in range(3))
    div_h = sum(dh[i][:, i] for i in range(3))
    return {"curl_e_relative": float(np.linalg.norm(curl_e/g.k0-1j*h)/np.linalg.norm(h)),
            "curl_h_relative": float(np.linalg.norm(curl_h/g.k0+1j*eps[:, None]*e)/np.linalg.norm(eps[:, None]*e)),
            "div_d_relative": float(np.linalg.norm(eps*div_e/g.k0)/np.linalg.norm(eps[:, None]*e)),
            "div_b_relative": float(np.linalg.norm(div_h/g.k0)/np.linalg.norm(h))}
