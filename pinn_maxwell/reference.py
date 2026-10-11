"""Independent analytic and transfer-matrix references, used only AFTER training."""

import numpy as np
from scipy.optimize import brentq

from .geometry import Waveguide


def symmetric_slab(geometry, x):
    g = geometry
    if len(g.indices) != 3 or not np.isclose(g.indices[0], g.indices[-1]):
        raise ValueError("Analytic slab requires equal claddings and one core")
    half = (g.boundaries_um[2] - g.boundaries_um[1]) / 2
    center = (g.boundaries_um[1] + g.boundaries_um[2]) / 2
    v = g.k0 * half * np.sqrt(g.indices[1]**2 - g.indices[0]**2)
    u = brentq(lambda u: u * np.tan(u) - np.sqrt(v*v - u*u),
               1e-10, min(v, np.pi/2) - 1e-10, xtol=1e-14)
    neff = np.sqrt(g.indices[1]**2 - (u / (g.k0 * half))**2)
    kx = u / half
    decay = g.k0 * np.sqrt(neff**2 - g.indices[0]**2)
    xc = np.asarray(x) - center
    inside = np.abs(xc) <= half
    e = np.where(inside, np.cos(kx*xc), np.cos(u)*np.exp(-decay*(np.abs(xc)-half)))
    de = np.where(inside, -kx*np.sin(kx*xc), -np.sign(xc)*decay*e)
    anchor = np.cos(kx*(g.anchor_um-center))
    return neff, e/anchor, de/(g.k0*anchor)


def layer_matrix(index, neff, k0, distance):
    k2 = k0*k0 * (index*index - neff*neff)
    if abs(k2) < 1e-12:
        return np.array([[1., distance], [0., 1.]])
    k = np.sqrt(abs(k2))
    if k2 > 0:
        c, s = np.cos(k*distance), np.sin(k*distance)
        return np.array([[c, s/k], [-k*s, c]])
    c, s = np.cosh(k*distance), np.sinh(k*distance)
    return np.array([[c, s/k], [k*s, c]])


def transfer_reference(geometry: Waveguide, x):
    g = geometry
    b = g.boundaries_um

    def shoot(neff):
        state = np.array([1., g.k0*np.sqrt(neff*neff-g.indices[0]**2)])
        for i in range(1, len(g.indices)-1):
            state = layer_matrix(g.indices[i], neff, g.k0, b[i+1]-b[i]) @ state
        return state[1] + g.k0*np.sqrt(neff*neff-g.indices[-1]**2)*state[0]

    grid = np.linspace(g.n_lower+1e-9, g.n_upper-1e-9, 1500)
    values = [shoot(n) for n in grid]
    roots = [brentq(shoot, a, c, xtol=1e-13) for a, c, fa, fc in
             zip(grid[:-1], grid[1:], values[:-1], values[1:]) if fa*fc < 0]
    if not roots:
        raise ValueError("No guided TE mode found in this configuration")
    neff = max(roots)
    alpha_l = g.k0*np.sqrt(neff*neff-g.indices[0]**2)
    alpha_r = g.k0*np.sqrt(neff*neff-g.indices[-1]**2)
    states = [np.array([1., alpha_l])]
    for i in range(1, len(g.indices)-1):
        states.append(layer_matrix(g.indices[i], neff, g.k0, b[i+1]-b[i]) @ states[-1])

    def evaluate(position):
        if position < b[1]:
            e = np.exp(alpha_l*(position-b[1]))
            return np.array([e, alpha_l*e])
        if position >= b[-2]:
            e = states[-1][0]*np.exp(-alpha_r*(position-b[-2]))
            return np.array([e, -alpha_r*e])
        i = int(np.searchsorted(b[1:-1], position, side="right"))
        return layer_matrix(g.indices[i], neff, g.k0, position-b[i]) @ states[i-1]

    norm = evaluate(g.anchor_um)[0]
    result = np.array([evaluate(float(t)) for t in np.asarray(x).ravel()])/norm
    return neff, result[:, 0], result[:, 1]/g.k0
