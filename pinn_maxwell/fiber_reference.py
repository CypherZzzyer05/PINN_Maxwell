"""Independent full-vector Bessel matching eigensolver, validation only."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv, ivp, jv, jvp, kv, kvp


class BesselReference:
    def __init__(self, geometry):
        self.geometry = geometry
        self.specs = [(0, component, "J") for component in (0, 1)]
        for i in range(1, len(geometry.indices)-1):
            self.specs.extend((i, component, kind) for component in (0, 1) for kind in ("I", "K"))
        self.specs.extend((len(geometry.indices)-1, component, "K") for component in (0, 1))
        grid = np.linspace(geometry.n_lower+1e-7, geometry.indices[0]-1e-7, 1400)
        determinants = [self.determinant(n) for n in grid]
        roots = [brentq(self.determinant, a, b, xtol=1e-13) for a, b, fa, fb in
                 zip(grid[:-1], grid[1:], determinants[:-1], determinants[1:]) if fa*fb < 0]
        if not roots:
            raise ValueError("No core-guided m=1 hybrid mode found above all cladding indices")
        self.roots = roots
        self.neff = max(roots)
        matrix = self.matching_matrix(self.neff)
        _, singular, vectors = np.linalg.svd(matrix)
        self.smallest_singular_value = float(singular[-1])
        self.coefficients = vectors[-1]
        _, _, p0, _ = self.axis_values()
        self.coefficients *= (1/np.sqrt(2))/p0

    def basis(self, layer, component, kind, radius, neff):
        g = self.geometry
        r = np.asarray(radius, dtype=float)
        kt2 = g.k0**2*(g.indices[layer]**2-neff**2)
        k = np.sqrt(abs(kt2))
        function, slope_function = {"J": (jv, jvp), "I": (iv, ivp), "K": (kv, kvp)}[kind]
        value = function(1, k*r)
        slope = k*slope_function(1, k*r)
        over_r = np.divide(value, r, out=np.full_like(r, k/2 if kind in ("J", "I") else np.nan), where=r != 0)
        zero = np.zeros_like(r)
        if component == 0:
            return np.stack((value, zero, -g.k0*neff*over_r/kt2,
                             g.k0*g.indices[layer]**2*slope/kt2), axis=-1)
        return np.stack((zero, value, g.k0*slope/kt2, -g.k0*neff*over_r/kt2), axis=-1)

    def matching_matrix(self, neff):
        g = self.geometry
        matrix = np.zeros((4*len(g.interface_radii_um), len(self.specs)))
        for i, radius in enumerate(g.interface_radii_um):
            for column, (layer, component, kind) in enumerate(self.specs):
                if layer in (i, i+1):
                    sign = 1 if layer == i else -1
                    matrix[4*i:4*i+4, column] = sign*self.basis(layer, component, kind, radius, neff)
        return matrix

    def determinant(self, neff):
        matrix = self.matching_matrix(neff)
        norm = np.linalg.norm(matrix, axis=0)
        return float(np.linalg.det(matrix/norm[None, :]))

    def axis_values(self):
        g = self.geometry
        k = g.k0*np.sqrt(g.indices[0]**2-self.neff**2)
        u0 = self.coefficients[0]*k/2
        w0 = self.coefficients[1]*k/2
        p0 = (-g.k0*self.neff*u0+g.k0*w0)/k**2
        q0 = (-g.k0*self.neff*w0+g.k0*g.indices[0]**2*u0)/k**2
        return u0, w0, p0, q0

    def state(self, radius, layer=None):
        r = np.asarray(radius).reshape(-1)
        layers = self.geometry.layer_at(r) if layer is None else np.full(len(r), layer)
        result = np.zeros((len(r), 4))
        for coefficient, (i, component, kind) in zip(self.coefficients, self.specs):
            mask = layers == i
            if mask.any():
                result[mask] += coefficient*self.basis(i, component, kind, r[mask], self.neff)
        return result
