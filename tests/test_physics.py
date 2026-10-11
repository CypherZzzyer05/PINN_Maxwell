import unittest

import numpy as np
import torch
from torch import nn

from pinn_maxwell.fields import Z0_OHM, gaussian_projection, poynting, propagate
from pinn_maxwell.geometry import Waveguide
from pinn_maxwell.model import MaxwellTEPINN
from pinn_maxwell.reference import symmetric_slab, transfer_reference


def slab():
    return Waveguide(1.55, (-2., -.25, .25, 2.), (1.45, 2., 1.45))


class PhysicsTests(unittest.TestCase):
    def test_independent_references(self):
        g = slab()
        x = np.linspace(-2, 2, 1001)
        n1, e1, q1 = symmetric_slab(g, x)
        n2, e2, q2 = transfer_reference(g, x)
        self.assertAlmostEqual(n1, n2, places=11)
        np.testing.assert_allclose(e1, e2, atol=1e-11)
        np.testing.assert_allclose(q1, q2, atol=1e-11)
        self.assertTrue(g.n_lower < n1 < g.n_upper)

    def test_exact_mode_satisfies_pinn_equations(self):
        # Substitute an analytic core only to test residual signs and BCs.
        g = slab()
        n, _, _ = symmetric_slab(g, np.array([0.]))
        kx = g.k0*np.sqrt(g.indices[1]**2-n*n)

        class ExactCore(nn.Module):
            def forward(self, x):
                return torch.cat((torch.cos(kx*x), -kx/g.k0*torch.sin(kx*x)), 1)

        model = MaxwellTEPINN(g)
        model.layers = nn.ModuleList([ExactCore()])
        ratio = (n-g.n_lower)/(g.n_upper-g.n_lower)
        with torch.no_grad():
            model.raw_neff.copy_(torch.tensor(np.log(ratio/(1-ratio))))
        terms = model.loss_terms([torch.linspace(-.249, .249, 101, dtype=torch.float64).reshape(-1, 1)])
        for value in terms.values():
            self.assertLess(float(value), 1e-20)

    def test_complex_fields_and_power(self):
        g = slab()
        x = np.linspace(-2, 2, 1001)
        n, e, q = symmetric_slab(g, x)
        z = np.array([0., 1., 5.])
        electric, magnetic = propagate(e, q, n, g.wavelength_um, z, amplitude=2.)
        np.testing.assert_allclose(magnetic[..., 0], -n*electric[..., 1]/Z0_OHM)
        np.testing.assert_allclose(magnetic[0, :, 2], -2j*q/Z0_OHM)
        s = poynting(electric, magnetic)
        np.testing.assert_allclose(s[..., 2], n*np.abs(electric[..., 1])**2/(2*Z0_OHM))
        self.assertTrue(np.all(s[..., 2] >= 0))
        np.testing.assert_allclose(s[0], s[-1], atol=1e-16)
        np.testing.assert_allclose(electric[1, :, 1], electric[0, :, 1]*np.exp(1j*n*g.k0))
        # Verify curl E = i k0 Z0 H, including both x and z components.
        np.testing.assert_allclose(-1j*n*g.k0*electric[0, :, 1], 1j*g.k0*Z0_OHM*magnetic[0, :, 0])
        np.testing.assert_allclose(2*g.k0*q, 1j*g.k0*Z0_OHM*magnetic[0, :, 2])

    def test_projection_and_interface_continuity(self):
        g = slab()
        x = np.linspace(-2, 2, 2001)
        n, e, q = symmetric_slab(g, x)
        coefficient, fraction, _ = gaussian_projection(x, e, .5)
        self.assertGreater(coefficient, 0)
        self.assertTrue(0 < fraction <= 1)
        for edge in (-.25, .25):
            _, ee, qq = symmetric_slab(g, np.array([edge-1e-9, edge+1e-9]))
            self.assertLess(abs(ee[0]-ee[1]), 1e-7)
            self.assertLess(abs(qq[0]-qq[1]), 1e-7)

    def test_geometry_rejects_invalid_physics(self):
        for indices in ((1.45, 1., 1.45), (1.45, -2., 1.45)):
            with self.assertRaises(ValueError):
                Waveguide(1.55, (-2., -.25, .25, 2.), indices)
        with self.assertRaises(ValueError):
            Waveguide(float("nan"), (-2., -.25, .25, 2.), (1.45, 2., 1.45))


if __name__ == "__main__":
    unittest.main()
