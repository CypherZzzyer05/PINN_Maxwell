import unittest

import numpy as np
from scipy.special import kve
import torch

from pinn_maxwell.fiber import FiberGeometry, FiberPINN
from pinn_maxwell.fiber_fields import fiber_fields, linear_basis, maxwell_diagnostics, stokes
from pinn_maxwell.fiber_reference import BesselReference


class FiberPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = FiberGeometry()
        cls.reference = BesselReference(cls.g)
        torch.set_num_threads(1)

    def test_full_vector_reference_satisfies_maxwell(self):
        points = [[.1, .2, 0.], [.6, .1, .7], [1., .1, 1.], [2., .3, 2.]]
        for residual in maxwell_diagnostics(self.reference, points).values():
            self.assertLess(residual, 1e-7)
        self.assertTrue(self.g.n_lower < self.reference.neff < self.g.indices[0])
        self.assertLess(self.reference.smallest_singular_value, 1e-10)

    def test_interface_conditions_and_normal_e_jump(self):
        for i, radius in enumerate(self.g.interface_radii_um):
            ei, hi = fiber_fields(self.reference, radius, 0., layer=i, cylindrical=True)
            eo, ho = fiber_fields(self.reference, radius, 0., layer=i+1, cylindrical=True)
            np.testing.assert_allclose(ei[1:], eo[1:], atol=1e-10)
            np.testing.assert_allclose(hi, ho, atol=1e-12)
            np.testing.assert_allclose(self.g.indices[i]**2*ei[0], self.g.indices[i+1]**2*eo[0], atol=1e-10)
            ratio = abs(eo[0]/ei[0])
            self.assertAlmostEqual(ratio, self.g.indices[i]**2/self.g.indices[i+1]**2, places=9)
        self.assertGreater(abs(eo[0]-ei[0]), 1e-7)

    def test_circular_excitation_is_quadrature_of_orthogonal_modes(self):
        x, y = np.array([0., .1, .5, 1.]), np.array([0., .2, .3, .1])
        ex, hx, ey, hy = linear_basis(self.reference, x, y)
        ec, hc = fiber_fields(self.reference, x, y)
        np.testing.assert_allclose(ec, (ex+1j*ey)/np.sqrt(2), atol=1e-13)
        np.testing.assert_allclose(hc, (hx+1j*hy)/np.sqrt(2), atol=1e-13)
        np.testing.assert_allclose(ex[0], [1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(ey[0], [0., 1., 0.], atol=1e-12)
        for sign in (-1, 1):
            e, _ = fiber_fields(self.reference, 0., 0., handedness=sign)
            self.assertAlmostEqual(stokes(e)[3]/stokes(e)[0], sign, places=12)

    def test_axis_regularity_and_phase(self):
        model = FiberPINN(self.g)
        for phi in np.linspace(0, 2*np.pi, 9):
            x, y = 1e-8*np.cos(phi), 1e-8*np.sin(phi)
            e, h = fiber_fields(model, x, y)
            e_axis, h_axis = fiber_fields(model, 0., 0.)
            np.testing.assert_allclose(e, e_axis, atol=1e-7)
            np.testing.assert_allclose(h, h_axis, atol=1e-9)
        e0, h0 = fiber_fields(self.reference, .2, .3)
        e1, h1 = fiber_fields(self.reference, .2, .3, 2.)
        phase = np.exp(2j*self.g.k0*self.reference.neff)
        np.testing.assert_allclose(e1, e0*phase, atol=1e-12)
        np.testing.assert_allclose(h1, h0*phase, atol=1e-12)

    def test_differentiable_open_boundary_matches_scipy(self):
        model = FiberPINN(self.g)
        x = torch.tensor([[1.], [2.], [5.], [10.], [20.]], dtype=torch.float64, requires_grad=True)
        k0, k1 = model.scaled_k(x)
        ratio = k0/k1
        np.testing.assert_allclose(ratio.detach().numpy().ravel(), kve(0, x.detach().numpy().ravel())/kve(1, x.detach().numpy().ravel()), rtol=1e-12)
        grad = torch.autograd.grad(ratio.sum(), x)[0].detach().numpy().ravel()
        xx = x.detach().numpy().ravel()
        numerical = ((kve(0, xx+1e-5)/kve(1, xx+1e-5))-(kve(0, xx-1e-5)/kve(1, xx-1e-5)))/(2e-5)
        np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-9)

    def test_open_cladding_extension_satisfies_full_maxwell(self):
        model = FiberPINN(self.g)
        for value in maxwell_diagnostics(model, [[2., .3, .2], [3., .1, .8]]).values():
            self.assertLess(value, 1e-7)

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            FiberGeometry(indices=(1.44, 1.6, 1.45))
        with self.assertRaises(ValueError):
            FiberGeometry(radius_um=1.)
        with self.assertRaises(ValueError):
            FiberGeometry(handedness=0)


if __name__ == "__main__":
    unittest.main()
