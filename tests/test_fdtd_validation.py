import unittest

import numpy as np

from fdtd_validation.analysis import one_sided_trace, trapezoid_weights, weighted_relative, zero_mesh_fit
from fdtd_validation.lumerical import rotate_vector_90, vector_dataset


class FDTDAnalysisTests(unittest.TestCase):
    def test_nonuniform_area_weights(self):
        x = np.array([0., .1, .7, 1.])
        w = trapezoid_weights(x)
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertAlmostEqual(np.sum(w*x), .5)
        p = np.ones((4, 4, 3), complex)
        weight = w[:, None]*w[None, :]
        self.assertAlmostEqual(weighted_relative(1.1*p, p, weight, np.ones((4, 4), bool)), .1)

    def test_vector_rotation_includes_coordinate_and_component_rotation(self):
        x = np.linspace(-1, 1, 5)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        field = np.stack((xx, 2*yy, np.ones_like(xx)), axis=-1)
        rotated = rotate_vector_90(x, x, field)
        expected = np.stack((2*xx, yy, np.ones_like(xx)), axis=-1)
        np.testing.assert_allclose(rotated, expected, atol=1e-14)

    def test_quadratic_same_medium_trace(self):
        # f(x,0)=1+2x+3x²: exact one-sided extrapolation at r=0.5.
        x = np.linspace(-2, 2, 401)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        field = np.stack((1+2*xx+3*xx**2, 0*yy, 0*yy), axis=-1).astype(complex)
        for side in (-1, 1):
            trace = one_sided_trace(x, x, field, .5, np.array([0.]), .1, side)
            self.assertAlmostEqual(trace[0, 0].real, 2.75, places=10)

    def test_mesh_extrapolation_for_known_error_orders(self):
        mesh = np.array([80., 60., 40., 30., 25.])
        h2 = (mesh/100)**2
        values = 1.5+0.03*h2+0.002*h2**2
        intercept, _ = zero_mesh_fit(mesh, values, 2)
        self.assertAlmostEqual(intercept, 1.5, places=12)
        complex_values = np.column_stack((values+1j*values, 2*values))
        intercept, _ = zero_mesh_fit(mesh, complex_values, 2)
        np.testing.assert_allclose(intercept, [1.5+1.5j, 3.], atol=1e-12)

    def test_vendor_dataset_axes_are_not_transposed(self):
        raw = {"x": np.array([0, 1, 2])*1e-6, "y": np.array([0, 1])*1e-6,
               "E": np.arange(18).reshape(3, 2, 1, 1, 3)}
        x, y, field = vector_dataset(raw, "E")
        self.assertEqual(field.shape, (3, 2, 3))
        np.testing.assert_array_equal(field[2, 1], [15, 16, 17])
        np.testing.assert_array_equal(x, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
