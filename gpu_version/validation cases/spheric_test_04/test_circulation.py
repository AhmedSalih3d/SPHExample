"""Regression checks for the Case 04 experimental measurement operators."""
import unittest

import numpy as np

from compare_circulation import CREST, CREST_HEIGHT, Vortex, derivative


class VortexMeasurements(unittest.TestCase):
    def test_five_point_derivative(self):
        x = np.arange(12, dtype=float)
        values = (x**4)[None, :]
        valid = np.ones_like(values, dtype=bool)
        valid[:, :2] = False
        measured = derivative(values, valid, 1, 1)
        np.testing.assert_allclose(measured[0, 2:], 4*x[2:]**3)
        self.assertTrue(np.isnan(measured[0, :2]).all())

    def test_own_peak_crest_sign_and_moments(self):
        spacing = 0.005
        xx, yy = np.meshgrid(CREST + np.arange(-30, 31)*spacing,
                             CREST_HEIGHT + np.arange(-5, 16)*spacing)
        omega = np.zeros_like(xx)
        omega[8, 36:39] = [-1, -10, -1]
        omega[8, 9] = -1000       # neighbouring ripple
        omega[10, 37] = 200       # opposite sign
        track = Vortex(-1)
        gamma, x, y, radius, peak = track.measure(xx, yy, omega, spacing, 0, True)
        np.testing.assert_allclose((gamma, x, y, radius, peak),
                                   (12*spacing**2, 7*spacing, 3*spacing, spacing/np.sqrt(6), 10))
        omega[8, 36:39] = 0
        self.assertTrue(np.isnan(track.measure(xx, yy, omega, spacing, 0, False)).all())


if __name__ == "__main__":
    unittest.main()
