import unittest

import numpy as np

from audit_simple_mdbc import reconstruct


class FrozenReconstructionTests(unittest.TestCase):
    def test_constant_and_linear_density_fields(self):
        h = 0.03
        alpha = 7 / (4*np.pi*h*h)
        grid = np.array([(x, y) for x in np.arange(-3, 4)*0.0125
                         for y in np.arange(-3, 4)*0.0125])
        fluid = grid[np.linalg.norm(grid, axis=1) < 2*h]
        ghost = np.zeros(2)
        wall = np.array([-0.025, 0.0])
        for slope in (np.zeros(2), np.array([3.0, -2.0])):
            rho = 1000 + fluid @ slope
            result = reconstruct(wall, ghost, fluid, rho, h, 0.15625, alpha)
            self.assertEqual(result["branch"], "MLS")
            self.assertAlmostEqual(result["density"], 1000 + wall @ slope, places=9)
            self.assertAlmostEqual(result["ghost_density"], 1000, places=9)
            self.assertAlmostEqual(result["extrapolation"], wall @ slope, places=9)
            self.assertAlmostEqual(result["amplification"], result["perturbation_gain"], places=5)
            self.assertGreaterEqual(result["shepard"], rho.min()-1e-10)
            self.assertLessEqual(result["shepard"], rho.max()+1e-10)

    def test_low_support_uses_shepard(self):
        h = 0.03
        ghost = np.zeros(2)
        fluid = np.array([[0.01, 0], [0.02, 0]])
        rho = np.array([1000.0, 1000.1])
        result = reconstruct(np.array([-0.01, 0]), ghost, fluid, rho, h, 0.15625, 7/(4*np.pi*h*h))
        self.assertEqual(result["branch"], "Shepard")
        self.assertEqual(result["density"], result["shepard"])
        self.assertGreaterEqual(result["density"], rho.min())
        self.assertLessEqual(result["density"], rho.max())
        self.assertAlmostEqual(result["amplification"], 1)


if __name__ == "__main__":
    unittest.main()
