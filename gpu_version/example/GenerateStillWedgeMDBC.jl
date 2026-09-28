using SPHExampleGPU
using Meshes

Geom1 = Triangle((0, 0), (1, 0), (0, 1))

outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
hole1 = [(0.2, 0.2), (0.2, 0.4), (0.4, 0.4), (0.4, 0.2)]
hole2 = [(0.6, 0.2), (0.6, 0.4), (0.8, 0.4), (0.8, 0.2)]
Geom2 = PolyArea([outer, hole1, hole2])