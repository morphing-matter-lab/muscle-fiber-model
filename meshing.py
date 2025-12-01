import numpy as np
import polyscope as ps
import fabsim_py
import pyvista as pv

vertices = [
[-4.428360, 3.451138],
[-4.267949, 3.464089],
[4.267949, 3.464089],
[-4.584617, 3.412629],
[-4.732672, 3.349545],
[-4.868691, 3.263532],
[-4.989151, 3.156816],
[-5.090932, 3.032155],
[-5.171400, 2.892785],
[5.260658, 2.584626],
[5.171400, 2.892785],
[5.090934, 3.032155],
[4.989152, 3.156816],
[4.868691, 3.263532],
[4.732673, 3.349545],
[4.584618, 3.412629],
[4.428361, 3.451138],
[-0.534466, -5.773404],
[-0.391966, -5.848193],
[0.239316, -5.899156],
[0.391967, -5.848193],
[0.534467, -5.773404],
[0.663123, -5.676724],
[5.228468, 2.742307],
[-5.247739, 2.264066],
[-5.228467, 2.742307],
[-5.260658, 2.584626],
[-5.267138, 2.423825],
[-0.239316, -5.899156],
[-0.080466, -5.924972],
[0.774605, -5.560660],
[0.866026, -5.428214],
[5.133975, 1.964089],
[-5.133975, 1.964089],
[-0.866025, -5.428214],
[-0.774605, -5.560660],
[-0.663122, -5.676724],
[0.080467, -5.924972],
[-5.202965, 2.109487],
[5.267138, 2.423825],
[5.247740, 2.264066],
[5.202966, 2.109487]
]
segments = [[5, 6], [15,16], [19,20], [9,23], [19,37], [9,39], [32,41], [0, 1], [0, 3], [6, 7], [20,21], [28,29], [33,34], [33,38], [1, 2], [7, 8], [24,27], [34,35], [29,37], [24,38], [10,11], [2,16], [21,22], [10,23], [8,25], [35,36], [11,12], [22,30], [39,40], [12,13], [17,18], [25,26], [30,31], [17,36], [3, 4], [13,14], [26,27], [18,28], [31,32], [40,41], [4, 5], [14, 15]]
n_boundary = len(vertices)
def make_circle(radius, centerX, centerY, n_circle):
  n = len(vertices)
  for i in range(n_circle):
    vertices.append([centerX + radius * np.cos(2 * i / n_circle * np.pi), centerY + radius * np.sin(2 * i / n_circle * np.pi)])
    segments.append([n + i, n + ((i + 1) % n_circle)])

# make_circle(0.75, -2.275, 2.627/2, 16)
# make_circle(0.75, 2.275, 2.627/2, 16)
# make_circle(0.75, 0, -2.627, 16)
make_circle(0.75, -2.275, 2.627/2, 24)
make_circle(0.75, 2.275, 2.627/2, 24)
make_circle(0.75, 0, -2.627, 24)
holes = [[-2.25, 1.314], [2.25, 1.314], [0, -2.63]]

P, F = fabsim_py.triangulate(np.array(vertices, dtype=np.float64), np.array(segments, dtype=np.int32), np.array(holes, dtype=np.int32), "pqa0.025")
# P, F = fabsim_py.triangulate(np.array(vertices, dtype=np.float64), np.array(segments, dtype=np.int32), np.array(holes, dtype=np.int32), "pqa0.1")

zeros = np.zeros((P.shape[0], 1))
P = np.hstack((np.array(P), zeros))

faces = np.hstack((3 * np.ones((F.shape[0], 1), dtype=np.int32), F)).flatten()
mesh = pv.PolyData(P, faces)
# mesh.save("top_surface_3posts_.obj")
mesh.save("top_surface_3posts_highres.obj")


ps.init()
ps_mesh = ps.register_surface_mesh("Initial mesh", P, F, smooth_shade=True, enabled=True)
ps.show()
