import numpy as np
import triangle as tr
import polyscope as ps
import fabsim_py

vertices = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
holes = []

def make_circle(radius, centerX, centerY):
  n = len(vertices)
  k = 10
  for i in range(k):
    vertices.append([centerX + radius * np.cos(2 * i / k * np.pi), centerY + radius * np.sin(2 * i / k * np.pi)])
    segments.append([n + i, n + ((i + 1) % k)])

make_circle(1.25, -8, -8)
holes.append([-8, -8])

make_circle(1.25, 8, -8)
holes.append([8, -8])

make_circle(1.25, 8, 8)
holes.append([8, 8])

make_circle(1.25, -8, 8)
holes.append([-8, 8])

A = dict(vertices=vertices, segments=segments, holes=holes)
B = tr.triangulate(A, 'pqa1')

# Initialize polyscope
ps.init()
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode("shadow_only")

zeros = np.zeros((len(B['vertices']), 1))
V = np.hstack((np.array(B['vertices']), zeros))

fixed_dofs = []
for i in range(4):
  fixed_dofs.append(3 * i + 0)
  fixed_dofs.append(3 * i + 1)
  fixed_dofs.append(3 * i + 2)

V, F = fabsim_py.simulate_membrane(V, np.array(B['triangles']), fixed_dofs, 0.7, 0.3)
ps.register_surface_mesh("my mesh", V, F, smooth_shade=True)

ps.show()
