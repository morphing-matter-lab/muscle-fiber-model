import numpy as np
import triangle as tr
import polyscope as ps
import fabsim_py

vertices = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
holes = []

n_circle = 15
def make_circle(radius, centerX, centerY):
  n = len(vertices)
  for i in range(n_circle):
    vertices.append([centerX + radius * np.cos(2 * i / n_circle * np.pi), centerY + radius * np.sin(2 * i / n_circle * np.pi)])
    segments.append([n + i, n + ((i + 1) % n_circle)])

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
for i in range(4, 4 + 4 * n_circle):
  fixed_dofs.append(3 * i + 0)
  fixed_dofs.append(3 * i + 1)
  fixed_dofs.append(3 * i + 2)

NV, F = fabsim_py.simulate_membrane(V, np.array(B['triangles']), fixed_dofs, 1.5, 0.5)
ps_mesh = ps.register_surface_mesh("my mesh", NV, F, smooth_shade=True)

# angles = fabsim_py.compute_stretch_angles(V, NV[:, :2], F)
# dirs = np.empty((len(angles), 2))
# for i in range(len(angles)):
#   dirs[i, 0] = np.cos(angles[i])
#   dirs[i, 1] = np.sin(angles[i])
# # dirs = fabsim_py.compute_membrane_forces(V, np.array(B['triangles']), NV, 1.5, 0.3)

# vec = fabsim_py.compute_membrane_energies(V, np.array(B['triangles']), NV, 1.5, 0.5)

# ps_mesh.add_scalar_quantity("energy values", vec, defined_on="faces", enabled=True)
# ps_mesh.add_vector_quantity("force directions", dirs, defined_on="faces", enabled=True)

ps.show()
