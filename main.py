import numpy as np
import triangle as tr
import polyscope as ps
import fabsim_py

vertices = [[-4, -7.5], [4, -7.5], [4, 7.5], [-4, 7.5]]
segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
holes = []

n_circle = 25
def make_circle(radius, centerX, centerY):
  n = len(vertices)
  for i in range(n_circle):
    vertices.append([centerX + radius * np.cos(2 * i / n_circle * np.pi), centerY + radius * np.sin(2 * i / n_circle * np.pi)])
    segments.append([n + i, n + ((i + 1) % n_circle)])

make_circle(2, 0, -4.5)
holes.append([0, -4.5])

make_circle(2, 0, 4.5)
holes.append([0, 4.5])

A = dict(vertices=vertices, segments=segments, holes=holes)
B = tr.triangulate(A, 'pqa0.5')

# Initialize polyscope
ps.init()
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode("shadow_only")

zeros = np.zeros((len(B['vertices']), 1))
V = np.hstack((np.array(B['vertices']), zeros))

fixed_dofs = []
for i in range(4, 4 + len(holes) * n_circle):
  fixed_dofs.append(i)

NV, F = fabsim_py.simulate_membrane(V, np.array(B['triangles']), fixed_dofs, 1.5, 0.25)
ps_mesh = ps.register_surface_mesh("initial", V, F, smooth_shade=True, enabled=False)
ps_mesh = ps.register_surface_mesh("deformed", NV, F, smooth_shade=True, color=(42/255, 53/255, 213/255))

angles, eigenvalues = fabsim_py.compute_stretch_angles(V, NV[:, :2], F)
# alignment = eigenvalues[:,0] /  eigenvalues[:,1]
dirs = np.empty((len(angles), 2))
for i in range(len(angles)):
  dirs[i, 0] = np.cos(angles[i])
  dirs[i, 1] = np.sin(angles[i])

ps_mesh.add_vector_quantity("strain directions", dirs, defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)
ps_mesh.add_vector_quantity("strain directions2", -dirs, defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)

ps.show()
