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
P = np.hstack((np.array(B['vertices']), zeros))
F = np.array(B['triangles'])

fixed_dofs = []
for i in range(4, 4 + len(holes) * n_circle):
  fixed_dofs.append(i)

V, F = fabsim_py.simulate_membrane(P, F, fixed_dofs, 1.5, 0.25)
# V = NV[:, :2]
ps_mesh = ps.register_surface_mesh("initial", P, F, smooth_shade=True, enabled=False)
ps_mesh = ps.register_surface_mesh("deformed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))

angles, eigenvalues = fabsim_py.compute_stretch_angles(V, P, F)
angles = np.array(angles) + np.pi / 2
# alignment = eigenvalues[:,0] /  eigenvalues[:,1]
dirs = np.empty((len(angles), 2))
for i in range(len(angles)):
  dirs[i, 0] = np.cos(angles[i])
  dirs[i, 1] = np.sin(angles[i])

# ps_mesh.add_vector_quantity("strain directions", dirs, defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)
# ps_mesh.add_vector_quantity("strain directions2", -dirs, defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)

# for i in range(4):
#   angle = np.pi / 4 * i
#   dirs = np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), F.shape[0], axis=0)
#   sigma = fabsim_py.directional_fiber_stress(V, P, F, angle)
#   dirs = dirs * sigma.reshape(sigma.shape[0], 1)
#   ps_mesh.add_vector_quantity(f"stress{i * 45}", dirs, defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)
#   ps_mesh.add_vector_quantity(f"stress{i * 45 + 180}", -dirs, defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)

# ps.show()

dt = 1 / 12
kd = 0.1 # rate of dissociation
k0 = 1e-4
k1 = 5e-2
frac_f = 0.7
frac_s = 0.25
n = 4

sigmas = fabsim_py.fiber_stress(V, P, F, n)
print(sigmas.shape)
polymer_frac = np.zeros(sigmas.shape)

for i in range(1000):
  fabsim_py.polymer_fraction_one_step(polymer_frac, sigmas, k0, k1, kd, frac_f, frac_s, dt)
  
dirs = []
for i in range(n):
  angle = np.pi / n * i
  dirs.append(np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), F.shape[0], axis=0))
dirs = np.array(dirs)

dirs[:, :, 0] = dirs[:, :, 0] * polymer_frac.T
dirs[:, :, 1] = dirs[:, :, 1] * polymer_frac.T

for i in range(4):
  ps_mesh.add_vector_quantity(f"frac_p{i * 45}", dirs[i, :, :], defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)
  ps_mesh.add_vector_quantity(f"frac_p{i * 45 + 180}", -dirs[i, :, :], defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)

ps.show()

def plot_convergence(face_id):
  sigmas = []
  for i in range(n):
    angle = np.pi / n * i
    sigma = fabsim_py.directional_fiber_stress(V, P, F, angle)  
    sigmas.append(sigma[face_id])
  sigmas = np.array(sigmas, ndmin=2)
  kf = k0 * np.ones((n, 1)) + k1 * sigmas.T # rate of formation

  phi_t = [np.array([0, 0, 0, 0])]
  I = np.identity(n)
  M = (1 + dt * kd) * I + dt / frac_f / n * kf.dot(np.ones((1, n)))
  print(M)

  for i in range(1000):
    b = phi_t[-1] + dt / frac_f * (1 - frac_s - frac_f) * kf.reshape(-1)
    x = np.linalg.solve(M, b)
    if np.allclose(np.dot(M, x), b):
      phi_t.append(x)
    else:
      print(f"Error: {np.linalg.norm(M * x)} != {np.linalg.norm(b)}")
      break

  phi_t = np.array(phi_t)

  import matplotlib.pyplot as plt
  plt.plot(phi_t[:, 0])
  plt.plot(phi_t[:, 1])
  plt.plot(phi_t[:, 2])
  plt.plot(phi_t[:, 3])
  plt.show()