import numpy as np
import triangle as tr
import polyscope as ps
import fabsim_py
import matplotlib.pyplot as plt

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
  fixed_dofs.append(3 * i)
  fixed_dofs.append(3 * i + 1)
  fixed_dofs.append(3 * i + 2)

Phi = np.zeros((F.shape[0], 6))

V = P.copy()
fabsim_py.simulate_membrane(V, P, F, Phi, fixed_dofs, 1.7, 0.25, 2e4, 0.12, 0.17)
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
n = 12

sigmas = 2e4 * fabsim_py.fiber_stress(V, P / 1.7, F, n)
print(sigmas.shape)
polymer_frac = np.zeros(sigmas.shape)

for i in range(4):
  fabsim_py.polymer_fraction_one_step(polymer_frac, polymer_frac * sigmas, k0, k1, kd, frac_f, frac_s, dt)
  fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, 1.7, 0.25, 2e4, 0.12, 0.17)
  sigmas = 2e4 * fabsim_py.fiber_stress(V, P / 1.7, F, n)

  
dirs = []
for i in range(n):
  angle = np.pi / n * i
  dirs.append(np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), F.shape[0], axis=0))
dirs = np.array(dirs)

dirs[:, :, 0] = dirs[:, :, 0] * polymer_frac.T
dirs[:, :, 1] = dirs[:, :, 1] * polymer_frac.T

for i in range(n):
  ps_mesh.add_vector_quantity(f"frac_p{i * 180 / n}", dirs[i, :, :], defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)
  ps_mesh.add_vector_quantity(f"frac_p{i * 180 / n + 180}", -dirs[i, :, :], defined_on="faces", enabled=True, color=(213/255, 202/255, 42/255), length=0.01)

ps.show()

def plot_convergence(face_id, n=4):
  sigmas = fabsim_py.fiber_stress(V, P / 1.7, F, n)

  phi_t = [np.zeros(n)]
  I = np.identity(n)

  for i in range(30000):
    kf = k0 * np.ones(n) + k1 * phi_t[-1] * sigmas[face_id, :] # rate of formation
    M = (1 + dt * kd) * I + dt / frac_f / n * np.array(kf, ndmin=2).T.dot(np.ones((1, n)))
    b = phi_t[-1] + dt / frac_f * (1 - frac_s - frac_f) * kf
    x = np.linalg.solve(M, b)
    if np.allclose(np.dot(M, x), b):
      phi_t.append(x)
    else:
      print(f"Error: {np.linalg.norm(M * x)} != {np.linalg.norm(b)}")
      break

  phi_t = np.array(phi_t)

  for i in range(n):
    plt.plot(phi_t[:, i])
  plt.show()


# test 103, 39
def polar_plot_phi(face_id):
  n = 100
  sigmas = 2e4 * fabsim_py.fiber_stress(V, P / 1.7, F, n)
  # print(sigmas[face_id, :].shape)
  # print(sigmas[face_id:face_id+1].shape)

  phi_t = [np.zeros(n)]

  for i in range(1000):
    kf = k0 * np.ones(n) + k1 * phi_t[-1] * sigmas[face_id] # rate of formation
    I = np.identity(n)
    M = (1 + dt * kd) * I + dt / frac_f / n * np.array(kf, ndmin=2).T.dot(np.ones((1, n)))
    b = phi_t[-1] + dt / frac_f * (1 - frac_s - frac_f) * kf
    x = np.linalg.solve(M, b)
    phi_t.append(x)

  phi_t = np.array(phi_t)

  # _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
  # ax.plot([np.pi / n * i for i in range(2 * n)], np.tile(phi_t[-1, :], 2))
  # ax.grid(True)
  # plt.plot([np.pi / n * i for i in range(2 * n)], np.tile(sigmas[face_id, :], 2))

  # Ds = np.array([V[F[face_id, 1], :2] - V[F[face_id, 0], :2], V[F[face_id, 2], :2] - V[F[face_id, 0], :2]]).T
  # R = np.array([P[F[face_id, 1], :2] - P[F[face_id, 0], :2], P[F[face_id, 2], :2] - P[F[face_id, 0], :2]]).T
  # deformation_gradient = Ds .dot(np.linalg.inv(R))
  # E = 0.5 * (deformation_gradient.T.dot(deformation_gradient) - np.identity(2))

  # epsilons = []
  # for i in range(2 * n):
  #   angle = np.pi / n * i
  #   u = np.array([np.cos(angle), np.sin(angle)])
  #   epsilons.append(u.dot(E.dot(u)))

  # plt.plot([np.pi / n * i for i in range(2 * n)], epsilons) 

  fig, axs = plt.subplots(2, subplot_kw={'projection': 'polar'})
  axs[0].plot([np.pi / n * i for i in range(2 * n)], np.tile(sigmas[face_id], 2))
  axs[0].set_title('Sigma')
  axs[1].plot([np.pi / n * i for i in range(2 * n)], np.tile(phi_t[-1], 2), 'tab:orange')
  axs[1].set_title('Phi')

  print(phi_t[-1].mean())

  plt.show()

def plot_approx(data):
  k = 10
  n = data.shape[0]
  rows = []
  for i in range(k):
    angle = i * np.pi / k
    rows.append([1 + 2 * np.cos(2 * angle), 1 - 2 * np.cos(2 * angle), 2 * np.sin(2 * angle)])
  A = np.array(rows)

  b = [data[m] for m in np.linspace(0, n, num=k, dtype=int, endpoint=False)]
  Phi = np.linalg.solve(A.T.dot(A), A.T.dot(b))

  thetas = np.arange(0, 2 * np.pi, 0.01)
  approx = [np.array([1 + 2 * np.cos(2 * angle), 1 - 2 * np.cos(2 * angle), 2 * np.sin(2 * angle)]).dot(Phi) for angle in thetas]

  _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
  ax.set_rmax(0.025)
  ax.plot(thetas, approx)
  ax.plot([np.pi / n * i for i in range(2 * n)], np.tile(data, 2))
  ax.grid(True)
  plt.show()

def compare_approximations(data):
  # # original idea (doesn't seem to be working)
  # k = 10
  # n = data.shape[0]
  # rows = []
  # for i in range(k):
  #   angle = i * np.pi / k
  #   rows.append([1 + 2 * np.cos(2 * angle), 1 - 2 * np.cos(2 * angle), 2 * np.sin(2 * angle)])
  # A = np.array(rows)

  # b = [data[m] for m in np.linspace(0, n, num=k, dtype=int, endpoint=False)]
  # Phi = np.linalg.solve(A.T.dot(A), A.T.dot(b))

  # thetas = np.arange(0, 2 * np.pi, 0.01)
  # approx3 = [np.array([1 + 2 * np.cos(2 * angle), 1 - 2 * np.cos(2 * angle), 2 * np.sin(2 * angle)]).dot(Phi) for angle in thetas]

  n = data.shape[0]
  Sigma = np.zeros(3)
  for i in range(n):
    angle = i * np.pi / n
    Sigma += data[i] / n * np.array([np.cos(angle)**2, np.sin(angle)**2, np.sin(2 * angle) / 2])

  thetas = np.arange(0, 2 * np.pi, 0.01)
  approx1 = [np.array([np.cos(angle)**2, np.sin(angle)**2, np.sin(2 * angle)]).dot(Sigma) for angle in thetas]

  k = 5
  Phi = np.zeros(3)
  b = [data[m] for m in np.linspace(0, n, num=k, dtype=int, endpoint=False)]
  for i in range(k):
    angle = i * np.pi / k
    Phi += b[i] / k * np.array([np.cos(angle)**2, np.sin(angle)**2, np.sin(2 * angle) / 2])

  approx2 = [np.array([np.cos(angle)**2, np.sin(angle)**2, np.sin(2 * angle)]).dot(Phi) for angle in thetas]

  _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
  ax.set_rmax(0.025)
  ax.plot(thetas, approx1)
  ax.plot(thetas, approx2)
  print(np.linalg.norm(np.array(approx1) - np.array(approx2)) / np.linalg.norm(np.array(approx1)) * 100)
  # ax.plot(thetas, approx3)
  # ax.plot([np.pi / n * i for i in range(2 * n)], np.tile(data, 2))
  ax.grid(True)
  plt.show()

# sigmas = fabsim_py.fiber_stress(V, P / 1.7, F, 100)

# for i in range(F.shape[0]):
#   polar_plot_phi(i)