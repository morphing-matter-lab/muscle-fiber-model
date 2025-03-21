import numpy as np
import triangle as tr
import polyscope as ps
import fabsim_py
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

vertices = [
[4.116003, 1.403990],
[4.034014, 1.537785],
[4.176056, 1.259017],
[3.678986, 1.841007],
[3.534014, 1.901057],
[3.932104, 1.657107],
[3.812784, 1.759017],
[3.381433, 1.937688],
[3.224997, 1.950000],
[0.000000, 1.950000],
[4.224998, 0.000000],
[4.224998, 0.950000],
[4.212687, 1.106434],
[-4.116003, 1.403990],
[-4.034014, 1.537785],
[-4.176056, 1.259017],
[-3.678986, 1.841007],
[-3.534014, 1.901057],
[-3.932104, 1.657107],
[-3.812784, 1.759017],
[-3.381433, 1.937688],
[-3.224997, 1.950000],
[-4.224998, 0.000000],
[-4.224998, 0.950000],
[-4.212687, 1.106434],
[4.116003, -1.403990],
[4.034014, -1.537785],
[4.176056, -1.259017],
[3.678986, -1.841007],
[3.534014, -1.901057],
[3.932104, -1.657107],
[3.812784, -1.759017],
[3.381433, -1.937688],
[3.224997, -1.950000],
[0.000000, -1.950000],
[4.224998, -0.950000],
[4.212687, -1.106434],
[-4.116003, -1.403990],
[-4.034014, -1.537785],
[-4.176056, -1.259017],
[-3.678986, -1.841007],
[-3.534014, -1.901057],
[-3.932104, -1.657107],
[-3.812784, -1.759017],
[-3.381433, -1.937688],
[-3.224997, -1.950000],
[-4.224998, -0.950000],
[-4.212687, -1.106434]
]
segments = [[11,12],[5,6],[7,8],[3,4],[3,6],[8,9],[0,1],[0,2],[4,7],[1,5],[10,11],[2,12],[23,24],[18,19],[20,21],[16,17],[16,19],[21,9],[13,14],[13,15],[17,20],[14,18],[22,23],[15,24],[35,36],[30,31],[32,33],[28,29],[28,31],[33,34],[25,26],[25,27],[29,32],[26,30],[10,35],[27,36],[46,47],[42,43],[44,45],[40,41],[40,43],[45,34],[37,38],[37,39],[41,44],[38,42],[22,46],[39,47]]
n_boundary = len(vertices)
n_circle = 16
def make_circle(radius, centerX, centerY):
  n = len(vertices)
  for i in range(n_circle):
    vertices.append([centerX + radius * np.cos(2 * i / n_circle * np.pi), centerY + radius * np.sin(2 * i / n_circle * np.pi)])
    segments.append([n + i, n + ((i + 1) % n_circle)])

make_circle(0.75, -4.55/2, 0)
make_circle(0.7, 4.55/2, 0)
holes = [[-2.25, 0], [2.25, 0]]

A = dict(vertices=vertices, segments=segments, holes=holes)
B = tr.triangulate(A, 'pqa0.1')

# Initialize polyscope
ps.init()
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode("shadow_only")

zeros = np.zeros((len(B['vertices']), 1))
P = np.hstack((np.array(B['vertices']), zeros))
F = np.array(B['triangles'])

fixed_dofs = []
for i in range(n_boundary, n_boundary + len(holes) * n_circle):
  fixed_dofs.append(3 * i)
  fixed_dofs.append(3 * i + 1)
  fixed_dofs.append(3 * i + 2)

# Phi = np.zeros((F.shape[0], 6))

# V = P.copy()
# fabsim_py.simulate_membrane(V, P, F, Phi, fixed_dofs, 1.5, 0.25, 1.0, e0, e1)
# # V = NV[:, :2]
# ps_mesh = ps.register_surface_mesh("initial", P, F, smooth_shade=True, enabled=False)
# ps_mesh = ps.register_surface_mesh("deformed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))

# angles, eigenvalues = fabsim_py.compute_stretch_angles(V, P, F)
# angles = np.array(angles) + np.pi / 2
# dirs = np.empty((len(angles), 2))
# for i in range(len(angles)):
#   dirs[i, 0] = np.cos(angles[i])
#   dirs[i, 1] = np.sin(angles[i])

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

dt = 1 / 24
k0 = 1e-4
k1 = 5e-2
kd = 0.1 # rate of dissociation
e0 = 1.2e-1
e1 = 1.7e-1
frac_f = 0.7
frac_s = 0.25
n = 16


def plot_convergence(face_id, stretch_factor=1.5, n=4):
  sigmas = fabsim_py.fiber_stress(V, P / stretch_factor, F, n)

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
def polar_plot_phi(face_id, stretch_factor=1.5):
  n = 16
  sigmas = 1.0 * fabsim_py.fiber_stress(V, P / stretch_factor, F, n)
  phi_t = np.zeros(n)

  for i in range(1000):
    kf = k0 * np.ones(n) + k1 * phi_t * sigmas[face_id] # rate of formation
    I = np.identity(n)
    M = (1 + dt * kd) * I + dt / frac_f / n * np.array(kf, ndmin=2).T.dot(np.ones((1, n)))
    b = phi_t + dt / frac_f * (1 - frac_s - frac_f) * kf
    phi_t = np.linalg.solve(M, b)

  fig, axs = plt.subplots(2, subplot_kw={'projection': 'polar'})
  axs[0].plot([np.pi / n * i for i in range(3 * n)], np.tile(sigmas[face_id], 3))
  axs[0].set_title('Sigma')
  axs[1].plot([np.pi / n * i for i in range(3 * n)], np.tile(phi_t, 3), 'tab:orange')
  axs[1].set_title('Phi')

  print(phi_t.mean())

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

# sigmas = fabsim_py.fiber_stress(V, P / 1.5, F, 100)

# for i in range(F.shape[0]):
#   polar_plot_phi(i)

def polygons(polymer_frac, scale):
  n = polymer_frac.shape[1]
  polygon = []
  rotated_polygon = []
  for i in range(n):
    angle = np.pi / n * i
    polygon.append(np.array([np.cos(angle), np.sin(angle), 0.0]))
    rotated_polygon.append(np.array([np.cos(angle + np.pi), np.sin(angle + np.pi), 0.0]))
  polygon = scale * np.array(polygon)
  rotated_polygon = scale * np.array(rotated_polygon)

  # faces = np.arange(F.shape[0] * 2 * n).reshape(F.shape[0], 2 * n)
  polygon_face = np.zeros((2 * n, 3))
  polygon_face[:, 1] = np.arange(1, 2 * n + 1)
  polygon_face[:, 2] = np.arange(2, 2 * n + 2)
  polygon_face[2 * n - 1, 2] = 1

  verts = []
  faces = []
  for i in range(F.shape[0]):
    center = (V[F[i, 0], :] + V[F[i, 1], :] + V[F[i, 2], :]) / 3
    verts.append(center.reshape((1, 3)))
    verts.append(polygon * polymer_frac[i, :][:, None] + center)
    verts.append(rotated_polygon * polymer_frac[i, :][:, None] + center)
    faces.append(polygon_face + i * (2 * n + 1))
  verts = np.concatenate(verts, axis=0)
  verts[:, 2] = 0.1
  faces = np.concatenate(faces, axis=0)

  return verts, faces

ps_mesh = ps.register_surface_mesh("initial", P, F, smooth_shade=True, enabled=False)
ps.set_navigation_style("planar")
ps.set_view_projection_mode("orthographic")
ps.set_SSAA_factor(3)
ps.set_view_from_json('{"farClipRatio":20.0,"fov":16.0,"nearClipRatio":0.005,"projectionMode":"Orthographic","viewMat":[1.0,-0.0,0.0,-0.0,0.0,0.997785151004791,-0.0665190145373344,0.000246047973632812,-0.0,0.0665190145373344,0.997785151004791,-25.5602951049805,0.0,0.0,0.0,1.0],"windowHeight":1964,"windowWidth":3456}')
polymer_frac = np.zeros((F.shape[0], n))
V = P.copy()

for k in range(9, 10):
  stretch_factor = 1 + 0.5 * (k + 1) / 10.
  fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
  ps_mesh = ps.register_surface_mesh("deformed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))
  sigmas = 1.0 * fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
  
  phi = fabsim_py.polymer_fraction_reduced(sigmas, k1 / k0, kd / k0, frac_f, frac_s)

  # phi = fabsim_py.polymer_fraction_steady_state(sigmas, k0, k1, kd, frac_f, frac_s)
  # verts, faces = polygons(phi, 0.5)
  # ps.register_surface_mesh("polymer frac", verts, faces, enabled=True, color=(213/255, 202/255, 42/255))
  # ps.show()
  # ps.screenshot()


  # phi_t = np.zeros(n)
  # for i in range(1000):
  #   kf = k0 * np.ones(n) + k1 * phi_t * sigmas[8] # rate of formation
  #   I = np.identity(n)
  #   M = (1 + dt * kd) * I + dt / frac_f / n * np.array(kf, ndmin=2).T.dot(np.ones((1, n)))
  #   b = phi_t + dt / frac_f * (1 - frac_s - frac_f) * kf
  #   phi_t = np.linalg.solve(M, b)

  # fig, axs = plt.subplots(2, subplot_kw={'projection': 'polar'})

  # phi_b = k0 / (kd - k1 * sigmas[8])
  # axs[0].plot([np.pi / n * i for i in range(3 * n)], np.tile(phi_b, 3))
  # axs[0].set_title('Phi B')
  # axs[1].plot([np.pi / n * i for i in range(3 * n)], np.tile(phi_t, 3), 'tab:orange')
  # axs[1].set_title('Phi A')
  # plt.show()

  for i in range(1000):
    fabsim_py.polymer_fraction_one_step(polymer_frac, sigmas, k0, k1, kd, frac_f, frac_s, dt)
    fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
    sigmas = 1.0 * fabsim_py.fiber_stress(V, P / stretch_factor, F, n)

  # polar_plot_phi(8)
  # polar_plot_phi(392)

  verts, faces = polygons(polymer_frac, 0.5)
  ps.register_surface_mesh("polymer frac", verts, faces, enabled=True, color=(213/255, 202/255, 42/255))
  ps.show()
  ps.screenshot()

# def stress(strain):
#   stress = np.exp(-(strain / e0)**2)
#   stress += np.heaviside(strain, 1) * (strain / e1)**2


def fitting(strains, phi_measured):
  def fun(params):
    k1 = params[0]
    kd = params[1]
    e0 = params[2]
    e1 = params[3]

    stress = 1.0 * fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
    phi = fabsim_py.polymer_fraction_reduced(stress, k1 / k0, kd / k0, frac_f, frac_s)

    return (phi - phi_measured).flatten()

  initial_guess = np.array([5e-2 / 1e-4, 0.1 / 1e-4, 1.2e-1, 1.7e-1])
  return least_squares(fun, initial_guess)


phi_measured = polymer_frac + 0.01 * np.random.default_rng().random(polymer_frac.shape)
stress = 1.0 * fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
phi = fabsim_py.polymer_fraction_reduced(stress, k1 / k0, kd / k0, frac_f, frac_s)
print(phi)
print(polymer_frac)
print(np.linalg.norm((phi - polymer_frac) / phi))


strains = fabsim_py.directional_strain(V, P, F, n)
params = fitting(strains, phi_measured)
print(params.x)
print(k1 / k0, kd / k0, e0, e1)
print(params.fun.reshape(phi_measured.shape) - phi_measured)

verts, faces = polygons(phi_measured, 0.5)
ps.register_surface_mesh("polymer frac", verts, faces, enabled=True, color=(213/255, 202/255, 42/255))
ps.show()
ps.screenshot()

k1 = params.x[0]
kd = params.x[1]
e0 = params.x[2]
e1 = params.x[3]

polymer_frac = np.zeros((F.shape[0], n))
for i in range(10000):
  sigmas = 1.0 * fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
  fabsim_py.polymer_fraction_one_step(polymer_frac, sigmas, k0, k0 * k1, k0 * kd, frac_f, frac_s, dt)
  fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)

ps_mesh = ps.register_surface_mesh("deformed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))
verts, faces = polygons(polymer_frac, 0.5)
ps.register_surface_mesh("polymer frac", verts, faces, enabled=True, color=(213/255, 202/255, 42/255))
ps.screenshot()
ps.show()