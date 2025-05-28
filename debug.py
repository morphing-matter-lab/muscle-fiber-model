import numpy as np
import polyscope as ps
import fabsim_py
import pyvista as pv
import triangle as tr


dt = 1 / 24
k0 = 1e-4
k1 = 5e-2
kd = 0.1 # rate of dissociation
e0 = 1.2e-1
e1 = 1.7e-1
frac_f = 0.7
frac_s = 0.25
n = 16

# fixed_dofs = []

# def fix_dofs_in_circle(radius, center, V):
#   for i in range(V.shape[0]):
#     if np.linalg.norm(center - V[i, :2]) < radius + 1e-6:
#       if center[0] * (center[0] - V[i, 0]) < 0:
#         fixed_dofs.append(3 * i)
#         fixed_dofs.append(3 * i + 1)

# mesh = pv.read("top_surface.obj")
# P = np.array(mesh.points, dtype=np.float64)
# # P = P[:, :2]
# F = np.array(mesh.faces, dtype=np.int32)
# F = F.reshape((F.shape[0] // 4, 4))[:, 1:]

# fixed_dofs = []
# fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), P)
# fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), P)


vertices = [
  [0.000000 , 1.950000 ],
  [-4.034014,  1.537785 ],
  [-4.176056,  1.259017 ],
  [-3.534014,  1.901057 ],
  [-3.812784,  1.759017 ],
  [-3.224997,  1.950000 ],
  [-4.224998,  0.000000],
  [-4.224998,  0.950000 ],
  [4.034014 , 1.537785 ],
  [4.176056 , 1.259017 ],
  [3.534014 , 1.901057 ],
  [3.812784 , 1.759017 ],
  [3.224997 , 1.950000 ],
  [4.224998 , 0.000000],
  [4.224998 , 0.950000 ],
  [0.000000 , -1.950000],
  [-4.034014,  -1.537785],
  [-4.176056,  -1.259017],
  [-3.534014,  -1.901057],
  [-3.812784,  -1.759017],
  [-3.224997,  -1.950000],
  [-4.224998,  -0.950000],
  [4.034014 , -1.537785],
  [4.176056 , -1.259017],
  [3.534014 , -1.901057],
  [3.812784 , -1.759017],
  [3.224997 , -1.950000],
  [4.224998 , -0.950000],
]
segments = [
 [ 0,  5],
 [ 1,  2],
 [ 1,  4],
 [ 6,  7],
 [ 2,  7],
 [ 3,  4],
 [ 3,  5],
 [ 0, 12],
 [ 8,  9],
 [ 8, 11],
 [13, 14],
 [ 9, 14],
 [10, 11],
 [10, 12],
 [15, 20],
 [16, 17],
 [16, 19],
 [ 6, 21],
 [17, 21],
 [18, 19],
 [18, 20],
 [15, 26],
 [22, 23],
 [22, 25],
 [13, 27],
 [23, 27],
 [24, 25],
 [24, 26]
]
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

P, F = fabsim_py.triangulate(np.array(vertices, dtype=np.float64), np.array(segments, dtype=np.int32), np.array(holes, dtype=np.int32))

zeros = np.zeros((P.shape[0], 1))
P = np.hstack((np.array(P), zeros))

# TODO save mesh as top_surface.obj







# fixed_dofs = []
# for i in range(n_boundary, n_boundary + len(holes) * n_circle):
#   fixed_dofs.append(3 * i)
#   fixed_dofs.append(3 * i + 1)
#   fixed_dofs.append(3 * i + 2)


# # Initialize polyscope
# ps.init()
# ps.set_give_focus_on_show(True)
# ps.set_ground_plane_mode("shadow_only")

# # ps_mesh = ps.register_surface_mesh("Initial mesh", P, F, smooth_shade=True, enabled=True)
# ps.set_navigation_style("planar")
# ps.set_view_projection_mode("orthographic")
# ps.set_SSAA_factor(3)
# ps.set_view_from_json('{"farClipRatio":20.0,"fov":16.0,"nearClipRatio":0.005,"projectionMode":"Orthographic","viewMat":[1.0,-0.0,0.0,-0.0,0.0,0.997785151004791,-0.0665190145373344,0.000246047973632812,-0.0,0.0665190145373344,0.997785151004791,-25.5602951049805,0.0,0.0,0.0,1.0],"windowHeight":1964,"windowWidth":3456}')
# # ps.show()

# stretch_factor = 1.5
# polymer_frac = np.zeros((F.shape[0], n))
# V = P.copy()

# for i in range(10):
#   fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
#   stress = fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
#   polymer_frac = fabsim_py.polymer_fraction_steady_state(stress, 1, k1 / k0, kd / k0, frac_f, frac_s)
# ps.register_surface_mesh("deformed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))


# P = P[:, :2]
# V = V[:, :2]
# V, P, F = fabsim_py.remesh(V, P, F)

# zeros = np.zeros((P.shape[0], 1))
# P = np.hstack((P[:], zeros))
# V = np.hstack((V[:], zeros))

# # boundary_loops = fabsim_py.boundary_loops(F)
# # max_size = 0
# # for i in range(len(boundary_loops)):
# #   if len(boundary_loops[i]) > max_size:
# #     max_size = len(boundary_loops[i])
# #     max_idx = i

# # fixed_dofs = []
# # fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), P)
# # fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), P)


# # for i in range(5):
# #   stretch_factor += 0.1
# #   fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
# #   stress = fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
# #   polymer_frac = fabsim_py.polymer_fraction_steady_state(stress, 1, k1 / k0, kd / k0, frac_f, frac_s)

# # ps.register_surface_mesh("deformed remeshed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))
# # # ps_mesh = ps.register_surface_mesh("Initial mesh remeshed", P, F, smooth_shade=True, enabled=True)


ps.show()
