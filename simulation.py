import numpy as np
import polyscope as ps
import polyscope.imgui as gui
import pyvista as pv
import tetgen
import fabsim_py

# dt = 1 / 24
# k0 = 1e-4
# k1 = 5e-2
# kd = 0.1 # rate of dissociation
e0 = 1.2e-1
e1 = 1.7e-1
# frac_f = 0.7
# frac_s = 0.25
n = 16

fixed_dofs = []

def fix_dofs_in_circle(radius, center, V):
  for i in range(V.shape[0]):
    if np.linalg.norm(center - V[i, :2]) < radius + 1e-6:
      fixed_dofs.append(3 * i)
      fixed_dofs.append(3 * i + 1)

mesh = pv.read("tissue.obj")
tet = tetgen.TetGen(mesh)
V_3D, F_3D = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)

fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), V_3D)
fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), V_3D)

NV_3D = V_3D.copy()
Phi_3D = np.zeros((F_3D.shape[0], n))
# fabsim_py.simulate3D(NV_3D, V_3D, F_3D, Phi_3D, fixed_dofs, 1.3, 0.3, 1, e0, e1)
# fabsim_py.simulate3D(NV_3D, V_3D, F_3D, Phi_3D, fixed_dofs, 1.4, 0.3, 1, e0, e1)
# fabsim_py.simulate3D(NV_3D, V_3D, F_3D, Phi_3D, fixed_dofs, 1.5, 0.3, 1, e0, e1)

stretch_factor = 1.5
young_modulus = 1
poisson_ratio = 0.3
sigma_max = 1
damping = 0.1
dt = 1 / 1000

model = fabsim_py.Model(V_3D, F_3D, Phi_3D, fixed_dofs, stretch_factor, young_modulus, poisson_ratio, sigma_max, e0, e1, damping, dt)

x = np.reshape(V_3D, -1)
v = np.zeros(x.shape)
a = np.zeros(x.shape)

# model.solve_timestep_newmark(x, v, a)

# NV_3D = np.reshape(x, V_3D.shape)

# print(fixed_dofs)

def callback_newmark():
  global x, v, a, model, V_3D

  # if gui.Button("Timestep"):
  for i in range(4):
    model.solve_timestep_newmark(x, v, a)
  ps.get_volume_mesh("sim").update_vertex_positions(np.reshape(x, V_3D.shape))
  ps.get_volume_mesh("sim").add_vector_quantity("v", np.reshape(v, V_3D.shape), defined_on="vertices")
  ps.get_volume_mesh("sim").add_vector_quantity("a", np.reshape(a, V_3D.shape), defined_on="vertices")

  ps.screenshot()


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps_vol = ps.register_volume_mesh("sim", V_3D, tets=F_3D)
ps.set_user_callback(callback_newmark)
ps.show()
