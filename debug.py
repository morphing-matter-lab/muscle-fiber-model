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

fixed_dofs = []

def fix_dofs_in_circle(radius, center, V):
  for i in range(V.shape[0]):
    if np.linalg.norm(center - V[i, :2]) < radius + 1e-6:
      if center[0] * (center[0] - V[i, 0]) < 1:
        fixed_dofs.append(3 * i)
        fixed_dofs.append(3 * i + 1)

mesh = pv.read("top_surface_highres.obj")
P = np.array(mesh.points, dtype=np.float64)
F = np.array(mesh.faces, dtype=np.int32)
F = F.reshape((F.shape[0] // 4, 4))[:, 1:]

fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), P)
fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), P)

# Initialize polyscope
ps.init()
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode("shadow_only")

# ps_mesh = ps.register_surface_mesh("Initial mesh", P, F, smooth_shade=True, enabled=True)
ps.set_navigation_style("planar")
ps.set_view_projection_mode("orthographic")
ps.set_SSAA_factor(3)
ps.set_view_from_json('{"farClipRatio":20.0,"fov":16.0,"nearClipRatio":0.005,"projectionMode":"Orthographic","viewMat":[1.0,-0.0,0.0,-0.0,0.0,0.997785151004791,-0.0665190145373344,0.000246047973632812,-0.0,0.0665190145373344,0.997785151004791,-25.5602951049805,0.0,0.0,0.0,1.0],"windowHeight":1964,"windowWidth":3456}')
# ps.show()

stretch_factor = 1.1
polymer_frac = np.zeros((F.shape[0], n))
V = P.copy()

for j in range(15):
  print(stretch_factor)
  for i in range(3):
    fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
    stress = fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
    polymer_frac = fabsim_py.polymer_fraction_steady_state(stress, 1, k1 / k0, kd / k0, frac_f, frac_s)
  stretch_factor += 0.1

  if j % 4 == 1:
    V, P, F = fabsim_py.remesh(V, P, F, "pqa0.01")

    fixed_dofs = []
    fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), P)
    fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), P)
    fixed_dofs = np.sort(fixed_dofs)
    polymer_frac = np.zeros((F.shape[0], n))

V, P, F = fabsim_py.remesh(V, P, F, "pqa0.005")

fixed_dofs = []
fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), P)
fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), P)
fixed_dofs = np.sort(fixed_dofs)
polymer_frac = np.zeros((F.shape[0], n))

for i in range(3):
  fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
  stress = fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
  polymer_frac = fabsim_py.polymer_fraction_steady_state(stress, 1, k1 / k0, kd / k0, frac_f, frac_s)

ps.register_surface_mesh("deformed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))

# V, P, F = fabsim_py.remesh(V, P, F, "pqa0.1")

# fixed_dofs = []
# fix_dofs_in_circle(0.75, np.array([-4.55/2, 0]), P)
# fix_dofs_in_circle(0.75, np.array([4.55/2, 0]), P)

# polymer_frac = np.zeros((F.shape[0], n))
# for i in range(3):
#   fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
#   stress = fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
#   polymer_frac = fabsim_py.polymer_fraction_steady_state(stress, 1, k1 / k0, kd / k0, frac_f, frac_s)


# for i in range(5):
#   stretch_factor += 0.1
#   fabsim_py.simulate_membrane(V, P, F, polymer_frac, fixed_dofs, stretch_factor, 0.25, 1.0, e0, e1)
#   stress = fabsim_py.fiber_stress(V, P / stretch_factor, F, n, e0, e1)
#   polymer_frac = fabsim_py.polymer_fraction_steady_state(stress, 1, k1 / k0, kd / k0, frac_f, frac_s)

# ps.register_surface_mesh("deformed remeshed", V, F, smooth_shade=True, color=(42/255, 53/255, 213/255))
# ps_mesh = ps.register_surface_mesh("Initial mesh remeshed", P, F, smooth_shade=True, enabled=True)


ps.show()
