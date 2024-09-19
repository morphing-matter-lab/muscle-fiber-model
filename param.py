import numpy as np
import polyscope as ps
import polyscope.imgui as gui
import fabsim_py
import potpourri3d as pp3d


# Initialize polyscope
ps.init()
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode("shadow_only")

V, F = pp3d.read_mesh("Frame 30.obj")
centerV = np.sum(V, axis=0) / V.shape[0]
V -= centerV
ps_mesh = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

ps.show()
ps.remove_all_structures()

P, F = fabsim_py.boundary_first_flattening(V, F)
centerP = np.sum(P, axis=0) / P.shape[0]
P -= centerP
ps_mesh = ps.register_surface_mesh("param", P, F, smooth_shade=True)
areas = fabsim_py.compute_area_distortion(V, P, F)
ps_mesh.add_scalar_quantity("area distortion", np.array(areas), defined_on="faces", cmap="coolwarm", enabled=True)
ps.reset_camera_to_home_view()

X = 0.9999 * P + 0.0001 * V # push out of plane

def callback():
  global V, P, X, F
  thickness = 0.01
  poisson_ratio = 0.3
  if gui.Button("Simulation"):
    X = fabsim_py.simulate_shell(V, X, F, [0, 1, 2], thickness, poisson_ratio)
    ps.get_surface_mesh("param").update_vertex_positions(X)
    ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

  if gui.Button("Dynamic time step"):
    delta_t = 1e-4
    X = fabsim_py.simulate_shell_timestep(V, P, X, F, thickness, poisson_ratio, delta_t)
    ps.get_surface_mesh("param").update_vertex_positions(X)

ps.set_user_callback(callback)

ps.show()
