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

# ps.show()
ps.remove_all_structures()

P, F = fabsim_py.boundary_first_flattening(V, F)
centerP = np.sum(P, axis=0) / P.shape[0]
P -= centerP
ps_mesh = ps.register_surface_mesh("param", P, F, smooth_shade=True)
areas = fabsim_py.compute_area_distortion(V, P, F)
ps_mesh.add_scalar_quantity("area distortion", np.array(areas), defined_on="faces", cmap="coolwarm", enabled=True)
ps.reset_camera_to_home_view()

X = 0.99 * P + 0.01 * V # push out of plane

running = False
i = 0
velocity = np.zeros(V.shape)
def callback():
  global V, P, X, F, running, i, velocity
  thickness = 0.02
  # source: https://www.researchgate.net/figure/Experimental-data-for-the-Poissons-ratio-n-of-two-bulk-PNIPAM-gels-with-different_fig2_317373615
  poisson_ratio = 0.3
  # source: https://www.researchgate.net/figure/Youngs-modulus-values-for-polyN-isopropylacrylamide-PNIPAM-hydrogels-synthesized-in_fig5_309955494
  young_modulus = 2e9
  delta_t = 1e-4
  # source: https://www.researchgate.net/figure/Rheological-measurements-of-viscosity-of-hydrogel-components-All-samples-were-measured_fig2_287147361
  viscosity = 5e-13
  # if gui.Button("Simulation (static)"):
  #   X = fabsim_py.simulate_shell(V, X, F, [0, 1, 2], thickness, poisson_ratio)
  #   ps.get_surface_mesh("param").update_vertex_positions(X)
  #   ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

  if gui.Button("Simulation (dynamic)"):
    running = True
  
  if running:
    i = i + 1
  
  if i % 10 == 0:
    velocity = fabsim_py.simulate_shell_timestep(V, P, X, F, velocity, young_modulus, thickness, poisson_ratio, viscosity, delta_t)
    X = X + velocity * delta_t
    ps.get_surface_mesh("param").update_vertex_positions(X)
    # ps.screenshot()

ps.set_user_callback(callback)

ps.show()
