import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2

world_coords_to_px = 1287.2

def read_csv_to_array(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]
    return np.array(data, dtype=np.float32)

def to_8bit_rgb(x):
    R = np.floor(x)
    G = np.floor((x - R) * 256)
    B = np.floor(((x - R) * 256 - G) * 256)
    return R, G, B

def from_8bit_rgb(R, G, B):
    return R + G / 256. + B / 256. / 256.


import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import pyvista as pv
import polyscope as ps

ps.init()

ps.set_navigation_style("planar")
ps.set_view_projection_mode("orthographic")
ps.set_SSAA_factor(3)
ps.set_give_focus_on_show(True)

eta_file = "data/sim/3post_eta.png"
theta_file = "data/sim/3post_theta.png"
# eta_file = "data/sim/3.5h_eta.png"
# theta_file = "data/sim/3.5h_theta.png"

# # V = np.load("V_80.npy")
# # Phi = np.load("Phi_80.npy")
# # V = np.load("V_720.npy")
# # Phi = np.load("Phi_720.npy")
V = np.load("V_3post_truearea.npy")
Phi = np.load("Phi_3post_truearea.npy")
R = np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]])
V = V @ R.T

# mesh = pv.read("data/top_surface_ultrahighres_deflect.obj")
mesh = pv.read("data/top_surface_3posts_ultrahighres.obj")
P = np.array(mesh.points, dtype=np.float64)
F = np.array(mesh.faces, dtype=np.int32)
F = F.reshape((F.shape[0] // 4, 4))[:, 1:]
P = P[:,:2]


phi_p = np.empty(F.shape[0])
eta = np.empty(F.shape[0])
dirs = np.empty((F.shape[0], 2))

for j in range(F.shape[0]):
  phi_p[j] = Phi[2 * j, 0] + Phi[2 * j + 1, 1]
  M = Phi[2 * j : 2 * (j + 1), :] / phi_p[j]
  w, v = np.linalg.eig(M)
  if w[0] > 0.5:
    dirs[j,:] = v[0]
    eta[j] = w[1]
  else:
    dirs[j] = v[1]
    eta[j] = w[0]

dirs[:,1] *= -1 


angles = np.atan2(dirs[:,1], dirs[:,0])
angles = np.mod(angles + np.pi / 6, np.pi)

ps_mesh = ps.register_surface_mesh("Simulation mesh", V, F, smooth_shade=True, enabled=True, color=(42/255, 53/255, 213/255), edge_width=0, material="flat")

# R, G, B = to_8bit_rgb(eta * 256 / 0.5)
# colors_face = np.array([B.T, G.T, R.T]).T
# ps_mesh.add_color_quantity("eta colors", colors_face / 256, defined_on='faces', enabled=True)

# ps.show()
# ps.screenshot(eta_file)

# R, G, B = to_8bit_rgb(angles * 256 / np.pi)
# colors_face = np.array([B.T, G.T, R.T]).T
# ps_mesh.add_color_quantity("theta colors", colors_face / 256, defined_on='faces', enabled=True)

# ps.screenshot(theta_file)

# min_eta = np.min(eta)
# max_eta = np.max(eta)
# min_angles = np.min(angles)
# max_angles = np.max(angles)

# print(np.min(eta), np.max(eta))
# print(np.min(angles), np.max(angles))

# img = np.array(cv2.imread(eta_file, cv2.IMREAD_UNCHANGED), dtype=np.float32)
# converted = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2])
# # converted = np.clip(converted, a_min=0, a_max=217)

# eta = converted * max_eta / np.max(converted)
# print(np.min(eta), min_eta)
# print(np.max(eta), max_eta)

# plt.imshow(eta, cmap='turbo', origin='lower', vmin=0, vmax=0.5, extent=[0, eta.shape[1] / world_coords_to_px, 0, eta.shape[0] / world_coords_to_px], aspect='equal', alpha=img[:,:,3] / 255)
# # plt.colorbar(label="$\eta$")
# # plt.title("Simulated compaction (3.5h)")
# plt.axis('off')
# plt.tight_layout()
# # plt.show()

# img = np.array(cv2.imread(theta_file, cv2.IMREAD_UNCHANGED), dtype=np.float32)
# converted = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2])
# # converted = np.clip(converted, a_min=0, a_max=217)

# theta = converted * max_angles / np.max(converted)
# print(np.min(theta), min_angles)
# print(np.max(theta), max_angles)

# height, width = theta.shape

# X, Y = np.meshgrid(np.arange(0, width, 66), np.arange(0, height, 66))
# # X, Y = np.meshgrid(np.arange(0, width, 59), np.arange(0, height, 59))
# U = np.cos(theta) * img[:,:,3] / 255
# V = -np.sin(theta) * img[:,:,3] / 255

# q = plt.quiver(X / world_coords_to_px,  Y / world_coords_to_px, U[Y, X], V[Y, X], color="white", headlength=0, headaxislength=0, headwidth=0, scale=50, units='width')

# plt.savefig("Sim.pdf", dpi=300)
# plt.show()



data_path = "data/3 post/"



# ps.init()
ps.load_color_map("twilight", "data/twilight_colormap.png");
# ps.set_give_focus_on_show(True)
# ps.set_ground_plane_mode("shadow_only")

# ps.set_navigation_style("planar")
# ps.set_view_projection_mode("orthographic")
# ps.set_SSAA_factor(3)


# ps_mesh = ps.register_surface_mesh("Simulation mesh", V, F, smooth_shade=True, enabled=True, color=(42/255, 53/255, 213/255), edge_width=0, material="flat")
ps_mesh.add_scalar_quantity("angles", angles, defined_on='faces', enabled=True, cmap="twilight")
ps_mesh.add_scalar_quantity("eta", eta, defined_on='faces', enabled=True, cmap="turbo")

# ps_mesh.add_vector_quantity("vectors", np.vstack((np.cos(angles), np.sin(angles))).T, defined_on='faces', enabled=True)
# ps.show()
# import fabsim_py

for i in range(1,5):
    # img = cv2.imread(f'{data_path}orientation{i}_dispersion_new.png', cv2.IMREAD_UNCHANGED)
    # dispersion_data = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * 0.5 / 256
    # eta_measured = fabsim_py.image_data_to_mesh(V, F, dispersion_data, world_coords_to_px)
    # mask_data = (cv2.imread(f'{data_path}mask{i}.png', cv2.IMREAD_UNCHANGED) > 0).astype(float)
    # mask = fabsim_py.image_data_to_mesh(V, F, mask_data, world_coords_to_px)

    # eta = mask * eta_measured
    # eta += (mask - 1)
    # np.save(f'data/3 post/eta{i}_measured_new.npy', eta)



    eta_measured = np.load(f'data/3 post/eta{i}_measured_new_new.npy')
    mask = (eta_measured > 0).astype(float)
    eta = mask * eta_measured
    
    # img = cv2.imread(f'data/3 post/orientation{i}_mean.png')
    # orientation_data = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * np.pi / 256 + np.pi / 2

    # orientation = fabsim_py.orientation_data_to_mesh(V, F, orientation_data, world_coords_to_px)
    # orientation = orientation * mask[:, None]
    # orientation[:,1] *= -1
    # ps_mesh.add_vector_quantity(f"orientation{i}", orientation, defined_on="faces", enabled=True)

    # theta = np.atan2(orientation[:,1], orientation[:,0])
    # theta = np.mod(theta, np.pi)

    # theta = theta * mask
    # theta += (mask - 1)

    ps_mesh.add_scalar_quantity(f"mask{i}", mask, defined_on="faces", enabled=True, cmap="turbo")
    ps_mesh.add_scalar_quantity(f"eta{i}", eta, defined_on="faces", enabled=True, cmap="turbo")
    # ps_mesh.add_scalar_quantity(f"theta{i}", theta, defined_on="faces", enabled=True, cmap="twilight")
    # np.save(f'data/3 post/theta{i}_measured.npy', theta)
    ps.show()

