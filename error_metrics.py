import numpy as np
import fabsim_py
import matplotlib.pyplot as plt
import cv2
import igl
import seaborn as sns
import csv
import pyvista as pv

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


world_coords_to_px = 1287.2

eta_measured = []

# V = np.load("V_40mean.npy")
# Phi = np.load("Phi_40mean.npy")
# V = np.load("V_720truearea.npy")
# Phi = np.load("Phi_720truearea.npy")
V = np.load("V_7D.npy")
Phi = np.load("Phi_7D.npy")
mesh = pv.read("data/top_surface_ultrahighres_deflect.obj")
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

area_weights = np.clip(igl.doublearea(V, F), a_min=0, a_max=np.inf)

# data_path = "data/3.5hrs/"
data_path = "data/72h/"
# data_path = "data/7D/"
n = 4


eta_errors = np.zeros((n,n))
eta_measured = []
for i in range(1, 4):
  # if i == 3:
  #   continue
  img = cv2.imread(f'{data_path}orientation{i}_dispersion_new.png', cv2.IMREAD_UNCHANGED)
  dispersion_data = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * 0.5 / 256
  eta_measured.append(fabsim_py.image_data_to_mesh(V, F, dispersion_data, world_coords_to_px))
  # if i == 4:
  #    i -= 1
  np.save(f'{data_path}/eta{i}_measured.npy', eta_measured[i-1])
  # eta_measured.append(np.load(f'{data_path}eta{i}_measured.npy'))
  mask = (eta_measured[i-1] > 0).astype(float)
  eta_errors[0,i] = np.sum(mask * area_weights * np.abs(eta - eta_measured[i-1])) / 0.5 / np.sum(mask * area_weights)
  eta_errors[i,0] = np.sum(mask * area_weights * np.abs(eta - eta_measured[i-1])) / 0.5 / np.sum(mask * area_weights)

for i in range(n-1):
  for j in range(n-1):
    mask = (eta_measured[i] > 0).astype(float) * (eta_measured[j] > 0).astype(float)
    eta_errors[i+1,j+1] = np.sum(mask * area_weights * np.abs(eta_measured[j] - eta_measured[i])) / 0.5 / np.sum(mask * area_weights)

print(eta_errors)
print(np.sum(eta_errors[0,:]) / (n-1))
print(np.sum(eta_errors[1:,1:]) / (n * (n-1) / 2))


labels = ["Sim.", "Exp. 1", "Exp. 2", "Exp. 3"]

n = len(eta_errors)
plt.figure(figsize=(n + 2, n + 1))

# Mask diagonal
mask = np.eye(n, dtype=bool)

ax = sns.heatmap(
    100 * eta_errors,
    mask=mask,
    annot=True,          # show numbers
    fmt=".1f",           # number format
    cmap="viridis",      # colormap
    xticklabels=labels,
    yticklabels=labels,
    square=True,
    cbar_kws={"label": "Difference (%)"},
    vmin=0,
    vmax=25
)

plt.title(r"Differences in dispersion factor $\eta$ (%)")
plt.tight_layout()
# plt.savefig("3.5hrs eta.pdf", dpi=150)
# plt.savefig("72h eta.pdf", dpi=150)
plt.savefig("7D eta.pdf", dpi=150)
# plt.show()




angle_measured = []

angles = np.atan2(-dirs[:,1], dirs[:,0])
angles = np.mod(angles, np.pi)

area_weights = np.clip(igl.doublearea(V, F), a_min=0, a_max=np.inf)

theta_errors = np.zeros((n,n))
for i in range(1, 4):
  # if i == 3:
  #   continue
  img = cv2.imread(data_path + f'orientation{i}_mean.png', cv2.IMREAD_UNCHANGED)
  theta_data = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * np.pi / 256
  vectors = fabsim_py.orientation_data_to_mesh(V, F, theta_data, world_coords_to_px)
  angle_measured.append(np.atan2(vectors[:,1], vectors[:,0]) + np.pi / 2)
  # if i == 4:
  #    i -= 1
  np.save(f'{data_path}/theta{i}_measured.npy', angle_measured[i-1])
  # angle_measured.append(np.load(f'{data_path}theta{i}_measured.npy'))
  error = np.abs(angle_measured[i - 1] - angles)
  error = area_weights * np.where(error > np.pi / 2, np.abs(error - np.pi), error)

  theta_errors[0,i] = np.sum(error) / (np.pi / 2) / np.sum(area_weights)
  theta_errors[i,0] = np.sum(error) / (np.pi / 2) / np.sum(area_weights)

for i in range(n-1):
  for j in range(n-1):
    error = np.abs(angle_measured[j] - angle_measured[i])
    error = area_weights * np.where(error > np.pi / 2, np.abs(error - np.pi), error)
    theta_errors[i+1,j+1] = np.sum(error) / (np.pi / 2) / np.sum(area_weights)

print(theta_errors)
print(np.sum(theta_errors[0,:]) / (n-1))
print(np.sum(theta_errors[1:,1:]) / (n * (n-1) / 2))


labels = ["Sim.", "Exp. 1", "Exp. 2", "Exp. 3"]

n = len(theta_errors)
plt.figure(figsize=(n + 2, n + 1))

# Mask diagonal
mask = np.eye(len(theta_errors), dtype=bool)

ax = sns.heatmap(
    100 * theta_errors,
    mask=mask,
    annot=True,          # show numbers
    fmt=".1f",           # number format
    cmap="viridis",      # colormap
    xticklabels=labels,
    yticklabels=labels,
    square=True,
    cbar_kws={"label": "Difference (%)"},
    vmin=0,
    vmax=25
)

plt.title(r"Differences in $\theta_0$ angle (%)")
plt.tight_layout()
# plt.savefig("3.5hrs theta.pdf", dpi=150)
# plt.savefig("72h theta.pdf", dpi=150)
plt.savefig("7D theta.pdf", dpi=150)
# plt.show()




