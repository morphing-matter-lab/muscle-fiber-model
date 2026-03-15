import numpy as np
import csv
import cv2
import numpy as np
import cv2
import csv
import pyvista as pv
import fabsim_py
import igl
from concurrent.futures import ProcessPoolExecutor
# import polyscope as ps
from joblib import Parallel, delayed
from tqdm import tqdm


sigma = 74.2
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

def fix_dofs_in_circle(radius, center, V, shift=0):
  for i in range(V.shape[0]):
    if np.linalg.norm(center - V[i, :2]) < radius + 1e-6:
      if (center @ center) - (V[i, :2] @ center) < shift * np.sqrt(center @ center):
        fixed_dofs.append(i)


def run_simulation(params):
    V_ = V.copy()
    Phi_ = Phi.copy()
    vel = np.zeros(2 * V.shape[0])

    n_iter = 7200
    stretch = 1 / np.sqrt(0.122252901) * np.ones(n_iter)

    fabsim_py.implicit_euler_sqrt(Phi_, V_, vel, P, F, fixed_dofs, stretch, 0.49, sigma, params[0], 0, params[1], params[2], 0.5, n_iter)

    eta = np.empty(F.shape[0])
    for l in range(F.shape[0]):
        phi_p = Phi_[2 * l, 0] + Phi_[2 * l + 1, 1]
        M = Phi_[2 * l : 2 * (l + 1), :] / phi_p
        w, v = np.linalg.eig(M)
        if w[0] > 0.5:
            eta[l] = w[1]
        else:
            eta[l] = w[0]

    # R = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)], [np.sin(np.pi/6), np.cos(np.pi/6)]])
    # V_ = V_ @ R.T 

    total_error = 0
    total_mask = ((eta_measured[1] > 0) + (eta_measured[2] > 0) + (eta_measured[3] > 0)).astype(float)

    for i in range(2, 5):
        mask = (eta_measured[i-1] > 0).astype(float)
        error = mask * area_weights * np.abs(eta - eta_measured[i-1])
        total_error += np.sum(error) / 0.5 / np.sum(mask * area_weights)
    
    return params[0], params[1], params[2], total_error, np.sum(area_weights * total_mask * eta) / np.sum(area_weights * total_mask), np.min(igl.doublearea(V_, F)) 

V = np.load("V_3post.npy")
Phi = np.load("Phi_3post.npy")
mesh = pv.read("data/top_surface_3posts_ultrahighres.obj")
P = np.array(mesh.points, dtype=np.float64)
F = np.array(mesh.faces, dtype=np.int32)
F = F.reshape((F.shape[0] // 4, 4))[:, 1:]
P = P[:,:2]


fixed_dofs = []
fix_dofs_in_circle(0.75, np.array([-2.275, 2.627/2]), P, 0.)
fix_dofs_in_circle(0.75, np.array([2.275, 2.627/2]), P, 0.)
fix_dofs_in_circle(0.75, np.array([0, -2.627]), P, 0.)
fixed_dofs = np.sort(fixed_dofs)

eta_measured = []
for i in range(1, 5):
  eta_measured.append(np.load(f'data/3 post/eta{i}_measured_new.npy'))

#   img = cv2.imread(f'data/3 post/orientation{i}_dispersion_new.png', cv2.IMREAD_UNCHANGED)
#   dispersion_data = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * 0.5 / 256
#   eta_measured.append(fabsim_py.image_data_to_mesh(V, F, dispersion_data, world_coords_to_px))
#   np.save(f'data/3 post/eta{i}_measured_new.npy', eta_measured[i-1])

area_weights = np.clip(igl.doublearea(V, F), a_min=0, a_max=np.inf)
print(np.sum(area_weights * (eta_measured[3]+eta_measured[1]+eta_measured[2])/3) / np.sum(area_weights))


k0 = np.array([5, 7, 20, 30, 40, 50]) * 1e-6
k1 = np.array([5, 7, 20, 30, 40, 50]) * 1e-3
kd = np.array([5, 7, 20, 30, 40, 50]) * 1e-2

param_list = [[0.0001, 0.0004, 0.001], [0.0001, 0.0004, 0.0005],
[0.0001, 0.0004, 0.005],
[0.0001, 0.0001, 0.001],
[0.0001, 0.0001, 0.0005],
[0.0001, 0.001, 0.001],
[0.0001, 0.001, 0.0005],
[0.0001, 0.001, 0.005],
[0.0005, 0.0004, 0.001],
[0.0005, 0.0004, 0.0005],
[0.0005, 0.001, 0.001],
[0.0005, 0.001, 0.0005],
[0.0005, 0.001, 0.005],
[0.00005, 0.004, 0.001],
[0.00005, 0.004, 0.0005],
[0.00005, 0.0001, 0.001],
[0.00005, 0.0001, 0.0005]
]

# for i in range(len(k0)):
#   for j in range(len(k1)):
#     for k in range(len(kd)):
#        param_list.append([k0[i], k1[j], kd[k]])

# print(run_simulation([0, 2e-2, 4e-1]))

results = Parallel(n_jobs=8)(
    delayed(run_simulation)(p) for p in tqdm(param_list)
)

print(results)
R = np.array(results)

np.set_printoptions(threshold=R.shape[0] * R.shape[1])
print(R)
print(np.min(R[:,3]))
print(np.min(R[:,4]))
print(R[np.argmin(R[:,3]), :])
print(R[np.argmin(R[:,4]), :])


# np.save('V_test.npy', V)
# np.save('Phi_test.npy', Phi)
