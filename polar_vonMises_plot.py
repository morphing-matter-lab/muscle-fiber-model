import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

# # Parameters
# mu = np.pi / 3
# kappa = 16.0
# n_bins = 300

# # Angular bin edges
# theta_edges = np.linspace(0, 2*np.pi, n_bins + 1)

# # Evaluate PDF at edges (not centers)
# pdf_edges = vonmises.pdf(theta_edges, kappa, loc=mu) + vonmises.pdf(theta_edges + np.pi, kappa, loc=mu)

# # Normalize for colormap
# values = pdf_edges / pdf_edges.max()

# # Radial extent (single ring)
# r_edges = np.array([0.0, 1.0])

# # 2D grid required by pcolormesh
# Theta, R = np.meshgrid(theta_edges, r_edges)
# Z = 1 - np.tile(values[:], (2, 1))  # shape (1, n_bins)

# # Plot
# fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
# pcm = ax.pcolormesh(
#     Theta,
#     R,
#     Z,
#     cmap="gray",
#     shading="gouraud"  # enables smooth angular gradient
# )

# # Remove all theta ticks and labels
# # ax.set_xticks([])
# ax.set_xticklabels([])
# ax.set_yticks([])

# # Draw the mean direction
# ax.plot([mu, mu], [0, 1.0], color="C0", lw=1.5, ls="--")
# ax.plot([mu + np.pi, mu + np.pi], [0, 1.0], color="C0", lw=1.5, ls="--")

# # Arc parameters
# r_arc = 0.5
# theta_arc = np.linspace(0, mu, 200)
# r_arc_vals = np.full_like(theta_arc, r_arc)

# # Draw curved arc
# ax.plot(
#     theta_arc,
#     r_arc_vals,
#     color="C0",
#     lw=1.5
# )

# # Arrowhead at the end of the arc
# # Small angular offset to define arrow direction
# dtheta = 0.01
# ax.annotate(
#     "",
#     xy=(mu, r_arc),
#     xytext=(mu - dtheta, r_arc),
#     arrowprops=dict(
#         arrowstyle="->",
#         lw=1.5,
#         color="C0"
#     )
# )

# ax.text(
#     mu / 2,
#     r_arc + 0.07,
#     r"$\theta_0$",
#     ha="center",
#     va="center"
# )

# ax.set_theta_zero_location("E")
# ax.set_theta_direction(-1)

# # plt.colorbar(pcm, ax=ax, pad=0.1, label="Normalized density")
# plt.show()

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import vonmises

# # # Parameters of the von Mises distribution
# # mu = np.pi        # mean direction
# # kappa = 3.0           # concentration parameter

# # # Angular bins
# # n_bins = 120
# # theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
# # width = 2 * np.pi / n_bins

# # # Von Mises PDF evaluated at bin centers
# # pdf = vonmises.pdf(theta, kappa, loc=mu)

# # # Normalize PDF to [0, 1] for opacity
# # alpha = pdf / pdf.max()

# # # Optional: constant radius for all bars
# # radius = np.ones_like(theta)

# # plt.plot(theta, pdf)

# # plt.show()

import cv2
import csv

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



# data_path = 'data/72h/'
# img = cv2.imread(f'{data_path}orientation3.png', cv2.IMREAD_UNCHANGED)
# arr = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * np.pi / 256

# height, width, _ = img.shape

# center_row = int(height/2)
# center_col = int(width/2)
# data = arr[center_row-200:center_row+200, center_col-350:center_col+50]


# cos_data = np.cos(2 * data) * img[center_row-200:center_row+200, center_col-350:center_col+50,3]
# sin_data = np.sin(2 * data) * img[center_row-200:center_row+200, center_col-350:center_col+50,3]

# # normalized convolution of image with mask
# sin_smooth = np.sum(sin_data) / np.sum(img[center_row-200:center_row+200, center_col-350:center_col+50,3])
# cos_smooth = np.sum(cos_data) / np.sum(img[center_row-200:center_row+200, center_col-350:center_col+50,3])

# complex_ab = np.sqrt(cos_smooth + 1j * sin_smooth)

# mean_orientation = np.pi / 2 - np.mod(np.angle(complex_ab), np.pi)
# print(mean_orientation)

# # mean resultant length
# R = np.sqrt(np.real(complex_ab)**2 + np.imag(complex_ab)**2)

# if R < 0.53:
#   kappa =   2 * R + R**3 + (5 * R**5) / 6
# elif R < 0.85:
#   kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
# else:
#   kappa = 1 / (R**3 - 4 * R**2 + 3 * R)


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import vonmises


# print(kappa)

# theta_samples = data[img[center_row-200:center_row+200, center_col-350:center_col+50,3] > 0].flatten()
# theta_samples = (np.pi / 2 - theta_samples) % (2*np.pi)

# bins = 128
# counts, edges = np.histogram(theta_samples, bins=bins, range=(0, 2*np.pi), density=True)

# centers = (edges[:-1] + edges[1:]) / 2
# width = edges[1] - edges[0]


# theta = np.linspace(0, 2*np.pi, 500)
# pdf = vonmises.pdf(theta, kappa, loc=mean_orientation)

# # polar plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')

# # histogram bars
# ax.bar(centers, counts, width=width, alpha=0.4, edgecolor="black", label="Angle histogram")


# ax.plot(theta, pdf)
# ax.set_title("Von Mises Distribution")

# plt.show()



import numpy as np
import pyvista as pv
import polyscope as ps
import fabsim_py
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import IPython.display
import csv
import cv2
import igl
from numba import njit, prange
import scipy.special


@njit(parallel=True, fastmath=True)
def integrate_field(P, x, dx):
    ny, nx = P.shape
    n_x = len(x)
    F = np.empty_like(P)

    for i in prange(ny):               # parallel loop over rows
        for j in range(nx):            # inner loop over columns
            p = P[i, j]
            acc = 0.0
            for k in range(n_x - 1):   # trapezoidal rule
                f1 = np.exp(p * np.cos(2 * x[k])) * np.sin(x[k])**2
                f2 = np.exp(p * np.cos(2 * x[k+1])) * np.sin(x[k+1])**2
                acc += 0.5 * (f1 + f2)
            F[i, j] = acc * dx
    return F

data_path = "data/7D/"
# data_path = "data/3 post/"

for i in range(1,5):
  if i == 3:
      continue
  img = cv2.imread(f'{data_path}orientation{i}.png', cv2.IMREAD_UNCHANGED)

  data = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2]) * np.pi / 256
  cos_data = np.cos(2 * data) * img[:,:,3]
  sin_data = np.sin(2 * data) * img[:,:,3]

  blur = 100

  # normalized convolution of image with mask
  sin_smooth = gaussian_filter(sin_data, sigma = blur)
  cos_smooth = gaussian_filter(cos_data, sigma = blur)

  weights = np.clip(gaussian_filter(img[:,:,3].astype(np.float64), sigma = blur), a_min=1, a_max=1000) 
  sin_smooth /= weights
  cos_smooth /= weights

  complex_ab = np.sqrt(cos_smooth + 1j * sin_smooth)

#   mean_orientation = np.mod(np.angle(complex_ab), np.pi) * 256 / np.pi

#   R, G, B = to_8bit_rgb(mean_orientation)
#   A = np.round((weights / np.max(weights)) * 255 ).astype(np.uint8)
#   new_img = np.array([R.T, G.T, B.T, A.T]).T

#   cv2.imwrite(f'{data_path}orientation{i}_mean.png', new_img)

  # mean resultant length
  R = np.sqrt(np.real(complex_ab)**2 + np.imag(complex_ab)**2)

  kappa = np.zeros_like(R)

  # approximation from Best & Fisher [1981] http://dx.doi.org/10.1080/03610918108812225
  lower_R = R[R < 0.53]
  mid_R = R[(R > 0.53) & (R < 0.85)]
  upper_R = R[R > 0.85]

  kappa[R < 0.53] = 2 * lower_R + lower_R**3 + (5 * lower_R**5) / 6
  kappa[(R > 0.53) & (R < 0.85)] = -0.4 + 1.39 * mid_R + 0.43 / (1 - mid_R)
  kappa[R > 0.85] = 1 / (upper_R**3 - 4 * upper_R**2 + 3 * upper_R)

  theta = np.linspace(-np.pi/2, np.pi/2, 256)

  eta = integrate_field(kappa, theta, theta[1] - theta[0])
  eta /= scipy.special.iv(0, kappa) * np.pi

  r, g, b = to_8bit_rgb(eta * 256 / 0.5)
  a = np.round((weights / np.max(weights)) * 255 ).astype(np.uint8)

  new_img = np.array([r.T, g.T, b.T, a.T]).T

  cv2.imwrite(f'{data_path}orientation{i}_dispersion_new.png', new_img)
IPython.display.Image(filename=data_path + 'orientation1_dispersion_new.png')