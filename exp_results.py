import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


# Folder stuff, change for different data sets

# data_path = "data/72h/"
# data_path = "data/3.5hrs/"
# data_path = "data/3 post/"
data_path = "data/7D/"

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


# img = np.array(cv2.imread('data/3 post/orientation_dispersion_mask.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
img = np.array(cv2.imread(data_path + 'orientation_dispersion_mask.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
converted = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2])

eta = converted * 0.5 / 256

print(np.min(converted), np.max(converted))
print(np.min(eta), np.max(eta))
height, width, _ = img.shape

fig, ax = plt.subplots()

im = ax.imshow(eta, cmap='turbo', origin='lower', vmin=0, vmax=0.5, extent=[0, width / world_coords_to_px, 0, height / world_coords_to_px], aspect='equal', alpha=img[:,:,3] / 255)
fig.colorbar(im, ax=ax, label=r"$\eta$")
ax.axis('off')

# ax.title("Measured compaction (3.5h)")
# ax.tight_layout()

# img = np.array(cv2.imread('data/3 post/orientation_mean_mask.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
img = np.array(cv2.imread(data_path + 'orientation_mean_mask.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
converted = from_8bit_rgb(img[:,:,0], img[:,:,1], img[:,:,2])


shape = converted.shape
print(converted.shape)
down = cv2.resize(
    converted,
    None,
    fx=0.1,
    fy=0.1,
    interpolation=cv2.INTER_AREA
)
print(down.shape)
converted = result = cv2.resize(
    down,
    (shape[1], shape[0]),
    interpolation=cv2.INTER_NEAREST
)
print(converted.shape)


# theta = converted * np.pi / 256 + np.pi / 2 - 152.1 / 180 * np.pi
theta = converted * np.pi / 256 + np.pi / 2
# theta = converted * np.pi / 256


X, Y = np.meshgrid(np.arange(0, width, 149), np.arange(0, height, 149))
U = np.cos(theta) * np.clip(img[:,:,3], a_min=0, a_max=1)
V = np.sin(theta) * np.clip(img[:,:,3], a_min=0, a_max=1)

q = ax.quiver(X / world_coords_to_px,  Y / world_coords_to_px, U[Y, X], V[Y, X], color="white", headlength=0, headaxislength=0, headwidth=0, scale=50, units='width')

scalebar = AnchoredSizeBar(
    ax.transData,
    size=1.0,                 # 1 cm in data units
    label="1 mm",
    loc="lower center",
    pad=0.3,
    color="black",
    frameon=False,
    size_vertical=0.02,
    fontproperties=fm.FontProperties(size=10),
    bbox_to_anchor=(0.9, -0.3),      # x centered, y below axes
    bbox_transform=ax.transAxes  # interpret anchor in axes coords
)

ax.add_artist(scalebar)

plt.show()