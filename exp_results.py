import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


# Folder stuff, change for different data sets

# data_path = "data/72h/"
# data_path = "data/3.5hrs/"
data_path = "data/3 post/"
# data_path = "data/7D/"

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

def plt_eta_theta(eta, theta, mask):
    fig, ax = plt.subplots()

    im = ax.imshow(eta, cmap='turbo', origin='lower', vmin=0, vmax=0.5, extent=[0, width / world_coords_to_px, 0, height / world_coords_to_px], aspect='equal', alpha=mask)
    fig.colorbar(im, ax=ax, label=r"$\eta$")
    fig.canvas.manager.full_screen_toggle()
    ax.axis('off')

    # ax.title("Measured compaction (3.5h)")
    # ax.tight_layout()

    X, Y = np.meshgrid(np.arange(0, width, 149), np.arange(0, height, 149))
    U = np.cos(theta) * mask
    V = np.sin(theta) * mask

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
    # plt.show()


# img = np.array(cv2.imread('data/3 post/orientation_dispersion_mask.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
mask_img = np.array(cv2.imread(data_path + 'mask.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
mask = (mask_img[:,:] > 128).astype(np.float32)
height, width = mask_img.shape

eta = np.zeros((height, width))
alpha = np.zeros((height, width))
z = np.zeros((height, width), dtype=np.complex128)

for i in range(1,5):
    # if i == 3:
    #     continue
    mask_img = np.array(cv2.imread(f"{data_path}mask{i}.png", cv2.IMREAD_UNCHANGED), dtype=np.float32)
    mask_i = (mask_img[:,:] > 128).astype(np.float32)

    input_img = np.array(cv2.imread(data_path + f'orientation{i}_dispersion.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
    eta_i = from_8bit_rgb(input_img[:,:,0], input_img[:,:,1], input_img[:,:,2]) * 0.5 / 256

    eta += eta_i * input_img[:,:,3]
    alpha += input_img[:,:,3]

    input_img = np.array(cv2.imread(data_path + f'orientation{i}_mean.png', cv2.IMREAD_UNCHANGED), dtype=np.float32)
    theta_i = from_8bit_rgb(input_img[:,:,0], input_img[:,:,1], input_img[:,:,2]) * np.pi / 256 + np.pi / 2

    z += (np.cos(theta_i) + 1j * np.sin(theta_i)) * input_img[:,:,3]

    plt_eta_theta(eta_i, theta_i, mask_i)
    plt.savefig(f"Sample{i}.pdf", dpi=150)

eta /= np.clip(alpha, a_min=1, a_max=1000000)
eta = eta * mask

theta = np.mod(np.angle(z), np.pi)

plt_eta_theta(eta, theta, mask)
plt.savefig("Averaged.pdf", dpi=150)