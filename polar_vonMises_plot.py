import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

# Parameters
mu = np.pi / 3
kappa = 16.0
n_bins = 300

# Angular bin edges
theta_edges = np.linspace(0, 2*np.pi, n_bins + 1)

# Evaluate PDF at edges (not centers)
pdf_edges = vonmises.pdf(theta_edges, kappa, loc=mu) + vonmises.pdf(theta_edges + np.pi, kappa, loc=mu)

# Normalize for colormap
values = pdf_edges / pdf_edges.max()

# Radial extent (single ring)
r_edges = np.array([0.0, 1.0])

# 2D grid required by pcolormesh
Theta, R = np.meshgrid(theta_edges, r_edges)
Z = 1 - np.tile(values[:], (2, 1))  # shape (1, n_bins)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
pcm = ax.pcolormesh(
    Theta,
    R,
    Z,
    cmap="gray",
    shading="gouraud"  # enables smooth angular gradient
)

# Remove all theta ticks and labels
# ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])

# Draw the mean direction
ax.plot([mu, mu], [0, 1.0], color="C0", lw=1.5, ls="--")
ax.plot([mu + np.pi, mu + np.pi], [0, 1.0], color="C0", lw=1.5, ls="--")

# Arc parameters
r_arc = 0.5
theta_arc = np.linspace(0, mu, 200)
r_arc_vals = np.full_like(theta_arc, r_arc)

# Draw curved arc
ax.plot(
    theta_arc,
    r_arc_vals,
    color="C0",
    lw=1.5
)

# Arrowhead at the end of the arc
# Small angular offset to define arrow direction
dtheta = 0.01
ax.annotate(
    "",
    xy=(mu, r_arc),
    xytext=(mu - dtheta, r_arc),
    arrowprops=dict(
        arrowstyle="->",
        lw=1.5,
        color="C0"
    )
)

ax.text(
    mu / 2,
    r_arc + 0.07,
    r"$\theta_0$",
    ha="center",
    va="center"
)

ax.set_theta_zero_location("E")
ax.set_theta_direction(-1)

# plt.colorbar(pcm, ax=ax, pad=0.1, label="Normalized density")
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import vonmises

# # Parameters of the von Mises distribution
# mu = np.pi        # mean direction
# kappa = 3.0           # concentration parameter

# # Angular bins
# n_bins = 120
# theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
# width = 2 * np.pi / n_bins

# # Von Mises PDF evaluated at bin centers
# pdf = vonmises.pdf(theta, kappa, loc=mu)

# # Normalize PDF to [0, 1] for opacity
# alpha = pdf / pdf.max()

# # Optional: constant radius for all bars
# radius = np.ones_like(theta)

# plt.plot(theta, pdf)

# plt.show()
