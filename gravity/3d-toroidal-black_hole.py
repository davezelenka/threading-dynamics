import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# TOROIDAL BLACK HOLE PARAMETERS
# ----------------------------
M0 = 1.0e30       # Peak memory density [arbitrary units]
R0 = 5.0e3        # Annulus radius
sigma = 1.0e3     # Radial/vertical thickness
k = 1.0           # Gravity proportionality constant

# ----------------------------
# GRID SETUP
# ----------------------------
grid_size = 200
rho = np.linspace(0, 2*R0, grid_size)
z = np.linspace(-2*R0, 2*R0, grid_size)
rho_grid, z_grid = np.meshgrid(rho, z)

# ----------------------------
# MEMORY DENSITY FUNCTION
# ----------------------------
def M(rho, z):
    return M0 * np.exp(-((rho - R0)**2 + z**2) / (2 * sigma**2))

# Compute memory density
M_grid = M(rho_grid, z_grid)

# ----------------------------
# GRADIENT FIELD
# ----------------------------
dM_drho, dM_dz = np.gradient(M_grid, rho[1]-rho[0], z[1]-z[0])
g_rho = k * dM_drho
g_z = k * dM_dz
g_magnitude = np.sqrt(g_rho**2 + g_z**2)

# ----------------------------
# PREDICTIVE PHOTON RING
# ----------------------------
# Photon ring expected near maximum gradient
max_idx = np.unravel_index(np.argmax(g_magnitude, axis=None), g_magnitude.shape)
photon_ring_rho = rho_grid[max_idx]
photon_ring_z = z_grid[max_idx]

print(f"Predicted photon ring location: rho = {photon_ring_rho:.2f}, z = {photon_ring_z:.2f}")

# ----------------------------
# 3D VISUALIZATION
# ----------------------------
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Create a surface plot for memory density (annulus)
ax.plot_surface(rho_grid, z_grid, M_grid, cmap='viridis', alpha=0.8)

# Add gradient vectors (subsample for clarity)
step = 10
ax.quiver(rho_grid[::step, ::step], z_grid[::step, ::step], M_grid[::step, ::step],
          g_rho[::step, ::step], g_z[::step, ::step], 0, color='red', length=500, normalize=True)

# Highlight photon ring location
ax.scatter(photon_ring_rho, photon_ring_z, M_grid[max_idx], color='orange', s=100, label='Predicted photon ring')

ax.set_xlabel('rho')
ax.set_ylabel('z')
ax.set_zlabel('Memory density M')
ax.set_title('Fabric Toroidal Black Hole: Memory Density and Gradient Field')
ax.legend()
plt.show()
