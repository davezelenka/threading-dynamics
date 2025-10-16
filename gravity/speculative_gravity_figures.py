import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- PARAMETERS ----------
M0 = 1.0          # Peak memory density
R0 = 1.0          # Toroid major radius
sigma = 0.2       # Toroid thickness
k = 1.0           # Gravity proportionality constant
beta = 1.0        # Gradient saturation parameter

# ---------- GRID ----------
N = 200
rho = np.linspace(0, 2.0, N)
z = np.linspace(-1.0, 1.0, N)
R, Z = np.meshgrid(rho, z)

# ---------- MEMORY DENSITY ----------
M = M0 * np.exp(-((R - R0)**2 + Z**2)/(2 * sigma**2))

# ---------- GRADIENTS ----------
dM_dR, dM_dZ = np.gradient(M, rho[1]-rho[0], z[1]-z[0])
g_mag = np.sqrt(dM_dR**2 + dM_dZ**2)

# ---------- SATURATED ENERGY DENSITY (Born-Infeld-like) ----------
E_density = beta**2 * (1 - np.sqrt(np.maximum(0, 1 - g_mag**2/beta**2)))  # avoid sqrt negative

# ---------- PHOTON RING (max gradient) ----------
max_idx = np.unravel_index(np.argmax(g_mag), g_mag.shape)
rho_photon = R[max_idx]
z_photon = Z[max_idx]
print(f"Photon ring located at rho={rho_photon:.3f}, z={z_photon:.3f}")

# ---------- INTEGRATED ENERGY ----------
# Simple trapezoid integration
integrated_energy = np.trapz(np.trapz(E_density * 2*np.pi*R, z, axis=0), rho)
print(f"Total integrated energy (toroid): {integrated_energy:.3f}")

# ---------- FIGURE 1: Toroid Memory Density ----------
plt.figure(figsize=(8,6))
plt.contourf(R, Z, M, levels=50, cmap='viridis')
plt.colorbar(label='Memory Density M')
plt.scatter(rho_photon, z_photon, color='red', label='Photon Ring')
plt.xlabel('rho')
plt.ylabel('z')
plt.title('Toroidal Memory Density and Photon Ring')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/toroid_core_placeholder.png', dpi=300)
plt.close()

# ---------- FIGURE 2: Gradient Magnitude ----------
plt.figure(figsize=(8,6))
plt.contourf(R, Z, g_mag, levels=50, cmap='plasma')
plt.colorbar(label='|∇M|')
plt.scatter(rho_photon, z_photon, color='white', label='Photon Ring')
plt.xlabel('rho')
plt.ylabel('z')
plt.title('Gradient Magnitude |∇M|')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/flow_vorticity_placeholder.png', dpi=300)
plt.close()

# ---------- FIGURE 3: Energy Density ----------
plt.figure(figsize=(8,6))
plt.contourf(R, Z, E_density, levels=50, cmap='inferno')
plt.colorbar(label='Energy Density E')
plt.scatter(rho_photon, z_photon, color='cyan', label='Photon Ring')
plt.xlabel('rho')
plt.ylabel('z')
plt.title('Saturated Energy Density')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/energy_density.png', dpi=300)
plt.close()