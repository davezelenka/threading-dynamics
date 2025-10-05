# ==========================================
# Fabric Dynamics — Toroidal Memory Density Model
# ==========================================
# Generates Figures 1–5 for the paper:
#   1. Toroidal Memory Density
#   2. Gradient Magnitude and Photon Ring
#   3. Photon Ring Radius vs Torus Thickness
#   4. Total Energy vs Torus Thickness
#   5. Optional: Total Energy vs β (Born–Infeld saturation)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
M0 = 1.0          # Peak memory density
R0 = 1.0          # Torus major radius
sigma_values = [0.1, 0.2, 0.5]  # Torus thicknesses to explore
beta_values = np.logspace(-1, 2, 10)  # Born–Infeld parameters (0.1 to 100)
grid_points = 300

# -------------------------------
# Grid setup
# -------------------------------
rho = np.linspace(0, 2.0, grid_points)
z = np.linspace(-1.0, 1.0, grid_points)
RHO, Z = np.meshgrid(rho, z)

# Storage for results
photon_ring_radii = []
total_energies = []

# -------------------------------
# Main computation loop over sigma
# -------------------------------
for sigma in sigma_values:

    # --- Toroidal memory density ---
    M = M0 * np.exp(-((RHO - R0)**2 + Z**2) / (2 * sigma**2))

    # --- Gradient magnitude ---
    dM_drho, dM_dz = np.gradient(M, rho[1] - rho[0], z[1] - z[0])
    g_mag = np.sqrt(dM_drho**2 + dM_dz**2)

    # --- Photon ring (maximum gradient) ---
    photon_ring_idx = np.unravel_index(np.argmax(g_mag), g_mag.shape)
    photon_ring_rho = RHO[photon_ring_idx]
    photon_ring_z = Z[photon_ring_idx]
    photon_ring_radii.append(photon_ring_rho)

    # --- Energy functional (Born–Infeld) ---
    beta = 1.0  # default β for baseline plots
    g_mag_clipped = np.clip(g_mag, 0, beta * 0.999)
    E_density = beta**2 * (1 - np.sqrt(1 - g_mag_clipped**2 / beta**2))
    E_total = np.sum(E_density) * (rho[1] - rho[0]) * (z[1] - z[0])
    total_energies.append(E_total)

    # -------------------------------
    # Plot 1: Toroidal Memory Density
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.contourf(RHO, Z, M, levels=80, cmap='viridis')
    plt.colorbar(label='Memory Density $M$')
    plt.title(f'Toroidal Memory Density ($\\sigma={sigma}$)')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$z$')
    plt.tight_layout()
    plt.savefig(f'figures/black_holes/figure_toroidal_density_sigma{sigma}.png', dpi=300)
    plt.close()

    # -------------------------------
    # Plot 2: Gradient Magnitude + Photon Ring
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.contourf(RHO, Z, g_mag, levels=80, cmap='inferno')
    plt.plot(photon_ring_rho, photon_ring_z, 'bo', label='Photon Ring')
    plt.colorbar(label=r'$|\nabla M|$')
    plt.title(f'Gradient Magnitude and Photon Ring ($\\sigma={sigma}$)')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$z$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/black_holes/figure_gradient_sigma{sigma}.png', dpi=300)
    plt.close()

    print(f"σ={sigma:.2f}: Photon ring at ρ={photon_ring_rho:.3f}, z={photon_ring_z:.3f}, E_total={E_total:.3f}")

# -------------------------------
# Plot 3: Photon Ring Radius vs Torus Thickness
# -------------------------------
plt.figure(figsize=(6, 5))
plt.plot(sigma_values, photon_ring_radii, 'o-', color='orange', linewidth=2)
plt.xlabel(r'Torus Thickness $\sigma$')
plt.ylabel(r'Photon Ring Radius $\rho_{\text{photon}}$')
plt.title('Photon Ring Radius vs Torus Thickness')
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/black_holes/figure_photon_ring_vs_sigma.png', dpi=300)
plt.close()

# -------------------------------
# Plot 4: Total Energy vs Torus Thickness
# -------------------------------
plt.figure(figsize=(6, 5))
plt.plot(sigma_values, total_energies, 'o-', color='teal', linewidth=2)
plt.xlabel(r'Torus Thickness $\sigma$')
plt.ylabel('Integrated Born–Infeld Energy')
plt.title('Total Energy vs Torus Thickness')
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/black_holes/figure_energy_vs_sigma.png', dpi=300)
plt.close()

# -------------------------------
# Optional Plot 5: Energy vs β
# -------------------------------
# Compute for fixed σ = 0.2
sigma = 0.2
M = M0 * np.exp(-((RHO - R0)**2 + Z**2) / (2 * sigma**2))
dM_drho, dM_dz = np.gradient(M, rho[1] - rho[0], z[1] - z[0])
g_mag = np.sqrt(dM_drho**2 + dM_dz**2)
g_max = np.max(g_mag)

E_total_beta = []
for beta in beta_values:
    g_mag_clipped = np.clip(g_mag, 0, beta * 0.999)
    E_density = beta**2 * (1 - np.sqrt(1 - g_mag_clipped**2 / beta**2))
    E_total = np.sum(E_density) * (rho[1] - rho[0]) * (z[1] - z[0])
    E_total_beta.append(E_total)

beta_norm = beta_values / g_max
E_total_beta = np.nan_to_num(E_total_beta, nan=0.0)

plt.figure(figsize=(6, 5))
plt.plot(beta_norm, E_total_beta, 'o-', color='purple', linewidth=2)
plt.xscale('log')
plt.xlabel(r'$\beta / g_{\max}$')
plt.ylabel('Integrated Energy')
plt.title('Total Energy vs Saturation Parameter β')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/black_holes/figure_energy_vs_beta.png', dpi=300)
plt.close()

print("All figures successfully generated.")
