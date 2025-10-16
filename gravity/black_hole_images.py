import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
M0 = 1.0       # peak memory density
R0 = 1.0       # torus major radius
sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # torus thicknesses
beta = 1.0     # Born-Infeld parameter for energy
grid_points = 200

# -------------------------------
# Grid setup
# -------------------------------
rho = np.linspace(0, 2.0, grid_points)
z = np.linspace(-1.0, 1.0, grid_points)
RHO, Z = np.meshgrid(rho, z)

# Storage for results
photon_ring_radii = []
E_totals = []

# -------------------------------
# Loop over thicknesses
# -------------------------------
for sigma in sigma_values:
    # Toroidal memory density
    M = M0 * np.exp(-((RHO - R0)**2 + Z**2)/(2*sigma**2))

    # Gradient magnitude
    dM_drho, dM_dz = np.gradient(M, rho[1]-rho[0], z[1]-z[0])
    g_mag = np.sqrt(dM_drho**2 + dM_dz**2)

    # Photon ring
    idx = np.unravel_index(np.argmax(g_mag), g_mag.shape)
    photon_ring_rho = RHO[idx]
    photon_ring_z = Z[idx]
    photon_ring_radii.append(photon_ring_rho)

    # Born-Infeld energy
    g_mag_clipped = np.clip(g_mag, 0, beta*0.999)
    E_density = beta**2 * (1 - np.sqrt(1 - g_mag_clipped**2 / beta**2))
    E_total = np.sum(E_density)*(rho[1]-rho[0])*(z[1]-z[0])
    E_totals.append(E_total)

    # -------------------------------
    # Plot memory density
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.contourf(RHO, Z, M, levels=50, cmap='viridis')
    plt.colorbar(label='Memory Density M')
    plt.title(f'Toroidal Memory Density (sigma={sigma})')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$z$')
    plt.savefig(f'torus_memory_density_sigma{sigma}.png', dpi=300)
    plt.close()

    # -------------------------------
    # Plot gradient magnitude + photon ring
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.contourf(RHO, Z, g_mag, levels=50, cmap='inferno')
    plt.plot(photon_ring_rho, photon_ring_z, 'ro', label='Photon Ring')
    plt.colorbar(label=r'$|\mathbf{g}|$')
    plt.title(f'Gradient Field (sigma={sigma})')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$z$')
    plt.legend()
    plt.savefig(f'gradient_field_sigma{sigma}.png', dpi=300)
    plt.close()

    print(f"sigma={sigma}: Photon ring at rho={photon_ring_rho:.3f}, z={photon_ring_z:.3f}, E_total={E_total:.3f}")

# -------------------------------
# Photon ring vs sigma
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(sigma_values, photon_ring_radii, 'bo-', label='Photon Ring Radius')
plt.xlabel(r'Torus Thickness $\sigma$')
plt.ylabel(r'Photon Ring Radius $\rho$')
plt.title('Photon Ring Radius vs Torus Thickness')
plt.grid(True)
plt.legend()
plt.savefig('photon_ring_vs_sigma.png', dpi=300)
plt.close()

# -------------------------------
# Total energy vs sigma
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(sigma_values, E_totals, 'ms-', label='Total Born-Infeld Energy')
plt.xlabel(r'Torus Thickness $\sigma$')
plt.ylabel('Integrated Energy')
plt.title('Born-Infeld Energy vs Torus Thickness')
plt.grid(True)
plt.legend()
plt.savefig('energy_vs_sigma.png', dpi=300)
plt.close()
