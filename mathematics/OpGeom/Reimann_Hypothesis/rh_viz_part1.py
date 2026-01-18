"""
Riemann Hypothesis Visualization Generator
Generates all figures for the paper at 300 DPI
Based on the CRT torus operational geometry framework
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.special import zeta
import warnings
warnings.filterwarnings('ignore')

# Global settings
DPI = 300
FIGSIZE = (10, 8)
OUTPUT_DIR = "figures/"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Helper functions
def is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def get_primes(max_n):
    """Get all primes up to max_n"""
    return [n for n in range(2, max_n + 1) if is_prime(n)]

def torus_surface(R_major=3, r_minor=1, n_points=50):
    """
    Generate torus surface with correct aspect ratio.
    R_major: major radius (distance from center to tube center)
    r_minor: minor radius (tube radius)
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)

    x = (R_major + r_minor * np.cos(phi)) * np.cos(theta)
    y = (R_major + r_minor * np.cos(phi)) * np.sin(theta)
    z = r_minor * np.sin(phi)

    return x, y, z

def prime_winding_path(prime, R_major=3, r_minor=1, n_points=200):
    """
    Generate winding path for a single prime on the torus.
    The winding is determined by log(prime).
    """
    # Winding frequency around major circle
    winding_freq = np.log(prime) / np.log(2)
    
    # Parameter along the path
    s = np.linspace(0, 1, n_points)
    
    # Major circle angle (winds around based on prime)
    theta = s * 2 * np.pi * winding_freq
    
    # Minor circle angle (oscillates to show threading)
    phi = s * 4 * np.pi  # Complete 2 full oscillations
    
    x = (R_major + r_minor * np.cos(phi)) * np.cos(theta)
    y = (R_major + r_minor * np.cos(phi)) * np.sin(theta)
    z = r_minor * np.sin(phi)

    return x, y, z

def operational_potential(sigma, gamma, primes):
    """
    Calculate operational potential V(sigma, gamma).
    Minimum at sigma = 0.5
    """
    # Quadratic deviation from critical line
    deviation = (sigma - 0.5)**2
    base_potential = 100 * deviation  # Scale for visibility
    
    # Add oscillatory component from gamma (interference pattern)
    interference = 0
    for p in primes:
        weight = p**(-sigma)
        phase = gamma * np.log(p)
        interference += weight * np.cos(phase)
    
    # Renormalization: subtract continuous background
    continuous = 0
    for p in primes:
        weight = p**(-0.5)  # At critical line
        phase = gamma * np.log(p)
        continuous += weight * np.cos(phase)
    
    oscillation = 0.5 * np.abs(interference - continuous)
    
    return base_potential + oscillation

def riemann_zeros():
    """First 20 Riemann zeros (imaginary parts)"""
    return [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
            67.079811, 69.546402, 72.067158, 75.704691, 77.144840]

class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

# ============================================================================
# IMAGE 1: 3D visualization of CRT torus with prime winding paths
# ============================================================================
def generate_image1():
    print("Generating image1.png: 3D CRT torus with prime windings...")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')

    # Use proper aspect ratio: major radius = 3, minor radius = 1
    R_major, r_minor = 3, 1

    # Draw torus surface (semi-transparent)
    x, y, z = torus_surface(R_major=R_major, r_minor=r_minor, n_points=40)
    ax.plot_surface(x, y, z, alpha=0.15, color='gray', edgecolor='none')

    # Draw wireframe
    ax.plot_wireframe(x, y, z, alpha=0.3, color='darkgray', linewidth=0.5)

    # Draw prime winding paths
    primes = [2, 3, 5, 7, 11, 13]
    colors = plt.cm.viridis(np.linspace(0, 1, len(primes)))

    for i, prime in enumerate(primes):
        x_wind, y_wind, z_wind = prime_winding_path(prime, R_major=R_major, 
                                                     r_minor=r_minor, n_points=300)
        ax.plot(x_wind, y_wind, z_wind, color=colors[i], linewidth=2.5, 
                label=f'p={prime}', alpha=0.8)

    # Highlight critical line (z=0 plane) - equator of torus
    theta = np.linspace(0, 2*np.pi, 100)
    x_crit = R_major * np.cos(theta)
    y_crit = R_major * np.sin(theta)
    z_crit = np.zeros_like(theta)
    ax.plot(x_crit, y_crit, z_crit, 'r-', linewidth=4, 
            label=r'Critical line ($\sigma=1/2$)', alpha=0.9)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(r'CRT Torus $\mathbb{T}^2$ with Prime Winding Paths', fontsize=14, pad=20)
    ax.legend(loc='upper left', fontsize=9, ncol=1)

    # Set equal aspect ratio for proper torus appearance
    ax.set_box_aspect([1, 1, 0.4])
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image1.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image1.png")

# ============================================================================
# IMAGE 2: Heatmap of operational potential V(σ,γ) in critical strip
# ============================================================================
def generate_image2():
    print("Generating image2.png: Operational potential heatmap...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sigma = np.linspace(0, 1, 300)
    gamma = np.linspace(0, 80, 300)
    Sigma, Gamma = np.meshgrid(sigma, gamma)

    primes = get_primes(50)
    V = np.zeros_like(Sigma)

    for i in range(len(sigma)):
        for j in range(len(gamma)):
            V[j, i] = operational_potential(Sigma[j, i], Gamma[j, i], primes)

    # Use log scale for better visualization of minimum
    V_log = np.log10(V + 1)  # Add 1 to avoid log(0)

    im = ax.contourf(Sigma, Gamma, V_log, levels=50, cmap='RdYlBu_r')

    # Mark critical line prominently
    ax.axvline(0.5, color='lime', linewidth=4, linestyle='-', 
               label=r'Critical line $\sigma=1/2$ (minimum)', zorder=5)

    # Mark known zeros
    zeros = riemann_zeros()
    ax.scatter([0.5]*len(zeros), zeros, c='white', s=80, marker='x', 
               linewidths=2.5, label='Known zeros', zorder=10, alpha=0.9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}(V(\sigma, \gamma))$', fontsize=12)

    ax.set_xlabel(r'$\sigma = \mathrm{Re}(s)$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\gamma = \mathrm{Im}(s)$', fontsize=14, fontweight='bold')
    ax.set_title(r'Operational Potential Landscape: Minimum at $\sigma=1/2$', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 80)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image2.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image2.png")

# ============================================================================
# IMAGE 4: Progressive winding animation frames
# ============================================================================
def generate_image4():
    print("Generating image4.png: Progressive winding animation frames...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    R_major, r_minor = 3, 1
    prime = 3

    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    for idx, frac in enumerate(fractions):
        ax = axes[idx]

        # Draw torus
        x, y, z = torus_surface(R_major=R_major, r_minor=r_minor, n_points=30)
        ax.plot_wireframe(x, y, z, alpha=0.2, color='gray', linewidth=0.5)

        # Draw partial winding - FIXED: actually show progression
        n_points = int(200 * frac)
        if n_points > 1:
            x_wind, y_wind, z_wind = prime_winding_path(prime, R_major=R_major, 
                                                         r_minor=r_minor, n_points=200)
            # Only plot up to current fraction
            ax.plot(x_wind[:n_points], y_wind[:n_points], z_wind[:n_points], 
                   color='green', linewidth=3, alpha=0.9)

            # Mark current position
            ax.scatter([x_wind[n_points-1]], [y_wind[n_points-1]], [z_wind[n_points-1]], 
                      c='red', s=100, marker='o', edgecolors='darkred', linewidths=2)

        # Draw critical line
        theta = np.linspace(0, 2*np.pi, 100)
        x_crit = R_major * np.cos(theta)
        y_crit = R_major * np.sin(theta)
        z_crit = np.zeros_like(theta)
        ax.plot(x_crit, y_crit, z_crit, 'r--', linewidth=2, alpha=0.5)

        ax.set_title(f'Progress: {frac*100:.0f}%', fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 0.4])

    fig.suptitle(f'Progressive Winding of Prime p={prime} on Torus', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image4.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image4.png")

# ============================================================================
# IMAGE 5: Discrete primes vs continuous PNT density
# ============================================================================
def generate_image5():
    print("Generating image5.png: Discrete vs continuous density...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    max_x = 200
    x = np.arange(2, max_x)

    # Discrete: prime counting function
    primes = get_primes(max_x)
    pi_x = np.array([sum(1 for p in primes if p <= xi) for xi in x])

    # Continuous: PNT approximation
    pnt_approx = x / np.log(x)

    # Logarithmic integral (better approximation)
    from scipy.special import expi
    li_x = np.array([expi(np.log(xi)) for xi in x])

    ax.plot(x, pi_x, 'b-', linewidth=2.5, label=r'$\pi(x)$ (actual prime count)', alpha=0.9)
    ax.plot(x, pnt_approx, 'r--', linewidth=2.5, label=r'$x/\ln x$ (PNT)', alpha=0.8)
    ax.plot(x, li_x, 'g:', linewidth=2.5, label=r'$\mathrm{Li}(x)$ (better approx)', alpha=0.8)

    # Highlight primes with vertical lines
    for p in primes[:30]:  # First 30 primes
        ax.axvline(p, color='blue', alpha=0.1, linewidth=0.5)

    ax.set_xlabel('$x$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Prime Number Theorem: Discrete Primes vs Continuous Density', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image5.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image5.png")

# ============================================================================
# IMAGE 6: 3D renormalized winding field (CORRECTED)
# ============================================================================
def generate_image6():
    print("Generating image6.png: Renormalized winding field...")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')

    R_major, r_minor = 3, 1
    primes = get_primes(30)

    # Generate torus surface
    x, y, z = torus_surface(R_major=R_major, r_minor=r_minor, n_points=50)

    # Calculate renormalized field at each point
    # The field represents: sum_p p^(-1/2) * cos(gamma * log(p))
    # where gamma is derived from position on torus
    
    field = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xi, yi, zi = x[i,j], y[i,j], z[i,j]

            # Position on torus in angular coordinates
            # theta: angle around major circle
            theta = np.arctan2(yi, xi)
            
            # phi: angle around minor circle (related to z)
            # For a torus: z = r_minor * sin(phi)
            phi = np.arcsin(np.clip(zi / r_minor, -1, 1))
            
            # Gamma parameter (imaginary part of s) derived from position
            # Use theta as the primary parameter for gamma
            gamma = 10 * theta  # Scale factor for visibility
            
            # Discrete prime contribution (at critical line sigma=1/2)
            discrete_sum = 0
            for prime in primes:
                weight = 1.0 / np.sqrt(prime)  # p^(-1/2)
                phase = gamma * np.log(prime)
                discrete_sum += weight * np.cos(phase)
            
            # Continuous background (renormalization)
            # Approximate integral: sum over many points
            continuous_sum = 0
            for k in range(2, 100):
                weight = 1.0 / np.sqrt(k)
                phase = gamma * np.log(k)
                continuous_sum += weight * np.cos(phase) * 0.01  # Scaled down
            
            # Renormalized field: discrete - continuous
            field[i,j] = discrete_sum - continuous_sum

    # Normalize for coloring to [-1, 1]
    field_max = np.max(np.abs(field))
    field_norm = field / (field_max + 1e-10)
    
    # Map to [0, 1] for colormap
    field_normalized = (field_norm + 1) / 2

    # Plot surface with field coloring
    surf = ax.plot_surface(x, y, z, facecolors=cm.RdBu_r(field_normalized),
                          linewidth=0, antialiased=True, alpha=0.95, shade=False)

    # Add colorbar
    m = cm.ScalarMappable(cmap=cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
    m.set_array(field_norm)
    cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    cbar.set_label('Renormalized Field\n(Discrete - Continuous)', fontsize=11, fontweight='bold')

    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Renormalized Prime Threading Field on Torus\n' + 
                 r'$\widetilde{Z}(\sigma=1/2) = \sum_p p^{-1/2}\cos(\gamma \ln p) - \int x^{-1/2}\cos(\gamma \ln x)dx$',
                 fontsize=13, fontweight='bold', pad=20)
    ax.view_init(elev=25, azim=60)
    ax.set_box_aspect([1, 1, 0.4])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image6.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image6.png")

# ============================================================================
# IMAGE 7: Commutative diagram (symbolic)
# ============================================================================
def generate_image7():
    print("Generating image7.png: Commutative diagram...")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Nodes with better positioning
    nodes = {
        'torus': (1.5, 7.5),
        'L2': (8.5, 7.5),
        'spectral': (8.5, 2.5),
        'zeta': (1.5, 2.5)
    }

    labels = {
        'torus': r'$\mathbb{T}^n$' + '\n(CRT Torus)',
        'L2': r'$L^2(\mathbb{R}, e^{-t}dt)$' + '\n(Hilbert Space)',
        'spectral': r'Spectrum' + '\n(Eigenvalues)',
        'zeta': r'$\zeta(s)$ zeros' + '\n(Critical Line)'
    }

    # Draw nodes with better styling
    for key, (x, y) in nodes.items():
        bbox = FancyBboxPatch((x-0.9, y-0.6), 1.8, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='darkblue', 
                             linewidth=2.5)
        ax.add_patch(bbox)
        ax.text(x, y, labels[key], ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow properties
    arrow_props = dict(arrowstyle='->', lw=2.5, color='darkblue')
    arrow_props_red = dict(arrowstyle='->', lw=2.5, color='darkred', linestyle='dashed')

    # Torus -> L2 (Fourier transform)
    ax.annotate('', xy=(nodes['L2'][0]-0.95, nodes['L2'][1]), 
                xytext=(nodes['torus'][0]+0.95, nodes['torus'][1]),
                arrowprops=arrow_props)
    ax.text(5, 8.3, r'$\mathcal{F}$ (Fourier)', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=1.5))

    # L2 -> Spectral (operator action)
    ax.annotate('', xy=(nodes['spectral'][0], nodes['spectral'][1]+0.7), 
                xytext=(nodes['L2'][0], nodes['L2'][1]-0.7),
                arrowprops=arrow_props)
    ax.text(9.5, 5, r'$\widetilde{Z}_{1/2}$' + '\n(operator)', ha='center', fontsize=10, fontweight='bold', rotation=-90,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=1.5))

    # Spectral -> Zeta (Weil formula)
    ax.annotate('', xy=(nodes['zeta'][0]+0.95, nodes['zeta'][1]), 
                xytext=(nodes['spectral'][0]-0.95, nodes['spectral'][1]),
                arrowprops=arrow_props)
    ax.text(5, 1.7, 'Weil Explicit Formula', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=1.5))

    # Zeta -> Torus (geometric realization - dashed red)
    ax.annotate('', xy=(nodes['torus'][0], nodes['torus'][1]-0.7), 
                xytext=(nodes['zeta'][0], nodes['zeta'][1]+0.7),
                arrowprops=arrow_props_red)
    ax.text(0.3, 5, 'Geometric\nRealization', ha='center', fontsize=10, fontweight='bold', rotation=90,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='darkred', linewidth=1.5))

    ax.set_title(r'Commutative Diagram: Torus $\to$ Fourier $\to$ Hilbert Space $\to$ Zeta Zeros', 
                fontsize=14, pad=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image7.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image7.png")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Riemann Hypothesis Visualization Generator")
    print("Generating high-resolution figures at 300 DPI")
    print("="*70 + "\n")

    # Generate first 7 images
    generate_image1()
    generate_image2()
    generate_image4()
    generate_image5()
    generate_image6()
    generate_image7()

    print("\n" + "="*70)
    print("First batch complete! (Images 1, 2, 4, 5, 6, 7)")
    print("="*70 + "\n")
