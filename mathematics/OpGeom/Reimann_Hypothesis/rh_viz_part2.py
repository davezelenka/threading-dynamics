"""
Riemann Hypothesis Visualization Generator - Part 2
Images 8-13 (CORRECTED)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# Global settings
DPI = 300
FIGSIZE = (10, 8)
OUTPUT_DIR = "figures/"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def get_primes(max_n):
    return [n for n in range(2, max_n + 1) if is_prime(n)]

def torus_surface(R_major=3, r_minor=1, n_points=50):
    """Generate torus with correct aspect ratio"""
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)
    
    x = (R_major + r_minor * np.cos(phi)) * np.cos(theta)
    y = (R_major + r_minor * np.cos(phi)) * np.sin(theta)
    z = r_minor * np.sin(phi)
    return x, y, z

def riemann_zeros():
    return [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
            67.079811, 69.546402, 72.067158, 75.704691, 77.144840]

# ============================================================================
# IMAGE 8: Conjugation symmetry on torus (FIXED ASPECT RATIO)
# ============================================================================
def generate_image8():
    print("Generating image8.png: Conjugation symmetry...")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')

    R_major, r_minor = 3, 1
    x, y, z = torus_surface(R_major=R_major, r_minor=r_minor, n_points=40)

    # Draw torus
    ax.plot_surface(x, y, z, alpha=0.2, color='lightgray', edgecolor='none')

    # Critical line (z=0 plane)
    theta = np.linspace(0, 2*np.pi, 100)
    x_crit = R_major * np.cos(theta)
    y_crit = R_major * np.sin(theta)
    z_crit = np.zeros_like(theta)
    ax.plot(x_crit, y_crit, z_crit, 'r-', linewidth=4, label=r'$\sigma=1/2$ (fixed line)')

    # Show symmetric points under conjugation s -> 1-s
    n_examples = 8
    for i in range(n_examples):
        angle = 2 * np.pi * i / n_examples
        z_val = 0.6

        # Point at sigma > 1/2
        x1 = R_major * np.cos(angle)
        y1 = R_major * np.sin(angle)
        z1 = z_val

        # Conjugate point at sigma < 1/2
        x2 = x1
        y2 = y1
        z2 = -z_val

        ax.scatter([x1], [y1], [z1], c='blue', s=100, alpha=0.7)
        ax.scatter([x2], [y2], [z2], c='green', s=100, alpha=0.7)

        # Connect with line
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'k--', alpha=0.4, linewidth=1)

    # Add reflection plane
    theta_plane = np.linspace(0, 2*np.pi, 20)
    r_plane = np.linspace(0, R_major*1.5, 10)
    Theta, R_plane = np.meshgrid(theta_plane, r_plane)
    X_plane = R_plane * np.cos(Theta)
    Y_plane = R_plane * np.sin(Theta)
    Z_plane = np.zeros_like(X_plane)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color='red')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(r'Conjugation Symmetry: $s \leftrightarrow 1-s$ reflects through $\sigma=1/2$', 
                fontsize=13, pad=20, fontweight='bold')
    ax.legend(fontsize=10)
    ax.view_init(elev=15, azim=30)
    ax.set_box_aspect([1, 1, 0.4])  # FIX: Correct aspect ratio

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image8.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image8.png")

# ============================================================================
# IMAGE 9: Cross-section showing parabolic minimum (FIXED LOGIC)
# ============================================================================
def generate_image9():
    print("Generating image9.png: Potential cross-section...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sigma = np.linspace(0, 1, 500)
    gamma_fixed = 14.134725  # First zero

    primes = get_primes(50)
    
    # Quadratic base (deviation from critical line)
    V = 100 * (sigma - 0.5)**2
    
    # Add small oscillatory component from interference
    for p in primes:
        weight = p**(-0.5)
        phase = gamma_fixed * np.log(p)
        oscillation = weight * np.cos(phase + sigma * np.pi)
        V += 2 * np.abs(oscillation)

    ax.plot(sigma, V, 'b-', linewidth=3, label=r'$V(\sigma, \gamma_1)$ where $\gamma_1 = 14.13$')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=3, label=r'Critical line $\sigma=1/2$')

    # Mark minimum
    min_idx = np.argmin(V)
    ax.scatter([sigma[min_idx]], [V[min_idx]], c='red', s=300, marker='*', 
              zorder=5, edgecolors='darkred', linewidths=2, label='Global minimum')

    ax.fill_between(sigma, 0, V, alpha=0.15, color='blue')

    ax.set_xlabel(r'$\sigma = \mathrm{Re}(s)$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$V(\sigma, \gamma)$', fontsize=14, fontweight='bold')
    ax.set_title(r'Operational Potential Cross-Section: Parabolic Minimum at $\sigma=1/2$', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add annotation
    ax.annotate('Unique gradient\nminimum', xy=(0.5, V[min_idx]), 
               xytext=(0.25, V.max()*0.6),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red'),
               fontsize=11, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='orange', linewidth=2))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image9.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image9.png")

# ============================================================================
# IMAGE 10: Phasor diagram (FIXED: Show full range)
# ============================================================================
def generate_image10():
    print("Generating image10.png: Phasor diagram...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    primes = get_primes(20)
    gamma = 14.134725  # First zero

    # Left: Constructive interference (off critical line)
    sigma1 = 0.6
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 1)
    ax1.set_ylim(-2, 1)
    ax1.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color='k', linewidth=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    cumulative = 0 + 0j
    for i, p in enumerate(primes):
        weight = p**(-sigma1)
        phase = gamma * np.log(p)
        phasor = weight * np.exp(1j * phase)

        # Draw phasor
        ax1.arrow(cumulative.real, cumulative.imag, phasor.real, phasor.imag,
                 head_width=0.08, head_length=0.08, fc=f'C{i%10}', ec=f'C{i%10}',
                 alpha=0.7, length_includes_head=True, linewidth=1.5)

        cumulative += phasor

    # Final sum
    ax1.scatter([cumulative.real], [cumulative.imag], c='red', s=250, marker='o',
               edgecolors='darkred', linewidths=2.5, zorder=10, label=f'Sum ≈ {cumulative.real:.2f}')

    # Draw unit circle for reference
    circle = plt.Circle((0, 0), 1, fill=False, linestyle=':', color='gray', linewidth=1, alpha=0.5)
    ax1.add_patch(circle)

    ax1.set_title(r'$\sigma = 0.6$ (Off critical line)' + '\nConstructive interference', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Real', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Imaginary', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')

    # Right: Destructive interference (on critical line)
    sigma2 = 0.5
    ax2.set_aspect('equal')
    ax2.set_xlim(-3, 1)
    ax2.set_ylim(-2, 1)
    ax2.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax2.axvline(0, color='k', linewidth=0.5, alpha=0.5)
    ax2.grid(True, alpha=0.3)

    cumulative = 0 + 0j
    for i, p in enumerate(primes):
        weight = p**(-sigma2)
        phase = gamma * np.log(p)
        phasor = weight * np.exp(1j * phase)

        ax2.arrow(cumulative.real, cumulative.imag, phasor.real, phasor.imag,
                 head_width=0.08, head_length=0.08, fc=f'C{i%10}', ec=f'C{i%10}',
                 alpha=0.7, length_includes_head=True, linewidth=1.5)

        cumulative += phasor

    # Final sum (near zero for a true zero)
    ax2.scatter([cumulative.real], [cumulative.imag], c='green', s=250, marker='o',
               edgecolors='darkgreen', linewidths=2.5, zorder=10, label=f'Sum ≈ {cumulative.real:.2f}')

    # Draw unit circle for reference
    circle = plt.Circle((0, 0), 1, fill=False, linestyle=':', color='gray', linewidth=1, alpha=0.5)
    ax2.add_patch(circle)

    ax2.set_title(r'$\sigma = 1/2$ (Critical line)' + '\nDestructive interference', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Real', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Imaginary', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')

    fig.suptitle('Phasor Diagram: Prime Threading Interference', fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image10.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image10.png")

# ============================================================================
# IMAGE 11: Winding paths for different sigma (FIXED: Show differences + aspect)
# ============================================================================
def generate_image11():
    print("Generating image11.png: Winding comparison...")

    fig = plt.figure(figsize=(16, 5))

    R_major, r_minor = 3, 1
    prime = 7
    sigmas = [0.4, 0.5, 0.6]
    titles = [r'$\sigma = 0.4$ (Under-weighted)', 
              r'$\sigma = 0.5$ (Balanced)',
              r'$\sigma = 0.6$ (Over-weighted)']

    for idx, (sigma, title) in enumerate(zip(sigmas, titles)):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # Torus
        x, y, z = torus_surface(R_major=R_major, r_minor=r_minor, n_points=30)
        ax.plot_wireframe(x, y, z, alpha=0.15, color='gray', linewidth=0.5)

        # Winding with weight factor - FIXED: Show actual difference
        n_points = 300
        s = np.linspace(0, 1, n_points)

        # Weight affects the pitch/frequency of winding
        winding_freq = np.log(prime) / np.log(2)
        
        # Sigma affects how much the path deviates from equator
        deviation = (sigma - 0.5) * 2  # Range from -1 to 1
        
        theta = s * 2 * np.pi * winding_freq
        phi = s * 4 * np.pi + deviation * np.pi  # Different starting phase
        
        x_wind = (R_major + r_minor * np.cos(phi)) * np.cos(theta)
        y_wind = (R_major + r_minor * np.cos(phi)) * np.sin(theta)
        z_wind = r_minor * np.sin(phi)

        # Color by position along path
        colors = plt.cm.viridis(s)
        for i in range(len(s)-1):
            ax.plot(x_wind[i:i+2], y_wind[i:i+2], z_wind[i:i+2], 
                   color=colors[i], linewidth=2.5, alpha=0.85)

        # Mark critical line
        theta_crit = np.linspace(0, 2*np.pi, 100)
        x_crit = R_major * np.cos(theta_crit)
        y_crit = R_major * np.sin(theta_crit)
        z_crit = np.zeros_like(theta_crit)
        ax.plot(x_crit, y_crit, z_crit, 'r-', linewidth=3, alpha=0.7, label='Critical line')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 0.4])  # FIX: Correct aspect ratio

    fig.suptitle(f'Prime Winding for p={prime}: Effect of Weight Parameter $\\sigma$', 
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image11.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image11.png")

# ============================================================================
# IMAGE 12: Gradient flow field (FIXED: Proper aspect and content)
# ============================================================================
def generate_image12():
    print("Generating image12.png: Gradient flow field...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sigma = np.linspace(0.1, 0.9, 25)
    gamma = np.linspace(5, 50, 25)
    Sigma, Gamma = np.meshgrid(sigma, gamma)

    # Calculate gradient: points toward sigma=0.5
    U = -(Sigma - 0.5) * 2  # Gradient in sigma direction
    V_field = np.zeros_like(Gamma)  # No gamma flow

    # Streamlines
    strm = ax.streamplot(Sigma, Gamma, U, V_field, color='blue', 
                        density=1.5, linewidth=1.2, arrowsize=1.8, arrowstyle='->')

    # Mark critical line
    ax.axvline(0.5, color='red', linewidth=4, linestyle='-', 
              label=r'Critical line $\sigma=1/2$ (attractor)', zorder=10)

    # Mark known zeros
    zeros = riemann_zeros()
    ax.scatter([0.5]*len(zeros), zeros, c='red', s=120, marker='x', 
              linewidths=2.5, label='Known zeros', zorder=15, alpha=0.9)

    ax.set_xlabel(r'$\sigma = \mathrm{Re}(s)$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\gamma = \mathrm{Im}(s)$', fontsize=14, fontweight='bold')
    ax.set_title('Operational Gradient Flow: All Paths Lead to Critical Line', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(5, 50)

    # Add annotations
    ax.annotate('Gradient minimum\n(attractor basin)', xy=(0.5, 25), 
               xytext=(0.75, 15),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='darkred'),
               fontsize=11, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image12.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image12.png")

# ============================================================================
# IMAGE 13: Reconstruction cost comparison (FIXED: Annotation position)
# ============================================================================
def generate_image13():
    print("Generating image13.png: Reconstruction cost...")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    N = np.arange(10, 201, 5)  # Number of primes

    # At sigma = 0.5: polynomial cost
    cost_half = N / np.log(N + 1)

    # At sigma = 0.6: exponential cost
    cost_off = np.exp(0.05 * N / np.log(N + 1))

    ax.semilogy(N, cost_half, 'b-', linewidth=3, label=r'$\sigma = 0.5$ (polynomial)', 
               marker='o', markersize=5, markevery=3)
    ax.semilogy(N, cost_off, 'r-', linewidth=3, label=r'$\sigma = 0.6$ (exponential)', 
               marker='s', markersize=5, markevery=3)

    # Shade the gap
    ax.fill_between(N, cost_half, cost_off, alpha=0.2, color='yellow', 
                    label='IOT overhead gap')

    ax.set_xlabel('Number of Primes $N$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Reconstruction Cost $C_{\mathrm{rec}}(N)$', fontsize=14, fontweight='bold')
    ax.set_title('Reconstruction Cost: On-Line vs Off-Line Zeros', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_ylim(1, 1e2)

    # Add annotation - FIXED: Position it inside the plot area
    ax.annotate('Exponential\nseparation\n(IOT bound)', xy=(150, cost_off[28]), 
               xytext=(80, 1e1),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='darkred', connectionstyle='arc3,rad=0.3'),
               fontsize=11, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='darkred', linewidth=2))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}image13.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved image13.png")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Riemann Hypothesis Visualization Generator - Part 2")
    print("Generating images 8-13 at 300 DPI (CORRECTED)")
    print("="*70 + "\n")

    generate_image8()
    generate_image9()
    generate_image10()
    generate_image11()
    generate_image12()
    generate_image13()

    print("\n" + "="*70)
    print("Images 8-13 complete!")
    print(f"Check the '{OUTPUT_DIR}' directory for PNG files")
    print("="*70 + "\n")