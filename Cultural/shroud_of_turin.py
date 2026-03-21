"""
Inverse-Square Projection onto a Cylinder
------------------------------------------
Qualitative demonstration: a self-luminous body-like object emits
light radially. Each surface point emits with intensity I = I0 / r^2
where r is the distance from that surface point to the cylinder wall.
The cylinder records the integrated flux at each point on its inner surface.

Three body-like implicit surfaces are modelled:
  Object 1: x^2 + y^2 + z^2 + sin(2y)cos(2z) = 1
  Object 2: x^2 + y^2 + z^2 + sin(4xyz)       = 1
  Object 3: x^2 + y^2 + z^2 + cos(4x)sin(4y)  = 1

Wrap cylinder: y^2 + z^2 = 2  (infinite along x-axis)

For each object:
  Left panel  — 3D view of the implicit surface inside the cylinder
  Right panel — unwrapped cylinder (theta vs x) showing flux falloff image

# Set A
def f1(x, y, z):
    return x**2 + y**2 + z**2 + np.sin(2*y)*np.cos(2*z) - 1

def f2(x, y, z):
    return x**2 + y**2 + z**2 + np.sin(4*x*y*z) - 1

def f3(x, y, z):
    return x**2 + y**2 + z**2 + np.cos(4*x)*np.sin(4*y) - 1

objects = [
    (f1, "Object 1\n$x^2+y^2+z^2+\\sin(2y)\\cos(2z)=1$"),
    (f2, "Object 2\n$x^2+y^2+z^2+\\sin(4xyz)=1$"),
    (f3, "Object 3\n$x^2+y^2+z^2+\\cos(4x)\\sin(4y)=1$"),
]

# Set B
def f1(x, y, z):
    return x**2 + y**2 + z**2 + 1.5*np.sin(4*x)*np.sin(4*y)*np.sin(4*z) - 1

def f2(x, y, z):
    return x**2 + y**2 + z**2 + np.sin(5*x*z + 3*y*z)*np.cos(5*y*z) - 1

def f3(x, y, z):
    return x**2 + y**2 + z**2 + 2*np.cos(3*x)*np.cos(3*y)*np.cos(3*z) - 1

objects = [
    (f1, "Object 4\n$x^2+y^2+z^2+1.5\\sin(4x)\\sin(4y)\\sin(4z)=1$"),
    (f2, "Object 5\n$x^2+y^2+z^2+2\\sin(5xz+3yz)\\cos(5xy)=1$"),
    (f3, "Object 6\n$x^2+y^2+z^2+2\\cos(3x)\\cos(3y)\\cos(3z)=1$"),
]

# Set C
def f1(x, y, z):
    return x**2 + y**2 + z**2 + 0.4*np.sin(3*x**2 + 2*y)*np.cos(3*z) - 1

def f2(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2 + 1e-9)
    return x**2 + y**2 + z**2 + 0.35*np.sin(6*r)*np.cos(4*np.arctan2(y, x)) - 1

def f3(x, y, z):
    return x**2 + y**2 + z**2 + 0.3*np.sin(4*x + np.sin(3*y))*np.cos(4*z + np.cos(3*x)) - 1

objects = [
    (f1, "Object 7\n$x^2+y^2+z^2+0.4\\sin(3x^2+2y)\\cos(3z)=1$"),
    (f2, "Object 8\n$x^2+y^2+z^2+0.35\\sin(6r)\\cos(4\\theta)=1$"),
    (f3, "Object 9\n$x^2+y^2+z^2+0.3\\sin(4x+\\sin 3y)\\cos(4z+\\cos 3x)=1$"),
]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage import measure

# ── implicit surface definitions ──────────────────────────────────────────────

# Set A
def f1(x, y, z):
    return x**2 + y**2 + z**2 + 2*x*np.sin(2*y*z)*np.cos(2*x*z) - 1

def f2(x, y, z):
    return x**2 + y**2 + z**2 + np.sin(4*x*y*z) - 1

def f3(x, y, z):
    return x**2 + y**2 + z**2 + np.cos(4*x)*np.sin(4*y) - 1

objects = [
    (f1, "Object 1\n$x^2+y^2+z^2+2x\sin(2yz)\cos(2xz)=1$"),
    (f2, "Object 2\n$x^2+y^2+z^2+\sin(4xyz)=1$"),
    (f3, "Object 3\n$x^2+y^2+z^2+\cos(4x)\sin(4y)=1$"),
]

# ── grid for marching cubes ────────────────────────────────────────────────────
N = 120
lim = 1.6
coords = np.linspace(-lim, lim, N)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

# cylinder radius
R_cyl = np.sqrt(2)

# ── cylinder unwrap grid ───────────────────────────────────────────────────────
N_theta = 300
N_x     = 300
theta_vals = np.linspace(-np.pi, np.pi, N_theta)
x_vals     = np.linspace(-lim, lim, N_x)
THETA, XX  = np.meshgrid(theta_vals, x_vals, indexing='ij')

# cylinder surface points
CY = R_cyl * np.sin(THETA)   # shape (N_theta, N_x)
CZ = R_cyl * np.cos(THETA)

# ── inverse-square flux accumulation ──────────────────────────────────────────

def compute_flux(func, X, Y, Z, coords, lim, CY, CZ, XX, N_theta, N_x):
    """
    1. Extract isosurface vertices via marching cubes.
    2. For each cylinder pixel, sum I0/r^2 contributions from all
       surface vertices (sub-sampled for speed).
    """
    vol = func(X, Y, Z)
    # marching cubes
    verts, faces, _, _ = measure.marching_cubes(
        vol, level=0,
        spacing=(coords[1]-coords[0],)*3
    )
    # re-centre verts to world coords
    verts = verts + coords[0]

    # subsample surface points for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(verts), size=min(4000, len(verts)), replace=False)
    pts = verts[idx]          # (M, 3)  columns: x, y, z

    # flux image  (N_theta x N_x)
    flux = np.zeros((N_theta, N_x))

    # vectorised over surface points in batches
    batch = 500
    for start in range(0, len(pts), batch):
        p = pts[start:start+batch]          # (b, 3)
        px = p[:, 0][:, None, None]         # (b,1,1)
        py = p[:, 1][:, None, None]
        pz = p[:, 2][:, None, None]

        dx = XX[None, :, :] - px            # (b, N_theta, N_x)  -- wait, axes
        # XX shape is (N_theta, N_x), so broadcast needs adjustment
        dx = XX[None] - px                  # (b, N_theta, N_x)
        dy = CY[None] - py
        dz = CZ[None] - pz

        r2 = dx**2 + dy**2 + dz**2
        r2 = np.where(r2 < 1e-6, 1e-6, r2)

        # only points inside cylinder contribute (r_source < R_cyl)
        r_src = np.sqrt(p[:, 1]**2 + p[:, 2]**2)   # (b,)
        mask = (r_src < R_cyl)[:, None, None]

        flux += np.sum(np.where(mask, 1.0 / r2, 0.0), axis=0)

    return verts, faces, flux

# ── plotting ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('white')

titles_3d   = []
titles_wrap = []

for col, (func, label) in enumerate(objects):

    verts, faces, flux = compute_flux(
        func, X, Y, Z, coords, lim,
        CY, CZ, XX, N_theta, N_x
    )

    # ── 3D panel ──────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(3, 2, col*2 + 1, projection='3d')

    # draw implicit surface
    ax3d.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap='Greys', alpha=0.55, linewidth=0, antialiased=True
    )

    # draw cylinder wireframe
    th = np.linspace(-np.pi, np.pi, 60)
    xc = np.linspace(-lim*0.8, lim*0.8, 2)
    TH, XC = np.meshgrid(th, xc)
    YC = R_cyl * np.sin(TH)
    ZC = R_cyl * np.cos(TH)
    ax3d.plot_surface(XC, YC, ZC, alpha=0.08, color='steelblue', linewidth=0)
    ax3d.plot_wireframe(XC, YC, ZC, color='steelblue', alpha=0.25,
                        rstride=1, cstride=4, linewidth=0.4)

    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    ax3d.tick_params(labelsize=14)
    ax3d.set_title(label + "\n3D body inside cylinder",
                   fontsize=14, pad=4)
    ax3d.view_init(elev=22, azim=-55)

    # ── unwrapped cylinder panel ───────────────────────────────────────────────
    ax2d = fig.add_subplot(3, 2, col*2 + 2)

    # normalise flux and apply gamma for visual clarity
    f_norm = flux / flux.max()
    f_gamma = f_norm ** 0.45        # brighten midtones slightly

    im = ax2d.imshow(
        f_gamma,
        origin='lower',
        extent=[x_vals[0], x_vals[-1],
                np.degrees(theta_vals[0]), np.degrees(theta_vals[-1])],
        aspect='auto',
        cmap='gray_r',              # dark = high flux (like a photographic negative)
        vmin=0, vmax=1
    )

    ax2d.set_xlabel('x (along cylinder axis)', fontsize=14)
    ax2d.set_ylabel('θ (degrees around cylinder)', fontsize=14)
    ax2d.set_title(label + "\nUnwrapped cylinder — flux (dark = high)",
                   fontsize=14, pad=4)
    ax2d.tick_params(labelsize=14)

    cb = fig.colorbar(im, ax=ax2d, fraction=0.03, pad=0.03)
    cb.set_label('Relative flux', fontsize=14)
    cb.ax.tick_params(labelsize=14)

fig.suptitle(
    "Inverse-Square Projection onto a Cylinder\n"
    r"Body surface emits $I \propto 1/r^2$  —  cylinder records flux image"
    "\n(qualitative / mathematical — no physical units)",
    fontsize=14, y=1.01
)

plt.tight_layout()
plt.savefig('shroud_inverse_square_model.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved.")