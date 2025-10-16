# Time-dependent Fabric torus evolution + thin-lens ray-tracing prototype
# - Axisymmetric (rho,z) evolution of M(rho,z,t) with Born-Infeld flux limiter
# - Project M to 2D surface density Sigma(x,y) (thin lens)
# - Compute projected potential Phi via FFT convolution, deflection angle alpha = 2 * grad(Phi)/c^2 (thin-lens approx)
# - Map source brightness (proportional to Sigma) to image plane using lens equation and display synthetic image
#
# This is a prototype for qualitative, rapid exploration. Parameters and scales are in arbitrary units;
# choose them to explore different regimes. The code is intentionally simple and readable so it can be
# adapted into a more realistic solver and ray-tracer later.
#
# Requirements: numpy, scipy, matplotlib
# Execution: runs here and produces several PNG-like inline figures and console outputs.
#
# Author: Assistant (prototype for user's Fabric project)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.interpolate import RegularGridInterpolator

# -------------------------------
# Physical / numerical parameters (dimensionless units for prototype)
# -------------------------------
G = 1.0          # effective gravitational constant (set to 1 for units convenience)
c = 1.0          # speed of light in code units
g_max = 50.0     # saturation gradient scale (choose large so limiter acts when needed)
beta = g_max     # Born-Infeld parameter ~ g_max
D0 = 0.1         # base diffusion/mobility
dt = 0.001       # time step
n_steps = 200    # number of evolution steps to run (adjust for runtime)
output_every = 40

# grid in cylindrical (rho,z) for evolution
nr = 128
nz = 128
rho_max = 2.0
z_max = 2.0
rho = np.linspace(0.0, rho_max, nr)
z = np.linspace(-z_max, z_max, nz)
RHO, Z = np.meshgrid(rho, z, indexing='xy')  # RHO.shape = (nz, nr)

# initial torus parameters (you can param-scan these)
M0 = 1.0
R0 = 1.0
sigma = 0.2

# initialize M(rho,z)
def torus_M(RHO, Z, M0=1.0, R0=1.0, sigma=0.2):
    return M0 * np.exp(-(((RHO - R0)**2 + Z**2) / (2.0*sigma**2)))

M = torus_M(RHO, Z, M0=M0, R0=R0, sigma=sigma)

# convenience grid spacing (rho,z)
dr = rho[1] - rho[0]
dz = z[1] - z[0]

# source function S(rho,z,t) - for now, simple central inflow that adds M in inner region for a short time
def source_func(RHO, Z, t):
    # gaussian-in-time accretion near axis to seed collapse
    tt = np.exp(-((RHO/(0.2))**2 + (Z/0.2)**2))
    # time-windowed source
    return 0.5 * tt * np.exp(-((t-0.02)/0.02)**2)

# divergence operator in cylindrical coords for axisymmetric J = (J_r, J_z)
def div_cylindrical(Jr, Jz, rho, dr, dz):
    # Jr, Jz shape (nz, nr)
    # compute (1/rho) d(rho Jr)/dr + dJz/dz ; careful at rho=0 (use forward/backward)
    nz, nr = Jr.shape
    div = np.zeros_like(Jr)
    # compute d(rho*Jr)/dr
    rho_arr = rho[None, :]  # shape (1,nr)
    rhoJ = rho_arr * Jr
    d_rhoJ_dr = np.zeros_like(rhoJ)
    # central differences for interior
    d_rhoJ_dr[:,1:-1] = (rhoJ[:,2:] - rhoJ[:,:-2])/(2.0*dr)
    # one-sided boundaries
    d_rhoJ_dr[:,0] = (rhoJ[:,1] - rhoJ[:,0])/(dr)
    d_rhoJ_dr[:,-1] = (rhoJ[:,-1] - rhoJ[:,-2])/(dr)
    # term (1/rho) * d_rhoJ_dr ; handle rho=0 -> limit gives dJr/dr at 0
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_rho = np.where(rho_arr==0.0, 0.0, 1.0/rho_arr)
    term_r = inv_rho * d_rhoJ_dr
    # dJz/dz
    dJz_dz = np.zeros_like(Jz)
    dJz_dz[1:-1,:] = (Jz[2:,:] - Jz[:-2,:])/(2.0*dz)
    dJz_dz[0,:] = (Jz[1,:] - Jz[0,:])/(dz)
    dJz_dz[-1,:] = (Jz[-1,:] - Jz[-2,:])/(dz)
    div = term_r + dJz_dz
    return div

# gradient in cylindrical coords (returns components df/drho, df/dz)
def grad_cyl(f, rho, dr, dz):
    nz, nr = f.shape
    dfdrho = np.zeros_like(f)
    dfdz = np.zeros_like(f)
    # d/d rho (central)
    dfdrho[:,1:-1] = (f[:,2:] - f[:,:-2])/(2.0*dr)
    dfdrho[:,0] = (f[:,1] - f[:,0])/(dr)
    dfdrho[:,-1] = (f[:,-1] - f[:,-2])/(dr)
    # d/d z (central)
    dfdz[1:-1,:] = (f[2:,:] - f[:-2,:])/(2.0*dz)
    dfdz[0,:] = (f[1,:] - f[0,:])/(dz)
    dfdz[-1,:] = (f[-1,:] - f[-2,:])/(dz)
    return dfdrho, dfdz

# BI-limited flux given M and its gradients
def compute_flux(M, dr, dz, D0=0.1, g_max=50.0):
    dM_drho, dM_dz = grad_cyl(M, rho, dr, dz)
    grad_mag = np.sqrt(dM_drho**2 + dM_dz**2)
    # limiter factor: 1/sqrt(1-(|grad|/g_max)^2) ; avoid divide by zero
    frac = np.clip(grad_mag / g_max, 0.0, 0.999999)
    limiter = 1.0/np.sqrt(1.0 - frac**2)
    D = D0  # could be made function of M
    Jr = -D * dM_drho * limiter
    Jz = -D * dM_dz * limiter
    return Jr, Jz, grad_mag

# helper: project to 2D Cartesian grid (x,y) from axisymmetric M by integrating over z and rotating rho->x
# we'll build a cartesian grid for lensing (size Lx x Ly), and compute Sigma(x,y) by mapping rho->sqrt(x^2+y^2)
nx = 128
ny = 128
L = 4.0  # x,y range [-L/2, L/2]
x = np.linspace(-L/2, L/2, nx)
y = np.linspace(-L/2, L/2, ny)
X, Y = np.meshgrid(x, y, indexing='xy')

def project_M_to_Sigma(M, rho, z, X, Y):
    # Project M(rho,z) onto Sigma(x,y) = \int M(rho= sqrt(x^2+y^2), z) dz
    # We use interpolation: for each rho compute M_rho(z) via indexing
    # Build interpolator for M in (z,rho) coords: note ordering (z,rho)
    interp = RegularGridInterpolator((z, rho), M, bounds_error=False, fill_value=0.0)
    pts = np.stack([np.zeros_like(X.ravel()), X.ravel(), Y.ravel()], axis=-1)  # placeholder shape
    # Instead, compute rho at each (x,y)
    R_xy = np.sqrt(X**2 + Y**2)
    Sigma = np.zeros_like(R_xy)
    # integrate over z (simple trapezoid over z grid)
    for iz, zval in enumerate(z):
        # sample M at (rho=R_xy, z=zval)
        pts2 = np.stack([np.full(R_xy.size, zval), R_xy.ravel()], axis=-1)
        Mvals = interp(pts2).reshape(R_xy.shape)
        Sigma += Mvals
    Sigma *= (z[1]-z[0])
    return Sigma

# compute projected potential via FFT convolution on Cartesian grid: Phi = -G * (Sigma * kernel) where kernel = 1/r
def compute_Phi_from_Sigma(Sigma, x, y, G=1.0, eps=1e-6):
    nx, ny = Sigma.shape
    dx = x[1]-x[0]; dy = y[1]-y[0]
    # build kernel in same grid (centered)
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    # create kernel in spatial domain
    Xk, Yk = np.meshgrid(x, y, indexing='xy')
    R = np.sqrt(Xk**2 + Yk**2) + eps
    kernel = 1.0 / R  # Green's function in 2D ~ log(r), but for projection we use 1/r proxy to capture scaling
    # Use FFT-based convolution (note: convolution with 1/r in finite domain is approximate)
    fSigma = fft2(Sigma)
    fKernel = fft2(fftshift(kernel))
    conv = np.real(ifft2(fSigma * fKernel))
    Phi = -G * conv * dx * dy
    return Phi

# compute deflection angle alpha = 2 * grad(Phi)/c^2 (thin-lens approx) on cartesian grid
def compute_deflection(Phi, x, y, c=1.0):
    dx = x[1]-x[0]; dy = y[1]-y[0]
    dPhidx = np.zeros_like(Phi)
    dPhidy = np.zeros_like(Phi)
    dPhidx[:,1:-1] = (Phi[:,2:] - Phi[:,:-2])/(2.0*dx)
    dPhidx[:,0] = (Phi[:,1] - Phi[:,0])/(dx)
    dPhidx[:,-1] = (Phi[:,-1] - Phi[:,-2])/(dx)
    dPhidy[1:-1,:] = (Phi[2:,:] - Phi[:-2,:])/(2.0*dy)
    dPhidy[0,:] = (Phi[1,:] - Phi[0,:])/(dy)
    dPhidy[-1,:] = (Phi[-1,:] - Phi[-2,:])/(dy)
    alpha_x = 2.0 * dPhidx / (c**2)
    alpha_y = 2.0 * dPhidy / (c**2)
    return alpha_x, alpha_y

# lens mapping: image plane grid angles theta_x, theta_y correspond to pixels x,y (small-angle approx)
# lens equation: beta = theta - alpha(theta) ; here we identify theta with x,y in code units
def lens_mapping_and_image(Sigma_source, alpha_x, alpha_y, x, y):
    # compute source coordinates beta_x = X - alpha_x, beta_y = Y - alpha_y
    Ximg, Yimg = np.meshgrid(x, y, indexing='xy')
    beta_x = Ximg - alpha_x
    beta_y = Yimg - alpha_y
    # interpolate source brightness (Sigma_source) at (beta_x,beta_y)
    interp_src = RegularGridInterpolator((y, x), Sigma_source, bounds_error=False, fill_value=0.0)
    pts = np.stack([beta_y.ravel(), beta_x.ravel()], axis=-1)
    I_img = interp_src(pts).reshape(beta_x.shape)
    return I_img

# -------------------------------
# Evolution loop (time stepping)
# -------------------------------
history = {}
for step in range(n_steps):
    t = step * dt
    # compute source
    S = source_func(RHO, Z, t)
    # compute flux with BI limiter
    Jr, Jz, gradmag = compute_flux(M, dr, dz, D0=D0, g_max=g_max)
    # compute divergence
    divJ = div_cylindrical(Jr, Jz, rho, dr, dz)
    # update M: dM/dt = -div J + S
    M = M + dt * (-divJ + S)
    # enforce non-negativity and small floor
    M = np.maximum(M, 0.0)
    # optional: small diffusion for numerical stability
    if step % 20 == 0:
        M = M + 1e-6 * np.random.randn(*M.shape)
        M = np.maximum(M, 0.0)
    # snapshot outputs
    if step % output_every == 0 or step == n_steps-1:
        # compute grad magnitude for diagnostics
        _, _, gradmag = compute_flux(M, dr, dz, D0=D0, g_max=g_max)
        # project M to Sigma on cartesian grid
        Sigma = project_M_to_Sigma(M, rho, z, X, Y)
        # compute Phi, deflection
        Phi = compute_Phi_from_Sigma(Sigma, x, y, G=G)
        alpha_x, alpha_y = compute_deflection(Phi, x, y, c=c)
        # build lensed image
        I_img = lens_mapping_and_image(Sigma, alpha_x, alpha_y, x, y)
        # store to history for later plotting
        history[step] = dict(M=M.copy(), gradmag=gradmag.copy(), Sigma=Sigma.copy(), Phi=Phi.copy(), I_img=I_img.copy())
        print(f"step {step}/{n_steps-1}: t={t:.4f}, M_max={M.max():.4f}, grad_max={gradmag.max():.4f}, E_proj_sum={Sigma.sum():.4f}")

# -------------------------------
# Plot a few snapshots (M, grad, Sigma, image)
# -------------------------------
snap_steps = sorted(list(history.keys()))
for s in snap_steps:
    data = history[s]
    M_snap = data['M']
    grad_snap = data['gradmag']
    Sigma_snap = data['Sigma']
    I_snap = data['I_img']
    # plot M (rho,z)
    plt.figure(figsize=(6,4))
    plt.contourf(RHO, Z, M_snap, levels=50)
    plt.colorbar(label='M (rho,z)')
    plt.title(f'M at step {s}')
    plt.xlabel(r'$\rho$'); plt.ylabel('z')
    plt.show()
    # plot grad magnitude
    plt.figure(figsize=(6,4))
    plt.contourf(RHO, Z, grad_snap, levels=50)
    plt.colorbar(label='|grad M|')
    plt.title(f'|grad M| at step {s}')
    plt.xlabel(r'$\rho$'); plt.ylabel('z')
    plt.show()
    # plot Sigma (projected)
    plt.figure(figsize=(5,5))
    plt.contourf(X, Y, Sigma_snap, levels=50)
    plt.colorbar(label='Sigma (projected M)')
    plt.title(f'Projected Sigma at step {s}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()
    # plot lensed image
    plt.figure(figsize=(5,5))
    plt.imshow(I_snap, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.colorbar(label='Lensed intensity (arb units)')
    plt.title(f'Lensed image (thin-lens approx) at step {s}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

print("Prototype run complete. History keys (snapshots):", snap_steps)
