import numpy as np
import matplotlib.pyplot as plt
from sympy import factorint

# -----------------------
# PARAMETERS TO ADJUST
# -----------------------
N_min = 1
N_max = 5000

p = 5        # first modulus (small prime)
q = 7        # second modulus (small prime)

R = 3.0      # major radius of torus
r = 1.0      # minor radius of torus

height_scale = 0.4   # how strongly Ω(n) lifts points
alpha = 0.8          # transparency

# -----------------------
# DATA
# -----------------------
N = np.arange(N_min, N_max)

# residues mapped to angles
theta_p = 2 * np.pi * (N % p) / p
theta_q = 2 * np.pi * (N % q) / q

# Ω(n): total number of prime factors (with multiplicity)
Omega = np.array([
    sum(factorint(n).values()) if n > 1 else 0
    for n in N
])

# -----------------------
# TORUS PARAMETRIZATION
# -----------------------
X = (R + r * np.cos(theta_q)) * np.cos(theta_p)
Y = (R + r * np.cos(theta_q)) * np.sin(theta_p)
Z = r * np.sin(theta_q) + height_scale * Omega

# -----------------------
# PLOT
# -----------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    X, Y, Z,
    c=Omega,
    cmap='viridis',
    s=5,
    alpha=alpha
)

ax.set_title(f"CRT Torus: mod {p} × mod {q} with Ω(n) lift")
ax.set_axis_off()

cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label("Ω(n)")

plt.show()
