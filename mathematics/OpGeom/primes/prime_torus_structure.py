# CRT Torus Visualization (Python)
# Replicates HTML logic exactly: pxq torus + Omega height
# Black points only; transparency adjustable via alpha

import math
import matplotlib.pyplot as plt

# -------------------- PARAMETERS --------------------
p = 87         # prime p
q = 91          # prime q (can be composite, as in HTML)
N = 100000     # max n
R = 150.0       # major radius
r = 50.0        # minor radius
height_scale = 15.0
alpha = 0.5    # point transparency
point_size = 0.1 # marker size

# -------------------- HELPERS --------------------
def omega(n: int) -> int:
    """Prime factor count with multiplicity"""
    if n <= 1:
        return 0
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

# -------------------- BUILD POINT CLOUD --------------------
xs, ys, zs = [], [], []

for n in range(2, N + 1):
    theta_p = 2 * math.pi * (n % p) / p
    theta_q = 2 * math.pi * (n % q) / q
    omega_n = omega(n)

    x = (R + r * math.cos(theta_q)) * math.cos(theta_p)
    y = (R + r * math.cos(theta_q)) * math.sin(theta_p)
    z = r * math.sin(theta_q) + height_scale * omega_n

    xs.append(x)
    ys.append(y)
    zs.append(z)

# -------------------- PLOT --------------------
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(projection='3d')

# Set the view to Front View
ax.view_init(elev=0, azim=-80)

ax.scatter(xs, ys, zs, c='black', s=point_size, alpha=alpha, linewidths=0)

ax.set_axis_off()
ax.set_title(f"CRT Toroidal Embedding (p={p}, q={q}, N={N})", fontsize=10)

plt.show()
