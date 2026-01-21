"""
Arithmetic Fluid Analysis
=========================

Interprets Omega(n) data as a 1D viscous compressible flow
on the integer line.

Input:  omega_field_100k.csv
Output: enriched CSV + diagnostic summaries
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# -------------------------
# Configuration
# -------------------------

DATA_FILE = Path("omega_field_100k.csv")
OUTPUT_FILE = Path("omega_fluid_analysis.csv")

SMOOTH_SIGMA = 3.0        # coarse-grain scale (viscosity proxy)
EPS = 1e-9               # numerical stability

# -------------------------
# Load data
# -------------------------

df = pd.read_csv(DATA_FILE)

n = df["n"].values
rho = df["Omega_total"].values.astype(float)   # mass density

moduli_cols = [c for c in df.columns if c.startswith("Omega_") and c != "Omega_total"]
rho_components = df[moduli_cols].values.astype(float)

# -------------------------
# Core fluid quantities
# -------------------------

# 1. Velocity field (gradient of density)
velocity = np.gradient(rho)

# 2. Acceleration (curvature)
acceleration = np.gradient(velocity)

# 3. Coarse-grained density (viscous smoothing)
rho_smooth = gaussian_filter1d(rho, sigma=SMOOTH_SIGMA)

# 4. Pressure ~ local compression / variance
pressure = gaussian_filter1d((rho - rho_smooth) ** 2, sigma=SMOOTH_SIGMA)

# 5. Effective viscosity (resistance to gradient)

grad_sq = velocity**2
viscosity = np.full_like(rho, np.nan)

active = grad_sq > 0
viscosity[active] = pressure[active] / grad_sq[active]

# 6. Flux (Fick / Navier–Stokes analogue)
flux = -viscosity * velocity

# 7. Divergence of flux (sources / sinks)
divergence = np.gradient(flux)

# -------------------------
# Constraint-channel dynamics
# -------------------------

# Total modular load
modular_load = rho_components.sum(axis=1)

# Channel coherence (how aligned the channels are)
channel_variance = rho_components.var(axis=1)
# -------------------------
# Channel entropy (robust)
# -------------------------

channel_entropy = np.full(len(rho), np.nan)

nonzero = modular_load > 0
p = rho_components[nonzero] / modular_load[nonzero, None]

# Mask zero probabilities inside channels
p_safe = np.where(p > 0, p, 1.0)   # log(1)=0 → contributes nothing

channel_entropy[nonzero] = -np.sum(
    p * np.log(p_safe),
    axis=1
)



# -------------------------
# Shock & release detection
# -------------------------

shock_strength = np.abs(acceleration)
release_rate = np.maximum(divergence, 0)
compression_rate = np.maximum(-divergence, 0)

# -------------------------
# Assemble output
# -------------------------

out = df.copy()

out["rho"] = rho
out["velocity"] = velocity
out["acceleration"] = acceleration
out["pressure"] = pressure
out["viscosity"] = viscosity
out["flux"] = flux
out["divergence"] = divergence

out["channel_entropy"] = channel_entropy
out["channel_variance"] = channel_variance

out["shock_strength"] = shock_strength
out["release_rate"] = release_rate
out["compression_rate"] = compression_rate

out.to_csv(OUTPUT_FILE, index=False)

# -------------------------
# Diagnostics
# -------------------------

summary = {
    "mean_density": rho.mean(),
    "mean_viscosity": np.nanmean(viscosity),
    "mean_pressure": pressure.mean(),
    "mean_shock": shock_strength.mean(),
    "mean_entropy": np.nanmean(channel_entropy),
    "fraction_release": np.mean(divergence > 0),
}

print("\nArithmetic Fluid Diagnostics")
print("=" * 32)
for k, v in summary.items():
    print(f"{k:20s}: {v:.6f}")
print("\nOutput written to:", OUTPUT_FILE)
