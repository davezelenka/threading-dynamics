import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Config
# -------------------------

CSV_PATH = "omega_fluid_analysis.csv"
OUT_DIR = "omega_fluid_figures"
ROLL = 300

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load
# -------------------------

df = pd.read_csv(CSV_PATH)

print("Columns:", list(df.columns))

# -------------------------
# Helpers
# -------------------------

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")

def smooth(x, w=ROLL):
    return x.rolling(w, center=True).mean()

# -------------------------
# 1. Density field (ρ)
# -------------------------

plt.figure()
plt.plot(df["rho"], alpha=0.25)
plt.plot(smooth(df["rho"]), linewidth=2)
plt.xlabel("n")
plt.ylabel("ρ (density)")
plt.title("Arithmetic Density Field")
savefig("density_rho.png")

# -------------------------
# 2. Pressure vs density (equation of state)
# -------------------------

plt.figure()
plt.scatter(df["rho"], df["pressure"], s=4, alpha=0.3)
plt.xlabel("ρ (density)")
plt.ylabel("pressure")
plt.title("Arithmetic Equation of State")
savefig("pressure_vs_density.png")

# -------------------------
# 3. Entropy distribution
# -------------------------

entropy = df["channel_entropy"].replace([np.inf, -np.inf], np.nan).dropna()

plt.figure()
plt.hist(entropy, bins=80, density=True)
plt.xlabel("Channel entropy")
plt.ylabel("PDF")
plt.title("Entropy Distribution (Composite Channels)")
savefig("entropy_distribution.png")

# -------------------------
# 4. Viscosity spectrum
# -------------------------

vis = df["viscosity"].replace([np.inf, -np.inf], np.nan).dropna()

plt.figure()
plt.hist(np.log10(vis), bins=80, density=True)
plt.xlabel("log10(viscosity)")
plt.ylabel("PDF")
plt.title("Viscosity Spectrum")
savefig("viscosity_distribution.png")

# -------------------------
# 5. Shock field
# -------------------------

plt.figure()
plt.plot(df["shock_strength"], alpha=0.3)
plt.plot(smooth(df["shock_strength"]), linewidth=2)
plt.xlabel("n")
plt.ylabel("shock strength")
plt.title("Arithmetic Shock Field")
savefig("shock_field.png")

# -------------------------
# 6. Entropy vs shock (turbulence proxy)
# -------------------------

plt.figure()
plt.scatter(
    df["channel_entropy"],
    df["shock_strength"],
    s=4,
    alpha=0.3
)
plt.xlabel("entropy")
plt.ylabel("shock strength")
plt.title("Entropy–Shock Coupling")
savefig("entropy_vs_shock.png")

# -------------------------
# 7. Pressure vs viscosity
# -------------------------

plt.figure()
plt.scatter(
    df["pressure"],
    np.log10(df["viscosity"]),
    s=4,
    alpha=0.3
)
plt.xlabel("pressure")
plt.ylabel("log10(viscosity)")
plt.title("Pressure–Viscosity Relation")
savefig("pressure_vs_viscosity.png")

# -------------------------
# 8. Divergence field (sources/sinks)
# -------------------------

plt.figure()
plt.plot(df["divergence"], alpha=0.3)
plt.plot(smooth(df["divergence"]), linewidth=2)
plt.xlabel("n")
plt.ylabel("divergence")
plt.title("Arithmetic Source–Sink Structure")
savefig("divergence_field.png")

# -------------------------
# 9. Release vs compression
# -------------------------

plt.figure()
plt.scatter(
    df["compression_rate"],
    df["release_rate"],
    s=4,
    alpha=0.3
)
plt.xlabel("compression rate")
plt.ylabel("release rate")
plt.title("Load Accumulation vs Release")
savefig("compression_vs_release.png")

# -------------------------
# 10. Ω-conditioned entropy
# -------------------------

plt.figure()
for k in sorted(df["Omega_total"].unique()):
    subset = df[df["Omega_total"] == k]
    if len(subset) < 50:
        continue
    plt.scatter(
        subset["rho"],
        subset["channel_entropy"],
        s=4,
        alpha=0.3,
        label=f"Ω={int(k)}"
    )

plt.xlabel("density")
plt.ylabel("entropy")
plt.title("Density–Entropy Stratified by Ω(n)")
plt.legend(markerscale=3)
savefig("density_entropy_by_omega.png")

# -------------------------
# Summary (paper-ready)
# -------------------------

summary = {
    "mean_density": df["rho"].mean(),
    "mean_pressure": df["pressure"].mean(),
    "mean_entropy": df["channel_entropy"].mean(),
    "mean_viscosity": df["viscosity"].mean(),
    "mean_shock": df["shock_strength"].mean(),
    "mean_release": df["release_rate"].mean(),
}

print("\nArithmetic Fluid Diagnostics")
print("=" * 40)
for k, v in summary.items():
    print(f"{k:20s}: {v:.6f}")
