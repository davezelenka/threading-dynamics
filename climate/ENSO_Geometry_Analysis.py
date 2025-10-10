"""
ENSO Geometry Analysis (Fabric Framework)

This script:
1. Loads Niño 3.4 monthly anomaly data from CSV (columns: date,34_anom).
2. Computes the power spectrum to extract dominant ENSO periods.
3. Calculates the Fabric-style geometric scale λ = V^(1/3) * Φ.
4. Solves for coupling Φ given observed dominant period and candidate c values.
5. Compares predicted harmonics (T_n) to observed spectral peaks.
6. Outputs tables + plots for interpretation.

Author: (your name)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend

# -------------------------------
# 1. Load Niño 3.4 data
# -------------------------------
# Your CSV should look like:
# date,34_anom
# 1870-01-01,-1
# 1870-02-01,-1.2
# ...

df = pd.read_csv("nino34.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

dates = df["date"]
anom = df["34_anom"].astype(float).values

# -------------------------------
# 2. Detrend the anomaly series
# -------------------------------
anom_detrended = detrend(anom)

# -------------------------------
# 3. Power Spectrum
# -------------------------------
# Use Welch method (robust for noisy climate data)
fs = 1.0  # sampling frequency = 1/month
f, Pxx = welch(anom_detrended, fs=fs, nperseg=512, detrend="linear")

# Convert frequency [cycles/month] to period [years]
periods = np.where(f > 0, 1.0 / f / 12.0, np.inf)

# Extract peaks in ENSO-relevant band (2–7 years)
mask = (periods >= 2) & (periods <= 20)  # extend to see decadal
periods_band = periods[mask]
power_band = Pxx[mask]

# Identify top peaks
peak_indices = power_band.argsort()[::-1][:5]
peak_periods = periods_band[peak_indices]
peak_powers = power_band[peak_indices]

# -------------------------------
# 4. Fabric-style geometry
# -------------------------------
# Define Niño 3.4 box geometry
R_earth = 6_371_000  # m
deg2rad = np.pi / 180
lat1, lat2 = -5 * deg2rad, 5 * deg2rad
lon_span = 50 * deg2rad  # 170W–120W
area = (R_earth**2) * lon_span * (np.sin(lat2) - np.sin(lat1))

# Active ocean depth (adjustable)
depth = 150  # m
V = area * depth
V13 = V ** (1/3)  # cube root

# Candidate propagation speeds (Kelvin/Rossby wave group speeds, m/s)
candidates = [0.05, 0.1, 0.2, 0.5, 1.0]

# Pick observed dominant ENSO period (first peak)
T_obs_years = peak_periods[0]
T_obs_sec = T_obs_years * 365.25 * 24 * 3600

# Compute coupling parameter Phi for each candidate c
phi_values = []
for c in candidates:
    phi = (c * T_obs_sec) / V13
    phi_values.append(phi)

# -------------------------------
# 5. Predicted harmonics
# -------------------------------
harmonics = {}
for c, phi in zip(candidates, phi_values):
    lam = V13 * phi
    T1 = lam / c / (365.25 * 24 * 3600)  # fundamental in years
    harmonics[c] = [T1 / n for n in range(1, 5)]  # first 4 harmonics

# -------------------------------
# 6. Outputs
# -------------------------------
print("\n=== Spectral Peaks (from data) ===")
for p, pw in zip(peak_periods, peak_powers):
    print(f"  Period ≈ {p:.2f} yr (Power={pw:.4f})")

print("\n=== Geometric Parameters ===")
print(f"Niño 3.4 region effective V^(1/3): {V13/1000:.2f} km")
print(f"Observed dominant ENSO period: {T_obs_years:.2f} yr")

print("\nCoupling Φ estimates:")
for c, phi in zip(candidates, phi_values):
    print(f"  c={c:.2f} m/s → Φ ≈ {phi:.3f}")

print("\nPredicted harmonics (years):")
for c, Ts in harmonics.items():
    Ts_fmt = ", ".join([f"{T:.2f}" for T in Ts])
    print(f"  c={c:.2f} m/s: {Ts_fmt}")

# -------------------------------
# 7. Charts
# -------------------------------
plt.figure(figsize=(10,4))
plt.plot(dates, anom, label="Niño 3.4 anomaly")
plt.title("Niño 3.4 SST Anomaly (monthly)")
plt.xlabel("Year")
plt.ylabel("Anomaly (°C)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.semilogy(periods_band, power_band)
plt.scatter(peak_periods, peak_powers, color="red", label="Top peaks")
plt.gca().invert_xaxis()  # longer periods on left
plt.title("ENSO Power Spectrum (Welch)")
plt.xlabel("Period (years)")
plt.ylabel("Spectral Power")
plt.legend()
plt.grid()
plt.show()
