import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

# ----------------------------------------------------
# 1. Load NASA GISS dataset
# ----------------------------------------------------
# File should be downloaded manually from:
# https://data.giss.nasa.gov/gistemp/
# Example filename: "GLB.Ts+dSST.csv"

file_path = "data/GLB.Ts+dSST.csv"

# Read CSV while skipping metadata lines at top
df = pd.read_csv(file_path, skiprows=1)

# Keep only relevant columns
# J-D = Annual mean, D-N = Dec–Nov, etc.
df = df[['Year','J-D']].copy()

# Convert anomalies to numeric, handle missing values
df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')

# Drop missing years
df = df.dropna()

years = df['Year'].values
temps = df['J-D'].values

# ----------------------------------------------------
# 2. Spectral analysis (Welch PSD)
# ----------------------------------------------------
fs = 1.0  # sampling frequency = 1/year
freqs, psd = welch(temps, fs=fs, nperseg=128)

# Convert frequency to period (years)
periods = 1 / freqs

# ----------------------------------------------------
# 3. Find spectral peaks
# ----------------------------------------------------
peaks, _ = find_peaks(psd)
peak_periods = periods[peaks]
peak_powers = psd[peaks]

# Sort by strongest peaks
sorted_idx = np.argsort(peak_powers)[::-1]
peak_periods = peak_periods[sorted_idx]
peak_powers = peak_powers[sorted_idx]

# ----------------------------------------------------
# 4. Threading predicted periods
# ----------------------------------------------------
threading_periods = np.array([16.2, 8.1, 4.0, 1.6, 0.8, 0.4])

# ----------------------------------------------------
# 5. Plot results
# ----------------------------------------------------
plt.figure(figsize=(10,6))
plt.semilogy(periods, psd, label="Global Temp PSD")
plt.scatter(peak_periods, peak_powers, color='red', label="Observed Peaks")

# Mark threading predictions
for T in threading_periods:
    plt.axvline(T, color='green', linestyle='--', alpha=0.7)
    plt.text(T, max(psd)*0.5, f"{T:.1f}y", rotation=90, va='bottom', ha='right', color='green')

plt.xlabel("Period (years)")
plt.ylabel("Power (log scale)")
plt.title("NASA GISS Global Temperature Anomalies — Spectral Analysis")
plt.legend()
plt.xlim(0, 50)  # focus on <50 years
plt.grid(True, alpha=0.3)
plt.show()

# ----------------------------------------------------
# 6. Print summary of strongest peaks
# ----------------------------------------------------
print("Top spectral peaks (period in years):")
for p, power in zip(peak_periods[:10], peak_powers[:10]):
    print(f"  {p:.2f} years (power={power:.4f})")
