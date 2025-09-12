# save as extract_threading_harmonics.py and run with: python extract_threading_harmonics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# --------------------------
# Path to data file
# --------------------------
file_path = "GLB.Ts+dSST.csv"   # put the NASA GISS file in same folder

# --------------------------
# Read file robustly
# --------------------------
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

header_row = None
for i, line in enumerate(lines[:40]):
    if line.strip().startswith("Year"):
        header_row = i
        break
if header_row is None:
    header_row = 0

df = pd.read_csv(file_path, skiprows=header_row)
df.columns = [c.strip() for c in df.columns]

if 'Year' not in df.columns or 'J-D' not in df.columns:
    raise ValueError("Expected columns 'Year' and 'J-D' in CSV. Found: " + ", ".join(df.columns))

df = df[['Year', 'J-D']].copy()
df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
df = df.dropna(subset=['J-D']).reset_index(drop=True)

years = df['Year'].values
temps = df['J-D'].values

# --------------------------
# PSD (Welch)
# --------------------------
fs = 1.0
freqs, psd = welch(temps, fs=fs, nperseg=min(256, len(temps)))
periods = np.where(freqs>0, 1.0/freqs, np.inf)
peak_idx = np.argsort(psd)[::-1]
top_periods = periods[peak_idx][:12]
top_powers = psd[peak_idx][:12]

print("Top spectral peaks (period in years):")
for p, power in zip(top_periods, top_powers):
    if np.isfinite(p):
        print(f"  {p:.2f} years (power={power:.4f})")
    else:
        print(f"  DC / trend (power={power:.4f})")

# --------------------------
# Bandpass function
# --------------------------
def bandpass_filter(data, lowcut, highcut, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(low, 1e-6)
    high = min(high, 0.9999)
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --------------------------
# Target bands (empirical extraction)
# --------------------------
targets = np.array([16.2, 8.1, 4.0])
bandwidth = 0.20   # ±20% around center frequency

components = {}
for T in targets:
    f0 = 1.0 / T
    lowcut = f0 * (1.0 - bandwidth)
    highcut = f0 * (1.0 + bandwidth)
    comp = bandpass_filter(temps, lowcut, highcut, fs=fs, order=4)
    components[T] = comp

recon = np.sum(np.vstack([components[T] for T in targets]), axis=0)
corrs = {T: np.corrcoef(temps, components[T])[0,1] for T in targets}
corr_recon = np.corrcoef(temps, recon)[0,1]

# --------------------------
# Detect large year-to-year steps (raw series)
# --------------------------
yearly_diff = np.diff(temps)
thresh = np.percentile(np.abs(yearly_diff), 90)
step_years = years[1:][np.where(np.abs(yearly_diff) >= thresh)]
step_values = yearly_diff[np.where(np.abs(yearly_diff) >= thresh)]

print("\nYear-to-year step detection threshold (90th pct abs diff):", thresh)
print("Detected large steps (year, delta):")
for y, d in zip(step_years.astype(int), step_values):
    print(f"  {y}: {d:.3f} °C")

# --------------------------
# Plots
# --------------------------
# 1) original vs reconstruction
plt.figure(figsize=(12,4))
plt.plot(years, temps, label='Annual Global Temp Anomaly (J-D)')
plt.plot(years, recon, label='Reconstruction (sum of bands)')
plt.scatter(step_years, temps[1:][np.where(np.abs(yearly_diff) >= thresh)], marker='o', s=40, label='Large year-to-year steps')
plt.xlabel('Year'); plt.ylabel('Temperature anomaly (°C)')
plt.title('Global Annual Temp Anomaly and Harmonic Reconstruction (16.2, 8.1, 4.0 yr bands)')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# 2) individual components
for T in targets:
    plt.figure(figsize=(12,2.2))
    plt.plot(years, components[T], label=f'Bandpassed ~{T:.1f} yr (±{int(bandwidth*100)}%)')
    plt.xlabel('Year'); plt.ylabel('Anomaly (°C)')
    plt.title(f'Component near {T:.1f} years')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

# 3) PSD with markers
plt.figure(figsize=(10,4))
plt.semilogy(periods[1:], psd[1:])
for T in targets:
    plt.axvline(T, linestyle='--')
plt.xlim(0, 60)
plt.xlabel('Period (years)'); plt.ylabel('PSD (log scale)')
plt.title('Welch PSD of Global Temp Anomaly with threading band markers')
plt.grid(True); plt.tight_layout()
plt.show()

# print correlations
print("\nCorrelation summary:")
for T in targets:
    print(f"  {T:.1f} yr component: corr = {corrs[T]:.3f}")
print("  Reconstruction (sum) correlation with raw series:", round(corr_recon, 3))
