import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

# === Load CSV ===
df = pd.read_csv("data/AMO_unsmoothed.csv")

# Compute annual mean
df["Annual"] = df.iloc[:, 1:13].mean(axis=1)
years = df["Year"].values
temps = df["Annual"].values

# === PSD analysis (Welch) ===
fs = 1.0  # 1 sample per year
nperseg = min(64, len(temps))  # smaller segment to avoid NaNs
freqs, psd = welch(temps, fs=fs, nperseg=nperseg)

# Avoid divide by zero for DC component
periods = np.where(freqs > 0, 1.0 / freqs, np.nan)

# Top spectral peaks
peak_idx = np.argsort(psd)[::-1]
print("Top spectral peaks (period in years):")
for p, power in zip(periods[peak_idx][:12], psd[peak_idx][:12]):
    if np.isfinite(p):
        print(f"  {p:.2f} years (power={power:.4f})")
    else:
        print(f"  DC / trend (power={power:.4f})")

# === Step detection (90th percentile year-to-year differences) ===
yearly_diff = np.diff(temps)
thresh = np.percentile(np.abs(yearly_diff), 90)
step_years = years[1:][np.abs(yearly_diff) >= thresh]
step_values = yearly_diff[np.abs(yearly_diff) >= thresh]

print(f"\nStep detection threshold (90th pct abs diff): {thresh:.3f}")
for y, d in zip(step_years, step_values):
    print(f"  {int(y)}: {d:.3f} °C")

# === Bandpass filter around threading harmonics ===
def bandpass_filter(data, lowcut, highcut, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

targets = np.array([16.2, 8.1, 4.0])
bandwidth = 0.20  # 20% around target frequency
components, corrs = {}, {}

for T in targets:
    f0 = 1.0 / T
    lowcut, highcut = f0*(1-bandwidth), f0*(1+bandwidth)
    comp = bandpass_filter(temps, lowcut, highcut)
    components[T] = comp
    corrs[T] = np.corrcoef(temps, comp)[0,1]

# Reconstruction
recon = np.sum(np.vstack([components[T] for T in targets]), axis=0)
corr_recon = np.corrcoef(temps, recon)[0,1]

# === Plots ===
plt.figure(figsize=(12,4))
plt.plot(years, temps, label="AMO annual mean")
plt.plot(years, recon, label="Reconstruction (16/8/4 yr bands)")
plt.scatter(step_years, temps[np.isin(years, step_years)], color='red', label='Detected steps')
plt.xlabel("Year")
plt.ylabel("AMO anomaly (°C)")
plt.legend()
plt.title("AMO Unsmooth Series + Threading Harmonics")
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.semilogy(periods[1:], psd[1:])  # skip DC
for T in targets:
    plt.axvline(T, linestyle='--', color='r', label=f"{T:.1f} yr band")
plt.xlabel("Period (years)")
plt.ylabel("PSD")
plt.title("AMO PSD with threading markers")
plt.xlim(0, 100)
plt.grid()
plt.legend()
plt.show()

# === Correlation summary ===
print("\nCorrelation summary:")
for T in targets:
    print(f"  {T:.1f} yr component: corr = {corrs[T]:.3f}")
print(f"  Reconstruction corr = {corr_recon:.3f}")