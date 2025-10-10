import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# === Load dataset ===
file_path = "sea-surface-temperature.csv"
df = pd.read_csv(file_path)

# Keep only Northern Hemisphere
df = df[df["Entity"] == "Northern Hemisphere"].copy()

# Convert Day column to datetime, then extract year
df["Day"] = pd.to_datetime(df["Day"])
df["Year"] = df["Day"].dt.year

# Annual means
annual = df.groupby("Year")["Monthly sea surface temperature anomalies"].mean().reset_index()
years = annual["Year"].values
temps = annual["Monthly sea surface temperature anomalies"].values

# === Spectral analysis ===
fs = 1.0  # one sample per year
freqs, psd = welch(temps, fs=fs, nperseg=min(256, len(temps)))
periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

# Top peaks
peak_idx = np.argsort(psd)[::-1]
print("Top spectral peaks (period in years):")
for p, power in zip(periods[peak_idx][:12], psd[peak_idx][:12]):
    if np.isfinite(p):
        print(f"  {p:.2f} years (power={power:.4f})")
    else:
        print(f"  DC / trend (power={power:.4f})")

# === Bandpass filter around threading harmonics ===
def bandpass_filter(data, lowcut, highcut, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

targets = np.array([16.2, 8.1, 4.0])
bandwidth = 0.20
components, corrs = {}, {}
for T in targets:
    f0 = 1.0 / T
    comp = bandpass_filter(temps, f0 * (1 - bandwidth), f0 * (1 + bandwidth))
    components[T] = comp
    corrs[T] = np.corrcoef(temps, comp)[0, 1]

recon = np.sum(np.vstack([components[T] for T in targets]), axis=0)
corr_recon = np.corrcoef(temps, recon)[0, 1]

# === Step detection ===
yearly_diff = np.diff(temps)
thresh = np.percentile(np.abs(yearly_diff), 90)
step_years = years[1:][np.abs(yearly_diff) >= thresh]
step_values = yearly_diff[np.abs(yearly_diff) >= thresh]

print(f"\nStep detection threshold (90th pct abs diff): {thresh:.3f}")
for y, d in zip(step_years, step_values):
    print(f"  {int(y)}: {d:.3f} °C")

# === Plots ===
plt.figure(figsize=(12, 4))
plt.plot(years, temps, label="NH SST anomaly (annual)")
plt.plot(years, recon, label="Reconstruction (16, 8, 4 yr bands)")
plt.scatter(step_years, temps[np.isin(years, step_years)], s=40, c="red", label="Detected steps")
plt.xlabel("Year")
plt.ylabel("Anomaly (°C)")
plt.title("Northern Hemisphere SST Anomaly + Harmonic Reconstruction")
plt.legend()
plt.grid()
plt.show()

# Individual components
for T in targets:
    plt.figure(figsize=(12, 2.5))
    plt.plot(years, components[T], label=f"~{T:.1f} yr band")
    plt.xlabel("Year")
    plt.ylabel("Anomaly (°C)")
    plt.legend()
    plt.grid()
    plt.show()

# PSD
plt.figure(figsize=(10, 4))
plt.semilogy(periods[1:], psd[1:])
for T in targets:
    plt.axvline(T, linestyle="--")
plt.xlim(0, 60)
plt.xlabel("Period (years)")
plt.ylabel("PSD")
plt.title("Welch PSD of NH SST anomalies with threading markers")
plt.grid()
plt.show()

print("\nCorrelation summary:")
for T in targets:
    print(f"  {T:.1f} yr component: corr = {corrs[T]:.3f}")
print(f"  Reconstruction corr = {corr_recon:.3f}")