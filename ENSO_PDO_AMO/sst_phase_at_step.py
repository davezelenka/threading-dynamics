import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# === Load data ===
file_path = "sea-surface-temperature.csv"
df = pd.read_csv(file_path)
df = df[df["Entity"] == "Northern Hemisphere"].copy()
df["Day"] = pd.to_datetime(df["Day"])
df["Year"] = df["Day"].dt.year
annual = df.groupby("Year")["Monthly sea surface temperature anomalies"].mean().reset_index()
years = annual["Year"].values
temps = annual["Monthly sea surface temperature anomalies"].values

# === Step detection (90th percentile) ===
yearly_diff = np.diff(temps)
thresh = np.percentile(np.abs(yearly_diff), 90)
step_years = years[1:][np.abs(yearly_diff) >= thresh]

print(f"Detected step years: {step_years.astype(int)}")

# === Bandpass filter function ===
def bandpass_filter(data, lowcut, highcut, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# === Threading periods ===
targets = np.array([16.2, 8.1, 4.0])
bandwidth = 0.20

# Store phases at step years
phase_at_steps = {T: [] for T in targets}

for T in targets:
    f0 = 1.0 / T
    lowcut, highcut = f0 * (1 - bandwidth), f0 * (1 + bandwidth)
    filtered = bandpass_filter(temps, lowcut, highcut)
    
    # Compute instantaneous phase
    analytic_signal = hilbert(filtered)
    phase = np.angle(analytic_signal)  # radians (-pi to pi)
    
    # Convert phase to 0-360 degrees
    phase_deg = (np.degrees(phase) + 360) % 360
    
    # Get phase at step years
    step_indices = [np.where(years == y)[0][0] for y in step_years]
    phase_at_steps[T] = phase_deg[step_indices]

# === Plot phase distribution ===
for T in targets:
    plt.figure(figsize=(6,6))
    plt.title(f"Phase at step years (~{T:.1f} yr band)")
    plt.subplot(111, polar=True)
    # Convert degrees to radians for polar plot
    plt.scatter(np.radians(phase_at_steps[T]), np.ones_like(phase_at_steps[T]), c='r', s=50)
    plt.yticks([])
    plt.show()

# === Circular statistics (mean vector length, clustering) ===
from scipy.stats import circmean, circstd

for T in targets:
    phases_rad = np.radians(phase_at_steps[T])
    mean_phase = circmean(phases_rad, high=np.pi, low=-np.pi)
    phase_std = circstd(phases_rad, high=np.pi, low=-np.pi)
    print(f"~{T:.1f} yr band: mean phase = {np.degrees(mean_phase):.1f}°, circular std = {np.degrees(phase_std):.1f}°")
