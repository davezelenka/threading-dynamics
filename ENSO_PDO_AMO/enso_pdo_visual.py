import numpy as np
import matplotlib.pyplot as plt

# Time range (approx 1870-2025)
time = np.arange(1870, 2026)

# Threading predicted periods (years)
threading_periods = np.array([0.4, 0.81, 1.62, 4.05, 8.1, 16.19])

# Observed PSD peaks
enso_psd_peaks = np.array([1.5, 3.6, 7.1])
pdo_psd_peaks = np.array([4.05, 8.11, 16.22])

# Wavelet peaks (ENSO)
wlt_peaks = [11.3, 5.3]
wlt_colors = ['#ff9999', '#99ccff']  # semi-transparent patches

fig, ax = plt.subplots(figsize=(12,6))

# Plot threading predictions (horizontal dashed lines)
for period in threading_periods:
    ax.hlines(period, time[0], time[-1], colors='gray', linestyles='dashed', alpha=0.6, label='_nolegend_')

# Plot PSD peaks (ENSO and PDO) as horizontal solid lines
for peak in enso_psd_peaks:
    ax.hlines(peak, time[0], time[-1], colors='red', linestyles='solid', linewidth=2, label='ENSO PSD Peaks' if peak==enso_psd_peaks[0] else '')

for peak in pdo_psd_peaks:
    ax.hlines(peak, time[0], time[-1], colors='green', linestyles='solid', linewidth=2, label='PDO PSD Peaks' if peak==pdo_psd_peaks[0] else '')

# Plot wavelet patches (ENSO)
for i, peak in enumerate(wlt_peaks):
    ax.fill_between(time, peak-0.2, peak+0.2, color=wlt_colors[i], alpha=0.3, label='ENSO Wavelet Peak' if i==0 else '')

# Labels and aesthetics
ax.set_yscale('log')
ax.set_xlabel('Year')
ax.set_ylabel('Period (years)')
ax.set_title('Threading Predictions vs Observed ENSO and PDO Oscillations (1870-2025)')
ax.grid(True, which='both', linestyle='--', alpha=0.4)

# Legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()