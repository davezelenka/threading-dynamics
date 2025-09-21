import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
import statsmodels.api as sm

# === Load data ===
csv_path = "data/enso_global_temp_anom_sunspots.csv"  # adjust path
df = pd.read_csv(csv_path, parse_dates=['date']).set_index('date').sort_index()

enso = df['34_anom'].astype(float).values
sun = df['sunspots'].astype(float).values

# === Detrend & standardize ===
enso_dt = signal.detrend(enso)
sun_dt = signal.detrend(sun)
enso_z = (enso_dt - np.mean(enso_dt)) / np.std(enso_dt, ddof=1)
sun_z = (sun_dt - np.mean(sun_dt)) / np.std(sun_dt, ddof=1)

# === Smooth series ===
window = 24  # months
enso_smooth = pd.Series(enso_z, index=df.index).rolling(window, center=True).mean()
sun_smooth = pd.Series(sun_z, index=df.index).rolling(window, center=True).mean()

# === Detect positive peaks ===
min_distance = 24  # months
enso_peaks, _ = find_peaks(enso_smooth.fillna(0), distance=min_distance)
sun_peaks, _ = find_peaks(sun_smooth.fillna(0), distance=min_distance)

print(f"Original ENSO peaks: {len(enso_peaks)}, Sunspot peaks: {len(sun_peaks)}")

enso_peak_dates = enso_smooth.index[enso_peaks]
sun_peak_dates = sun_smooth.index[sun_peaks]

# === Compute forward lags ≤10 years ===
lags = []
for ep in enso_peak_dates:
    diffs = (ep - sun_peak_dates).days / 30.44  # months
    diffs = diffs[(diffs > 0) & (diffs <= 120)]
    if len(diffs) > 0:
        lags.append(np.min(diffs))
lags = np.array(lags)
print(f"Number of ENSO peaks matched to prior sunspot peaks: {len(lags)}")
print(f"Observed median lag: {np.median(lags):.1f} months")
print(f"Observed mean lag: {np.mean(lags):.1f} months")

# === AR(1) surrogate function ===
def ar1_surrogate(series, n_surr=1000):
    model = sm.tsa.ARIMA(series, order=(1,0,0)).fit()
    phi = model.arparams[0]
    resid = series[1:] - phi*series[:-1]
    sigma = np.std(resid, ddof=1)
    surr = []
    for _ in range(n_surr):
        noise = np.random.normal(0, sigma, len(series))
        x = np.zeros(len(series))
        for t in range(1, len(series)):
            x[t] = phi*x[t-1] + noise[t]
        surr.append(x)
    return np.array(surr)

# === Generate surrogates ===
n_surr = 5000
enso_surr = ar1_surrogate(enso_smooth.fillna(0).values, n_surr=n_surr)
sun_surr = ar1_surrogate(sun_smooth.fillna(0).values, n_surr=n_surr)

# === Check peak counts in first few surrogates ===
print("\nSurrogate peak counts (first 10):")
for i in range(10):
    ep_surr, _ = find_peaks(enso_surr[i], distance=min_distance)
    sp_surr, _ = find_peaks(sun_surr[i], distance=min_distance)
    print(f"Surrogate {i}: ENSO peaks={len(ep_surr)}, Sun peaks={len(sp_surr)}")

# === Compute surrogate median lags ===
surr_median_lags = []
for i in range(n_surr):
    ep_surr, _ = find_peaks(enso_surr[i], distance=min_distance)
    sp_surr, _ = find_peaks(sun_surr[i], distance=min_distance)
    surr_lags = []
    for e in ep_surr:
        diffs = sp_surr - e
        diffs = diffs[(diffs < 0) & (diffs >= -120)] * -1
        if len(diffs) > 0:
            surr_lags.append(np.min(diffs))
    if len(surr_lags) > 0:
        surr_median_lags.append(np.median(surr_lags))
surr_median_lags = np.array(surr_median_lags)

# === Continuity-corrected p-value ===
pval = (np.sum(surr_median_lags >= np.median(lags)) + 1) / (len(surr_median_lags) + 1)
print(f"\nContinuity-corrected surrogate p-value: {pval:.4f}")

# === Plots ===
plt.figure(figsize=(12,5))
plt.plot(enso_smooth, label="ENSO (smoothed)", color='b')
plt.plot(sun_smooth, label="Sunspots (smoothed)", color='orange')
plt.plot(enso_peak_dates, enso_smooth.iloc[enso_peaks], "bo", label="ENSO peaks")
plt.plot(sun_peak_dates, sun_smooth.iloc[sun_peaks], "ro", label="Sunspot peaks")
plt.legend()
plt.title("ENSO vs Sunspot Positive Peaks (24-mo smoothed)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(lags, bins=20, alpha=0.7, color='gray')
plt.axvline(np.median(lags), color='r', linestyle='--', label=f"Median = {np.median(lags):.1f} mo")
plt.xlabel("Lag (months): ENSO peak minus Sunspot peak")
plt.ylabel("Frequency")
plt.title("Distribution of ENSO Lags After Sunspot Peaks (≤10 yr)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(surr_median_lags, bins=30, alpha=0.6, color='gray')
plt.axvline(np.median(lags), color='r', linestyle='--', label=f"Observed median = {np.median(lags):.1f} mo")
plt.xlabel("Median lag (months) in surrogates")
plt.ylabel("Frequency")
plt.title("AR(1) Surrogate Test for ENSO-Sunspot Lag")
plt.legend()
plt.tight_layout()
plt.show()
