import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
import statsmodels.api as sm

# Compatibility wrapper for binomial test (not needed for cross-corr, included if needed later)
try:
    from scipy.stats import binomtest
except ImportError:
    pass

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
enso_smooth = pd.Series(enso_z, index=df.index).rolling(window, center=True).mean().fillna(0).values
sun_smooth = pd.Series(sun_z, index=df.index).rolling(window, center=True).mean().fillna(0).values

# === Detect ENSO maxima + minima ===
min_distance = 24  # months
enso_max_peaks, _ = find_peaks(enso_smooth, distance=min_distance, height=0)
enso_min_peaks, _ = find_peaks(-enso_smooth, distance=min_distance)
enso_peaks_all = np.sort(np.concatenate([enso_max_peaks, enso_min_peaks]))
enso_peak_dates = df.index[enso_peaks_all]
print(f"ENSO peaks (max+min): {len(enso_peaks_all)}")

# === Cross-correlation: Sunspots → ENSO ===
max_lag = 120  # months (look back)
lags = np.arange(0, max_lag+1)  # only positive lags (Sunspots lead ENSO)
corrs = []

for lag in lags:
    c = np.corrcoef(enso_smooth[lag:], sun_smooth[:-lag])[0,1] if lag > 0 else np.corrcoef(enso_smooth, sun_smooth)[0,1]
    corrs.append(c)
corrs = np.array(corrs)

best_idx = np.argmax(corrs)
best_lag = lags[best_idx]
best_corr = corrs[best_idx]
print(f"Best lag (Sunspots → ENSO): {best_lag} months, correlation = {best_corr:.3f}")

# === AR(1) surrogate function for ENSO ===
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

# === Generate surrogates and max correlation for each ===
n_surr = 1000
enso_surr = ar1_surrogate(enso_smooth, n_surr=n_surr)
surr_max_corrs = []

for surr in enso_surr:
    surr_corrs = []
    for lag in lags:
        c = np.corrcoef(surr[lag:], sun_smooth[:-lag])[0,1] if lag > 0 else np.corrcoef(surr, sun_smooth)[0,1]
        surr_corrs.append(c)
    surr_max_corrs.append(np.max(surr_corrs))
surr_max_corrs = np.array(surr_max_corrs)

# === Continuity-corrected p-value ===
pval = (np.sum(surr_max_corrs >= best_corr) + 1) / (n_surr + 1)
print(f"Significance of best lag: p = {pval:.4f}")

# === Plot cross-correlation ===
plt.figure(figsize=(8,4))
plt.plot(lags, corrs, label='Cross-correlation')
plt.axvline(best_lag, color='r', linestyle='--', label=f'Best lag: {best_lag} mo')
plt.xlabel("Lag (months, Sunspots → ENSO)")
plt.ylabel("Correlation")
plt.title("Cross-Correlation: Sunspots lead ENSO")
plt.legend()
plt.tight_layout()
plt.show()

# === Optional: histogram of surrogate max correlations ===
plt.figure(figsize=(6,4))
plt.hist(surr_max_corrs, bins=30, alpha=0.6, color='gray')
plt.axvline(best_corr, color='r', linestyle='--', label=f'Observed max corr = {best_corr:.3f}')
plt.xlabel("Maximum correlation (Sunspots → ENSO)")
plt.ylabel("Frequency")
plt.title("Distribution of max correlation in AR(1) surrogates")
plt.legend()
plt.tight_layout()
plt.show()
