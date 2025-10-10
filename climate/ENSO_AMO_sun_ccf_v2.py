import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
import statsmodels.api as sm

# === Load data ===
csv_path = "data/enso_global_temp_anom_sunspots_amo.csv"  # adjust path
df = pd.read_csv(csv_path, parse_dates=['date']).set_index('date').sort_index()

# Drop any rows with NaNs in relevant columns
df = df.dropna(subset=['34_anom', 'SSTA_AMO', 'sunspots'])

# === Extract series ===
enso = df['34_anom'].astype(float).values
amo = df['SSTA_AMO'].astype(float).values
sun = df['sunspots'].astype(float).values

# === Detrend & standardize ===
def preprocess(series):
    dt = signal.detrend(series)
    return (dt - np.mean(dt)) / np.std(dt, ddof=1)

enso_z = preprocess(enso)
amo_z = preprocess(amo)
sun_z = preprocess(sun)

# === Smooth series ===
window = 32
enso_smooth = pd.Series(enso_z, index=df.index).rolling(window, center=True).mean().fillna(0).values
amo_smooth = pd.Series(amo_z, index=df.index).rolling(window, center=True).mean().fillna(0).values
sun_smooth = pd.Series(sun_z, index=df.index).rolling(window, center=True).mean().fillna(0).values

# === Detect maxima + minima ===
def get_peaks(series, min_distance=24):
    max_peaks, _ = find_peaks(series, distance=min_distance, height=0)
    min_peaks, _ = find_peaks(-series, distance=min_distance)
    return np.sort(np.concatenate([max_peaks, min_peaks]))

enso_peaks = get_peaks(enso_smooth)
amo_peaks = get_peaks(amo_smooth)

print(f"ENSO peaks (max+min): {len(enso_peaks)}")
print(f"AMO peaks (max+min): {len(amo_peaks)}")

# === Cross-correlation function ===
def cross_corr(series, reference, max_lag=120):
    lags = np.arange(0, max_lag+1)
    corrs = []
    for lag in lags:
        if lag == 0:
            c = np.corrcoef(series, reference)[0,1]
        else:
            c = np.corrcoef(series[lag:], reference[:-lag])[0,1]
        corrs.append(c)
    corrs = np.array(corrs)
    best_idx = np.argmax(corrs)
    return lags, corrs, lags[best_idx], corrs[best_idx]

enso_lags, enso_corrs, enso_best_lag, enso_best_corr = cross_corr(enso_smooth, sun_smooth)
amo_lags, amo_corrs, amo_best_lag, amo_best_corr = cross_corr(amo_smooth, sun_smooth)

print(f"Best lag (Sunspots → ENSO): {enso_best_lag} mo, corr = {enso_best_corr:.3f}")
print(f"Best lag (Sunspots → AMO): {amo_best_lag} mo, corr = {amo_best_corr:.3f}")

# === AR(1) surrogate function ===
from statsmodels.regression.linear_model import yule_walker

def ar1_surrogate(series, n_surr=1000):
    rho, sigma = yule_walker(series, order=1)
    phi = rho[0]
    surr = []
    for _ in range(n_surr):
        noise = np.random.normal(0, sigma, len(series))
        x = np.zeros(len(series))
        for t in range(1, len(series)):
            x[t] = phi*x[t-1] + noise[t]
        surr.append(x)
    return np.array(surr)

# === Shifted overlay plots ===
def plot_shifted_overlay(series_smooth, series_name, sun_smooth, best_lag, peaks_idx):
    shifted_sun = np.roll(sun_smooth, best_lag)  # shift sun forward in time
    time = df.index

    plt.figure(figsize=(12,5))
    plt.plot(time, series_smooth, label=f"{series_name} (smoothed)", color='b')
    plt.plot(time, shifted_sun, label=f"Sunspots shifted {best_lag} mo", color='orange', alpha=0.7)
    plt.scatter(time[peaks_idx], series_smooth[peaks_idx], color='red', marker='o', s=30, label='Peaks')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.title(f"{series_name} vs Sunspots (shifted by {best_lag} months)")
    plt.xlabel("Year")
    plt.ylabel("Standardized anomalies")
    plt.legend()
    plt.tight_layout()
    plt.show()



n_surr = 1000

# === Surrogate significance ===
def surrogate_pval(series_smooth, reference_smooth, best_corr):
    surr = ar1_surrogate(series_smooth, n_surr=n_surr)
    surr_max_corr = []
    for s in surr:
        _, surr_corrs, _, _ = cross_corr(s, reference_smooth)
        surr_max_corr.append(np.max(surr_corrs))
    surr_max_corr = np.array(surr_max_corr)
    pval = (np.sum(surr_max_corr >= best_corr)+1)/(n_surr+1)
    return pval, surr_max_corr

enso_pval, enso_surr_corrs = surrogate_pval(enso_smooth, sun_smooth, enso_best_corr)
amo_pval, amo_surr_corrs = surrogate_pval(amo_smooth, sun_smooth, amo_best_corr)

print(f"ENSO lag significance: p = {enso_pval:.4f}")
print(f"AMO lag significance: p = {amo_pval:.4f}")

# === Plot cross-correlation curves ===
plt.figure(figsize=(10,5))
plt.plot(enso_lags, enso_corrs, label='ENSO', color='b')
plt.plot(amo_lags, amo_corrs, label='AMO', color='g')
plt.axvline(enso_best_lag, color='b', linestyle='--', label=f'ENSO best lag: {enso_best_lag} mo')
plt.axvline(amo_best_lag, color='g', linestyle='--', label=f'AMO best lag: {amo_best_lag} mo')
# Optional: harmonic reference
plt.axvline(2*enso_best_lag, color='r', linestyle=':', label='2×ENSO lag')
plt.xlabel("Lag (months, Sunspots → series)")
plt.ylabel("Correlation")
plt.title("Sunspots lead ENSO and AMO: Cross-Correlation")
plt.legend()
plt.tight_layout()
plt.show()

# === ENSO overlay ===
plot_shifted_overlay(enso_smooth, "ENSO", sun_smooth, enso_best_lag, enso_peaks)

# === AMO overlay ===
plot_shifted_overlay(amo_smooth, "AMO", sun_smooth, amo_best_lag, amo_peaks)

# === Histogram of surrogate max correlations ===
plt.figure(figsize=(10,4))
plt.hist(enso_surr_corrs, bins=30, alpha=0.5, color='b', label='ENSO surrogates')
plt.hist(amo_surr_corrs, bins=30, alpha=0.5, color='g', label='AMO surrogates')
plt.axvline(enso_best_corr, color='b', linestyle='--', label='ENSO observed')
plt.axvline(amo_best_corr, color='g', linestyle='--', label='AMO observed')
plt.xlabel("Maximum correlation (Sunspots → series)")
plt.ylabel("Frequency")
plt.title("Distribution of max correlation in AR(1) surrogates")
plt.legend()
plt.tight_layout()
plt.show()
