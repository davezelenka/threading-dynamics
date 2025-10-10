"""
ENSO THREADING STRESS TEST
Based on CPC Niño3.4 anomaly file (enso34.csv).
Assumes columns:
    Date    Nino Anom 3.4 Index
with missing values flagged as -9999.

Steps:
1. Load anomaly series 
2. Clean missing values, detrend
3. Compute predicted threading periods T = λ/c
4. Run Welch PSD and AR(1) red-noise significance
5. Compare observed peaks vs predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.stattools import acf
import os

# ---------------- USER PARAMETERS ----------------
filename = "data/enso34.csv"   # anomaly file
lambda_km = 25551.0       # threading wavelength (Earth scale, from your doc)
lambda_m = lambda_km * 1e3

# Candidate propagation speeds (m/s)
c_candidates = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])

# Sampling step: monthly
dt_month = 1.0
fs = 1.0 / dt_month   # samples per month

# -------------------------------------------------

def ar1_red_noise_spectrum(ts, fs, sims=500):
    """Monte Carlo AR(1) surrogate spectra, return 95th percentile curve."""
    rho = acf(ts, nlags=1, fft=False)[1]
    n = len(ts)
    rng = np.random.default_rng(0)
    psd_sims = []
    for i in range(sims):
        noise = np.zeros(n)
        eps = rng.normal(scale=np.std(ts)*np.sqrt(1-rho**2), size=n)
        for t in range(1, n):
            noise[t] = rho * noise[t-1] + eps[t]
        f, P = signal.periodogram(noise, fs=fs)
        psd_sims.append(P)
    psd_sims = np.array(psd_sims)
    psd95 = np.percentile(psd_sims, 95, axis=0)
    return f, psd95

# ---------------- MAIN ----------------

print("Loading data from", filename)

# File is comma-separated, format: YYYY-MM-DD,value
df = pd.read_csv(
    filename,
    sep=",",
    header=None,
    names=["date", "anom"],
    na_values=[-9999, -99.99]  # handle missing codes
)

# Parse ISO dates
df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d", errors="coerce")

# Drop rows with missing or bad values
df = df.dropna(subset=["date", "anom"])

print("Data length after cleaning:", len(df))
print(df.head())

anom = pd.Series(signal.detrend(df['anom'].values), index=df['date'])

# candidate threading periods
T_days = lambda_m / c_candidates / (3600*24)
T_years = T_days / 365.25
print("Candidate speeds (m/s):", c_candidates)
print("Predicted periods (years):", np.round(T_years, 2))

# Welch PSD
f_w, Pxx = signal.welch(anom.values, fs=fs, window='hann',
                        nperseg=256, noverlap=128, detrend='linear', scaling='density')
period_years = np.where(f_w > 0, 1.0 / f_w / 12.0, np.inf) # convert cycles/month to period in years

# Red-noise significance
f_red, psd95 = ar1_red_noise_spectrum(anom.values, fs)

# Compare predicted T with PSD peaks
results = []
for c, T_y in zip(c_candidates, T_years):
    # find peak within ±40% of predicted
    mask = (period_years > T_y*0.6) & (period_years < T_y*1.4)
    if np.any(mask):
        idx = np.argmax(Pxx[mask])
        sel_period = period_years[mask][idx]
        sel_power = Pxx[mask][idx]
        results.append((c, T_y, sel_period, sel_power))
    else:
        results.append((c, T_y, None, None))

# Plot
plt.figure(figsize=(10,5))
plt.loglog(period_years[1:], Pxx[1:], label="Welch PSD (Niño3.4)")
plt.loglog(1.0/(f_red[1:]*12.0), psd95[1:], '--', label="95% AR(1) threshold")

for c, T_y, sel_period, _ in results:
    if sel_period is not None:
        plt.axvline(sel_period, linestyle=":", label=f"c={c} m/s → pred {T_y:.2f} yr, peak {sel_period:.2f} yr")
    else:
        plt.axvline(T_y, linestyle="--", alpha=0.3, label=f"c={c} m/s → pred {T_y:.2f} yr")

plt.gca().invert_xaxis()
plt.xlabel("Period (years, log scale)")
plt.ylabel("Power spectral density")
plt.title("ENSO Threading Stress Test (Niño3.4)")
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# Summary table
summary = []
for c, T_y, sel_period, powv in results:
    note = f"peak {sel_period:.2f} yr" if sel_period else "no peak found"
    summary.append({"c (m/s)": c, "T_pred (yr)": round(T_y,2),
                    "peak_period (yr)": None if sel_period is None else round(sel_period,2),
                    "note": note})

df_summary = pd.DataFrame(summary)
print("\nSUMMARY TABLE")
print(df_summary.to_string(index=False))

# Save outputs
outdir = "data"
os.makedirs(outdir, exist_ok=True)
anom.to_csv(os.path.join(outdir,"nino34_anom_clean.csv"))
df_summary.to_csv(os.path.join(outdir,"enso_summary_table.csv"), index=False)
print("Outputs saved to:", outdir)