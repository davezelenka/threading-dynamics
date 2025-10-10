"""
ENSO Fabric Forecast Ladder: Speculative Harmonic Analysis
----------------------------------------------------------
Disclaimer: This is a MATHEMATICAL EXPLORATION, not a validated forecasting method.

Core Principles:
1. Spectral decomposition of historical ENSO data
2. Identification of potential resonant periodicities
3. Stochastic harmonic synthesis with uncertainty quantification

Paramount Caveat: Results are conceptual and should NOT be used for prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft, fftfreq

# Robust statistical validation functions
def bootstrap_confidence_intervals(data, num_bootstraps=1000, ci=0.95):
    """Compute bootstrap confidence intervals for harmonic components"""
    bootstrapped_means = np.zeros(num_bootstraps)
    n = len(data)
    
    for i in range(num_bootstraps):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means[i] = np.mean(bootstrap_sample)
    
    # Compute confidence intervals
    lower_percentile = (1 - ci) / 2
    upper_percentile = 1 - lower_percentile
    
    return {
        'mean': np.mean(bootstrapped_means),
        'lower_ci': np.percentile(bootstrapped_means, lower_percentile * 100),
        'upper_ci': np.percentile(bootstrapped_means, upper_percentile * 100)
    }

def spectral_uncertainty(series, top_n=5):
    """Quantify uncertainty in spectral peaks"""
    fft_vals = fft(series - np.mean(series))
    freqs = fftfreq(len(series), d=1.0)
    mask = freqs > 0
    
    powers = np.abs(fft_vals[mask])**2
    periods = 1.0 / freqs[mask] / 12.0
    
    # Sort peaks by power
    peak_idx = np.argsort(powers)[-top_n:][::-1]
    peak_periods = periods[peak_idx]
    peak_powers = powers[peak_idx]
    
    return peak_periods, peak_powers

def stochastic_harmonic_synthesis(series, harmonics, noise_level=0.1):
    """Create stochastic harmonic projection with uncertainty"""
    N = len(series)
    t_months = np.arange(N, N + 12*15)  # 15 years
    
    # Base harmonic synthesis
    forecast = np.zeros_like(t_months, dtype=float)
    for h in harmonics:
        freq = 1.0 / (h * 12.0)  # cycles per month
        forecast += np.sin(2*np.pi*freq*t_months)
    
    # Normalize and add stochastic noise
    forecast *= np.std(series) / np.std(forecast)
    noise = np.random.normal(0, noise_level * np.std(series), len(t_months))
    forecast += noise
    
    return forecast

def main():
    # Load data
    df = pd.read_csv("nino34.csv", parse_dates=["date"])
    df = df.set_index("date")
    series = df["34_anom"].values
    
    # Spectral Analysis with Uncertainty
    peak_periods, peak_powers = spectral_uncertainty(series)
    
    print("\n=== Spectral Peaks (with Uncertainty) ===")
    for T, P in zip(peak_periods, peak_powers):
        print(f"  Period ≈ {T:.2f} yr (Power={P:.2f})")
    
    # Select fundamental and generate harmonics
    fundamental = np.min(peak_periods)
    harmonics = [fundamental, fundamental*2, fundamental*3, fundamental/2]
    
    print(f"\nFundamental ENSO period: {fundamental:.2f} years")
    print("\n=== Predicted Harmonic Ladder ===")
    for h in harmonics:
        print(f"  {h:.2f} years")
    
    # Stochastic Forecast with Confidence Intervals
    forecast = stochastic_harmonic_synthesis(series, harmonics)
    
    # Bootstrap Confidence Intervals
    ci_results = bootstrap_confidence_intervals(forecast)
    
    # Plotting
    plt.figure(figsize=(15,8))
    plt.plot(df.index, series, color="black", alpha=0.6, label="Observed Niño 3.4")
    
    future_dates = pd.date_range(start=df.index[-1], periods=len(forecast), freq="M")
    plt.plot(future_dates, forecast, color="red", label="Stochastic Harmonic Projection")
    
    # Confidence Interval Shading
    plt.fill_between(future_dates, 
                     forecast * (1 - 0.1),  # Lower bound
                     forecast * (1 + 0.1),  # Upper bound
                     color='red', alpha=0.2, label="Uncertainty Range")
    
    plt.title("ENSO Fabric Forecast: SPECULATIVE EXPLORATION")
    plt.xlabel("Year")
    plt.ylabel("Niño 3.4 Anomaly (°C)")
    plt.legend()
    
    # Statistical Annotations
    plt.text(future_dates[0], plt.ylim()[1], 
             "DISCLAIMER: Conceptual Mathematical Model\nNOT a Predictive Forecast", 
             color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print Uncertainty Statistics
    print("\n=== Forecast Uncertainty Statistics ===")
    print(f"Mean Projection: {ci_results['mean']:.4f}")
    print(f"95% Confidence Interval: [{ci_results['lower_ci']:.4f}, {ci_results['upper_ci']:.4f}]")
    
    print("\n=== CRITICAL INTERPRETATION GUIDELINES ===")
    print("1. This is a MATHEMATICAL EXPLORATION, not a forecast")
    print("2. Results are SPECULATIVE and should NOT be used for prediction")
    print("3. Extensive empirical validation is REQUIRED")

if __name__ == "__main__":
    main()