import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.stats import pearsonr
import os

def load_enso_data(filepath='enso_data.csv'):
    """Load ENSO data from CSV file"""
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"ERROR: File {filepath} not found in current directory.")
            print(f"Current directory contents: {os.listdir('.')}")
            raise FileNotFoundError(f"No file found at {filepath}")
        
        # Read CSV with flexible parsing
        df = pd.read_csv(filepath, header=None, names=['date', 'nino34'])
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Sort index to ensure chronological order
        df.sort_index(inplace=True)
        
        print(f"Loaded ENSO data: {len(df)} months from {df.index[0].date()} to {df.index[-1].date()}")
        return df['nino34']
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration.")
        return generate_synthetic_enso()

def generate_synthetic_enso():
    """Generate synthetic ENSO-like data for testing"""
    np.random.seed(42)
    t = np.arange(0, 155*12)  # 155 years, monthly (matching your data length)
    dates = pd.date_range('1870-01-01', periods=len(t), freq='MS')  # MS = Month Start
    
    # Threading predicted periods (years): [16.19, 8.1, 4.05, 1.62, 0.81, 0.4]
    # Convert to monthly periods
    periods_months = np.array([16.19, 8.1, 4.05, 1.62, 0.81, 0.4]) * 12
    
    signal_data = np.zeros(len(t))
    for period in periods_months:
        amplitude = np.random.uniform(0.3, 1.0)
        phase = np.random.uniform(0, 2*np.pi)
        signal_data += amplitude * np.sin(2*np.pi*t/period + phase)
    
    # Add noise and trend
    noise = np.random.normal(0, 0.5, len(t))
    trend = 0.001 * t  # slight warming trend
    
    synthetic_enso = signal_data + noise + trend
    return pd.Series(synthetic_enso, index=dates)

# Rest of the previous script remains the same



def continuous_wavelet_transform(data, dt=1.0, dj=0.1, s0=2, J=None):
    """
    Perform continuous wavelet transform using Morlet wavelet
    
    Parameters:
    data: time series data
    dt: time step (1 for monthly data in units of months)
    dj: scale resolution (smaller = higher resolution)
    s0: smallest scale (2 months minimum)
    J: number of scales (if None, auto-calculate)
    """
    N = len(data)
    if J is None:
        # Calculate J to reach ~20 year maximum period (240 months)
        J = int(np.log2(240 / s0) / dj)
    
    # Create scales - ensure they're reasonable for monthly data
    scales = s0 * 2**(np.arange(0, J+1) * dj)
    
    # Filter out scales that are too small or too large
    valid_scales = scales[(scales >= 2) & (scales <= 300)]  # 2 months to 25 years
    
    print(f"Using {len(valid_scales)} scales from {valid_scales[0]:.1f} to {valid_scales[-1]:.1f} months")
    
    # Perform CWT
    coefficients, frequencies = pywt.cwt(data, valid_scales, 'morl', dt)
    
    # Convert frequencies to periods (in months)
    periods = 1.0 / frequencies
    
    return coefficients, periods, valid_scales

def plot_wavelet_analysis(data, coefficients, periods, threading_periods, title="ENSO Threading Wavelet Analysis"):
    """Plot wavelet transform results with threading predictions"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Time series
    ax1.plot(data.index, data.values, 'b-', linewidth=0.8)
    ax1.set_ylabel('Niño 3.4 Anomaly (°C)')
    ax1.set_title(f'{title} ({data.index[0].year}-{data.index[-1].year})')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Wavelet power spectrum
    power = np.abs(coefficients)**2
    
    # Create time array for plotting (years since start)
    time_years = np.array([(date - data.index[0]).days / 365.25 for date in data.index])
    
    # Plot wavelet power (log scale for better visualization)
    power_log = np.log10(power + 1e-10)  # Add small value to avoid log(0)
    im = ax2.contourf(time_years, periods/12, power_log, levels=50, cmap='viridis')
    ax2.set_ylabel('Period (years)')
    ax2.set_yscale('log')
    ax2.set_ylim([0.5, 20])
    ax2.set_xlabel('Years since start')
    
    # Add threading predictions as horizontal lines
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'magenta']
    for i, period in enumerate(threading_periods):
        if 0.5 <= period <= 20:  # Only plot if in visible range
            ax2.axhline(y=period, color=colors[i % len(colors)], 
                       linestyle='--', linewidth=2, alpha=0.9,
                       label=f'Threading: {period:.1f}yr')
    
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Wavelet Power Spectrum (Log Scale)')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, label='Log₁₀(Power)')
    
    # Global wavelet spectrum (time-averaged)
    global_power = np.mean(power, axis=1)
    ax3.loglog(periods/12, global_power, 'b-', linewidth=2, label='Observed Power')
    ax3.set_xlabel('Period (years)')
    ax3.set_ylabel('Average Power')
    ax3.set_xlim([0.5, 20])
    ax3.grid(True, alpha=0.3)
    
    # Mark threading predictions
    for i, period in enumerate(threading_periods):
        if 0.5 <= period <= 20:
            ax3.axvline(x=period, color=colors[i % len(colors)], 
                       linestyle='--', linewidth=2, alpha=0.9,
                       label=f'{period:.1f}yr')
    
    ax3.set_title('Global Wavelet Spectrum')
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    return fig, power, global_power

def analyze_threading_correlation(periods, global_power, threading_periods):
    """Analyze correlation between threading predictions and observed power peaks"""
    
    # Find peaks in global power spectrum
    # Use relative height threshold
    peak_height = np.percentile(global_power, 75)  # Top 25% of power values
    peak_indices, properties = signal.find_peaks(global_power, 
                                                height=peak_height,
                                                distance=5)  # Minimum separation
    observed_periods = periods[peak_indices] / 12  # Convert to years
    observed_powers = global_power[peak_indices]
    
    print("THREADING vs OBSERVED ANALYSIS")
    print("="*60)
    print(f"Threading Predictions (years): {threading_periods}")
    print(f"Observed Peaks (years): {np.round(observed_periods, 2)}")
    print()
    
    # Match threading predictions to nearest observed peaks
    matches = []
    for t_period in threading_periods:
        if 0.5 <= t_period <= 20:  # Within analysis range
            # Find closest observed peak
            distances = np.abs(observed_periods - t_period)
            if len(distances) > 0:  # Make sure we have peaks to match
                closest_idx = np.argmin(distances)
                closest_period = observed_periods[closest_idx]
                closest_power = observed_powers[closest_idx]
                error_pct = 100 * abs(closest_period - t_period) / t_period
                
                matches.append({
                    'threading': t_period,
                    'observed': closest_period,
                    'error_pct': error_pct,
                    'power': closest_power,
                    'power_rank': np.sum(observed_powers > closest_power) + 1
                })
                
                print(f"Threading {t_period:5.1f}yr → Observed {closest_period:5.1f}yr (error: {error_pct:4.1f}%, power rank: {np.sum(observed_powers > closest_power) + 1}/{len(observed_powers)})")
    
    return matches, observed_periods, observed_powers

def statistical_significance_test(matches, observed_periods, n_bootstrap=1000):
    """Test statistical significance of threading matches"""
    
    if len(matches) == 0:
        print("No matches found for significance testing.")
        return None, []
    
    # Calculate observed match quality
    observed_errors = [m['error_pct'] for m in matches]
    observed_mean_error = np.mean(observed_errors)
    
    # Bootstrap test: random predictions
    bootstrap_errors = []
    n_periods = len(matches)
    
    for _ in range(n_bootstrap):
        # Generate random periods in same range as threading predictions
        random_periods = np.random.uniform(0.4, 16.2, n_periods)
        
        # Match to observed peaks
        random_errors = []
        for r_period in random_periods:
            if len(observed_periods) > 0:
                distances = np.abs(observed_periods - r_period)
                closest_period = observed_periods[np.argmin(distances)]
                error_pct = 100 * abs(closest_period - r_period) / r_period
                random_errors.append(error_pct)
        
        if len(random_errors) > 0:
            bootstrap_errors.append(np.mean(random_errors))
    
    if len(bootstrap_errors) == 0:
        print("Could not perform significance test.")
        return None, []
    
    # Calculate p-value
    p_value = np.sum(np.array(bootstrap_errors) <= observed_mean_error) / len(bootstrap_errors)
    
    print(f"\nSTATISTICAL SIGNIFICANCE TEST:")
    print(f"Observed mean error: {observed_mean_error:.1f}%")
    print(f"Random prediction mean error: {np.mean(bootstrap_errors):.1f}% ± {np.std(bootstrap_errors):.1f}%")
    print(f"P-value (threading better than random): {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ SIGNIFICANT: Threading predictions significantly better than random")
    else:
        print("❌ NOT SIGNIFICANT: Could be due to chance")
    
    return p_value, bootstrap_errors

def main(filepath=None):
    """Main analysis function"""
    print("ENSO Threading Wavelet Analysis")
    print("="*50)
    
    # Threading predictions from your results (years)
    threading_periods = [16.19, 8.1, 4.05, 1.62, 0.81, 0.4]
    
    # Load data
    if filepath:
        print(f"Loading ENSO data from: {filepath}")
        nino34 = load_enso_data(filepath)
    else:
        print("No filepath provided. Using synthetic data.")
        nino34 = generate_synthetic_enso()
    
    # Use original data (anomalies are already detrended)
    nino34_detrended = nino34
    
    # Perform wavelet analysis
    print("Performing continuous wavelet transform...")
    coefficients, periods, scales = continuous_wavelet_transform(
        nino34_detrended.values, dt=1, dj=0.1, s0=6  # Monthly data, 6-month minimum period
    )
    
    # Plot results
    fig, power, global_power = plot_wavelet_analysis(nino34_detrended, coefficients, periods, threading_periods)
    plt.show()
    
    # Analyze correlations
    matches, observed_periods, observed_powers = analyze_threading_correlation(periods, global_power, threading_periods)
    
    # Statistical significance test
    p_value, bootstrap_errors = statistical_significance_test(matches, observed_periods)
    
    # Summary statistics
    if len(matches) > 0:
        errors = [m['error_pct'] for m in matches]
        print(f"\nSUMMARY STATISTICS:")
        print(f"Average error: {np.mean(errors):.1f}%")
        print(f"Median error: {np.median(errors):.1f}%")
        print(f"Max error: {np.max(errors):.1f}%")
        print(f"Matches within 10% error: {sum(1 for e in errors if e < 10)}/{len(errors)}")
        print(f"Matches within 20% error: {sum(1 for e in errors if e < 20)}/{len(errors)}")
    else:
        print("\nNo matches found for analysis.")
    
    return nino34_detrended, coefficients, periods, matches

# Usage:
# For your data file:
# data, coefficients, periods, matches = main('path/to/your/enso_data.csv')

# For synthetic data (testing):
if __name__ == "__main__":
    # Run with real ENSO data
    data, coefficients, periods, matches = main('enso_data.csv')