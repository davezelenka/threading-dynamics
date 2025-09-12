import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
import os

def load_enso_data(filepath='enso_data.csv'):
    """Load ENSO data from CSV file"""
    try:
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
        return None

def advanced_spectral_analysis(data, threading_periods):
    """Comprehensive spectral analysis with multiple statistical tests"""
    # Prepare results dictionary
    results = {
        'periods': threading_periods,
        'detailed_analysis': []
    }
    
    # Compute periodogram
    frequencies, power_spectrum = signal.periodogram(data)
    periods = 1 / frequencies[1:]  # Exclude zero frequency
    power = power_spectrum[1:]
    
    # Normalize power spectrum
    power_normalized = power / np.max(power)
    
    # Detailed analysis for each threading period
    for t_period in threading_periods:
        # Find closest observed period
        period_index = np.argmin(np.abs(periods - t_period))
        closest_period = periods[period_index]
        closest_power = power_normalized[period_index]
        
        # Compute local spectral characteristics
        # Look at ±20% range around the predicted period
        range_mask = np.abs(periods - t_period) / t_period <= 0.2
        local_periods = periods[range_mask]
        local_power = power_normalized[range_mask]
        
        # Statistical tests
        # 1. Peak significance using chi-square test
        peak_significance = 1 - stats.chi2.cdf(closest_power, df=2)
        
        # 2. Spectral concentration
        spectral_concentration = np.sum(local_power) / np.sum(power_normalized)
        
        # 3. Periodicity stability
        # Compute windowed coherence to assess temporal stability
        window_size = len(data) // 4  # 4 segments
        windowed_coherence = compute_windowed_coherence(data, t_period, window_size)
        
        # Compile detailed results
        period_analysis = {
            'predicted_period': t_period,
            'observed_period': closest_period,
            'period_error_pct': 100 * abs(closest_period - t_period) / t_period,
            'peak_power': closest_power,
            'peak_significance': peak_significance,
            'spectral_concentration': spectral_concentration,
            'windowed_coherence': windowed_coherence
        }
        
        results['detailed_analysis'].append(period_analysis)
    
    # Overall threading hypothesis test
    results['hypothesis_test'] = compute_threading_hypothesis_test(results)
    
    return results

def compute_windowed_coherence(data, target_period, window_size):
    """Compute coherence of a specific period across time windows"""
    # Number of windows
    n_windows = len(data) // window_size
    coherence_values = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = data[start:end]
        
        # Compute periodogram for window
        frequencies, power = signal.periodogram(window_data)
        periods = 1 / frequencies[1:]  # Exclude zero frequency
        
        # Find closest period
        period_index = np.argmin(np.abs(periods - target_period))
        coherence_values.append(power[period_index+1])  # +1 to skip zero frequency
    
    return {
        'mean_coherence': np.mean(coherence_values),
        'coherence_std': np.std(coherence_values),
        'coherence_variation': np.std(coherence_values) / np.mean(coherence_values) if np.mean(coherence_values) != 0 else np.inf
    }

def compute_threading_hypothesis_test(spectral_results):
    """Compute overall hypothesis test for threading theory"""
    # Aggregate metrics
    period_errors = [p['period_error_pct'] for p in spectral_results['detailed_analysis']]
    peak_significances = [p['peak_significance'] for p in spectral_results['detailed_analysis']]
    spectral_concentrations = [p['spectral_concentration'] for p in spectral_results['detailed_analysis']]
    
    # Compute composite metrics
    mean_error = np.mean(period_errors)
    mean_significance = np.mean(peak_significances)
    mean_concentration = np.mean(spectral_concentrations)
    
    # Hypothesis test
    # Lower error, higher significance, higher concentration suggest stronger support
    threading_score = (
        (20 - mean_error) / 20 *  # Error term (lower is better)
        mean_significance *        # Significance term (higher is better)
        mean_concentration         # Concentration term (higher is better)
    )
    
    return {
        'mean_period_error': mean_error,
        'mean_peak_significance': mean_significance,
        'mean_spectral_concentration': mean_concentration,
        'threading_score': threading_score
    }

def main(filepath='enso_data.csv'):
    """Main analysis function"""
    print("ENSO Threading Spectral Analysis")
    print("="*50)
    
    # Threading predictions from your results (years)
    threading_periods = [16.19, 8.1, 4.05, 1.62, 0.81, 0.4]
    
    # Load data
    nino34 = load_enso_data(filepath)
    if nino34 is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Perform advanced spectral analysis
    spectral_results = advanced_spectral_analysis(nino34.values, threading_periods)
    
    # Print detailed results
    print("\nDETAILED THREADING PERIOD ANALYSIS")
    print("="*40)
    for analysis in spectral_results['detailed_analysis']:
        print(f"Predicted Period: {analysis['predicted_period']} years")
        print(f"  Observed Period: {analysis['observed_period']:.2f} years")
        print(f"  Period Error: {analysis['period_error_pct']:.2f}%")
        print(f"  Peak Power: {analysis['peak_power']:.4f}")
        print(f"  Peak Significance: {analysis['peak_significance']:.4f}")
        print(f"  Spectral Concentration: {analysis['spectral_concentration']:.4f}")
        print(f"  Windowed Coherence: {analysis['windowed_coherence']['mean_coherence']:.4f} ± {analysis['windowed_coherence']['coherence_std']:.4f}")
        print()
    
    # Print hypothesis test results
    print("THREADING HYPOTHESIS TEST")
    print("="*30)
    hypothesis_test = spectral_results['hypothesis_test']
    print(f"Mean Period Error: {hypothesis_test['mean_period_error']:.2f}%")
    print(f"Mean Peak Significance: {hypothesis_test['mean_peak_significance']:.4f}")
    print(f"Mean Spectral Concentration: {hypothesis_test['mean_spectral_concentration']:.4f}")
    print(f"Threading Score: {hypothesis_test['threading_score']:.4f}")
    
    return spectral_results

if __name__ == "__main__":
    # Run analysis
    results = main('enso_data.csv')