"""
ENSO Geometric Event Analysis - Robust Event Detection
------------------------------------------------------
Improved event detection with error handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import hilbert

class ENSOGeometricAnalysis:
    def __init__(self, data_path):
        # Load data
        self.df = pd.read_csv(data_path, parse_dates=['date'])
        self.df.set_index('date', inplace=True)
        self.series = self.df['34_anom'].values
        self.dates = self.df.index
        
        # Geometric analysis parameters
        self.fundamental_periods = None
        self.geometric_signatures = None
    
    def compute_spectral_geometry(self, top_n=5):
        # Compute Fourier Transform
        fft_vals = np.fft.fft(self.series - np.mean(self.series))
        freqs = np.fft.fftfreq(len(self.series), d=1/12)  # Monthly data
        
        # Positive frequencies
        mask = freqs > 0
        powers = np.abs(fft_vals[mask])**2
        periods = 1 / (freqs[mask] / 12)  # Convert to years
        
        # Top periods
        top_indices = np.argsort(powers)[-top_n:][::-1]
        top_periods = periods[top_indices]
        top_powers = powers[top_indices]
        
        # Geometric characteristics
        geometric_analysis = {
            'periods': top_periods,
            'powers': top_powers,
            'fundamental': np.min(top_periods),
            'harmonic_ladder': [
                np.min(top_periods),      # Fundamental
                np.min(top_periods) * 2,  # First harmonic
                np.min(top_periods) * 3,  # Second harmonic
                np.min(top_periods) / 2   # Subharmonic
            ]
        }
        
        self.fundamental_periods = geometric_analysis
        return geometric_analysis
    
    def geometric_coherence_analysis(self):
        if self.fundamental_periods is None:
            self.compute_spectral_geometry()
        
        # Hilbert transform for instantaneous characteristics
        analytic_signal = hilbert(self.series)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * 12  # cycles per year
        
        # Geometric coherence metrics
        coherence_metrics = {
            'phase_variability': np.std(instantaneous_phase),
            'frequency_range': {
                'min': np.min(instantaneous_freq),
                'max': np.max(instantaneous_freq)
            },
            'frequency_entropy': stats.entropy(np.abs(instantaneous_freq)),
            'geometric_scaling': np.std(instantaneous_freq) / np.mean(instantaneous_freq)
        }
        
        self.geometric_signatures = coherence_metrics
        return coherence_metrics
    
    def generate_report(self):
        print("=== ENSO GEOMETRIC ANALYSIS REPORT ===\n")
        
        # Spectral Geometry
        print("1. SPECTRAL GEOMETRY")
        if self.fundamental_periods:
            print("Dominant Periods:")
            for p, power in zip(self.fundamental_periods['periods'], 
                                 self.fundamental_periods['powers']):
                print(f"  - {p:.2f} years (Power: {power:.2f})")
            print(f"\nFundamental Period: {self.fundamental_periods['fundamental']:.2f} years")
            print("Harmonic Ladder:", 
                  [f"{h:.2f}" for h in self.fundamental_periods['harmonic_ladder']])
        
        # Geometric Coherence
        print("\n2. GEOMETRIC COHERENCE")
        if self.geometric_signatures:
            for key, value in self.geometric_signatures.items():
                # Handle different value types
                if key == 'frequency_range':
                    print(f"  Frequency Range: Min {value['min']:.4f}, Max {value['max']:.4f}")
                elif isinstance(value, (int, float)):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # ENSO Events
        events = self.event_geometric_signature()
        print(f"\n3. ENSO EVENTS (Total: {len(events)})")
        for event in events:
            print(f"  {event['type']} Event:")
            print(f"    Start: {event['start'].date()}")
            print(f"    End: {event['end'].date()}")
            print(f"    Duration: {event['duration']:.2f} years")
            print(f"    Peak Amplitude: {event['peak_amplitude']:.2f}°C")
            print(f"    Dominant Period: {event.get('dominant_period', 'N/A'):.2f} years\n")
        
        print("=== CRITICAL INTERPRETATION GUIDELINES ===")
        print("1. This is a SPECULATIVE geometric exploration")
        print("2. Results are CONCEPTUAL and NOT predictive")
        print("3. Extensive validation is REQUIRED")
    
    def event_geometric_signature(self, threshold=0.5, min_event_months=6):
        # Identify El Niño and La Niña events
        events = []
        in_event = False
        event_start = None
        event_series_list = []
        
        for i, value in enumerate(self.series):
            # El Niño events
            if value > threshold and not in_event:
                event_start = self.dates[i]
                in_event = True
                event_type = 'El Niño'
                event_series_list = [value]
            
            # La Niña events
            elif value < -threshold and not in_event:
                event_start = self.dates[i]
                in_event = True
                event_type = 'La Niña'
                event_series_list = [value]
            
            # Continue event
            elif in_event:
                event_series_list.append(value)
            
            # End of event
            elif abs(value) <= threshold and in_event:
                event_end = self.dates[i-1]
                duration = (event_end - event_start).days / 365.25  # years
                event_series = np.array(event_series_list)
                
                # Only process events longer than minimum duration
                if len(event_series) >= min_event_months:
                    # Compute event geometric characteristics
                    event_fft = np.fft.fft(event_series - np.mean(event_series))
                    event_freqs = np.fft.fftfreq(len(event_series), d=1/12)
                    
                    # Safely find dominant period
                    positive_freqs_mask = event_freqs > 0
                    if np.any(positive_freqs_mask):
                        event_periods = 1 / (event_freqs[positive_freqs_mask] / 12)
                        event_fft_positive = np.abs(event_fft[positive_freqs_mask])
                        
                        if len(event_fft_positive) > 0:
                            dominant_period_idx = np.argmax(event_fft_positive)
                            dominant_period = event_periods[dominant_period_idx]
                        else:
                            dominant_period = np.nan
                    else:
                        dominant_period = np.nan
                    
                    events.append({
                        'type': event_type,
                        'start': event_start,
                        'end': event_end,
                        'duration': duration,
                        'peak_amplitude': np.max(np.abs(event_series)),
                        'dominant_period': dominant_period
                    })
                
                in_event = False
                event_series_list = []
        
        return events

def main():
    # Assumes a CSV with columns: date,34_anom
    analysis = ENSOGeometricAnalysis('nino34.csv')
    
    # Perform analyses
    analysis.compute_spectral_geometry()
    analysis.geometric_coherence_analysis()
    
    # Generate report
    analysis.generate_report()

if __name__ == "__main__":
    main()