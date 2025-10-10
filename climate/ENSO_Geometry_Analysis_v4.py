"""
ENSO Fabric Dynamics Analysis - Geometric Threading Framework
-------------------------------------------------------------
Applying Fabric Dynamics principles to ENSO event analysis
WARNING: This is a speculative mathematical exploration, not a validated forecasting method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import hilbert, savgol_filter
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class ENSOFabricAnalysis:
    def __init__(self, data_path):
        """Initialize with ENSO data and Fabric Dynamics parameters"""
        # Load data
        self.df = pd.read_csv(data_path, parse_dates=['date'])
        self.df.set_index('date', inplace=True)
        self.series = self.df['34_anom'].values
        self.dates = self.df.index
        
        # Fabric Dynamics state variables
        self.fabric_state = {
            'phi': None,          # Configuration space (Φ)
            'tau': None,          # Fabric depth (τ)
            'c_threading': None,  # Threading rate (c = ΔΦ/Δτ)
            'memory_active': None, # M_active (expressed constraints)
            'memory_latent': None, # M_latent (dormant memory)
            'coherence': None,    # C (system coherence)
            'resonance': None,    # R (stability measure)
            'beauty': None,       # B (coherence gradient)
            'agency': None        # A (choice/activation potential)
        }
        
        # Analysis results
        self.harmonic_structure = None
        self.fabric_events = None
        
    def compute_fabric_configuration(self, window_size=24):
        """
        Compute fabric configuration space Φ from ENSO data
        
        Parameters:
        -----------
        window_size : int
            Window for local geometric analysis
        """
        n = len(self.series)
        
        # Configuration space Φ: multi-scale geometric representation
        phi_local = np.zeros(n)
        phi_global = np.zeros(n)
        
        for i in range(window_size, n - window_size):
            # Local configuration: geometric complexity in neighborhood
            local_data = self.series[i-window_size//2:i+window_size//2]
            phi_local[i] = np.std(local_data) * np.mean(np.abs(np.diff(local_data)))
            
            # Global configuration: position in phase space
            phi_global[i] = np.cumsum(self.series[:i+1])[-1] / (i+1)
        
        # Combined configuration
        self.fabric_state['phi'] = phi_local + 0.1 * phi_global
        
        return self.fabric_state['phi']
    
    def compute_fabric_depth(self):
        """
        Compute fabric depth τ representing temporal embedding
        """
        # Fabric depth as cumulative information content
        tau = np.zeros(len(self.series))
        
        for i in range(1, len(self.series)):
            # Depth increases with accumulated geometric complexity
            delta_info = abs(self.series[i] - self.series[i-1])
            tau[i] = tau[i-1] + delta_info * (1 + 0.01 * i)  # Time-weighted accumulation
        
        self.fabric_state['tau'] = tau
        return tau
    
    def compute_threading_rate(self):
        """
        Compute threading rate c = ΔΦ/Δτ
        """
        if self.fabric_state['phi'] is None:
            self.compute_fabric_configuration()
        if self.fabric_state['tau'] is None:
            self.compute_fabric_depth()
        
        phi = self.fabric_state['phi']
        tau = self.fabric_state['tau']
        
        # Threading rate with numerical stability
        delta_phi = np.gradient(phi)
        delta_tau = np.gradient(tau)
        
        # Avoid division by zero
        delta_tau_safe = np.where(np.abs(delta_tau) < 1e-10, 1e-10, delta_tau)
        c_threading = delta_phi / delta_tau_safe
        
        # Smooth to remove numerical noise
        c_threading = savgol_filter(c_threading, window_length=min(51, len(c_threading)//4*2+1), polyorder=3)
        
        self.fabric_state['c_threading'] = c_threading
        return c_threading
    
    def compute_memory_dynamics(self):
        """
        Compute active and latent memory states
        """
        # Active memory: currently expressed constraints (recent variability)
        window = 12  # 1 year window
        m_active = pd.Series(self.series).rolling(window=window).std().fillna(0).values
        
        # Latent memory: stored potential (long-term mean tendencies)
        long_window = 60  # 5 year window
        m_latent = pd.Series(self.series).rolling(window=long_window).mean().fillna(0).values
        
        self.fabric_state['memory_active'] = m_active
        self.fabric_state['memory_latent'] = m_latent
        
        return m_active, m_latent
    
    def compute_coherence_field(self):
        """
        Compute system coherence C
        """
        if self.fabric_state['c_threading'] is None:
            self.compute_threading_rate()
        
        c = self.fabric_state['c_threading']
        
        # Coherence as stability of threading rate
        coherence = 1.0 / (1.0 + np.abs(np.gradient(c)))
        
        # Smooth coherence field
        coherence = savgol_filter(coherence, window_length=min(25, len(coherence)//4*2+1), polyorder=2)
        
        self.fabric_state['coherence'] = coherence
        return coherence
    
    def compute_resonance_structure(self):
        """
        Compute resonance R = Σcos(Δφ) for harmonic stability
        """
        # Use Hilbert transform to get phase information
        analytic_signal = hilbert(self.series - np.mean(self.series))
        phases = np.angle(analytic_signal)
        
        # Compute phase differences for resonance calculation
        n = len(phases)
        resonance = np.zeros(n)
        
        window = 24  # 2-year window for resonance calculation
        
        for i in range(window, n):
            # Phase differences in local window
            local_phases = phases[i-window:i]
            phase_diffs = np.diff(local_phases)
            
            # Resonance as coherence of phase evolution
            resonance[i] = np.mean(np.cos(phase_diffs))
        
        self.fabric_state['resonance'] = resonance
        return resonance
    
    def compute_beauty_field(self):
        """
        Compute beauty B = ∇C (coherence gradient)
        """
        if self.fabric_state['coherence'] is None:
            self.compute_coherence_field()
        
        coherence = self.fabric_state['coherence']
        beauty = -np.abs(np.gradient(coherence))  # Negative gradient magnitude
        
        self.fabric_state['beauty'] = beauty
        return beauty
    
    def compute_agency_potential(self):
        """
        Compute agency potential A (unmeasurable choice activation)
        Represented as system unpredictability / information gain
        """
        # Agency as local unpredictability
        window = 6
        agency = np.zeros(len(self.series))
        
        for i in range(window, len(self.series)):
            local_data = self.series[i-window:i]
            # Information gain as proxy for agency
            if len(np.unique(local_data)) > 1:
                agency[i] = stats.entropy(np.histogram(local_data, bins=5)[0] + 1e-10)
            else:
                agency[i] = 0
        
        self.fabric_state['agency'] = agency
        return agency
    
    def fabric_harmonic_analysis(self):
        """
        Perform Fabric-enhanced harmonic analysis
        """
        # Ensure all fabric states are computed
        self.compute_threading_rate()
        self.compute_memory_dynamics()
        self.compute_coherence_field()
        self.compute_resonance_structure()
        self.compute_beauty_field()
        self.compute_agency_potential()
        
        # Enhanced spectral analysis using fabric threading rate
        c_threading = self.fabric_state['c_threading']
        
        # Fabric-weighted Fourier transform
        coherence_weights = self.fabric_state['coherence']
        weighted_signal = self.series * coherence_weights
        
        # Standard FFT
        fft_vals = np.fft.fft(weighted_signal - np.mean(weighted_signal))
        freqs = np.fft.fftfreq(len(self.series), d=1/12)  # Monthly data
        
        # Positive frequencies
        mask = freqs > 0
        powers = np.abs(fft_vals[mask])**2
        periods = 1 / (freqs[mask] / 12)  # Convert to years
        
        # Find fundamental and harmonics
        top_n = 10
        top_indices = np.argsort(powers)[-top_n:][::-1]
        top_periods = periods[top_indices]
        top_powers = powers[top_indices]
        
        # Fabric-specific harmonic ladder
        fundamental = np.min(top_periods)
        harmonic_ladder = [
            fundamental,           # Fundamental
            fundamental * 2,       # First harmonic
            fundamental * 3,       # Second harmonic
            fundamental / 2,       # Subharmonic
            fundamental * 1.618,   # Golden ratio harmonic (Fabric signature)
            fundamental / 1.618    # Inverse golden ratio
        ]
        
        self.harmonic_structure = {
            'periods': top_periods,
            'powers': top_powers,
            'fundamental': fundamental,
            'harmonic_ladder': harmonic_ladder,
            'fabric_resonance': np.mean(self.fabric_state['resonance']),
            'fabric_coherence': np.mean(self.fabric_state['coherence'])
        }
        
        return self.harmonic_structure
    
    def fabric_event_detection(self):
        """
        Detect ENSO events using Fabric Dynamics principles
        """
        # Ensure fabric states are computed
        if self.fabric_state['coherence'] is None:
            self.fabric_harmonic_analysis()
        
        events = []
        
        # Fabric-based event detection criteria
        coherence = self.fabric_state['coherence']
        beauty = self.fabric_state['beauty']
        resonance = self.fabric_state['resonance']
        agency = self.fabric_state['agency']
        
        # Event detection based on fabric instabilities
        # Events occur when beauty (coherence gradient) is high AND agency is elevated
        event_threshold = np.percentile(np.abs(beauty), 85)
        agency_threshold = np.percentile(agency, 75)
        
        in_event = False
        event_start = None
        event_values = []
        
        for i in range(len(self.series)):
            # Event trigger: high beauty (instability) + elevated agency
            fabric_instability = (np.abs(beauty[i]) > event_threshold and 
                                agency[i] > agency_threshold)
            
            # Also check traditional threshold
            traditional_event = np.abs(self.series[i]) > 0.5
            
            event_condition = fabric_instability or traditional_event
            
            if event_condition and not in_event:
                event_start = self.dates[i]
                event_type = 'El Niño' if self.series[i] > 0 else 'La Niña'
                in_event = True
                event_values = [self.series[i]]
                
            elif event_condition and in_event:
                event_values.append(self.series[i])
                
            elif not event_condition and in_event:
                # End event
                event_end = self.dates[i-1]
                duration = (event_end - event_start).days / 365.25
                
                if duration > 0.25:  # At least 3 months
                    # Fabric event characteristics
                    event_idx_start = list(self.dates).index(event_start)
                    event_idx_end = list(self.dates).index(event_end)
                    
                    event_coherence = np.mean(coherence[event_idx_start:event_idx_end+1])
                    event_beauty = np.mean(beauty[event_idx_start:event_idx_end+1])
                    event_agency = np.mean(agency[event_idx_start:event_idx_end+1])
                    
                    events.append({
                        'type': event_type,
                        'start': event_start,
                        'end': event_end,
                        'duration': duration,
                        'peak_amplitude': np.max(np.abs(event_values)),
                        'fabric_coherence': event_coherence,
                        'fabric_beauty': event_beauty,
                        'fabric_agency': event_agency,
                        'threading_stability': event_coherence - np.abs(event_beauty)
                    })
                
                in_event = False
                event_values = []
        
        self.fabric_events = events
        return events
    
    def fabric_forecast_potential(self, forecast_months=24):
        """
        Assess fabric-based forecast potential (NOT actual forecasting)
        """
        if self.harmonic_structure is None:
            self.fabric_harmonic_analysis()
        
        # Current fabric state
        current_coherence = self.fabric_state['coherence'][-12:].mean()
        current_beauty = self.fabric_state['beauty'][-12:].mean()
        current_agency = self.fabric_state['agency'][-12:].mean()
        
        # Fabric stability indicators
        stability_metrics = {
            'coherence_trend': np.polyfit(range(12), self.fabric_state['coherence'][-12:], 1)[0],
            'beauty_level': current_beauty,
            'agency_potential': current_agency,
            'resonance_strength': self.fabric_state['resonance'][-12:].mean(),
            'threading_rate_stability': np.std(self.fabric_state['c_threading'][-12:])
        }
        
        # Forecast confidence based on fabric stability
        stability_score = (
            abs(stability_metrics['coherence_trend']) * 0.3 +
            abs(stability_metrics['beauty_level']) * 0.3 +
            stability_metrics['agency_potential'] * 0.2 +
            (1 - stability_metrics['resonance_strength']) * 0.2
        )
        
        forecast_confidence = 1.0 / (1.0 + stability_score)
        
        return {
            'stability_metrics': stability_metrics,
            'forecast_confidence': forecast_confidence,
            'fabric_state': 'stable' if forecast_confidence > 0.6 else 'unstable',
            'warning': 'This is speculative analysis, not predictive forecasting'
        }
    
    def generate_fabric_report(self):
        """
        Generate comprehensive Fabric Dynamics analysis report
        """
        print("=" * 60)
        print("ENSO FABRIC DYNAMICS ANALYSIS REPORT")
        print("=" * 60)
        print("WARNING: This is speculative mathematical exploration")
        print("NOT a validated forecasting method")
        print("=" * 60)
        
        # Fabric harmonic analysis
        if self.harmonic_structure is None:
            self.fabric_harmonic_analysis()
        
        print("\n1. FABRIC HARMONIC STRUCTURE")
        print(f"Fundamental Period: {self.harmonic_structure['fundamental']:.2f} years")
        print(f"Fabric Resonance: {self.harmonic_structure['fabric_resonance']:.3f}")
        print(f"Fabric Coherence: {self.harmonic_structure['fabric_coherence']:.3f}")
        
        print("\nHarmonic Ladder (including Fabric signatures):")
        for i, period in enumerate(self.harmonic_structure['harmonic_ladder']):
            labels = ['Fundamental', '1st Harmonic', '2nd Harmonic', 'Subharmonic', 
                     'Golden Ratio', 'Inverse Golden']
            print(f"  {labels[i]}: {period:.2f} years")
        
        # Fabric event analysis
        events = self.fabric_event_detection()
        print(f"\n2. FABRIC-DETECTED ENSO EVENTS ({len(events)} total)")
        
        if events:
            for event in events[-10:]:  # Show last 10 events
                print(f"\n{event['type']} Event:")
                print(f"  Period: {event['start'].strftime('%Y-%m')} to {event['end'].strftime('%Y-%m')}")
                print(f"  Duration: {event['duration']:.2f} years")
                print(f"  Peak Amplitude: {event['peak_amplitude']:.2f}°C")
                print(f"  Fabric Coherence: {event['fabric_coherence']:.3f}")
                print(f"  Fabric Beauty: {event['fabric_beauty']:.3f}")
                print(f"  Threading Stability: {event['threading_stability']:.3f}")
        
        # Forecast potential
        forecast_potential = self.fabric_forecast_potential()
        print(f"\n3. CURRENT FABRIC STATE ANALYSIS")
        print(f"Forecast Confidence: {forecast_potential['forecast_confidence']:.3f}")
        print(f"Fabric State: {forecast_potential['fabric_state']}")
        
        print(f"\nStability Metrics:")
        for metric, value in forecast_potential['stability_metrics'].items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print("\n" + "=" * 60)
        print("CRITICAL DISCLAIMERS:")
        print("1. This is SPECULATIVE mathematical exploration")
        print("2. Fabric Dynamics concepts require extensive validation")
        print("3. NOT suitable for operational forecasting")
        print("4. Results are conceptual, not predictive")
        print("5. Traditional ENSO forecasting methods remain authoritative")
        print("=" * 60)
    
    def plot_fabric_analysis(self):
        """
        Create visualization of Fabric Dynamics analysis
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Ensure all states are computed
        if self.fabric_state['coherence'] is None:
            self.fabric_harmonic_analysis()
        
        time_years = [d.year + d.month/12 for d in self.dates]
        
        # Plot 1: Original ENSO data with fabric threading rate
        axes[0].plot(time_years, self.series, 'b-', alpha=0.7, label='Niño 3.4 Anomaly')
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(time_years, self.fabric_state['c_threading'], 'r-', alpha=0.5, label='Threading Rate')
        axes[0].set_ylabel('Temperature Anomaly (°C)')
        ax0_twin.set_ylabel('Threading Rate c', color='red')
        axes[0].set_title('ENSO Data with Fabric Threading Rate')
        axes[0].legend(loc='upper left')
        ax0_twin.legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fabric coherence and beauty
        axes[1].plot(time_years, self.fabric_state['coherence'], 'g-', label='Coherence C')
        axes[1].plot(time_years, -self.fabric_state['beauty'], 'm-', alpha=0.7, label='Beauty -B')
        axes[1].set_ylabel('Fabric Fields')
        axes[1].set_title('Fabric Coherence and Beauty Fields')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Memory dynamics
        axes[2].plot(time_years, self.fabric_state['memory_active'], 'orange', label='Active Memory')
        axes[2].plot(time_years, self.fabric_state['memory_latent'], 'brown', label='Latent Memory')
        axes[2].set_ylabel('Memory States')
        axes[2].set_title('Fabric Memory Dynamics')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Resonance and agency
        axes[3].plot(time_years, self.fabric_state['resonance'], 'purple', label='Resonance R')
        axes[3].plot(time_years, self.fabric_state['agency'], 'cyan', alpha=0.7, label='Agency A')
        axes[3].set_ylabel('Resonance/Agency')
        axes[3].set_xlabel('Year')
        axes[3].set_title('Fabric Resonance and Agency Fields')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main analysis function
    """
    print("Loading ENSO Fabric Dynamics Analysis...")
    
    # Initialize analysis (assumes CSV file with date,34_anom columns)
    try:
        analysis = ENSOFabricAnalysis('nino34.csv')
        
        # Run comprehensive analysis
        analysis.generate_fabric_report()
        
        # Create visualizations
        analysis.plot_fabric_analysis()
        
    except FileNotFoundError:
        print("Error: nino34.csv file not found")
        print("Please ensure the file exists with columns: date,34_anom")
    except Exception as e:
        print(f"Analysis error: {e}")
        print("Check data format and try again")

if __name__ == "__main__":
    main()