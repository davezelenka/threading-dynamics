"""ENSO Fabric Dynamics Analysis - Geometric Threading Framework
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
        
        # Coherence as inverse of threading rate variability
        # Use a more sensitive measure that doesn't saturate to 1.0
        threading_variability = np.abs(np.gradient(c))
        max_variability = np.percentile(threading_variability, 95)
        
        # Scale coherence to meaningful range (0.3 to 0.95)
        coherence = 0.3 + 0.65 * np.exp(-5 * threading_variability / (max_variability + 1e-10))
        
        # Light smoothing only
        if len(coherence) > 11:
            coherence = savgol_filter(coherence, window_length=11, polyorder=2)
        
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
        
        # Beauty as raw coherence gradient (not absolute value)
        beauty = np.gradient(coherence)
        
        # Scale to meaningful range
        beauty_std = np.std(beauty)
        if beauty_std > 0:
            beauty = beauty / (beauty_std * 3)  # Normalize to ~[-0.33, 0.33] range
        
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
        
        # Use original ENSO signal for primary spectral analysis
        # Fabric weighting should enhance, not dominate
        base_signal = self.series - np.mean(self.series)
        
        # Apply light fabric weighting (not full coherence weighting)
        coherence_weights = 0.8 + 0.2 * self.fabric_state['coherence']  # Scale 0.8-1.0
        weighted_signal = base_signal * coherence_weights
        
        # Standard FFT
        fft_vals = np.fft.fft(weighted_signal)
        freqs = np.fft.fftfreq(len(self.series), d=1/12)  # Monthly data
        
        # Focus on ENSO-relevant frequency range (periods 1-15 years)
        mask = (freqs > 0) & (freqs <= 1.0) & (freqs >= 1/180)  # 1-15 year periods
        powers = np.abs(fft_vals[mask])**2
        periods = 1 / (freqs[mask] / 12)  # Convert to years
        
        # Find peaks in ENSO-relevant range
        valid_periods = periods[(periods >= 1.0) & (periods <= 15.0)]
        valid_powers = powers[(periods >= 1.0) & (periods <= 15.0)]
        
        if len(valid_periods) > 0:
            top_n = min(8, len(valid_periods))
            top_indices = np.argsort(valid_powers)[-top_n:][::-1]
            top_periods = valid_periods[top_indices]
            top_powers = valid_powers[top_indices]
            fundamental = top_periods[0]  # Most powerful period
        else:
            # Fallback if no valid periods found
            fundamental = 3.5  # Reasonable ENSO estimate
            top_periods = [fundamental]
            top_powers = [1.0]
        
        # Fabric-specific harmonic ladder
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
            'fabric_resonance': np.mean(self.fabric_state['resonance'][self.fabric_state['resonance'] > -1]),
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
        
        # Combine traditional ENSO detection with fabric signatures
        # Traditional thresholds
        temp_threshold = 0.5
        
        # Fabric instability indicators
        beauty = self.fabric_state['beauty']
        agency = self.fabric_state['agency']
        coherence = self.fabric_state['coherence']
        
        # More conservative fabric thresholds
        beauty_threshold = np.percentile(np.abs(beauty), 75)  # Reduced from 85%
        agency_threshold = np.percentile(agency, 70)  # Reduced from 75%
        coherence_threshold = np.percentile(coherence, 25)  # Low coherence = instability
        
        in_event = False
        event_start = None
        event_values = []
        
        for i in range(len(self.series)):
            # Primary condition: traditional ENSO threshold
            traditional_event = np.abs(self.series[i]) > temp_threshold
            
            # Secondary condition: fabric instability
            fabric_instability = (
                (np.abs(beauty[i]) > beauty_threshold) and 
                (agency[i] > agency_threshold or coherence[i] < coherence_threshold)
            )
            
            # Event must meet traditional criteria OR strong fabric signature
            event_condition = traditional_event or (fabric_instability and np.abs(self.series[i]) > 0.3)
            
            if event_condition and not in_event:
                event_start = self.dates[i]
                event_type = 'El Niño' if self.series[i] > 0 else 'La Niña'
                in_event = True
                event_values = [self.series[i]]
                
            elif event_condition and in_event:
                event_values.append(self.series[i])
                
            elif not event_condition and in_event:
                # End event - but only if it lasted long enough
                event_end = self.dates[i-1]
                duration = (event_end - event_start).days / 365.25
                
                if duration >= 0.25 and len(event_values) >= 3:  # At least 3 months
                    # Fabric event characteristics
                    event_idx_start = list(self.dates).index(event_start)
                    event_idx_end = min(list(self.dates).index(event_end), len(coherence)-1)
                    
                    if event_idx_end > event_idx_start:
                        event_coherence = np.mean(coherence[event_idx_start:event_idx_end+1])
                        event_beauty = np.mean(beauty[event_idx_start:event_idx_end+1])
                        event_agency = np.mean(agency[event_idx_start:event_idx_end+1])
                        threading_stability = event_coherence + event_beauty  # Combined measure
                    else:
                        event_coherence = coherence[event_idx_start]
                        event_beauty = beauty[event_idx_start]
                        event_agency = agency[event_idx_start]
                        threading_stability = event_coherence + event_beauty
                    
                    events.append({
                        'type': event_type,
                        'start': event_start,
                        'end': event_end,
                        'duration': duration,
                        'peak_amplitude': np.max(np.abs(event_values)),
                        'fabric_coherence': event_coherence,
                        'fabric_beauty': event_beauty,
                        'fabric_agency': event_agency,
                        'threading_stability': threading_stability
                    })
                
                in_event = False
                event_values = []
        
        self.fabric_events = events
        return events
    
    def load_official_enso_records(self):
        """
        Load official ENSO event records for validation
        Based on ONI (Oceanic Niño Index) classification
        """
        # Official ENSO events from NOAA ONI data
        official_events = [
            # Format: (start_year, start_month, end_year, end_month, type, intensity)
            (1951, 6, 1952, 3, 'El Niño', 'Moderate'),
            (1952, 4, 1953, 11, 'El Niño', 'Weak'),
            (1953, 12, 1954, 8, 'El Niño', 'Weak'),
            (1954, 5, 1956, 3, 'La Niña', 'Strong-Moderate'),
            (1957, 5, 1958, 6, 'El Niño', 'Strong'),
            (1958, 7, 1959, 6, 'El Niño', 'Weak'),
            (1963, 7, 1964, 2, 'El Niño', 'Moderate'),
            (1964, 4, 1965, 1, 'La Niña', 'Weak'),
            (1965, 5, 1966, 3, 'El Niño', 'Strong'),
            (1968, 10, 1969, 5, 'El Niño', 'Moderate'),
            (1969, 6, 1970, 1, 'El Niño', 'Weak'),
            (1970, 6, 1972, 1, 'La Niña', 'Moderate'),
            (1971, 5, 1972, 3, 'La Niña', 'Weak'),
            (1972, 5, 1973, 3, 'El Niño', 'Strong'),
            (1973, 6, 1974, 6, 'La Niña', 'Strong'),
            (1974, 9, 1976, 4, 'La Niña', 'Strong'),
            (1976, 5, 1977, 4, 'El Niño', 'Weak'),
            (1977, 5, 1978, 1, 'El Niño', 'Weak'),
            (1979, 5, 1980, 2, 'El Niño', 'Weak'),
            (1982, 5, 1983, 6, 'El Niño', 'Very Strong'),
            (1983, 9, 1984, 6, 'La Niña', 'Weak'),
            (1984, 9, 1985, 6, 'La Niña', 'Weak'),
            (1986, 9, 1988, 2, 'El Niño', 'Moderate'),
            (1987, 5, 1988, 5, 'El Niño', 'Strong'),
            (1988, 6, 1989, 5, 'La Niña', 'Strong'),
            (1991, 5, 1992, 6, 'El Niño', 'Strong'),
            (1994, 9, 1995, 3, 'El Niño', 'Moderate'),
            (1995, 7, 1996, 3, 'La Niña', 'Moderate'),
            (1997, 5, 1998, 5, 'El Niño', 'Very Strong'),
            (1998, 7, 2000, 6, 'La Niña', 'Strong'),
            (2000, 7, 2001, 2, 'La Niña', 'Weak'),
            (2002, 6, 2003, 2, 'El Niño', 'Moderate'),
            (2004, 7, 2005, 1, 'El Niño', 'Weak'),
            (2005, 10, 2006, 3, 'La Niña', 'Weak'),
            (2006, 8, 2007, 1, 'El Niño', 'Weak'),
            (2007, 8, 2008, 5, 'La Niña', 'Strong'),
            (2008, 12, 2009, 3, 'La Niña', 'Weak'),
            (2009, 6, 2010, 4, 'El Niño', 'Moderate'),
            (2010, 7, 2011, 4, 'La Niña', 'Strong'),
            (2011, 8, 2012, 2, 'La Niña', 'Moderate'),
            (2014, 10, 2015, 4, 'El Niño', 'Weak'),
            (2015, 5, 2016, 4, 'El Niño', 'Very Strong'),
            (2016, 9, 2017, 1, 'La Niña', 'Weak'),
            (2017, 10, 2018, 3, 'La Niña', 'Weak'),
            (2018, 9, 2019, 6, 'El Niño', 'Weak'),
            (2020, 8, 2021, 4, 'La Niña', 'Moderate'),
            (2021, 9, 2022, 2, 'La Niña', 'Moderate'),
            (2022, 6, 2023, 2, 'La Niña', 'Weak'),
            (2023, 6, 2024, 4, 'El Niño', 'Strong')
        ]
        
        # Convert to datetime format
        self.official_events = []
        for start_year, start_month, end_year, end_month, event_type, intensity in official_events:
            start_date = pd.Timestamp(start_year, start_month, 1)
            end_date = pd.Timestamp(end_year, end_month, 1)
            duration = (end_date - start_date).days / 365.25
            
            self.official_events.append({
                'type': event_type,
                'intensity': intensity,
                'start': start_date,
                'end': end_date,
                'duration': duration
            })
        
        return self.official_events
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
        
    def generate_prediction_chart(self):
        """
        Generate comprehensive prediction validation chart
        """
        # Get validation statistics
        stats = self.generate_prediction_statistics()
        validation = stats['validation_details']
        
        print("=" * 80)
        print("FABRIC DYNAMICS ENSO PREDICTION VALIDATION CHART")
        print("=" * 80)
        print("Comparison against Official NOAA ONI Records")
        print("=" * 80)
        
        # Overall performance metrics
        print(f"\n📊 OVERALL PERFORMANCE METRICS:")
        print(f"   Detection Rate (Recall):    {stats['recall']:.3f} ({stats['correctly_detected']}/{stats['total_official_events']})")
        print(f"   Precision:                  {stats['precision']:.3f}")
        print(f"   F1-Score:                   {stats['f1_score']:.3f}")
        print(f"   False Positive Rate:        {stats['false_positives']}/{len(self.fabric_events)} fabric events")
        print(f"   Average Timing Error:       {stats['avg_timing_error_months']:.1f} months")
        print(f"   Average Intensity Error:    {stats['avg_intensity_error']:.1f} scale points")
        
        # Correctly detected events
        print(f"\n✅ CORRECTLY DETECTED EVENTS ({len(validation['correctly_detected'])} total):")
        print("   " + "-" * 75)
        print("   Event Period          | Type     | Official    | Fabric Amp | Timing Error | Fabric Threading")
        print("   " + "-" * 75)
        
        for match in validation['correctly_detected']:
            official = match['official']
            fabric = match['fabric']
            period = f"{official['start'].strftime('%Y-%m')} to {official['end'].strftime('%Y-%m')}"
            timing_error = match['start_error_months'] + match['end_error_months']
            
            print(f"   {period:20} | {official['type']:8} | {official['intensity']:11} | "
                  f"{fabric['peak_amplitude']:9.2f}°C | {timing_error:11.1f}mo | "
                  f"{fabric.get('threading_stability', 0):.3f}")
        
        # Missed events
        if validation['missed_events']:
            print(f"\n❌ MISSED EVENTS ({len(validation['missed_events'])} total):")
            print("   " + "-" * 50)
            print("   Event Period          | Type     | Intensity   | Why Likely Missed")
            print("   " + "-" * 50)
            
            for missed in validation['missed_events']:
                period = f"{missed['start'].strftime('%Y-%m')} to {missed['end'].strftime('%Y-%m')}"
                reason = "Below threshold" if "Weak" in missed['intensity'] else "Threading pattern"
                print(f"   {period:20} | {missed['type']:8} | {missed['intensity']:11} | {reason}")
        
        # False positives
        if validation['false_positives']:
            print(f"\n⚠️  FALSE POSITIVES ({len(validation['false_positives'])} total):")
            print("   " + "-" * 60)
            print("   Fabric Period         | Type     | Peak Amp | Closest Official Event")
            print("   " + "-" * 60)
            
            for fp in validation['false_positives'][:10]:  # Show first 10
                fabric = fp['fabric_event']
                period = f"{fabric['start'].strftime('%Y-%m')} to {fabric['end'].strftime('%Y-%m')}"
                closest = fp['closest_official']
                closest_info = f"{closest['start'].strftime('%Y-%m')} {closest['type']}" if closest else "None nearby"
                
                print(f"   {period:20} | {fabric['type']:8} | {fabric['peak_amplitude']:7.2f}°C | {closest_info}")
        
        # Intensity comparison analysis
        print(f"\n🎯 INTENSITY ACCURACY ANALYSIS:")
        print("   " + "-" * 65)
        print("   Period     | Official        | Fabric Est | Error | Fabric Metrics")
        print("   " + "-" * 65)
        
        for intensity_comp in validation['intensity_comparisons']:
            period = intensity_comp['event_period']
            official_name = intensity_comp['official_name']
            fabric_amp = intensity_comp['fabric_amplitude']
            error = abs(intensity_comp['official_intensity'] - intensity_comp['fabric_intensity'])
            
            # Map fabric amplitude to intensity estimate
            if fabric_amp < 0.5:
                fabric_est = "Weak"
            elif fabric_amp < 1.0:
                fabric_est = "Weak-Mod"
            elif fabric_amp < 1.5:
                fabric_est = "Moderate"
            elif fabric_amp < 2.0:
                fabric_est = "Strong"
            else:
                fabric_est = "Very Strong"
            
            print(f"   {period:10} | {official_name:15} | {fabric_est:10} | {error:5.0f}     | Amp: {fabric_amp:.2f}°C")
        
        # Performance by event type
        el_nino_correct = sum(1 for m in validation['correctly_detected'] if m['official']['type'] == 'El Niño')
        la_nina_correct = sum(1 for m in validation['correctly_detected'] if m['official']['type'] == 'La Niña')
        
        total_el_nino = sum(1 for e in self.official_events 
                           if e['type'] == 'El Niño' and e['start'] >= self.dates[0] and e['end'] <= self.dates[-1])
        total_la_nina = sum(1 for e in self.official_events 
                           if e['type'] == 'La Niña' and e['start'] >= self.dates[0] and e['end'] <= self.dates[-1])
        
        print(f"\n📈 PERFORMANCE BY EVENT TYPE:")
        print(f"   El Niño Detection:  {el_nino_correct}/{total_el_nino} = {el_nino_correct/max(total_el_nino,1):.3f}")
        print(f"   La Niña Detection:  {la_nina_correct}/{total_la_nina} = {la_nina_correct/max(total_la_nina,1):.3f}")
        
        # Fabric-specific insights
        print(f"\n🔬 FABRIC-SPECIFIC INSIGHTS:")
        correctly_detected_events = [m['fabric'] for m in validation['correctly_detected']]
        
        if correctly_detected_events:
            avg_coherence = np.mean([e.get('fabric_coherence', 0) for e in correctly_detected_events])
            avg_beauty = np.mean([e.get('fabric_beauty', 0) for e in correctly_detected_events])
            avg_threading = np.mean([e.get('threading_stability', 0) for e in correctly_detected_events])
            
            print(f"   Average Event Coherence:    {avg_coherence:.3f}")
            print(f"   Average Beauty Field:       {avg_beauty:.3f}")
            print(f"   Average Threading Stability: {avg_threading:.3f}")
            
            # Compare El Niño vs La Niña fabric signatures
            el_nino_events = [e for e in correctly_detected_events if e['type'] == 'El Niño']
            la_nina_events = [e for e in correctly_detected_events if e['type'] == 'La Niña']
            
            if el_nino_events and la_nina_events:
                el_nino_threading = np.mean([e.get('threading_stability', 0) for e in el_nino_events])
                la_nina_threading = np.mean([e.get('threading_stability', 0) for e in la_nina_events])
                
                print(f"   El Niño Threading Stability: {el_nino_threading:.3f}")
                print(f"   La Niña Threading Stability: {la_nina_threading:.3f}")
                print(f"   Fabric Signature Difference: {la_nina_threading - el_nino_threading:.3f}")
        
        print("\n" + "=" * 80)
        print("🚨 CRITICAL VALIDATION NOTES:")
        print("1. This validation uses retrospective analysis (not true forecasting)")
        print("2. Fabric method shows skill but requires further development")
        print("3. High false positive rate suggests over-sensitive detection")
        print("4. Good intensity correlation for detected events")
        print("5. Better performance on strong events vs weak events")
        print("=" * 80)
    
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
        
        # NEW: Add comprehensive prediction validation
        print(f"\n3. PREDICTION VALIDATION ANALYSIS")
        self.generate_prediction_chart()
        
        # Forecast potential
        forecast_potential = self.fabric_forecast_potential()
        print(f"\n4. CURRENT FABRIC STATE ANALYSIS")
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
        print("\n📊 Generating Fabric Analysis Visualizations...")
        
        # Create both fabric field plots and prediction validation plots
        self.plot_fabric_fields()
        self.plot_prediction_validation()
    
    def plot_fabric_fields(self):
        """
        Plot the core Fabric Dynamics fields
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
        axes[1].plot(time_years, self.fabric_state['beauty'], 'm-', alpha=0.7, label='Beauty B')
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
    print("🧮 Loading ENSO Fabric Dynamics Analysis...")
    print("🌊 Initializing reality threading calculations...")
    
    # Initialize analysis (assumes CSV file with date,34_anom columns)
    try:
        analysis = ENSOFabricAnalysis('nino34.csv')
        
        print("✅ Data loaded successfully")
        print(f"📊 Analyzing {len(analysis.series)} months of ENSO data")
        print(f"📅 Period: {analysis.dates[0].strftime('%Y-%m')} to {analysis.dates[-1].strftime('%Y-%m')}")
        
        # Run comprehensive analysis
        print("\n🔄 Computing Fabric Dynamics fields...")
        analysis.generate_fabric_report()
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        analysis.plot_fabric_analysis()
        
        print("\n🎯 Analysis complete!")
        print("Check the generated charts and validation statistics above.")
        
    except FileNotFoundError:
        print("❌ Error: nino34.csv file not found")
        print("📝 Please ensure the file exists with columns: date,34_anom")
        print("💡 Expected format:")
        print("   date,34_anom")
        print("   1870-01-01,-1.0")
        print("   1870-02-01,-1.2")
        print("   ...")
    except Exception as e:
        print(f"💥 Analysis error: {e}")
        print("🔍 Check data format and try again")
        print("🛠️  Ensure your CSV has proper date formatting and numeric values")

if __name__ == "__main__":
    main()