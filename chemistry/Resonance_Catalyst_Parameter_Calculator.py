# Resonance Catalyst Parameter Calculator
# Author: Fabric Framework Project
# Date: Sept 24, 2025

# This script encodes vibrational resonance parameters relevant to the
# proposed CO₂ → Hydrocarbon resonance catalysis system.
# All vibrational frequency values are taken from spectroscopy data
# (gas-phase fundamentals for CO₂, and liquid-phase / hydrogen-bond-shifted bands for H₂O).

import numpy as np

# ----------------------------
# Thermal background (kT in cm⁻¹)
# Formula: E/k_B = hc·wavenumber
# Approximate conversions:
#   300 K → ~208 cm⁻¹
#   200 K → ~139 cm⁻¹
#   77 K  → ~53.5 cm⁻¹
# Values derived from k_B T / (hc) using standard constants.
thermal_background = {
    "300K": 208,
    "200K": 139,
    "77K": 53.5
}

# ----------------------------
# Vibrational modes of CO₂ (cm⁻¹)
# Source: Infrared spectroscopy
#   - Asymmetric stretch (ν₃): ~2349 cm⁻¹ (strong IR band, greenhouse-active)
#   - Symmetric stretch (ν₁): ~1333 cm⁻¹ (IR inactive in free CO₂, but may couple in structured states)
#   - Bending mode (ν₂): ~667 cm⁻¹ (also IR active, longwave contribution)
CO2_modes = {
    "asymmetric_stretch": 2349,
    "symmetric_stretch": 1333,
    "bend": 667
}

# ----------------------------
# Vibrational modes of H₂O (cm⁻¹)
# Source: Infrared + Raman spectroscopy
#   - Symmetric stretch (ν₁): ~3657 cm⁻¹ (gas-phase)
#   - Asymmetric stretch (ν₃): ~3756 cm⁻¹ (gas-phase)
#   - Bend (ν₂): ~1595 cm⁻¹ (gas-phase)
# In liquid water, hydrogen bonding redshifts & broadens:
#   - Stretches → ~3200–3400 cm⁻¹
#   - Bend → ~1640 cm⁻¹
H2O_modes = {
    "symmetric_stretch_gas": 3657,
    "asymmetric_stretch_gas": 3756,
    "bend_gas": 1595,
    "stretch_liquid": 3400,
    "bend_liquid": 1640
}

# ----------------------------
# Experimental pulse parameters
# These are engineering assumptions for resonance-coupled catalysis.
pulse_width_range = (1e-6, 1e-3)  # seconds (1 µs – 1 ms)
repetition_rate_range = (1e2, 1e4)  # Hz (100 Hz – 10 kHz)
duty_cycle_range = (0.01, 0.10)  # 1–10 %

# ----------------------------
# Resonator parameters
# Resonator quality factor (Q) requirement for coherence:
#   Q ≥ 50–200 (to sustain vibrational alignment long enough to impact chemistry)
resonator_Q_min = 50
resonator_Q_max = 200
resonator_materials = ["polar dielectrics", "doped semiconductors", "phononic crystals"]

# ----------------------------
# Utility: calculate beat frequencies between CO₂ and H₂O modes
def calculate_beat_frequencies(modes_A, modes_B):
    """
    Calculate difference frequencies (beat notes) between vibrational modes
    of two species (CO₂ and H₂O).
    Returns dictionary of mode pairs and their beat frequency in cm⁻¹ and THz.
    Conversion: 1 cm⁻¹ ≈ 29.9792458 GHz
    """
    results = {}
    for name_A, freq_A in modes_A.items():
        for name_B, freq_B in modes_B.items():
            beat_cm = abs(freq_A - freq_B)
            beat_THz = beat_cm * 0.0299792458  # GHz per cm⁻¹, then /1000 for THz
            results[f"{name_A} - {name_B}"] = {
                "beat_cm^-1": beat_cm,
                "beat_THz": beat_THz
            }
    return results

# Calculate CO₂-H₂O beat frequencies
beat_results = calculate_beat_frequencies(CO2_modes, H2O_modes)

# ----------------------------
# Print results
print("Thermal background (cm⁻¹):")
for T, val in thermal_background.items():
    print(f"  {T}: {val}")

print("\nBeat Frequencies CO₂–H₂O:")
for pair, data in beat_results.items():
    print(f"  {pair}: {data['beat_cm^-1']:.1f} cm⁻¹ ({data['beat_THz']:.3f} THz)")

print("\nPulse Parameters:")
print(f"  Width: {pulse_width_range[0]} – {pulse_width_range[1]} s")
print(f"  Repetition rate: {repetition_rate_range[0]} – {repetition_rate_range[1]} Hz")
print(f"  Duty cycle: {duty_cycle_range[0]*100} – {duty_cycle_range[1]*100}%")

print("\nResonator Requirements:")
print(f"  Q factor: {resonator_Q_min} – {resonator_Q_max}")
print(f"  Candidate materials: {', '.join(resonator_materials)}")
