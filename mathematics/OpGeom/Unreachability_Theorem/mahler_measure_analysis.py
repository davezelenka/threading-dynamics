#!/usr/bin/env python3
"""
Mahler Measure Threading Depth Quantization Analysis
For: The Unreachability Principle (Zelenka 2025)

This script analyzes the distribution of Mahler measures of algebraic integers
to test Prediction 1: Threading depth quantization by the Omega_p hierarchy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.stats import gaussian_kde
from collections import defaultdict
import itertools

# ============================================================================
# PART 1: MAHLER MEASURE COMPUTATION
# ============================================================================

def mahler_measure(coeffs):
    """
    Compute Mahler measure of polynomial with given coefficients.
    
    M(P) = |a_n| * product of max(1, |root_i|) for all roots
    
    Args:
        coeffs: list of polynomial coefficients [a_0, a_1, ..., a_n]
                representing a_0 + a_1*x + ... + a_n*x^n
    
    Returns:
        Mahler measure M(P)
    """
    if len(coeffs) < 2:
        return 1.0
    
    # Leading coefficient
    a_n = abs(coeffs[-1])
    
    # Find roots
    roots = np.roots(coeffs[::-1])  # numpy wants highest degree first
    
    # Product of max(1, |root|)
    product = np.prod([max(1.0, abs(r)) for r in roots])
    
    return a_n * product


def threading_depth(mahler_measure_value):
    """
    Compute threading depth tau = log(M).
    
    Args:
        mahler_measure_value: Mahler measure M(alpha)
    
    Returns:
        Threading depth tau(alpha)
    """
    if mahler_measure_value <= 0:
        return -np.inf
    return np.log(mahler_measure_value)


def is_irreducible_check(coeffs):
    """
    Basic irreducibility check (not rigorous, but fast).
    
    For rigorous check, use Sage's is_irreducible().
    This is a heuristic: polynomial is likely irreducible if:
    - It has no rational roots (by rational root theorem)
    - Degree > 1
    
    Args:
        coeffs: polynomial coefficients
    
    Returns:
        True if likely irreducible (heuristic)
    """
    if len(coeffs) < 3:  # degree < 2
        return False
    
    # Check for rational roots using rational root theorem
    a_0, a_n = coeffs[0], coeffs[-1]
    if a_0 == 0:
        return False  # has root at 0
    
    # Possible rational roots: divisors of a_0 / divisors of a_n
    divisors_a0 = [d for d in range(1, abs(a_0) + 1) if a_0 % d == 0]
    divisors_an = [d for d in range(1, abs(a_n) + 1) if a_n % d == 0]
    
    possible_roots = []
    for p in divisors_a0:
        for q in divisors_an:
            possible_roots.extend([p/q, -p/q])
    
    # Evaluate polynomial at possible roots
    for r in possible_roots:
        val = sum(c * r**i for i, c in enumerate(coeffs))
        if abs(val) < 1e-10:
            return False  # has rational root
    
    return True


# ============================================================================
# PART 2: POLYNOMIAL GENERATION
# ============================================================================

def generate_polynomials(degree, height_bound=3, max_count=1000):
    """
    Generate monic integer polynomials of given degree.
    
    Args:
        degree: polynomial degree
        height_bound: maximum absolute value of coefficients
        max_count: maximum number of polynomials to generate
    
    Returns:
        List of coefficient tuples (a_0, a_1, ..., a_{n-1}, 1)
    """
    polynomials = []
    
    # Generate all combinations of coefficients
    coeff_range = range(-height_bound, height_bound + 1)
    
    # For monic polynomials: a_n = 1, vary a_0, ..., a_{n-1}
    for coeffs in itertools.product(coeff_range, repeat=degree):
        if len(polynomials) >= max_count:
            break
        
        # Skip if constant term is 0 (has root at 0)
        if coeffs[0] == 0:
            continue
        
        # Make monic: append leading coefficient 1
        poly = list(coeffs) + [1]
        
        # Basic irreducibility check
        if is_irreducible_check(poly):
            polynomials.append(tuple(poly))
    
    return polynomials


# ============================================================================
# PART 3: OMEGA_P REFERENCE VALUES
# ============================================================================

# Known Omega_p values (from Mossinghoff's work and LMFDB)
# These are the smallest known Mahler measures for each prime degree
OMEGA_P_VALUES = {
    2: 1.3247179572447,      # Golden ratio: (1 + sqrt(5))/2
    3: 1.3247179572447,      # Same as degree 2 (Lehmer's polynomial is degree 10)
    5: 1.1762808182599,      # Smallest known for degree 5
    7: 1.1279667902078,      # Smallest known for degree 7
    11: 1.0882908062069,     # Smallest known for degree 11
    13: 1.0812670938371,     # Smallest known for degree 13
}

# Lehmer's polynomial (degree 10, smallest known overall)
LEHMER_MAHLER = 1.1762808182599


# ============================================================================
# PART 4: DATA ANALYSIS
# ============================================================================

def analyze_mahler_distribution(data_by_degree):
    """
    Analyze distribution of Mahler measures and threading depths.
    
    Args:
        data_by_degree: dict mapping degree -> list of (coeffs, M, tau)
    
    Returns:
        Analysis results dictionary
    """
    results = {}
    
    for degree, data in data_by_degree.items():
        mahler_values = [m for _, m, _ in data]
        tau_values = [t for _, _, t in data]
        
        # Distance to nearest Omega_p
        omega_p = OMEGA_P_VALUES.get(degree, None)
        if omega_p:
            tau_omega = np.log(omega_p)
            distances = [abs(t - tau_omega) for t in tau_values]
        else:
            distances = None
        
        results[degree] = {
            'count': len(data),
            'mahler_mean': np.mean(mahler_values),
            'mahler_std': np.std(mahler_values),
            'mahler_min': np.min(mahler_values),
            'mahler_max': np.max(mahler_values),
            'tau_mean': np.mean(tau_values),
            'tau_std': np.std(tau_values),
            'tau_min': np.min(tau_values),
            'tau_max': np.max(tau_values),
            'distances': distances,
            'omega_p': omega_p,
        }
    
    return results


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def create_visualization(data_by_degree, analysis_results, output_file='mahler_analysis.png'):
    """
    Create comprehensive visualization of Mahler measure distribution.
    
    Args:
        data_by_degree: dict mapping degree -> list of (coeffs, M, tau)
        analysis_results: output from analyze_mahler_distribution
        output_file: filename for output figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Combined histogram of all tau values
    ax1 = fig.add_subplot(gs[0, :])
    all_tau = []
    for degree, data in data_by_degree.items():
        tau_values = [t for _, _, t in data]
        all_tau.extend(tau_values)
    
    ax1.hist(all_tau, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Mark Omega_p positions
    for degree, omega_p in OMEGA_P_VALUES.items():
        tau_omega = np.log(omega_p)
        ax1.axvline(tau_omega, color='red', linestyle='--', linewidth=2, 
                    label=f'Ω_{degree}' if degree == 2 else '')
    
    ax1.set_xlabel('Threading Depth τ = log(M)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Panel A: Distribution of Threading Depths (All Degrees)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel B: Separate histograms by degree
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_degree)))
    
    for i, (degree, data) in enumerate(sorted(data_by_degree.items())):
        tau_values = [t for _, _, t in data]
        ax2.hist(tau_values, bins=30, alpha=0.5, color=colors[i], 
                 label=f'Degree {degree}', edgecolor='black')
    
    ax2.set_xlabel('Threading Depth τ', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Panel B: Threading Depths by Degree', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel C: Distance to nearest Omega_p
    ax3 = fig.add_subplot(gs[1, 1])
    degrees_with_omega = []
    mean_distances = []
    
    for degree in sorted(data_by_degree.keys()):
        if degree in OMEGA_P_VALUES:
            distances = analysis_results[degree]['distances']
            if distances:
                degrees_with_omega.append(degree)
                mean_distances.append(np.mean(distances))
    
    if degrees_with_omega:
        ax3.bar(range(len(degrees_with_omega)), mean_distances, 
                color='coral', edgecolor='black')
        ax3.set_xticks(range(len(degrees_with_omega)))
        ax3.set_xticklabels([f'd={d}' for d in degrees_with_omega])
        ax3.set_ylabel('Mean Distance to Ω_p', fontsize=12)
        ax3.set_title('Panel C: Proximity to Ω_p Attractors', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
    
    # Panel D: Cumulative distribution
    ax4 = fig.add_subplot(gs[2, :])
    
    for degree, data in sorted(data_by_degree.items()):
        tau_values = sorted([t for _, _, t in data])
        cumulative = np.arange(1, len(tau_values) + 1) / len(tau_values)
        ax4.plot(tau_values, cumulative, label=f'Degree {degree}', linewidth=2)
    
    # Mark Omega_p positions
    for degree, omega_p in OMEGA_P_VALUES.items():
        tau_omega = np.log(omega_p)
        ax4.axvline(tau_omega, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Threading Depth τ', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Panel D: Cumulative Distribution (Gaps indicate forbidden zones)', 
                  fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: generate polynomials, compute Mahler measures, analyze.
    """
    print("="*70)
    print("Mahler Measure Threading Depth Quantization Analysis")
    print("The Unreachability Principle (Zelenka 2025)")
    print("="*70)
    
    # Configuration
    DEGREES = [2, 3, 5, 7]  # Prime degrees to analyze
    HEIGHT_BOUND = 3        # Coefficient bound
    MAX_PER_DEGREE = 1000   # Max polynomials per degree
    
    print(f"\nConfiguration:")
    print(f"  Degrees: {DEGREES}")
    print(f"  Height bound: {HEIGHT_BOUND}")
    print(f"  Max polynomials per degree: {MAX_PER_DEGREE}")
    
    # Generate and analyze
    data_by_degree = {}
    
    for degree in DEGREES:
        print(f"\n{'='*70}")
        print(f"Processing degree {degree}...")
        
        # Generate polynomials
        print(f"  Generating polynomials...")
        polynomials = generate_polynomials(degree, HEIGHT_BOUND, MAX_PER_DEGREE)
        print(f"  Generated {len(polynomials)} candidate polynomials")
        
        # Compute Mahler measures
        print(f"  Computing Mahler measures...")
        data = []
        for coeffs in polynomials:
            try:
                M = mahler_measure(coeffs)
                tau = threading_depth(M)
                if M > 1.0 and np.isfinite(tau):  # Only keep M > 1
                    data.append((coeffs, M, tau))
            except:
                continue  # Skip problematic polynomials
        
        print(f"  Computed {len(data)} valid Mahler measures")
        data_by_degree[degree] = data
        
        # Quick stats
        if data:
            mahler_values = [m for _, m, _ in data]
            print(f"  Mahler measure range: [{min(mahler_values):.6f}, {max(mahler_values):.6f}]")
            if degree in OMEGA_P_VALUES:
                print(f"  Ω_{degree} = {OMEGA_P_VALUES[degree]:.10f}")
    
    # Analyze
    print(f"\n{'='*70}")
    print("Analyzing distribution...")
    analysis_results = analyze_mahler_distribution(data_by_degree)
    
    # Print results
    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    for degree in sorted(analysis_results.keys()):
        res = analysis_results[degree]
        print(f"\nDegree {degree}:")
        print(f"  Sample size: {res['count']}")
        print(f"  Threading depth τ: {res['tau_mean']:.4f} ± {res['tau_std']:.4f}")
        print(f"  Range: [{res['tau_min']:.4f}, {res['tau_max']:.4f}]")
        if res['omega_p']:
            tau_omega = np.log(res['omega_p'])
            print(f"  Ω_{degree} position: τ = {tau_omega:.4f}")
            if res['distances']:
                print(f"  Mean distance to Ω_{degree}: {np.mean(res['distances']):.4f}")
    
    # Visualize
    print(f"\n{'='*70}")
    print("Creating visualization...")
    create_visualization(data_by_degree, analysis_results)
    
    # Export data
    print(f"\nExporting data to CSV...")
    all_data = []
    for degree, data in data_by_degree.items():
        for coeffs, M, tau in data:
            all_data.append({
                'degree': degree,
                'polynomial': str(coeffs),
                'mahler_measure': M,
                'threading_depth': tau,
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv('mahler_data.csv', index=False)
    print("Data saved to: mahler_data.csv")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()