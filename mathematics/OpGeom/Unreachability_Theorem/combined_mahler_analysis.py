#!/usr/bin/env python3
"""
Combined Analysis: Generated Data + Known Minimal Mahler Measures
Tests Prediction 1: Threading Depth Quantization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """
    Load generated data and reference data.
    """
    print("Loading data...")
    
    # Load your generated data
    try:
        df_generated = pd.read_csv('mahler_data.csv')
        print(f"  Loaded {len(df_generated)} generated polynomials")
    except FileNotFoundError:
        print("  WARNING: mahler_data.csv not found")
        df_generated = pd.DataFrame()
    
    # Load reference data
    try:
        df_reference = pd.read_csv('reference_mahler_measures.csv')
        print(f"  Loaded {len(df_reference)} reference polynomials")
    except FileNotFoundError:
        print("  WARNING: reference_mahler_measures.csv not found")
        df_reference = pd.DataFrame()
    
    # Load Omega_p values
    try:
        df_omega = pd.read_csv('omega_p_values.csv')
        print(f"  Loaded {len(df_omega)} Omega_p values")
    except FileNotFoundError:
        print("  WARNING: omega_p_values.csv not found")
        df_omega = pd.DataFrame()
    
    return df_generated, df_reference, df_omega


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_combined_data(df_generated, df_reference, df_omega):
    """
    Analyze combined dataset.
    """
    print("\n" + "="*70)
    print("COMBINED ANALYSIS")
    print("="*70)
    
    # Combine datasets
    df_generated['source'] = 'generated'
    df_reference['source'] = 'reference'
    
    # Ensure both have same columns
    cols_to_keep = ['degree', 'mahler_measure', 'threading_depth', 'source']
    df_gen_subset = df_generated[cols_to_keep].copy()
    df_ref_subset = df_reference[cols_to_keep].copy()
    
    df_combined = pd.concat([df_gen_subset, df_ref_subset], ignore_index=True)
    
    print(f"\nTotal polynomials: {len(df_combined)}")
    print(f"  Generated: {len(df_generated)}")
    print(f"  Reference: {len(df_reference)}")
    
    # Analysis by degree
    print(f"\n{'='*70}")
    print("ANALYSIS BY DEGREE")
    print(f"{'='*70}")
    
    results = {}
    
    for degree in sorted(df_combined['degree'].unique()):
        subset = df_combined[df_combined['degree'] == degree]
        
        gen_subset = subset[subset['source'] == 'generated']
        ref_subset = subset[subset['source'] == 'reference']
        
        min_M = subset['mahler_measure'].min()
        min_tau = subset['threading_depth'].min()
        
        print(f"\nDegree {degree}:")
        print(f"  Total: {len(subset)} polynomials")
        print(f"    Generated: {len(gen_subset)}")
        print(f"    Reference: {len(ref_subset)}")
        print(f"  Mahler measure range: [{min_M:.6f}, {subset['mahler_measure'].max():.6f}]")
        print(f"  Threading depth range: [{min_tau:.6f}, {subset['threading_depth'].max():.6f}]")
        
        # Compare to Omega_p
        if not df_omega.empty:
            omega_row = df_omega[df_omega['prime'] == degree]
            if not omega_row.empty:
                omega_p = omega_row.iloc[0]['omega_p']
                tau_p = omega_row.iloc[0]['tau_p']
                print(f"  Ω_{degree}: {omega_p:.10f} (τ = {tau_p:.6f})")
                print(f"  Distance to Ω_{degree}: {abs(min_M - omega_p):.10f}")
        
        results[degree] = {
            'count': len(subset),
            'min_M': min_M,
            'max_M': subset['mahler_measure'].max(),
            'min_tau': min_tau,
            'max_tau': subset['threading_depth'].max(),
            'mean_tau': subset['threading_depth'].mean(),
        }
    
    return df_combined, results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_combined_visualization(df_combined, df_omega, output_file='combined_mahler_analysis.png'):
    """
    Create comprehensive visualization.
    """
    print(f"\nCreating visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # ========== Panel A: All data combined ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    for degree in sorted(df_combined['degree'].unique()):
        subset = df_combined[df_combined['degree'] == degree]
        
        # Plot generated vs reference
        gen = subset[subset['source'] == 'generated']
        ref = subset[subset['source'] == 'reference']
        
        if len(gen) > 0:
            ax1.scatter(gen['threading_depth'], [degree]*len(gen), 
                       alpha=0.5, s=50, marker='o', label=f'Generated (d={degree})' if degree == 2 else '')
        
        if len(ref) > 0:
            ax1.scatter(ref['threading_depth'], [degree]*len(ref), 
                       alpha=0.8, s=100, marker='*', color='red', 
                       label=f'Reference (d={degree})' if degree == 2 else '')
    
    # Mark Omega_p positions
    if not df_omega.empty:
        for _, row in df_omega.iterrows():
            p = row['prime']
            tau_p = row['tau_p']
            ax1.axvline(tau_p, color='green', linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Threading Depth τ = log(M)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Degree', fontsize=12, fontweight='bold')
    ax1.set_title('Panel A: Combined Data - Generated vs Reference', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')
    
    # ========== Panel B: Histogram by degree ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_combined['degree'].unique())))
    
    for i, degree in enumerate(sorted(df_combined['degree'].unique())):
        subset = df_combined[df_combined['degree'] == degree]
        ax2.hist(subset['threading_depth'], bins=20, alpha=0.5, 
                color=colors[i], label=f'd={degree}', edgecolor='black')
    
    ax2.set_xlabel('Threading Depth τ', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Panel B: Distribution by Degree', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # ========== Panel C: Minimum M by degree ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    degrees = []
    min_M_values = []
    
    for degree in sorted(df_combined['degree'].unique()):
        subset = df_combined[df_combined['degree'] == degree]
        degrees.append(degree)
        min_M_values.append(subset['mahler_measure'].min())
    
    ax3.plot(degrees, min_M_values, 'o-', linewidth=2, markersize=8, color='steelblue')
    
    # Add Omega_p reference
    if not df_omega.empty:
        omega_degrees = df_omega['prime'].values
        omega_values = df_omega['omega_p'].values
        ax3.scatter(omega_degrees, omega_values, s=200, marker='*', 
                   color='red', zorder=5, label='Ω_p')
    
    ax3.set_xlabel('Degree', fontsize=11)
    ax3.set_ylabel('Minimum Mahler Measure', fontsize=11)
    ax3.set_title('Panel C: Minimum M by Degree', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    # ========== Panel D: Mahler measure distribution ==========
    ax4 = fig.add_subplot(gs[1, 2])
    
    ax4.hist(df_combined['mahler_measure'], bins=50, alpha=0.7, 
            color='steelblue', edgecolor='black')
    
    # Mark Omega_p positions
    if not df_omega.empty:
        for _, row in df_omega.iterrows():
            ax4.axvline(row['omega_p'], color='red', linestyle='--', 
                       linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('Mahler Measure M', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Panel D: Mahler Measure Distribution', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # ========== Panel E: Cumulative distribution ==========
    ax5 = fig.add_subplot(gs[2, :])
    
    for degree in sorted(df_combined['degree'].unique()):
        subset = df_combined[df_combined['degree'] == degree]
        tau_sorted = sorted(subset['threading_depth'].values)
        cumulative = np.arange(1, len(tau_sorted) + 1) / len(tau_sorted)
        ax5.plot(tau_sorted, cumulative, label=f'd={degree}', linewidth=2)
    
    # Mark Omega_p positions
    if not df_omega.empty:
        for _, row in df_omega.iterrows():
            ax5.axvline(row['tau_p'], color='red', linestyle='--', alpha=0.3)
    
    ax5.set_xlabel('Threading Depth τ', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax5.set_title('Panel E: Cumulative Distribution (Gaps = Forbidden Zones)', 
                 fontsize=14, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution.
    """
    print("="*70)
    print("Combined Analysis: Generated + Reference Mahler Measures")
    print("Testing Prediction 1: Threading Depth Quantization")
    print("="*70)
    
    # Load data
    df_generated, df_reference, df_omega = load_data()
    
    if df_generated.empty and df_reference.empty:
        print("\nERROR: No data files found!")
        print("Please run:")
        print("  1. python mahler_measure_analysis.py")
        print("  2. python reference_mahler_database.py")
        return
    
    # Analyze
    df_combined, results = analyze_combined_data(df_generated, df_reference, df_omega)
    
    # Visualize
    if not df_combined.empty:
        create_combined_visualization(df_combined, df_omega)
    
    # Save combined data
    df_combined.to_csv('combined_mahler_analysis.csv', index=False)
    print(f"\nCombined data saved to: combined_mahler_analysis.csv")
    
    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    
    print("\n1. ATTRACTOR DETECTION:")
    for degree in sorted(results.keys()):
        res = results[degree]
        print(f"   Degree {degree}: min M = {res['min_M']:.10f}")
    
    print("\n2. THREADING DEPTH HIERARCHY:")
    for degree in sorted(results.keys()):
        res = results[degree]
        print(f"   Degree {degree}: τ_min = {res['min_tau']:.6f}, τ_mean = {res['mean_tau']:.6f}")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()