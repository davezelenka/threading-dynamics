#!/usr/bin/env python3
"""
Test Conductor as Alternative Organizing Principle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

def load_lmfdb_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [col.split('"')[-2] if '"' in col else col for col in df.columns]
    
    df['conductor'] = pd.to_numeric(df['conductor'], errors='coerce')
    df['disc'] = pd.to_numeric(df['disc'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['regulator'] = pd.to_numeric(df['regulator'], errors='coerce')
    
    df = df.dropna(subset=['conductor', 'disc', 'rank', 'regulator'])
    
    return df

def compute_properties(df):
    df['abs_disc'] = np.abs(df['disc'])
    df['log_disc'] = np.log10(df['abs_disc'])
    df['tau_disc'] = np.log(np.log(df['abs_disc']))
    
    df['log_conductor'] = np.log10(df['conductor'])
    df['tau_conductor'] = np.log(np.log(df['conductor']))
    
    df['log_regulator'] = np.log10(df['regulator'] + 1e-10)
    
    return df

def analyze_conductor(df):
    print("="*70)
    print("ALTERNATIVE HYPOTHESIS: Conductor as Organizing Principle")
    print("="*70)
    
    results = {}
    
    # Test 1: Regulator vs Conductor
    print(f"\n{'='*70}")
    print("TEST 1: Regulator vs Conductor (Rank-Stratified)")
    print(f"{'='*70}")
    
    for rank_val in sorted(df['rank'].unique()):
        df_rank = df[df['rank'] == rank_val]
        
        if len(df_rank) > 10:
            slope, intercept, r_value, p_value, _ = linregress(
                df_rank['log_conductor'], df_rank['log_regulator']
            )
            
            print(f"\nRank {int(rank_val)}: n={len(df_rank)}")
            print(f"  Slope: {slope:.6f}")
            print(f"  R²: {r_value**2:.6f}")
            print(f"  p-value: {p_value:.2e}")
            
            results[f'reg_vs_cond_rank{int(rank_val)}'] = {
                'slope': slope,
                'r2': r_value**2,
                'p_value': p_value,
            }
    
    # Test 2: Regulator vs Discriminant (for comparison)
    print(f"\n{'='*70}")
    print("TEST 2: Regulator vs Discriminant (Rank-Stratified) [FOR COMPARISON]")
    print(f"{'='*70}")
    
    for rank_val in sorted(df['rank'].unique()):
        df_rank = df[df['rank'] == rank_val]
        
        if len(df_rank) > 10:
            slope, intercept, r_value, p_value, _ = linregress(
                df_rank['log_disc'], df_rank['log_regulator']
            )
            
            print(f"\nRank {int(rank_val)}: n={len(df_rank)}")
            print(f"  Slope: {slope:.6f}")
            print(f"  R²: {r_value**2:.6f}")
            print(f"  p-value: {p_value:.2e}")
    
    # Test 3: Rank vs Conductor Threading Depth
    print(f"\n{'='*70}")
    print("TEST 3: Rank vs Conductor Threading Depth")
    print(f"{'='*70}")
    
    corr_cond, pval_cond = spearmanr(df['rank'], df['tau_conductor'])
    slope_cond, _, r_value_cond, _, _ = linregress(df['tau_conductor'], df['rank'])
    
    print(f"\nSpearman correlation: ρ = {corr_cond:.6f} (p={pval_cond:.2e})")
    print(f"Linear fit: rank = {slope_cond:.6f} * τ_conductor + ...")
    print(f"R²: {r_value_cond**2:.6f}")
    
    results['rank_vs_tau_cond'] = {
        'correlation': corr_cond,
        'slope': slope_cond,
        'r2': r_value_cond**2,
    }
    
    # Test 4: Rank vs Discriminant Threading Depth (for comparison)
    print(f"\n{'='*70}")
    print("TEST 4: Rank vs Discriminant Threading Depth [FOR COMPARISON]")
    print(f"{'='*70}")
    
    corr_disc, pval_disc = spearmanr(df['rank'], df['tau_disc'])
    slope_disc, _, r_value_disc, _, _ = linregress(df['tau_disc'], df['rank'])
    
    print(f"\nSpearman correlation: ρ = {corr_disc:.6f} (p={pval_disc:.2e})")
    print(f"Linear fit: rank = {slope_disc:.6f} * τ_discriminant + ...")
    print(f"R²: {r_value_disc**2:.6f}")
    
    return results

def visualize_comparison(df, output_file='conductor_analysis.png'):
    print(f"\nCreating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Regulator vs Conductor (Rank 1)
    ax1 = fig.add_subplot(gs[0, 0])
    df_r1 = df[df['rank'] == 1]
    ax1.scatter(df_r1['log_conductor'], df_r1['log_regulator'], 
               alpha=0.5, s=20, color='steelblue')
    
    slope1, intercept1, r_value1, _, _ = linregress(
        df_r1['log_conductor'], df_r1['log_regulator']
    )
    x_fit = np.linspace(df_r1['log_conductor'].min(), df_r1['log_conductor'].max(), 100)
    y_fit = slope1 * x_fit + intercept1
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'slope = {slope1:.4f}, R² = {r_value1**2:.4f}')
    
    ax1.set_xlabel('log₁₀(Conductor)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('log₁₀(Regulator)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Panel A: Regulator vs Conductor - Rank 1 (n={len(df_r1)})',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Panel B: Regulator vs Discriminant (Rank 1) [for comparison]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df_r1['log_disc'], df_r1['log_regulator'], 
               alpha=0.5, s=20, color='coral')
    
    slope2, intercept2, r_value2, _, _ = linregress(
        df_r1['log_disc'], df_r1['log_regulator']
    )
    x_fit = np.linspace(df_r1['log_disc'].min(), df_r1['log_disc'].max(), 100)
    y_fit = slope2 * x_fit + intercept2
    ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'slope = {slope2:.4f}, R² = {r_value2**2:.4f}')
    
    ax2.set_xlabel('log₁₀(|Δ|)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('log₁₀(Regulator)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Panel B: Regulator vs Discriminant - Rank 1 (n={len(df_r1)})',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Panel C: Rank vs Conductor Threading Depth
    ax3 = fig.add_subplot(gs[1, 0])
    jitter = np.random.normal(0, 0.05, len(df))
    ax3.scatter(df['tau_conductor'], df['rank'] + jitter, 
               alpha=0.5, s=20, color='green')
    
    slope3, intercept3, r_value3, _, _ = linregress(df['tau_conductor'], df['rank'])
    x_fit = np.linspace(df['tau_conductor'].min(), df['tau_conductor'].max(), 100)
    y_fit = slope3 * x_fit + intercept3
    ax3.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'slope = {slope3:.4f}, R² = {r_value3**2:.4f}')
    
    ax3.set_xlabel('Threading Depth τ = log(log(Conductor))', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax3.set_title(f'Panel C: Rank vs Conductor Threading (n={len(df)})',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # Panel D: Rank vs Discriminant Threading Depth [for comparison]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df['tau_disc'], df['rank'] + jitter, 
               alpha=0.5, s=20, color='purple')
    
    slope4, intercept4, r_value4, _, _ = linregress(df['tau_disc'], df['rank'])
    x_fit = np.linspace(df['tau_disc'].min(), df['tau_disc'].max(), 100)
    y_fit = slope4 * x_fit + intercept4
    ax4.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'slope = {slope4:.4f}, R² = {r_value4**2:.4f}')
    
    ax4.set_xlabel('Threading Depth τ = log(log(|Δ|))', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax4.set_title(f'Panel D: Rank vs Discriminant Threading (n={len(df)})',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python conductor_analysis.py <lmfdb_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print("="*70)
    print("Conductor vs Discriminant: Which is Fundamental?")
    print("="*70)
    
    df = load_lmfdb_data(csv_file)
    df = compute_properties(df)
    
    results = analyze_conductor(df)
    
    visualize_comparison(df)
    
    print(f"\n{'='*70}")
    print("SUMMARY: Conductor vs Discriminant")
    print(f"{'='*70}")
    
    print(f"\nRegulator vs Conductor (Rank 1):")
    if 'reg_vs_cond_rank1' in results:
        print(f"  R² = {results['reg_vs_cond_rank1']['r2']:.6f}")
    
    print(f"\nRegulator vs Discriminant (Rank 1):")
    print(f"  R² = 0.032542 (from previous analysis)")
    
    print(f"\nRank vs Conductor Threading:")
    print(f"  ρ = {results['rank_vs_tau_cond']['correlation']:.6f}")
    print(f"  R² = {results['rank_vs_tau_cond']['r2']:.6f}")
    
    print(f"\nRank vs Discriminant Threading:")
    print(f"  ρ = -0.186603 (from previous analysis)")
    print(f"  R² = 0.031894")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    if results['rank_vs_tau_cond']['r2'] > 0.031894:
        print(f"\n✓ Conductor shows BETTER correlation with rank than discriminant")
    else:
        print(f"\n✗ Conductor shows WORSE correlation with rank than discriminant")
    
    print(f"\nBoth variables are weak predictors of rank.")
    print(f"The organizing principle for elliptic curves is NOT threading depth.")


if __name__ == '__main__':
    main()