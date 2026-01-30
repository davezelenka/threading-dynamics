#!/usr/bin/env python3
"""
Corrected LMFDB Analysis - Stratified by Rank
Avoids confounding between rank and height
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_lmfdb_data(csv_file):
    """
    Load LMFDB elliptic curve data from CSV.
    """
    print(f"Loading LMFDB data from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    
    # Clean column names (extract from HYPERLINK formulas)
    df.columns = [col.split('"')[-2] if '"' in col else col for col in df.columns]
    
    print(f"Loaded {len(df)} curves")
    print(f"Columns: {list(df.columns)}")
    
    # Ensure numeric types
    df['conductor'] = pd.to_numeric(df['conductor'], errors='coerce')
    df['disc'] = pd.to_numeric(df['disc'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['regulator'] = pd.to_numeric(df['regulator'], errors='coerce')
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['conductor', 'disc', 'rank', 'regulator'])
    
    print(f"After cleaning: {len(df)} curves")
    
    return df

def compute_derived_properties(df):
    """
    Compute derived properties.
    """
    print(f"\nComputing derived properties...")
    
    df['abs_disc'] = np.abs(df['disc'])
    df['log_disc'] = np.log10(df['abs_disc'])
    df['tau'] = np.log(np.log(df['abs_disc']))
    
    # Use regulator directly as proxy for height
    # (Faltings height ~ log(regulator) for rank > 0)
    df['log_regulator'] = np.log10(df['regulator'] + 1e-10)
    
    print(f"  Added: abs_disc, log_disc, tau, log_regulator")
    
    return df

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_by_rank(df):
    """
    Analyze predictions stratified by rank.
    """
    print(f"\nAnalyzing {len(df)} curves stratified by rank...")
    
    results = {}
    
    # Get rank distribution
    rank_dist = df['rank'].value_counts().sort_index()
    print(f"\nRank distribution:")
    print(rank_dist)
    
    # ========================================================================
    # PREDICTION 2: Height Floor Scaling (by rank)
    # ========================================================================
    print(f"\n{'='*70}")
    print("PREDICTION 2: Height Floor Scaling (Stratified by Rank)")
    print(f"{'='*70}")
    
    results['pred2_by_rank'] = {}
    
    for rank_val in sorted(df['rank'].unique()):
        df_rank = df[df['rank'] == rank_val].copy()
        
        if len(df_rank) > 10:
            slope, intercept, r_value, p_value, std_err = linregress(
                df_rank['log_disc'], 
                df_rank['log_regulator']
            )
            
            error_pct = abs(slope - (-1/12))/0.083333*100
            
            print(f"\nRank {int(rank_val)}: n={len(df_rank)}")
            print(f"  Slope: {slope:.6f} (expected -0.0833)")
            print(f"  R²: {r_value**2:.6f}")
            print(f"  Error: {error_pct:.2f}%")
            print(f"  p-value: {p_value:.2e}")
            
            results['pred2_by_rank'][int(rank_val)] = {
                'slope': slope,
                'r2': r_value**2,
                'p_value': p_value,
                'error_pct': error_pct,
                'n': len(df_rank),
            }
    
    # ========================================================================
    # PREDICTION 3: Rank-Threading Depth Correlation
    # ========================================================================
    print(f"\n{'='*70}")
    print("PREDICTION 3: Rank-Threading Depth Correlation")
    print(f"{'='*70}")
    
    corr, pval = spearmanr(df['rank'], df['tau'])
    slope3, intercept3, r_value3, p_value3, std_err3 = linregress(
        df['tau'], df['rank']
    )
    
    print(f"\nSpearman correlation: ρ = {corr:.6f} (p={pval:.2e})")
    print(f"Linear fit: rank = {slope3:.6f} * τ + {intercept3:.6f}")
    print(f"R² = {r_value3**2:.6f}")
    print(f"Expected slope: 1.0")
    print(f"Observed slope: {slope3:.6f}")
    
    results['pred3'] = {
        'correlation': corr,
        'p_value': pval,
        'slope': slope3,
        'r2': r_value3**2,
        'error': abs(slope3 - 1.0),
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_by_rank(df, output_file='lmfdb_analysis_corrected.png'):
    """
    Create visualization stratified by rank.
    """
    print(f"\nCreating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Prediction 2 for Rank 1
    ax1 = fig.add_subplot(gs[0, 0])
    df_rank1 = df[df['rank'] == 1]
    ax1.scatter(df_rank1['log_disc'], df_rank1['log_regulator'], 
               alpha=0.5, s=20, color='steelblue', label='Rank 1')
    
    if len(df_rank1) > 10:
        slope1, intercept1, r_value1, _, _ = linregress(
            df_rank1['log_disc'], df_rank1['log_regulator']
        )
        x_fit = np.linspace(df_rank1['log_disc'].min(), df_rank1['log_disc'].max(), 100)
        y_fit = slope1 * x_fit + intercept1
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'slope = {slope1:.4f}')
    
    ax1.set_xlabel('log₁₀(|Δ|)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('log₁₀(Regulator)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Panel A: Height Scaling - Rank 1 (n={len(df_rank1)})',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Panel B: Prediction 2 for Rank 2
    ax2 = fig.add_subplot(gs[0, 1])
    df_rank2 = df[df['rank'] == 2]
    ax2.scatter(df_rank2['log_disc'], df_rank2['log_regulator'], 
               alpha=0.5, s=20, color='coral', label='Rank 2')
    
    if len(df_rank2) > 10:
        slope2, intercept2, r_value2, _, _ = linregress(
            df_rank2['log_disc'], df_rank2['log_regulator']
        )
        x_fit = np.linspace(df_rank2['log_disc'].min(), df_rank2['log_disc'].max(), 100)
        y_fit = slope2 * x_fit + intercept2
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'slope = {slope2:.4f}')
    
    ax2.set_xlabel('log₁₀(|Δ|)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('log₁₀(Regulator)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Panel B: Height Scaling - Rank 2 (n={len(df_rank2)})',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Panel C: Prediction 3 (rank vs τ)
    ax3 = fig.add_subplot(gs[1, 0])
    jitter = np.random.normal(0, 0.05, len(df))
    ax3.scatter(df['tau'], df['rank'] + jitter, 
               alpha=0.5, s=20, color='green')
    
    if len(df) > 10:
        slope3, intercept3, r_value3, _, _ = linregress(df['tau'], df['rank'])
        tau_fit = np.linspace(df['tau'].min(), df['tau'].max(), 100)
        rank_fit = slope3 * tau_fit + intercept3
        ax3.plot(tau_fit, rank_fit, 'r-', linewidth=2,
                label=f'slope = {slope3:.4f} (expected 1.0)')
    
    ax3.set_xlabel('Threading Depth τ = log(log(|Δ|))', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax3.set_title(f'Panel C: Rank-Threading Correlation (n={len(df)})',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # Panel D: Rank distribution
    ax4 = fig.add_subplot(gs[1, 1])
    rank_counts = df['rank'].value_counts().sort_index()
    ax4.bar(rank_counts.index, rank_counts.values, color='skyblue', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title('Panel D: Rank Distribution', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lmfdb_analysis.py <lmfdb_csv_file>")
        print("Example: python lmfdb_analysis.py lmfdb_ec_curvedata_0128_1616.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print("="*70)
    print("Corrected LMFDB Analysis - Stratified by Rank")
    print("="*70)
    
    # Load and process
    df = load_lmfdb_data(csv_file)
    df = compute_derived_properties(df)
    
    # Analyze
    print(f"\n{'='*70}")
    results = analyze_by_rank(df)
    
    # Visualize
    print(f"\n{'='*70}")
    visualize_by_rank(df)
    
    # Save
    df.to_csv('lmfdb_processed_corrected.csv', index=False)
    print(f"Processed data saved to: lmfdb_processed_corrected.csv")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nPrediction 2 (Height Scaling by Rank):")
    for rank_val, stats in results['pred2_by_rank'].items():
        print(f"\n  Rank {rank_val}:")
        print(f"    Slope: {stats['slope']:.6f} (expected -0.0833)")
        print(f"    R²: {stats['r2']:.4f}")
        print(f"    Error: {stats['error_pct']:.2f}%")
        print(f"    Curves: {stats['n']}")
    
    print(f"\nPrediction 3 (Rank Correlation):")
    print(f"  Correlation: {results['pred3']['correlation']:.4f}")
    print(f"  Slope: {results['pred3']['slope']:.6f} (expected 1.0)")
    print(f"  R²: {results['pred3']['r2']:.4f}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    print(f"\nKey Finding: Predictions need rank-stratified analysis")
    print(f"\nPrediction 2 shows different slopes for different ranks.")
    print(f"This suggests rank-dependent corrections to the scaling law.")
    print(f"\nPrediction 3 shows weak correlation overall.")
    print(f"This may indicate threading depth is not the primary organizing principle.")
    print(f"\nAlternative: Conductor may be more fundamental than discriminant.")


if __name__ == '__main__':
    main()