import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_and_process_data(mass_models_path, sparc_path):
    """Load and process Mass Models and SPARC datasets"""
    column_names = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    mass_models_df = pd.read_csv(mass_models_path, sep='\s+', names=column_names)
    
    mass_models_agg = mass_models_df.groupby('ID').agg({
        'Vobs': 'max',
        'Vgas': 'max',
        'Vdisk': 'max',
        'Vbul': 'max',
        'D': 'first',
        'R': 'max'
    }).reset_index()
    
    sparc_df = pd.read_csv(sparc_path)
    return mass_models_df, mass_models_agg, sparc_df

def test_refined_structural_model(merged_data):
    """
    CRITICAL TEST: Does including more structural parameters eliminate the 'dark matter' signal?
    
    If including SBeff, Rdisk, Inc improves predictions → STRUCTURAL explanation
    If they don't help → Genuine dark matter signal
    """
    
    print("=" * 80)
    print("REFINED STRUCTURAL MODEL TEST")
    print("Question: Can detailed baryonic structure explain the 'dark matter' residuals?")
    print("=" * 80)
    
    # Calculate basic baryonic prediction
    merged_data['V_basic'] = np.sqrt(
        merged_data['Vgas']**2 + 
        merged_data['Vdisk']**2 + 
        merged_data['Vbul']**2
    )
    
    merged_data['residual_basic'] = merged_data['Vobs'] - merged_data['V_basic']
    
    # Required columns for refined model
    required_cols = ['Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBeff', 'Rdisk', 'Inc', 'mass_luminosity_ratio']
    
    # Clean data
    clean_data = merged_data[required_cols].dropna()
    
    print(f"\nSample size: {len(clean_data)} galaxies with complete structural data")
    
    # Model 1: Basic (just velocity components)
    print("\n" + "=" * 80)
    print("MODEL 1: Basic Baryonic Model (V_gas, V_disk, V_bulge only)")
    print("=" * 80)
    
    X_basic = clean_data[['Vgas', 'Vdisk', 'Vbul']].values
    y = clean_data['Vobs'].values
    
    # Linear regression
    reg_basic = LinearRegression()
    reg_basic.fit(X_basic, y)
    y_pred_basic = reg_basic.predict(X_basic)
    
    r2_basic = reg_basic.score(X_basic, y)
    rmse_basic = np.sqrt(np.mean((y - y_pred_basic)**2))
    
    print(f"\nR² = {r2_basic:.4f}")
    print(f"RMSE = {rmse_basic:.2f} km/s")
    print(f"Coefficients: V_gas={reg_basic.coef_[0]:.3f}, V_disk={reg_basic.coef_[1]:.3f}, V_bulge={reg_basic.coef_[2]:.3f}")
    
    # Model 2: Structural Enhancement
    print("\n" + "=" * 80)
    print("MODEL 2: Enhanced Structural Model (+ SBeff, Rdisk, Inc, M/L)")
    print("=" * 80)
    
    X_enhanced = clean_data[['Vgas', 'Vdisk', 'Vbul', 'SBeff', 'Rdisk', 'Inc', 'mass_luminosity_ratio']].values
    
    reg_enhanced = LinearRegression()
    reg_enhanced.fit(X_enhanced, y)
    y_pred_enhanced = reg_enhanced.predict(X_enhanced)
    
    r2_enhanced = reg_enhanced.score(X_enhanced, y)
    rmse_enhanced = np.sqrt(np.mean((y - y_pred_enhanced)**2))
    
    print(f"\nR² = {r2_enhanced:.4f}")
    print(f"RMSE = {rmse_enhanced:.2f} km/s")
    
    # Improvement
    r2_improvement = r2_enhanced - r2_basic
    rmse_improvement = rmse_basic - rmse_enhanced
    pct_improvement = 100 * rmse_improvement / rmse_basic
    
    print("\n" + "-" * 80)
    print("IMPROVEMENT FROM STRUCTURAL PARAMETERS:")
    print("-" * 80)
    print(f"ΔR² = +{r2_improvement:.4f} ({100*r2_improvement:.2f}% relative improvement)")
    print(f"ΔRMSE = -{rmse_improvement:.2f} km/s ({pct_improvement:.1f}% reduction)")
    
    # Statistical test: Are models significantly different?
    # F-test for nested models
    n = len(y)
    k_basic = 3
    k_enhanced = 7
    
    ss_res_basic = np.sum((y - y_pred_basic)**2)
    ss_res_enhanced = np.sum((y - y_pred_enhanced)**2)
    
    f_stat = ((ss_res_basic - ss_res_enhanced) / (k_enhanced - k_basic)) / (ss_res_enhanced / (n - k_enhanced - 1))
    p_value_f = 1 - stats.f.cdf(f_stat, k_enhanced - k_basic, n - k_enhanced - 1)
    
    print(f"\nF-test for model comparison:")
    print(f"F-statistic = {f_stat:.3f}")
    print(f"p-value = {p_value_f:.6f}")
    
    if p_value_f < 0.001:
        print("*** HIGHLY SIGNIFICANT! Structural parameters add explanatory power!")
    elif p_value_f < 0.05:
        print("** SIGNIFICANT! Structural parameters matter!")
    else:
        print("Not significant - structural parameters don't help much")
    
    # Show individual parameter importance
    print("\n" + "-" * 80)
    print("PARAMETER CONTRIBUTIONS (Enhanced Model):")
    print("-" * 80)
    param_names = ['Vgas', 'Vdisk', 'Vbul', 'SBeff', 'Rdisk', 'Inc', 'M/L ratio']
    for name, coef in zip(param_names, reg_enhanced.coef_):
        print(f"  {name:<15} β = {coef:>8.4f}")
    
    # Key insight: Does M/L ratio coefficient match dark matter prediction?
    print("\n" + "-" * 80)
    print("CRITICAL INTERPRETATION:")
    print("-" * 80)
    
    ml_coef = reg_enhanced.coef_[6]
    
    if abs(ml_coef) > 5:
        print(f"M/L ratio has LARGE effect (β={ml_coef:.2f})")
        print("→ Could indicate genuine dark matter signal")
    elif abs(ml_coef) < 2:
        print(f"M/L ratio has SMALL effect (β={ml_coef:.2f})")
        print("→ 'Dark matter' signal is actually captured by other structure!")
    
    # Most importantly: Are residuals now explained?
    clean_data = clean_data.copy()
    clean_data['residual_enhanced'] = y - y_pred_enhanced
    clean_data['residual_basic'] = y - y_pred_basic
    
    # Test if residuals are now uncorrelated with galaxy properties
    print("\n" + "=" * 80)
    print("RESIDUAL ANALYSIS: Are 'unexplained' velocities still systematic?")
    print("=" * 80)
    
    props_to_test = ['mass_luminosity_ratio', 'SBeff', 'Rdisk', 'Inc']
    
    print("\nCorrelations with residuals:")
    print("-" * 80)
    print(f"{'Property':<25} {'Basic Model':<20} {'Enhanced Model':<20}")
    print("-" * 80)
    
    for prop in props_to_test:
        r_basic, p_basic = stats.spearmanr(clean_data[prop], abs(clean_data['residual_basic']))
        r_enhanced, p_enhanced = stats.spearmanr(clean_data[prop], abs(clean_data['residual_enhanced']))
        
        reduction = 100 * (abs(r_basic) - abs(r_enhanced)) / abs(r_basic) if abs(r_basic) > 0.01 else 0
        
        sig_basic = "*" if p_basic < 0.05 else ""
        sig_enhanced = "*" if p_enhanced < 0.05 else ""
        
        print(f"{prop:<25} r={r_basic:>6.3f}{sig_basic:<2}         r={r_enhanced:>6.3f}{sig_enhanced:<2}  ({reduction:>5.1f}% reduction)")
    
    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    # Check if M/L correlation was eliminated
    r_ml_basic, _ = stats.spearmanr(clean_data['mass_luminosity_ratio'], abs(clean_data['residual_basic']))
    r_ml_enhanced, _ = stats.spearmanr(clean_data['mass_luminosity_ratio'], abs(clean_data['residual_enhanced']))
    
    ml_reduction = 100 * (abs(r_ml_basic) - abs(r_ml_enhanced)) / abs(r_ml_basic)
    
    if ml_reduction > 50:
        print(f"✓ M/L correlation ELIMINATED ({ml_reduction:.0f}% reduction)")
        print("  → The 'dark matter' signal was actually STRUCTURAL!")
        print("  → Detailed baryonic physics explains the residuals")
        print("  → STRONG support for your hypothesis!")
    elif ml_reduction > 25:
        print(f"○ M/L correlation REDUCED ({ml_reduction:.0f}% reduction)")
        print("  → Structure explains PART of the 'dark matter' signal")
        print("  → But some residual correlation remains")
    else:
        print(f"✗ M/L correlation PERSISTS ({ml_reduction:.0f}% reduction)")
        print("  → Structural details don't explain the M/L effect")
        print("  → May indicate genuine dark matter or other physics (MOND, etc.)")
    
    return clean_data, r2_basic, r2_enhanced, rmse_basic, rmse_enhanced

def visualize_model_comparison(clean_data, r2_basic, r2_enhanced, rmse_basic, rmse_enhanced):
    """
    Visualize the comparison between basic and enhanced models
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. R² comparison
    ax1 = plt.subplot(2, 3, 1)
    models = ['Basic\n(Velocity only)', 'Enhanced\n(+ Structure)']
    r2_values = [r2_basic, r2_enhanced]
    colors = ['steelblue', 'coral']
    
    bars = ax1.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('R² (Variance Explained)', fontsize=12, weight='bold')
    ax1.set_ylim([0.85, 1.0])
    ax1.set_title('Model Performance Comparison', fontsize=14, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # 2. RMSE comparison
    ax2 = plt.subplot(2, 3, 2)
    rmse_values = [rmse_basic, rmse_enhanced]
    
    bars = ax2.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('RMSE (km/s)', fontsize=12, weight='bold')
    ax2.set_title('Prediction Error Comparison', fontsize=14, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # 3. Residual distribution comparison
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.hist(clean_data['residual_basic'], bins=30, alpha=0.5, label='Basic Model', 
             color='steelblue', edgecolor='black')
    ax3.hist(clean_data['residual_enhanced'], bins=30, alpha=0.5, label='Enhanced Model',
             color='coral', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual (km/s)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Residuals', fontsize=14, weight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Residuals vs M/L ratio (Basic model)
    ax4 = plt.subplot(2, 3, 4)
    
    ax4.scatter(clean_data['mass_luminosity_ratio'], abs(clean_data['residual_basic']),
                alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    r_basic, p_basic = stats.spearmanr(clean_data['mass_luminosity_ratio'], 
                                        abs(clean_data['residual_basic']))
    
    ax4.set_xlabel('Mass-to-Light Ratio', fontsize=12)
    ax4.set_ylabel('|Residual| (km/s)', fontsize=12)
    ax4.set_title('Basic Model: "Dark Matter" Signal?', fontsize=14, weight='bold')
    ax4.text(0.05, 0.95, f'r={r_basic:.3f}\np={p_basic:.4f}',
             transform=ax4.transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax4.grid(alpha=0.3)
    
    # 5. Residuals vs M/L ratio (Enhanced model)
    ax5 = plt.subplot(2, 3, 5)
    
    ax5.scatter(clean_data['mass_luminosity_ratio'], abs(clean_data['residual_enhanced']),
                alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
    
    r_enhanced, p_enhanced = stats.spearmanr(clean_data['mass_luminosity_ratio'],
                                              abs(clean_data['residual_enhanced']))
    
    ax5.set_xlabel('Mass-to-Light Ratio', fontsize=12)
    ax5.set_ylabel('|Residual| (km/s)', fontsize=12)
    ax5.set_title('Enhanced Model: Signal Explained?', fontsize=14, weight='bold')
    ax5.text(0.05, 0.95, f'r={r_enhanced:.3f}\np={p_enhanced:.4f}',
             transform=ax5.transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax5.grid(alpha=0.3)
    
    # 6. Summary panel
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    improvement_r2 = 100 * (r2_enhanced - r2_basic) / r2_basic
    improvement_rmse = 100 * (rmse_basic - rmse_enhanced) / rmse_basic
    
    r_ml_basic, _ = stats.spearmanr(clean_data['mass_luminosity_ratio'], 
                                     abs(clean_data['residual_basic']))
    r_ml_enhanced, _ = stats.spearmanr(clean_data['mass_luminosity_ratio'],
                                        abs(clean_data['residual_enhanced']))
    ml_reduction = 100 * (abs(r_ml_basic) - abs(r_ml_enhanced)) / abs(r_ml_basic)
    
    summary_text = f"""
    STRUCTURAL REFINEMENT RESULTS
    {'='*48}
    
    Model Improvements:
      • R² improvement: +{improvement_r2:.2f}%
      • RMSE reduction: -{improvement_rmse:.1f}%
    
    "Dark Matter" Signal Test:
      • Basic model M/L corr: r={r_ml_basic:.3f}
      • Enhanced model M/L corr: r={r_ml_enhanced:.3f}
      • Reduction: {ml_reduction:.1f}%
    
    {'='*48}
    VERDICT:
    
    """
    
    if ml_reduction > 50:
        summary_text += "    ✓ STRUCTURAL EXPLANATION!\n"
        summary_text += "      The 'dark matter signal' was\n"
        summary_text += "      actually missing baryonic\n"
        summary_text += "      structure parameters.\n\n"
        summary_text += "    ✓ Strong support for your\n"
        summary_text += "      hypothesis!"
    elif ml_reduction > 25:
        summary_text += "    ○ PARTIAL STRUCTURAL EFFECT\n"
        summary_text += "      Structure explains some but\n"
        summary_text += "      not all of the signal.\n\n"
        summary_text += "    ○ Mixed evidence"
    else:
        summary_text += "    ✗ PERSISTENT SIGNAL\n"
        summary_text += "      Structural parameters don't\n"
        summary_text += "      eliminate M/L correlation.\n\n"
        summary_text += "    ✗ May indicate genuine DM\n"
        summary_text += "      or alternative physics"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('structural_refinement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Visualization saved: structural_refinement_analysis.png")

def main():
    """Main execution"""
    
    base_dir = 'data'
    mass_models_path = os.path.join(base_dir, 'MassModels_Lelli2016c_header_removed.mrt')
    sparc_path = os.path.join(base_dir, 'spark.csv')
    
    # Load data
    mass_models_df, mass_models_agg, sparc_df = load_and_process_data(
        mass_models_path, sparc_path
    )
    
    merged_data = pd.merge(mass_models_agg, sparc_df, on='ID', how='inner')
    merged_data['mass_luminosity_ratio'] = merged_data['Mbar'] / merged_data['L[3.6]']
    
    # Run refined structural model test
    clean_data, r2_basic, r2_enhanced, rmse_basic, rmse_enhanced = test_refined_structural_model(merged_data)
    
    # Visualize
    visualize_model_comparison(clean_data, r2_basic, r2_enhanced, rmse_basic, rmse_enhanced)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return clean_data

if __name__ == '__main__':
    results = main()