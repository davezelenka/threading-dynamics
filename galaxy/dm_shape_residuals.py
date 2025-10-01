import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_single_rotation_curve(filepath):
    """Load individual galaxy rotation curve"""
    galaxy_name = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    distance = None
    with open(filepath, 'r') as f:
        for line in f:
            if 'Distance' in line:
                distance = float(line.split('=')[1].split('Mpc')[0].strip())
                break
    
    data = pd.read_csv(filepath, sep='\s+', comment='#', 
                       names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
    
    data['Galaxy'] = galaxy_name
    data['Distance'] = distance
    
    return data

def load_all_rotation_curves(data_dir='data'):
    """Load all rotation curve files"""
    files = glob.glob(os.path.join(data_dir, '*_rotmod.dat'))
    
    print(f"Found {len(files)} rotation curve files\n")
    
    all_curves = []
    
    for filepath in files:
        try:
            data = load_single_rotation_curve(filepath)
            all_curves.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if len(all_curves) > 0:
        combined = pd.concat(all_curves, ignore_index=True)
        return combined
    else:
        return None

def calculate_residuals(df):
    """Calculate what dark matter would need to provide"""
    df['V_baryonic'] = np.sqrt(df['Vgas']**2 + df['Vdisk']**2 + df['Vbul']**2)
    
    # The "missing" velocity that dark matter must provide
    # V_total² = V_baryonic² + V_DM²
    # So: V_DM² = V_obs² - V_baryonic²
    
    df['V_DM_squared'] = df['Vobs']**2 - df['V_baryonic']**2
    
    # Set negative values to zero (regions where baryons overshoot)
    df['V_DM_squared'] = df['V_DM_squared'].clip(lower=0)
    df['V_DM'] = np.sqrt(df['V_DM_squared'])
    
    # Also calculate simple residual for reference
    df['V_residual'] = df['Vobs'] - df['V_baryonic']
    df['V_residual_pct'] = 100 * df['V_residual'] / df['Vobs']
    
    # Normalize radius by maximum for each galaxy
    df['Rad_norm'] = df.groupby('Galaxy')['Rad'].transform(lambda x: x / x.max())
    
    return df

def fit_halo_models(radius, v_dm):
    """
    Fit standard dark matter halo profiles
    """
    
    # Remove invalid points
    valid = (radius > 0) & (v_dm > 0) & np.isfinite(v_dm)
    r = radius[valid]
    v = v_dm[valid]
    
    if len(r) < 3:
        return None, None, None
    
    results = {}
    
    # Model 1: NFW halo (Navarro-Frenk-White)
    # V_DM(r) = V_0 * sqrt[ln(1+r/r_s) - (r/r_s)/(1+r/r_s)] / sqrt[ln(2) - 0.5]
    def nfw_velocity(r, v0, rs):
        x = r / rs
        numerator = np.log(1 + x) - x / (1 + x)
        denominator = np.log(2) - 0.5
        return v0 * np.sqrt(numerator / denominator)
    
    try:
        popt_nfw, _ = curve_fit(nfw_velocity, r, v, 
                                p0=[v.max(), r.max()/2], 
                                maxfev=5000,
                                bounds=([0, 0.1], [500, 50]))
        v_pred_nfw = nfw_velocity(r, *popt_nfw)
        r2_nfw = 1 - np.sum((v - v_pred_nfw)**2) / np.sum((v - v.mean())**2)
        results['NFW'] = {'params': popt_nfw, 'r2': r2_nfw}
    except:
        results['NFW'] = None
    
    # Model 2: Isothermal sphere
    # V_DM(r) = V_0 (constant velocity)
    v_iso = v.mean()
    r2_iso = 1 - np.sum((v - v_iso)**2) / np.sum((v - v.mean())**2)
    results['Isothermal'] = {'params': [v_iso], 'r2': r2_iso}
    
    # Model 3: Power law
    # V_DM(r) = A * r^α
    def power_law(r, A, alpha):
        return A * r**alpha
    
    try:
        popt_power, _ = curve_fit(power_law, r, v, 
                                  p0=[v.mean(), 0.5],
                                  maxfev=5000)
        v_pred_power = power_law(r, *popt_power)
        r2_power = 1 - np.sum((v - v_pred_power)**2) / np.sum((v - v.mean())**2)
        results['PowerLaw'] = {'params': popt_power, 'r2': r2_power}
    except:
        results['PowerLaw'] = None
    
    # Model 4: Burkert profile (cored halo)
    # V_DM(r) ∝ [ln(1+r/r_0) + 0.5*ln(1+(r/r_0)²) - arctan(r/r_0)]^0.5
    def burkert_velocity(r, v0, r0):
        x = r / r0
        term = np.log(1 + x) + 0.5 * np.log(1 + x**2) - np.arctan(x)
        return v0 * np.sqrt(term)
    
    try:
        popt_burkert, _ = curve_fit(burkert_velocity, r, v,
                                    p0=[v.max(), r.max()/3],
                                    maxfev=5000,
                                    bounds=([0, 0.1], [500, 50]))
        v_pred_burkert = burkert_velocity(r, *popt_burkert)
        r2_burkert = 1 - np.sum((v - v_pred_burkert)**2) / np.sum((v - v.mean())**2)
        results['Burkert'] = {'params': popt_burkert, 'r2': r2_burkert}
    except:
        results['Burkert'] = None
    
    return results, r, v

def analyze_halo_shape(df, output_dir='residuals'):
    """
    Analyze the shape of dark matter residuals for each galaxy
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DARK MATTER HALO SHAPE ANALYSIS")
    print("=" * 80)
    print("\nTesting if residuals match standard halo profiles:\n")
    
    summary_results = []
    
    for i, galaxy in enumerate(df['Galaxy'].unique()):
        gal_data = df[df['Galaxy'] == galaxy].sort_values('Rad').copy()
        
        if len(gal_data) < 5:
            continue
        
        # Fit halo models
        halo_fits, r_valid, v_valid = fit_halo_models(
            gal_data['Rad'].values, 
            gal_data['V_DM'].values
        )
        
        if halo_fits is None:
            continue
        
        # Determine best fit
        best_model = None
        best_r2 = -999
        
        for model_name, fit_result in halo_fits.items():
            if fit_result is not None and fit_result['r2'] > best_r2:
                best_r2 = fit_result['r2']
                best_model = model_name
        
        # Create figure for this galaxy
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{galaxy} - Dark Matter Profile Analysis', 
                     fontsize=14, weight='bold')
        
        # Plot 1: Full rotation curve
        ax1 = axes[0, 0]
        ax1.errorbar(gal_data['Rad'], gal_data['Vobs'], 
                    yerr=gal_data['errV'], fmt='o', 
                    label='Observed', alpha=0.7, capsize=3)
        ax1.plot(gal_data['Rad'], gal_data['V_baryonic'], 
                'r--', linewidth=2, label='Baryonic')
        ax1.plot(gal_data['Rad'], gal_data['Vgas'], 
                'g:', linewidth=1.5, label='Gas', alpha=0.7)
        ax1.plot(gal_data['Rad'], gal_data['Vdisk'], 
                'b:', linewidth=1.5, label='Disk', alpha=0.7)
        ax1.set_xlabel('Radius (kpc)', fontsize=11)
        ax1.set_ylabel('Velocity (km/s)', fontsize=11)
        ax1.set_title('Rotation Curve Components', fontsize=12, weight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Dark matter velocity profile
        ax2 = axes[0, 1]
        ax2.plot(gal_data['Rad'], gal_data['V_DM'], 
                'ko-', linewidth=2, markersize=6, label='Inferred V_DM')
        
        # Overlay fitted models
        r_plot = np.linspace(gal_data['Rad'].min(), gal_data['Rad'].max(), 100)
        
        if halo_fits['NFW'] is not None:
            v0, rs = halo_fits['NFW']['params']
            x = r_plot / rs
            v_nfw = v0 * np.sqrt((np.log(1+x) - x/(1+x)) / (np.log(2) - 0.5))
            ax2.plot(r_plot, v_nfw, '--', linewidth=2, 
                    label=f"NFW (R²={halo_fits['NFW']['r2']:.3f})", alpha=0.7)
        
        if halo_fits['Isothermal'] is not None:
            v_iso = halo_fits['Isothermal']['params'][0]
            ax2.axhline(v_iso, linestyle=':', linewidth=2,
                       label=f"Isothermal (R²={halo_fits['Isothermal']['r2']:.3f})", 
                       alpha=0.7)
        
        if halo_fits['PowerLaw'] is not None:
            A, alpha = halo_fits['PowerLaw']['params']
            v_power = A * r_plot**alpha
            ax2.plot(r_plot, v_power, '-.', linewidth=2,
                    label=f"Power α={alpha:.2f} (R²={halo_fits['PowerLaw']['r2']:.3f})",
                    alpha=0.7)
        
        ax2.set_xlabel('Radius (kpc)', fontsize=11)
        ax2.set_ylabel('V_DM (km/s)', fontsize=11)
        ax2.set_title('Dark Matter Velocity Profile', fontsize=12, weight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        
        # Plot 3: Residuals vs radius
        ax3 = axes[1, 0]
        ax3.plot(gal_data['Rad'], gal_data['V_residual'], 
                'o-', linewidth=2, markersize=6, color='purple')
        ax3.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Radius (kpc)', fontsize=11)
        ax3.set_ylabel('V_obs - V_baryonic (km/s)', fontsize=11)
        ax3.set_title('Velocity Residuals', fontsize=12, weight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Normalized profiles
        ax4 = axes[1, 1]
        
        # Normalize both radius and velocity for comparison
        if gal_data['V_DM'].max() > 0:
            v_dm_norm = gal_data['V_DM'] / gal_data['V_DM'].max()
            ax4.plot(gal_data['Rad_norm'], v_dm_norm, 
                    'o-', linewidth=2, markersize=6, label='V_DM (normalized)')
        
        ax4.set_xlabel('Normalized Radius (r/r_max)', fontsize=11)
        ax4.set_ylabel('Normalized V_DM', fontsize=11)
        ax4.set_title('Normalized DM Profile', fontsize=12, weight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # Add text summary
        textstr = f"Best fit: {best_model} (R²={best_r2:.3f})\n"
        textstr += f"Max V_DM: {gal_data['V_DM'].max():.1f} km/s\n"
        textstr += f"V_DM at r_max: {gal_data['V_DM'].iloc[-1]:.1f} km/s"
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{galaxy}_halo_profile.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store results
        summary_results.append({
            'Galaxy': galaxy,
            'N_points': len(gal_data),
            'Max_radius': gal_data['Rad'].max(),
            'Max_V_DM': gal_data['V_DM'].max(),
            'V_DM_final': gal_data['V_DM'].iloc[-1],
            'Best_model': best_model,
            'Best_R2': best_r2,
            'NFW_R2': halo_fits['NFW']['r2'] if halo_fits['NFW'] else np.nan,
            'Isothermal_R2': halo_fits['Isothermal']['r2'] if halo_fits['Isothermal'] else np.nan,
            'PowerLaw_R2': halo_fits['PowerLaw']['r2'] if halo_fits['PowerLaw'] else np.nan,
            'Burkert_R2': halo_fits['Burkert']['r2'] if halo_fits['Burkert'] else np.nan,
            'PowerLaw_alpha': halo_fits['PowerLaw']['params'][1] if halo_fits['PowerLaw'] else np.nan
        })
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1} galaxies...")
    
    summary_df = pd.DataFrame(summary_results)
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("HALO PROFILE STATISTICS")
    print("=" * 80)
    
    print(f"\nAnalyzed {len(summary_df)} galaxies")
    
    print("\nBest-fit model distribution:")
    print(summary_df['Best_model'].value_counts())
    
    print("\nAverage R² by model:")
    for model in ['NFW', 'Isothermal', 'PowerLaw', 'Burkert']:
        col = f'{model}_R2'
        if col in summary_df.columns:
            mean_r2 = summary_df[col].mean()
            print(f"  {model:<15} {mean_r2:.4f}")
    
    # Power law exponent analysis
    power_alpha = summary_df['PowerLaw_alpha'].dropna()
    if len(power_alpha) > 0:
        print(f"\nPower law exponent (α) statistics:")
        print(f"  Mean: {power_alpha.mean():.3f}")
        print(f"  Median: {power_alpha.median():.3f}")
        print(f"  Std: {power_alpha.std():.3f}")
        
        # Test if α is consistent with flat rotation (α ≈ 0)
        t_stat, p_value = stats.ttest_1samp(power_alpha, 0)
        print(f"\n  t-test vs α=0: t={t_stat:.3f}, p={p_value:.6f}")
        
        if p_value > 0.05:
            print("  → Consistent with FLAT dark matter profile (α ≈ 0)")
        elif power_alpha.mean() > 0:
            print(f"  → INCREASING profile (α > 0)")
        else:
            print(f"  → DECREASING profile (α < 0)")
    
    # Test universality
    print("\n" + "=" * 80)
    print("UNIVERSALITY TEST")
    print("=" * 80)
    
    # Do all galaxies follow the same profile shape?
    model_consistency = summary_df['Best_model'].value_counts(normalize=True).iloc[0]
    print(f"\nFraction with most common profile: {model_consistency:.1%}")
    
    if model_consistency > 0.7:
        print("  → UNIVERSAL profile shape (>70% agreement)")
    elif model_consistency > 0.5:
        print("  → Moderate consistency (50-70% agreement)")
    else:
        print("  → NO universal shape (highly diverse)")
    
    # Export results
    summary_df.to_csv(os.path.join(output_dir, 'halo_profile_summary.csv'), index=False)
    
    print(f"\n✓ Saved {len(summary_df)} individual galaxy plots to {output_dir}/")
    print(f"✓ Saved summary statistics to {output_dir}/halo_profile_summary.csv")
    
    return summary_df

def create_composite_plots(summary_df, output_dir='residuals'):
    """Create summary composite plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dark Matter Halo Profile Summary', fontsize=14, weight='bold')
    
    # Plot 1: Distribution of best-fit models
    ax1 = axes[0, 0]
    model_counts = summary_df['Best_model'].value_counts()
    ax1.bar(range(len(model_counts)), model_counts.values, 
           tick_label=model_counts.index)
    ax1.set_ylabel('Number of Galaxies', fontsize=11)
    ax1.set_title('Best-Fit Halo Model Distribution', fontsize=12, weight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Plot 2: R² comparison
    ax2 = axes[0, 1]
    r2_data = []
    r2_labels = []
    for model in ['NFW', 'Isothermal', 'PowerLaw', 'Burkert']:
        col = f'{model}_R2'
        if col in summary_df.columns:
            vals = summary_df[col].dropna()
            if len(vals) > 0:
                r2_data.append(vals)
                r2_labels.append(model)
    
    ax2.boxplot(r2_data, labels=r2_labels)
    ax2.set_ylabel('R² (goodness of fit)', fontsize=11)
    ax2.set_title('Model Performance Comparison', fontsize=12, weight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='R²=0.9')
    ax2.legend()
    
    # Plot 3: Power law exponent distribution
    ax3 = axes[1, 0]
    power_alpha = summary_df['PowerLaw_alpha'].dropna()
    ax3.hist(power_alpha, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='α=0 (flat)')
    ax3.axvline(power_alpha.mean(), color='blue', linestyle='-', 
               linewidth=2, label=f'Mean α={power_alpha.mean():.2f}')
    ax3.set_xlabel('Power Law Exponent (α)', fontsize=11)
    ax3.set_ylabel('Number of Galaxies', fontsize=11)
    ax3.set_title('Dark Matter Profile Slope', fontsize=12, weight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Max V_DM vs galaxy size
    ax4 = axes[1, 1]
    ax4.scatter(summary_df['Max_radius'], summary_df['Max_V_DM'], 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Maximum Radius (kpc)', fontsize=11)
    ax4.set_ylabel('Maximum V_DM (km/s)', fontsize=11)
    ax4.set_title('DM Amplitude vs Galaxy Size', fontsize=12, weight='bold')
    ax4.grid(alpha=0.3)
    
    # Add correlation
    r, p = stats.spearmanr(summary_df['Max_radius'], summary_df['Max_V_DM'])
    ax4.text(0.05, 0.95, f'r={r:.3f}\np={p:.4f}',
            transform=ax4.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'halo_profile_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved composite summary plot")

def main():
    """Main execution"""
    
    print("=" * 80)
    print("DARK MATTER HALO SHAPE ANALYSIS")
    print("=" * 80)
    print("\nIf dark matter halos cause rotation curves,")
    print("residuals should match standard halo profiles (NFW, isothermal, etc.)\n")
    
    # Load data
    df = load_all_rotation_curves('data')
    
    if df is None:
        print("\nERROR: No rotation curve files found")
        return
    
    # Calculate residuals
    df = calculate_residuals(df)
    
    # Analyze halo shapes
    summary_df = analyze_halo_shape(df)
    
    # Create composite plots
    create_composite_plots(summary_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df, summary_df

if __name__ == '__main__':
    df, summary = main()