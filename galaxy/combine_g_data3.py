import os
import glob
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.simplefilter(action='ignore', category=FutureWarning)

# MOND acceleration scale (m/s²)
A0_MOND = 1.2e-10  # m/s²
# Convert to convenient units: km²/s²/kpc
# 1 kpc = 3.086e19 m
# a₀ = 1.2e-10 m/s² = 1.2e-10 * (1000)² / 3.086e19 km²/s²/kpc
A0_CONVENIENT = 3.7  # km²/s²/kpc

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

def calculate_basic_quantities(df):
    """Calculate baryonic prediction and residuals"""
    df['V_baryonic'] = np.sqrt(df['Vgas']**2 + df['Vdisk']**2 + df['Vbul']**2)
    df['V_residual'] = df['Vobs'] - df['V_baryonic']
    df['V_residual_pct'] = 100 * df['V_residual'] / df['Vobs']
    
    # Calculate centripetal acceleration
    df['accel'] = df['Vobs']**2 / df['Rad']  # km²/s²/kpc
    df['accel_ratio'] = df['accel'] / A0_CONVENIENT  # Normalized by a₀
    
    return df

def test_1_acceleration_scale(df, output_dir='output'):
    """
    TEST 1: Acceleration Scale Analysis
    Does residual behavior change at a/a₀ ≈ 1?
    """
    print("=" * 80)
    print("TEST 1: ACCELERATION SCALE ANALYSIS (MOND Test)")
    print("=" * 80)
    
    # Filter valid data
    clean = df[(df['accel_ratio'] > 0) & (df['V_residual_pct'].notna())].copy()
    
    # Bin by acceleration ratio
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
    clean['accel_bin'] = pd.cut(clean['accel_ratio'], bins=bins)
    
    # Statistics by bin
    accel_stats = clean.groupby('accel_bin', observed=True).agg({
        'V_residual_pct': ['mean', 'std', 'count'],
        'Vobs': 'mean',
        'Rad': 'mean'
    }).reset_index()
    
    print("\nResiduals by Acceleration Ratio (a/a₀):")
    print("-" * 80)
    print(f"{'a/a₀ Range':<20} {'Mean Resid %':<15} {'Std':<10} {'N':<8} {'<Vobs>':<10} {'<Rad>'}")
    print("-" * 80)
    
    for _, row in accel_stats.iterrows():
        print(f"{str(row['accel_bin']):<20} "
              f"{row[('V_residual_pct', 'mean')]:>13.2f}  "
              f"{row[('V_residual_pct', 'std')]:>8.2f}  "
              f"{int(row[('V_residual_pct', 'count')]):>6}  "
              f"{row[('Vobs', 'mean')]:>8.1f}  "
              f"{row[('Rad', 'mean')]:>6.2f}")
    
    # Test for transition at a/a₀ ≈ 1
    low_accel = clean[clean['accel_ratio'] < 1]['V_residual_pct'].dropna()
    high_accel = clean[clean['accel_ratio'] > 1]['V_residual_pct'].dropna()
    
    if len(low_accel) > 10 and len(high_accel) > 10:
        t_stat, p_value = stats.ttest_ind(low_accel, high_accel)
        print(f"\nLow acceleration (a<a₀) vs High acceleration (a>a₀):")
        print(f"  Low:  {low_accel.mean():>7.2f}% (n={len(low_accel)})")
        print(f"  High: {high_accel.mean():>7.2f}% (n={len(high_accel)})")
        print(f"  Difference: {high_accel.mean() - low_accel.mean():>7.2f}%")
        print(f"  t-test p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            if low_accel.mean() > high_accel.mean():
                print("\n  ★ MOND SIGNATURE: Low-acceleration regions show HIGHER residuals")
            else:
                print("\n  ○ Opposite pattern from MOND expectation")
        else:
            print("\n  ○ No significant difference across acceleration scale")
    
    # Export raw data
    os.makedirs(output_dir, exist_ok=True)
    accel_stats.to_csv(os.path.join(output_dir, 'test1_acceleration_bins.csv'), index=False)
    clean[['Galaxy', 'Rad', 'Vobs', 'accel', 'accel_ratio', 'V_residual_pct']].to_csv(
        os.path.join(output_dir, 'test1_acceleration_raw.csv'), index=False
    )
    
    print(f"\n✓ Exported: test1_acceleration_bins.csv, test1_acceleration_raw.csv")
    
    return clean

def test_2_component_interaction(df, output_dir='output'):
    """
    TEST 2: Component Interaction Analysis
    Do velocity components interact non-linearly?
    """
    print("\n" + "=" * 80)
    print("TEST 2: COMPONENT INTERACTION ANALYSIS")
    print("=" * 80)
    
    # Clean data
    clean = df[['Vobs', 'Vgas', 'Vdisk', 'Vbul']].dropna()
    
    # Model 1: Standard superposition (no interaction)
    # Vobs² = Vgas² + Vdisk² + Vbul²
    clean['V_standard'] = np.sqrt(clean['Vgas']**2 + clean['Vdisk']**2 + clean['Vbul']**2)
    r2_standard = np.corrcoef(clean['Vobs'], clean['V_standard'])[0,1]**2
    
    print(f"\nModel 1 - Standard Superposition: Vobs² = Vgas² + Vdisk² + Vbul²")
    print(f"  R² = {r2_standard:.6f}")
    
    # Model 2: Linear regression (allows for scaling)
    # Vobs² = α·Vgas² + β·Vdisk² + γ·Vbul²
    X_squared = np.column_stack([clean['Vgas']**2, clean['Vdisk']**2, clean['Vbul']**2])
    y_squared = clean['Vobs']**2
    
    reg_squared = LinearRegression(fit_intercept=False)
    reg_squared.fit(X_squared, y_squared)
    
    alpha, beta, gamma = reg_squared.coef_
    r2_scaled = reg_squared.score(X_squared, y_squared)
    
    print(f"\nModel 2 - Scaled Components: Vobs² = α·Vgas² + β·Vdisk² + γ·Vbul²")
    print(f"  α (gas):   {alpha:.4f}")
    print(f"  β (disk):  {beta:.4f}")
    print(f"  γ (bulge): {gamma:.4f}")
    print(f"  R² = {r2_scaled:.6f}")
    print(f"  Improvement: ΔR² = {r2_scaled - r2_standard:.6f}")
    
    # Model 3: Include interaction terms
    # Vobs² = α·Vgas² + β·Vdisk² + γ·Vbul² + δ·Vgas·Vdisk + ε·Vgas·Vbul + ζ·Vdisk·Vbul
    X_interact = np.column_stack([
        clean['Vgas']**2, 
        clean['Vdisk']**2, 
        clean['Vbul']**2,
        clean['Vgas'] * clean['Vdisk'],
        clean['Vgas'] * clean['Vbul'],
        clean['Vdisk'] * clean['Vbul']
    ])
    
    reg_interact = LinearRegression(fit_intercept=False)
    reg_interact.fit(X_interact, y_squared)
    
    r2_interact = reg_interact.score(X_interact, y_squared)
    
    print(f"\nModel 3 - With Interactions:")
    print(f"  Vobs² = α·Vgas² + β·Vdisk² + γ·Vbul² + interaction terms")
    print(f"  Coefficients:")
    param_names = ['α (Vgas²)', 'β (Vdisk²)', 'γ (Vbul²)', 
                   'δ (Vgas·Vdisk)', 'ε (Vgas·Vbul)', 'ζ (Vdisk·Vbul)']
    for name, coef in zip(param_names, reg_interact.coef_):
        print(f"    {name:<20} {coef:>10.6f}")
    print(f"  R² = {r2_interact:.6f}")
    print(f"  Improvement over scaled: ΔR² = {r2_interact - r2_scaled:.6f}")
    
    # F-test for interaction terms
    n = len(y_squared)
    k_scaled = 3
    k_interact = 6
    
    ss_res_scaled = np.sum((y_squared - reg_squared.predict(X_squared))**2)
    ss_res_interact = np.sum((y_squared - reg_interact.predict(X_interact))**2)
    
    f_stat = ((ss_res_scaled - ss_res_interact) / (k_interact - k_scaled)) / (ss_res_interact / (n - k_interact))
    p_value = 1 - stats.f.cdf(f_stat, k_interact - k_scaled, n - k_interact)
    
    print(f"\n  F-test for interaction terms:")
    print(f"    F = {f_stat:.3f}, p = {p_value:.6f}")
    
    if p_value < 0.001:
        print("    ★ HIGHLY SIGNIFICANT interaction effects!")
        print("    → Components do NOT simply add in quadrature")
        print("    → Suggests non-linear coupling or gradient flow effects")
    elif p_value < 0.05:
        print("    ○ Marginally significant interactions")
    else:
        print("    ○ No significant interaction - linear superposition holds")
    
    # Export results
    results = pd.DataFrame({
        'Model': ['Standard', 'Scaled', 'With Interactions'],
        'R²': [r2_standard, r2_scaled, r2_interact],
        'N_params': [0, 3, 6]
    })
    results.to_csv(os.path.join(output_dir, 'test2_component_models.csv'), index=False)
    
    coef_df = pd.DataFrame({
        'Parameter': param_names,
        'Coefficient': reg_interact.coef_
    })
    coef_df.to_csv(os.path.join(output_dir, 'test2_interaction_coefficients.csv'), index=False)
    
    print(f"\n✓ Exported: test2_component_models.csv, test2_interaction_coefficients.csv")
    
    return clean

def test_3_density_gradient(df, output_dir='output'):
    """
    TEST 3: Density Gradient Analysis
    Does the steepness of surface brightness gradient correlate with residuals?
    """
    print("\n" + "=" * 80)
    print("TEST 3: DENSITY GRADIENT ANALYSIS")
    print("=" * 80)
    
    results_by_galaxy = []
    
    for galaxy in df['Galaxy'].unique():
        gal_data = df[df['Galaxy'] == galaxy].sort_values('Rad').copy()
        
        if len(gal_data) < 3:
            continue
        
        # Calculate gradient dSB/dr using finite differences
        gal_data['dSB_dr'] = np.gradient(gal_data['SBdisk'], gal_data['Rad'])
        gal_data['abs_dSB_dr'] = np.abs(gal_data['dSB_dr'])
        
        # Correlation between gradient strength and residuals
        valid = gal_data[['abs_dSB_dr', 'V_residual_pct']].dropna()
        
        if len(valid) > 3:
            r, p = stats.spearmanr(valid['abs_dSB_dr'], valid['V_residual_pct'])
            
            results_by_galaxy.append({
                'Galaxy': galaxy,
                'N_points': len(valid),
                'correlation': r,
                'p_value': p,
                'mean_gradient': valid['abs_dSB_dr'].mean(),
                'mean_residual': valid['V_residual_pct'].mean()
            })
    
    results_df = pd.DataFrame(results_by_galaxy)
    
    print(f"\nAnalyzed {len(results_df)} galaxies")
    print(f"\nGalaxies with significant gradient-residual correlation (p<0.05):")
    
    significant = results_df[results_df['p_value'] < 0.05].sort_values('correlation', ascending=False)
    
    if len(significant) > 0:
        print(f"  {len(significant)} galaxies show significant correlation")
        print("\nTop 10 by correlation strength:")
        print("-" * 80)
        print(f"{'Galaxy':<15} {'Corr':<8} {'p-value':<12} {'Mean Grad':<12} {'Mean Resid %'}")
        print("-" * 80)
        
        for _, row in significant.head(10).iterrows():
            print(f"{row['Galaxy']:<15} {row['correlation']:>6.3f}  {row['p_value']:>10.6f}  "
                  f"{row['mean_gradient']:>10.2f}  {row['mean_residual']:>10.2f}")
    else:
        print("  No galaxies show significant correlation")
    
    # Overall pattern
    print(f"\nOverall distribution of correlations:")
    print(f"  Median correlation: {results_df['correlation'].median():.3f}")
    print(f"  Mean correlation: {results_df['correlation'].mean():.3f}")
    print(f"  Positive correlations: {(results_df['correlation'] > 0).sum()} / {len(results_df)}")
    
    # Meta-analysis: Are correlations systematically non-zero?
    # One-sample t-test: is mean correlation significantly different from 0?
    t_stat, p_meta = stats.ttest_1samp(results_df['correlation'], 0)
    
    print(f"\nMeta-analysis (one-sample t-test):")
    print(f"  t = {t_stat:.3f}, p = {p_meta:.6f}")
    
    if p_meta < 0.05:
        if results_df['correlation'].mean() > 0:
            print("  ★ Systematic POSITIVE correlation across galaxies")
            print("  → Steeper gradients → larger residuals")
        else:
            print("  ★ Systematic NEGATIVE correlation across galaxies")
            print("  → Steeper gradients → smaller residuals")
    else:
        print("  ○ No systematic gradient effect")
    
    # Export
    results_df.to_csv(os.path.join(output_dir, 'test3_gradient_correlations.csv'), index=False)
    
    print(f"\n✓ Exported: test3_gradient_correlations.csv")
    
    return results_df

def test_4_radial_pattern_classification(df, output_dir='output'):
    """
    TEST 4: Radial Pattern Classification
    Characterize the shape of residual profiles for each galaxy
    """
    print("\n" + "=" * 80)
    print("TEST 4: RADIAL PATTERN CLASSIFICATION")
    print("=" * 80)
    
    classifications = []
    
    for galaxy in df['Galaxy'].unique():
        gal_data = df[df['Galaxy'] == galaxy].sort_values('Rad').copy()
        
        if len(gal_data) < 5:
            continue
        
        # Normalize radius
        gal_data['Rad_norm'] = gal_data['Rad'] / gal_data['Rad'].max()
        
        # Split into inner/outer regions
        inner = gal_data[gal_data['Rad_norm'] < 0.4]['V_residual_pct']
        outer = gal_data[gal_data['Rad_norm'] > 0.6]['V_residual_pct']
        
        if len(inner) < 2 or len(outer) < 2:
            continue
        
        inner_mean = inner.mean()
        outer_mean = outer.mean()
        
        # Fit linear trend to residuals vs radius
        valid = gal_data[['Rad_norm', 'V_residual_pct']].dropna()
        
        if len(valid) > 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid['Rad_norm'], valid['V_residual_pct']
            )
            
            # Classify pattern
            if abs(slope) < 5:
                pattern = 'Flat'
            elif slope > 5:
                pattern = 'Increasing' if p_value < 0.1 else 'Flat'
            else:
                pattern = 'Decreasing' if p_value < 0.1 else 'Flat'
            
            # Further classification
            if inner_mean > 5 and outer_mean < -5:
                subpattern = 'Inner_excess_outer_deficit'
            elif inner_mean < -5 and outer_mean > 5:
                subpattern = 'Inner_deficit_outer_excess'
            elif abs(inner_mean) < 5 and abs(outer_mean) < 5:
                subpattern = 'Well_matched'
            elif inner_mean > 5 and outer_mean > 5:
                subpattern = 'Uniformly_high'
            elif inner_mean < -5 and outer_mean < -5:
                subpattern = 'Uniformly_low'
            else:
                subpattern = 'Mixed'
            
            classifications.append({
                'Galaxy': galaxy,
                'N_points': len(gal_data),
                'Max_radius': gal_data['Rad'].max(),
                'Inner_mean': inner_mean,
                'Outer_mean': outer_mean,
                'Difference': outer_mean - inner_mean,
                'Slope': slope,
                'Slope_pvalue': p_value,
                'Pattern': pattern,
                'Subpattern': subpattern,
                'Overall_mean': gal_data['V_residual_pct'].mean(),
                'Overall_std': gal_data['V_residual_pct'].std()
            })
    
    class_df = pd.DataFrame(classifications)
    
    print(f"\nClassified {len(class_df)} galaxies")
    print(f"\nPattern distribution:")
    print(class_df['Pattern'].value_counts())
    
    print(f"\nSubpattern distribution:")
    print(class_df['Subpattern'].value_counts())
    
    print(f"\nMean inner vs outer residuals:")
    print(f"  Inner: {class_df['Inner_mean'].mean():>7.2f}%")
    print(f"  Outer: {class_df['Outer_mean'].mean():>7.2f}%")
    print(f"  Systematic difference: {class_df['Difference'].mean():>7.2f}%")
    
    # Test if difference is systematic
    t_stat, p_value = stats.ttest_1samp(class_df['Difference'], 0)
    print(f"\n  t-test (difference vs 0): t={t_stat:.3f}, p={p_value:.6f}")
    
    if p_value < 0.05:
        if class_df['Difference'].mean() > 0:
            print("  ★ Outer regions systematically show HIGHER residuals")
            print("  → Classic dark matter pattern (increasing with radius)")
        else:
            print("  ★ Inner regions systematically show HIGHER residuals")
            print("  → Unexpected pattern (baryons underestimate in centers)")
    else:
        print("  ○ No systematic radial trend across galaxies")
    
    # Export
    class_df.to_csv(os.path.join(output_dir, 'test4_radial_patterns.csv'), index=False)
    
    print(f"\n✓ Exported: test4_radial_patterns.csv")
    
    return class_df

def main():
    """Main execution"""
    
    print("=" * 80)
    print("ROTATION CURVE ANALYSIS - SIGNATURE TESTS")
    print("=" * 80)
    
    # Load rotation curves
    df = load_all_rotation_curves('data')
    
    if df is None:
        print("\nERROR: No rotation curve files found in 'data/' directory")
        return
    
    print(f"Loaded {df['Galaxy'].nunique()} galaxies with {len(df)} total data points\n")
    
    # Calculate basic quantities
    df = calculate_basic_quantities(df)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Run all tests
    test_1_acceleration_scale(df)
    test_2_component_interaction(df)
    test_3_density_gradient(df)
    test_4_radial_pattern_classification(df)
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nRaw data exported to output/ directory:")
    print("  - test1_acceleration_*.csv (MOND test)")
    print("  - test2_component_*.csv (interaction effects)")
    print("  - test3_gradient_*.csv (density gradients)")
    print("  - test4_radial_*.csv (pattern classification)")
    print("=" * 80)
    
    return df

if __name__ == '__main__':
    df = main()