import pandas as pd
import numpy as np
from scipy import stats

def load_and_prepare_data(filepath):
    """
    Load and preprocess the fire stand data with enhanced memory density calculations
    """
    df = pd.read_csv(filepath)
    
    # Memory density calculations with error handling
    df['pre_coniferous_memory'] = df['bs.biomass.pre'].fillna(0)
    df['pre_deciduous_memory'] = df['decid.bio.pre'].fillna(0)
    df['pre_total_memory'] = df['pre_coniferous_memory'] + df['pre_deciduous_memory']
    
    df['post_coniferous_memory'] = df['bs.bio.post'].fillna(0)
    df['post_deciduous_memory'] = df['decid.bio.post'].fillna(0)
    df['post_total_memory'] = df['post_coniferous_memory'] + df['post_deciduous_memory']
    
    # Memory gradient metrics
    df['memory_gradient_di'] = df['DI.post'] - df['DI.pre']
    df['memory_gradient_rate'] = (df['memory_gradient_di'] / (df['DI.pre'] + 1e-10)) * 100
    
    return df

def test_hypothesis_2_memory_gradient(df):
    """
    Enhanced H2: Memory Gradient Triggers Disturbance
    """
    print("\n===== HYPOTHESIS 2: MEMORY GRADIENT TRIGGERS DISTURBANCE =====")
    
    # Composition change as binary and ordinal
    df['composition_binary_change'] = (df['class.pre'] != df['class.post']).astype(int)
    
    # Ordinal mapping of composition change
    composition_order = {'Spruce': 0, 'Mixed': 1, 'Decid': 2}
    df['composition_ordinal_change'] = np.abs(
        df['class.pre'].map(composition_order) - 
        df['class.post'].map(composition_order)
    )
    
    # Non-parametric correlations
    binary_correlation, binary_p = stats.spearmanr(
        df['composition_binary_change'], 
        df['memory_gradient_di']
    )
    
    ordinal_correlation, ordinal_p = stats.spearmanr(
        df['composition_ordinal_change'], 
        df['memory_gradient_di']
    )
    
    print("Memory Gradient Correlation Analysis:")
    print(f"  Binary Change Correlation: {binary_correlation:.4f} (p = {binary_p:.4f})")
    print(f"  Ordinal Change Correlation: {ordinal_correlation:.4f} (p = {ordinal_p:.4f})")
    
    gradient_stats = {
        'mean_di_gradient': df['memory_gradient_di'].mean(),
        'median_di_gradient': df['memory_gradient_di'].median(),
        'std_di_gradient': df['memory_gradient_di'].std(),
        'gradient_rate_mean': df['memory_gradient_rate'].mean(),
        'positive_gradient_rate': (df['memory_gradient_di'] > 0).mean() * 100
    }
    
    print("\nMemory Gradient Statistics:")
    for key, value in gradient_stats.items():
        print(f"  {key}: {value:.4f}")
    
    return gradient_stats

def test_hypothesis_4_post_disturbance_enhancement(df):
    """
    Enhanced H4: Post-Disturbance Enhancement
    """
    print("\n===== HYPOTHESIS 4: POST-DISTURBANCE ENHANCEMENT =====")
    
    # Safe enhancement calculation
    def safe_enhancement(pre, post):
        return np.where(
            pre > 0, 
            (post - pre) / pre * 100, 
            np.nan
        )
    
    enhancements = {
        'total_memory': safe_enhancement(df['pre_total_memory'], df['post_total_memory']),
        'coniferous_memory': safe_enhancement(df['pre_coniferous_memory'], df['post_coniferous_memory']),
        'deciduous_memory': safe_enhancement(df['pre_deciduous_memory'], df['post_deciduous_memory'])
    }
    
    print("Memory Enhancement Analysis:")
    for name, enhancement in enhancements.items():
        valid_enh = enhancement[~np.isnan(enhancement)]
        print(f"\n{name.capitalize()} Memory Enhancement:")
        print(f"  Mean: {valid_enh.mean():.4f}%")
        print(f"  Median: {np.median(valid_enh):.4f}%")
        print(f"  Positive Rate: {(valid_enh > 0).mean() * 100:.2f}%")
        
        t_test = stats.ttest_1samp(valid_enh, 0)
        print(f"  t-statistic: {t_test.statistic:.4f}")
        print(f"  p-value: {t_test.pvalue:.4f}")
    
    return enhancements

def main(filepath):
    print("===== THREADING ECOLOGY ANALYSIS =====")
    print(f"Dataset: {filepath}")
    
    df = load_and_prepare_data(filepath)
    
    test_hypothesis_2_memory_gradient(df)
    test_hypothesis_4_post_disturbance_enhancement(df)

if __name__ == "__main__":
    filepath = '804_PrePostFireStandData_DiVA_XJW.csv'
    main(filepath)