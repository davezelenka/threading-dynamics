# Fixed and re-run the paired-analysis script for the pika dataset.
# This version corrects the previous syntax error and provides clearer printed output.
import pandas as pd
import numpy as np
from scipy import stats

def load_data(path='data.csv'):
    df = pd.read_csv(path)
    return df

def find_pairs(df):
    # Extract prefix letters and numeric part to form pair keys (e.g., A1 -> prefix=A, number=1)
    df = df.copy()
    df['prefix'] = df['plot_id'].str.extract(r'^([A-Za-z]+)')[0]
    df['number'] = df['plot_id'].str.extract(r'([0-9]+)')[0]
    df['pair_key'] = df['prefix'].fillna('') + df['number'].fillna('')
    counts = df['pair_key'].value_counts()
    paired_keys = counts[counts==2].index.tolist()
    paired_df = df[df['pair_key'].isin(paired_keys)].copy()
    # Pivot so we have With and Without columns per variable
    try:
        paired_df = paired_df.pivot(index='pair_key', columns='Pika_type')
        paired_df.columns = ['_'.join(map(str, col)).strip() for col in paired_df.columns.values]
        paired_df = paired_df.reset_index()
    except Exception as e:
        print("Pivot failed:", e)
    return paired_df, df, paired_keys

def compute_paired_differences(paired_df):
    vars_of_interest = [
        'AGB', 'SOC', 'N', 'P', 'NO3', 'NH4', 'AP', 'SM', 
        'Biodiversity', 'Mean_EMF', 'Effective_EMF'
    ]
    summary = {}
    for var in vars_of_interest:
        col_with = f'{var}_With'
        col_without = f'{var}_Without'
        if col_with in paired_df.columns and col_without in paired_df.columns:
            diff = paired_df[col_with] - paired_df[col_without]
            paired_df[f'diff_{var}'] = diff
            # summary stats
            mean_diff = diff.mean()
            median_diff = diff.median()
            std_diff = diff.std()
            count = diff.count()
            # paired t-test (With vs Without)
            try:
                tstat, pval = stats.ttest_rel(paired_df[col_with], paired_df[col_without], nan_policy='omit')
            except Exception as e:
                tstat, pval = np.nan, np.nan
            summary[var] = {
                'mean_diff': mean_diff,
                'median_diff': median_diff,
                'std_diff': std_diff,
                'count': int(count),
                'paired_t_stat': float(tstat) if not np.isnan(tstat) else np.nan,
                'paired_p_value': float(pval) if not np.isnan(pval) else np.nan
            }
    return paired_df, summary

def test_memory_gradient_prediction(paired_df, memory_proxy='SOC'):
    res = {}
    mem_col = f'{memory_proxy}_Without'
    response_vars = [c for c in paired_df.columns if c.startswith('diff_')]
    for resp in response_vars:
        resp_series = paired_df[resp]
        mem_series = paired_df.get(mem_col, pd.Series(np.nan, index=paired_df.index))
        mask = (~resp_series.isna()) & (~mem_series.isna())
        if mask.sum() < 3:
            continue
        corr, p = stats.spearmanr(mem_series[mask], resp_series[mask].abs())
        res[resp] = {'spearman_r': float(corr), 'p_value': float(p), 'n': int(mask.sum())}
    return res

def compare_multifunctionality_vs_biodiversity(paired_df):
    if 'diff_Mean_EMF' in paired_df.columns and 'diff_Biodiversity' in paired_df.columns:
        mask = (~paired_df['diff_Mean_EMF'].isna()) & (~paired_df['diff_Biodiversity'].isna())
        if mask.sum() >= 3:
            corr, p = stats.spearmanr(paired_df.loc[mask, 'diff_Biodiversity'], paired_df.loc[mask, 'diff_Mean_EMF'])
            return {'spearman_r': float(corr), 'p_value': float(p), 'n': int(mask.sum())}
    return None

def main(path='data.csv'):
    df = load_data(path)
    paired_df, full_df, paired_keys = find_pairs(df)
    print(f'Found {len(paired_keys)} paired sampling units.')
    paired_df, summary = compute_paired_differences(paired_df)
    print('\nPaired differences summary (selected variables):')
    for var, statsum in summary.items():
        print(f"{var}: mean_diff={statsum['mean_diff']:.4f}, t={statsum['paired_t_stat']:.3f}, p={statsum['paired_p_value']:.4f}, n={statsum['count']}")
    mem_pred = test_memory_gradient_prediction(paired_df, memory_proxy='SOC')
    print('\nMemory gradient (SOC) vs abs(change) correlations:')
    if mem_pred:
        for k,v in mem_pred.items():
            print(f"{k}: r={v['spearman_r']:.3f}, p={v['p_value']:.4f}, n={v['n']}")
    else:
        print("Not enough data to compute correlations.")
    emf_vs_bio = compare_multifunctionality_vs_biodiversity(paired_df)
    print('\nMean_EMF vs Biodiversity change correlation:')
    print(emf_vs_bio)
    paired_df.to_csv('output/paired_differences.csv', index=False)
    print('\nPaired differences saved to paired_differences.csv')
    return paired_df, summary, mem_pred, emf_vs_bio

# Run main if executed as script
if __name__ == '__main__':
    try:
        paired_df, summary, mem_pred, emf_vs_bio = main('data.csv')
    except FileNotFoundError:
        print("data.csv not found in working directory. Please place the dataset file in the current directory or specify the correct path.")