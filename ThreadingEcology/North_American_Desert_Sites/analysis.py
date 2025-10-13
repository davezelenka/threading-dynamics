#!/usr/bin/env python3
"""
analysis.py

Analysis for Hosna & Faist seedbank dataset (use seedbank_fctnlgrpsummary.csv
and seedbank_attribute_spreadsheet.csv). Produces summaries, tests, and simple plots.

Outputs (saved to working directory):
 - seedbank_merged.csv            : merged raw table
 - per_sample_fg_pivot.csv       : functional group counts per sample (pivot)
 - per_sample_summary.csv        : richness, total_counts, TSF, burn, microsite, desert, etc.
 - burned_vs_control_tests.json  : major test results (Wilcoxon/Mann-Whitney, regressions)
 - figures: richness_by_burn.png, richness_by_microsite.png, tsf_vs_richness.png

Requires:
    pandas, numpy, scipy, statsmodels, matplotlib, seaborn

Run:
    python analysis.py
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

# --------- USER CONFIG ----------
FG_FILE = "seedbank_fctnlgrpsummary.csv"
ATTR_FILE = "seedbank_attribute_spreadsheet.csv"
OUT_PREFIX = "output/seedbank_analysis"
# --------------------------------

def safe_read_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    # try multiple encodings if needed
    return pd.read_csv(path)

def merge_tables(fg_df, attr_df):
    # Columns expected in both: sample, tray_num, block_num, desert, plot_num maybe
    common_keys = ['sample', 'tray_num', 'block_num', 'desert']
    keys_avail = [k for k in common_keys if k in fg_df.columns and k in attr_df.columns]
    if not keys_avail:
        # fallback to 'sample' only if present
        if 'sample' in fg_df.columns and 'sample' in attr_df.columns:
            keys_avail = ['sample']
        else:
            raise ValueError("No merging keys found between files (expect sample, tray_num, block_num, desert).")
    merged = pd.merge(fg_df, attr_df, on=keys_avail, how='left', suffixes=('_fg','_attr'))
    return merged

def pivot_functional_groups(merged):
    # pivot so each sample is a row, columns are functional groups with counts (sum)
    if 'fctngrp' not in merged.columns or 'count' not in merged.columns:
        raise ValueError("Expected columns 'fctngrp' and 'count' in functional group file.")
    pivot = merged.pivot_table(index='sample', columns='fctngrp', values='count', aggfunc='sum', fill_value=0)
    
    # Ensure numeric (sometimes read as strings)
    pivot = pivot.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # flatten columns
    pivot.columns = [str(c) for c in pivot.columns]
    pivot.reset_index(inplace=True)
    return pivot


def build_sample_summary(pivot, attr_df, merged):
    # merge pivot with attribute metadata - prefer attr_df deduplicated by sample
    # dedupe attributes by sample: first occurrence
    attr_sub = attr_df.drop_duplicates(subset=['sample']).set_index('sample')
    pivot = pivot.set_index('sample')
    summary = pivot.join(attr_sub, how='left')
    # compute derived metrics
    # total count across functional groups
    fg_cols = [c for c in pivot.columns if c in pivot.columns]  # all pivot columns
    summary['total_seed_count'] = pivot.sum(axis=1)
    # richness = number of functional groups with count > 0
    summary['fg_richness'] = (pivot > 0).sum(axis=1)
    # presence of burn column etc
    for col in ['TSF', 'TSF2']:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce')

    # normalize names
    for col in ['burn', 'TSF', 'overstory', 'desert', 'plot_num', 'fullplot_name', 'TSF2']:
        if col not in summary.columns:
            summary[col] = np.nan
    summary.reset_index(inplace=True)
    return summary

def summarize_by_group(summary):
    # group summaries for quick inspection
    group_cols = ['desert', 'burn', 'overstory']
    numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
    agg = summary.groupby(group_cols)[numeric_cols].agg(['mean','std','count'])
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)
    return agg


def compare_burned_control(summary, pair_on='fullplot_name'):
    """
    If pairing information is available (e.g., same fullplot_name has both burned/control),
    perform paired test (Wilcoxon) on richness. Otherwise perform Mann-Whitney.
    Return dict of results for richness and total_count.
    """
    results = {}
    # attempt to find pairs: same 'plot_num' or 'fullplot_name' with both burn statuses
    if 'fullplot_name' in summary.columns:
        # group by fullplot_name and select pairs with both burn statuses present
        gb = summary.groupby('fullplot_name')
        pairs = []
        for name, g in gb:
            if g['burn'].nunique() >= 2:
                # select one burned and one control sample per fullplot_name - take means if multiple
                b = g[g['burn'].str.lower().str.contains('burn', na=False)]
                c = g[~g['burn'].str.lower().str.contains('burn', na=False)]
                if not b.empty and not c.empty:
                    pairs.append({'name':name,
                                  'rich_b': b['fg_richness'].mean(),
                                  'rich_c': c['fg_richness'].mean(),
                                  'total_b': b['total_seed_count'].mean(),
                                  'total_c': c['total_seed_count'].mean()})
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            # paired Wilcoxon signed-rank
            try:
                w_r, p_r = stats.wilcoxon(pairs_df['rich_b'], pairs_df['rich_c'])
            except Exception:
                w_r, p_r = (np.nan, np.nan)
            try:
                w_t, p_t = stats.wilcoxon(pairs_df['total_b'], pairs_df['total_c'])
            except Exception:
                w_t, p_t = (np.nan, np.nan)
            results['paired'] = {
                'n_pairs': len(pairs_df),
                'rich_w_stat': float(w_r) if not np.isnan(w_r) else None,
                'rich_p': float(p_r) if not np.isnan(p_r) else None,
                'total_w_stat': float(w_t) if not np.isnan(w_t) else None,
                'total_p': float(p_t) if not np.isnan(p_t) else None,
            }
            return results
    # fallback: unpaired test across burned vs control
    burned = summary[summary['burn'].str.lower().str.contains('burn', na=False)]
    control = summary[~summary['burn'].str.lower().str.contains('burn', na=False)]
    if burned.empty or control.empty:
        results['unpaired'] = {'error': 'No burned or no control samples found for unpaired test.'}
        return results
    try:
        u_r, p_r = stats.mannwhitneyu(burned['fg_richness'], control['fg_richness'], alternative='two-sided')
    except Exception:
        u_r, p_r = (np.nan, np.nan)
    try:
        u_t, p_t = stats.mannwhitneyu(burned['total_seed_count'], control['total_seed_count'], alternative='two-sided')
    except Exception:
        u_t, p_t = (np.nan, np.nan)
    results['unpaired'] = {
        'n_burned': int(len(burned)),
        'n_control': int(len(control)),
        'rich_u_stat': float(u_r) if not np.isnan(u_r) else None,
        'rich_p': float(p_r) if not np.isnan(p_r) else None,
        'total_u_stat': float(u_t) if not np.isnan(u_t) else None,
        'total_p': float(p_t) if not np.isnan(p_t) else None,
    }
    return results

def microsite_test(summary):
    # overstory column often coded 'I' (interspace) and 'S' (shrub)
    s = summary.dropna(subset=['overstory'])
    s['overstory_clean'] = s['overstory'].astype(str).str.strip().str.upper().str[0]
    shrub = s[s['overstory_clean']=='S']
    inter = s[s['overstory_clean']=='I']
    res = {}
    if len(shrub)>0 and len(inter)>0:
        try:
            u_r, p_r = stats.mannwhitneyu(shrub['fg_richness'], inter['fg_richness'], alternative='two-sided')
        except Exception:
            u_r, p_r = (np.nan, np.nan)
        res = {
            'n_shrub': int(len(shrub)),
            'n_inter': int(len(inter)),
            'rich_u_stat': float(u_r) if not np.isnan(u_r) else None,
            'rich_p': float(p_r) if not np.isnan(p_r) else None
        }
    else:
        res['error'] = 'Insufficient shrub/interspace samples for test'
    return res

def tsf_regression(summary):
    # Simple OLS: richness ~ TSF for burned samples only
    res = {}
    # ensure numeric TSF
    summary['TSF_num'] = pd.to_numeric(summary['TSF'], errors='coerce')
    burned = summary[summary['burn'].str.lower().str.contains('burn', na=False)].dropna(subset=['TSF_num','fg_richness'])
    if len(burned) >= 5:
        X = sm.add_constant(burned['TSF_num'])
        y = burned['fg_richness']
        model = sm.OLS(y, X, missing='drop').fit()
        res['n'] = int(len(burned))
        res['params'] = model.params.to_dict()
        res['pvalues'] = model.pvalues.to_dict()
        res['rsquared'] = float(model.rsquared)
    else:
        res['error'] = 'Not enough burned samples with TSF for regression'
    return res

def save_figures(summary, out_prefix):
    # richness by burn
    plt.figure(figsize=(6,4))
    sns.boxplot(x='burn', y='fg_richness', data=summary, order=sorted(summary['burn'].dropna().unique()))
    sns.stripplot(x='burn', y='fg_richness', data=summary, color='0.2', size=3, jitter=True, alpha=0.6)
    plt.title('Functional-group richness by burn status')
    plt.tight_layout()
    fn1 = f"{out_prefix}_richness_by_burn.png"
    plt.savefig(fn1, dpi=200)
    plt.close()

    # richness by microsite
    if 'overstory' in summary.columns:
        plt.figure(figsize=(6,4))
        s = summary.copy()
        s['overstory_clean'] = s['overstory'].astype(str).str.strip().str.upper().str[0]
        sns.boxplot(x='overstory_clean', y='fg_richness', data=s)
        sns.stripplot(x='overstory_clean', y='fg_richness', data=s, color='0.2', size=3, jitter=True, alpha=0.6)
        plt.title('Richness by microsite (S=shrub, I=interspace)')
        plt.tight_layout()
        fn2 = f"{out_prefix}_richness_by_microsite.png"
        plt.savefig(fn2, dpi=200)
        plt.close()

    # TSF vs richness scatter
    if 'TSF' in summary.columns:
        plt.figure(figsize=(6,4))
        s = summary.copy()
        s['TSF_num'] = pd.to_numeric(s['TSF'], errors='coerce')
        sns.scatterplot(x='TSF_num', y='fg_richness', hue='burn', data=s, alpha=0.7)
        plt.xlabel('Time since fire (TSF)')
        plt.ylabel('FG richness')
        plt.title('TSF vs FG richness')
        plt.tight_layout()
        fn3 = f"{out_prefix}_tsf_vs_richness.png"
        plt.savefig(fn3, dpi=200)
        plt.close()

def main():
    print("Loading tables...")
    fg = safe_read_csv(FG_FILE)
    attr = safe_read_csv(ATTR_FILE)

    print(f"Functional groups rows: {len(fg)}, attributes rows: {len(attr)}")

    print("Merging...")
    merged = merge_tables(fg, attr)
    merged.to_csv(f"{OUT_PREFIX}_merged.csv", index=False)

    print("Pivoting functional groups per sample...")
    pivot = pivot_functional_groups(merged)
    pivot.to_csv(f"{OUT_PREFIX}_per_sample_fg_pivot.csv", index=False)

    print("Building per-sample summary...")
    summary = build_sample_summary(pivot, attr, merged)
    summary.to_csv(f"{OUT_PREFIX}_per_sample_summary.csv", index=False)

    print("Group summaries...")
    group_summ = summarize_by_group(summary)
    group_summ.to_csv(f"{OUT_PREFIX}_group_summary_by_desert_burn_overstory.csv", index=False)

    print("Burned vs control tests...")
    burn_tests = compare_burned_control(summary)
    print("Microsite tests...")
    microsite_tests = microsite_test(summary)
    print("TSF regression on richness (burned samples)...")
    tsf_tests = tsf_regression(summary)

    test_results = {
        'burn_tests': burn_tests,
        'microsite_tests': microsite_tests,
        'tsf_regression': tsf_tests
    }

    with open(f"{OUT_PREFIX}_tests.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("Saving plots...")
    save_figures(summary, OUT_PREFIX)

    print("Done. Outputs:")
    for fn in Path('.').glob(f"{OUT_PREFIX}*"):
        print("  -", fn.name)

if __name__ == "__main__":
    main()
