import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Barton Plant & Soil Data Summer 2021.csv")

# Filter for numeric variables of interest
soil_vars = ['aboveground_biomass', 'soil_moisture', 'ph', 'ec',
             'nh4i', 'no3i', 'nh4f', 'no3f', 'po4']
# species columns (all plant codes from the dataset)
species_cols = df.columns[23:]  # adjust if necessary

# Create species richness column
df['species_richness'] = (df[species_cols].fillna(0) > 0).sum(axis=1)

# Function for non-parametric group comparisons
def group_comparison(var, group):
    groups = [df[df[group]==val][var].dropna() for val in df[group].unique()]
    if len(groups) > 1:
        try:
            stat, p = stats.kruskal(*groups)
            return stat, p
        except ValueError:
            return np.nan, np.nan
    return np.nan, np.nan

# Summary table for soil variables by legacy, seed, shrub
summary_table = []
for var in soil_vars + ['species_richness']:
    for factor in ['F.legacy', 'F.seed', 'F.shrub']:
        stat, p = group_comparison(var, factor)
        summary_table.append({
            'variable': var,
            'factor': factor,
            'kruskal_stat': stat,
            'p_value': p
        })

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv("nonparametric_summary_table.csv", index=False)

# Compute correlations of biomass vs soil variables and species richness
corr_table = []
for var in soil_vars + ['species_richness']:
    valid = df[['aboveground_biomass', var]].dropna()
    if len(valid) > 2:
        r, p = stats.spearmanr(valid['aboveground_biomass'], valid[var])
        corr_table.append({'variable': var, 'spearman_r': r, 'p_value': p})

corr_df = pd.DataFrame(corr_table)
corr_df.to_csv("biomass_correlations.csv", index=False)

# Optional: visualize biomass by legacy treatment
plt.figure(figsize=(10,6))
sns.boxplot(x='F.legacy', y='aboveground_biomass', data=df)
sns.swarmplot(x='F.legacy', y='aboveground_biomass', data=df, color=".25")
plt.title("Aboveground Biomass by Legacy Treatment")
plt.savefig("biomass_by_legacy.png", dpi=300)
plt.close()

# Optional: visualize species richness by legacy
plt.figure(figsize=(10,6))
sns.boxplot(x='F.legacy', y='species_richness', data=df)
sns.swarmplot(x='F.legacy', y='species_richness', data=df, color=".25")
plt.title("Species Richness by Legacy Treatment")
plt.savefig("richness_by_legacy.png", dpi=300)
plt.close()

print("Analysis complete. Summary tables and plots saved.")
