# Threading Ecology Analysis Repository

This repository contains the analysis scripts and supporting data for three ecological studies, each examined within the **Threading Ecology** framework. The analyses focus on how disturbances interact with ecological memory, redistribution, and recovery dynamics across multiple ecosystems.

---

## Folder Structure and Contents

### 1. Bonanza Creek Fire Regimes (Boreal Forests) 
- **Folder:** `Bonanza_Creek_LTER`
- **Contents:**
  - `analysis.py` — Script for analyzing pre- and post-fire forest composition and biomass.
  - `804_PrePostFireStandData_DiVA_XJW.csv` — Original dataset with tree species densities, biomass, and deciduous index (DI) per transect.
- **Analysis Highlights:**
  - Tested hypotheses on disturbance cycles, memory gradients, post-disturbance coherence, threading density recovery, and multi-phase memory recovery.
  - Outputs include summaries of memory redistribution and recovery trajectories.

- **Data Reference:**
  - Walker, X. et al. (2023). *Bonanza Creek Long-Term Fire Study Dataset*. [LTER Data Portal](http://www.lter.uaf.edu/data/data-detail/id/1). DOI: 10.6073/pasta/725db90d86686be13e6d6b2da5d61217.

---

### 2. Alpine Grassland Disturbance (Qinghai–Tibet Plateau)
- **Folder:** `Qinghai_Tibet_Plateau`
- **Contents:**
  - `analysis.py` — Script for paired analysis of pika-disturbed and undisturbed plots.
  - Original dataset: `Li_2025_AlpineGrassland.csv` (90 paired plots) including biodiversity, biomass, soil nutrients, and multifunctionality metrics.
- **Analysis Highlights:**
  - Paired comparisons for aboveground biomass, soil carbon, nutrients, and multifunctionality.
  - Correlation of soil organic carbon gradients with disturbance impact.
  - Evidence of post-disturbance coherence enhancement and phased memory recovery.

- **Data Reference:**
  - Li, J. (2025). *Paired alpine grassland vegetation–soil dataset under plateau pika disturbance for biodiversity and multifunctionality analysis*. Mendeley Data, V1. [Dataset Link](https://data.mendeley.com/datasets/6cxvvr2bmn/1).

---

### 3. North American Desert Seed Banks
- **Folder:** `North_American_Desert_Sites`
- **Contents:**
  - `analysis.py` — Script processing seed bank functional group summaries and attributes.
  - Input files:
    - `seedbank_fctnlgrpsummary.csv` — Functional group counts per sample.
    - `seedbank_attribute_spreadsheet.csv` — Sample metadata including desert, fire status, and microsite.
  - Outputs:
    - `paired_differences.csv` — Burn vs. control differences by functional group.
    - `summary_by_sample.csv` — Per-sample summaries.
    - `summary_by_group.csv` — Aggregated statistics by burn status, microsite, and desert.
    - `tsf_regression_results.json` — Regression of richness vs. time-since-fire.
- **Analysis Highlights:**
  - Examined seed bank responses to wildfire, microsite effects, and long-term richness recovery.
  - Tested hypotheses on disturbance cycles, memory gradients, post-disturbance coherence, and multi-phase memory recovery.

- **Data Reference:**
  - Hosna, R. & Faist, A. (2022). *Long-term relationships between seed banks and wildfire across four North American desert sites, ver 1*. Environmental Data Initiative. Accessed 2025-10-08.

---

## General Notes
- Each `analysis.py` script contains code for:
  - Loading and preprocessing the original dataset.
  - Computing per-sample and per-group summaries.
  - Performing hypothesis-specific tests and regressions.
  - Exporting summary CSVs and regression results.
- All analyses are structured to support **Threading Ecology hypotheses**, including:
  - H1: Threading–Disturbance Cycles
  - H2: Memory Gradient Triggers
  - H4: Post-Disturbance Coherence Enhancement
  - H5: Threading Density Recovery Before Composition
  - H6: Multi-Phase Memory Recovery

---

## Citation
Please cite the original datasets and studies when using these analyses:

1. Walker, X. et al. (2023). *Bonanza Creek Long-Term Fire Study Dataset*. LTER Data Portal. DOI: 10.6073/pasta/725db90d86686be13e6d6b2da5d61217.
2. Li, J. (2025). *Paired alpine grassland vegetation–soil dataset under plateau pika disturbance*. Mendeley Data, V1. https://data.mendeley.com/datasets/6cxvvr2bmn/1
3. Hosna, R., & Faist, A. (2022). *Long-term relationships between seed banks and wildfire across four North American desert sites*. Environmental Data Initiative.

