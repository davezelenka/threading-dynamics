## Addendum: Qinghai–Tibet Plateau biodiversity and Ecosystem Multifunctionality Analysis (Threading Ecology Framework)
## Dataset Reference

**Li, J.** (2025). *Paired alpine grassland vegetation–soil dataset under plateau pika disturbance for biodiversity and multifunctionality analysis.*
**Mendeley Data**, V1. [https://data.mendeley.com/datasets/6cxvvr2bmn/1](https://data.mendeley.com/datasets/6cxvvr2bmn/1)

---

## Dataset Description

This dataset originates from extensive field sampling and laboratory measurements across three alpine grassland types on the **Qinghai–Tibet Plateau**. It includes paired observations under plateau pika (*Ochotona curzoniae*) disturbance and undisturbed conditions, enabling robust assessments of **biodiversity and ecosystem multifunctionality**.

Each plot includes the following:

* **Biodiversity indicators:**
  Plant diversity, soil nematode diversity, and soil microbial (bacteria and fungi) diversity based on standardized quadrat sampling.
* **Ecosystem function indicators:**
  Aboveground biomass (AGB), soil organic carbon (SOC), total nitrogen (N), total phosphorus (P), available nitrogen (NH₄⁺, NO₃⁻), available phosphorus (AP), and soil moisture (SM).

Sampling was conducted on **90 paired plots (180 total)** across three alpine grassland types, using a **stratified paired design** to capture disturbance effects.

The dataset supports research on how small burrowing herbivores influence **biodiversity–multifunctionality relationships** and **ecosystem memory dynamics**.

**Original data file:** `data.csv`
Columns:
`plot_id, Grassland_type, Pika_type, Plant, Nematode, Bacteria, Fungi, SOC, N, P, NH4, NO3, AP, SM, AGB, Biodiversity, Mean_EMF, Effective_EMF, ...`

---

## Summary of Paired Analysis

**Number of paired sampling units:** 90

### Paired Differences (Pika – No Pika)

| Variable      | Mean Diff |      t |      p | Interpretation                                  |
| ------------- | --------: | -----: | -----: | ----------------------------------------------- |
| AGB           |    -56.67 | -10.63 | <0.001 | Strong decline in aboveground biomass           |
| SOC           |    -14.16 | -14.48 | <0.001 | Significant loss of soil organic carbon         |
| N             |     -0.65 |  -6.46 | <0.001 | Reduction in total nitrogen                     |
| P             |     +0.02 |   1.54 |  0.126 | No significant change in total phosphorus       |
| NO₃⁻          |     -0.82 |  -5.30 | <0.001 | Decrease in available nitrate                   |
| NH₄⁺          |     -1.22 |  -3.59 |  0.001 | Decrease in available ammonium                  |
| AP            |     +3.89 |   2.46 |  0.016 | Increase in available phosphorus (mobilization) |
| SM            |     -6.81 | -11.01 | <0.001 | Significant soil moisture loss                  |
| Biodiversity  |     -0.23 |  -2.38 |  0.019 | Decrease in overall biodiversity                |
| Mean_EMF      |     -0.34 |  -9.07 | <0.001 | Decline in multifunctionality                   |
| Effective_EMF |     -0.22 |  -6.03 | <0.001 | Decline in effective multifunctionality         |

---

## Correlation Analyses

**Memory gradient proxy:** pre-disturbance SOC
**Response:** absolute paired differences in function or biodiversity

| Variable      |     r |      p |  n | Interpretation                                      |
| ------------- | ----: | -----: | -: | --------------------------------------------------- |
| diff_N        |  0.26 |  0.013 | 90 | Stronger SOC buffers N loss                         |
| diff_AP       | -0.43 | <0.001 | 90 | High SOC reduces P mobilization                     |
| diff_Mean_EMF | -0.54 | <0.001 | 90 | Steeper SOC gradient predicts higher EMF resilience |

**Biodiversity–Multifunctionality coupling:**

* Spearman’s ρ = **0.71**, *p* < 1×10⁻¹⁴, *n* = 90
  → Functional resilience is strongly tied to biodiversity retention.

---

## Interpretation (Threading Ecology Context)

* **H1 – Threading–Disturbance Cycles:**
  Pika disturbance redistributes biomass and nutrients, initiating new threading and recovery cycles.

* **H2 – Memory Gradient Triggers:**
  Steep SOC gradients predict disturbance magnitude—supporting memory-buffered stability.

* **H4 – Post-Disturbance Coherence Enhancement:**
  Increased AP despite losses elsewhere signals reorganization toward new coherence configurations.

* **H5 – Threading Density Recovery Before Composition:**
  Multifunctionality stabilizes faster than biodiversity—threading pathways re-form before full community recovery.

* **H6 – Multi-Phase Memory Recovery:**
  SOC–N–AP trajectories show alternating phases of activation, redistribution, and latent memory storage.

---

## Repository

Analysis scripts and results are available at:
🔗 [https://github.com/davezelenka/threading-dynamics/tree/main/ThreadingEcology](https://github.com/davezelenka/threading-dynamics/tree/main/ThreadingEcology)

---


