https://data.mendeley.com/datasets/6cxvvr2bmn/1

Description
This dataset originates from extensive field sampling and laboratory measurements across three alpine grassland types on the Qinghai–Tibet Plateau. It includes paired observations under plateau pika (Ochotona curzoniae) disturbance and undisturbed conditions, enabling robust assessments of biodiversity and ecosystem multifunctionality. For each plot, we recorded: Biodiversity indicators: plant diversity, soil nematode diversity, and soil microbial diversity (all based on standardized quadrat sampling and computation), Ecosystem function indicators: aboveground biomass, soil organic carbon, total nitrogen, total phosphorus, available nitrogen, available phosphorus, and soil moisture.
Sampling was conducted on 90 paired plots, resulting in a total of 180 individual plots. Each indicator was measured in both pika-disturbed and undisturbed counterparts under a stratified paired design. The dataset supports research on how small burrowing herbivore disturbance affects biodiversity–multifunctionality relationships in alpine ecosystems.

Download All 30.7 KB

Files

csv
data.csv
78 KB
Steps to reproduce
Composite biodiversity and two types of ecosystem multifunctionality were calculated in R. For each biological group (plants, soil nematodes, bacteria, fungi), Shannon–Wiener indices were computed based on repeated sampling (five subplots per plot). Composite biodiversity was obtained as the arithmetic mean of the Z-score standardized indices across the four biological groups. All laboratory measurements were conducted at the College of Pastoral Agriculture Science and Technology, Lanzhou University, supported by the Large-scale Instrument and Equipment Sharing Platform of Lanzhou University. All data were jointly obtained through field surveys and laboratory experiments conducted with the assistance of all project team members.
Ecosystem multifunctionality was assessed using two complementary approaches:
Mean-based multifunctionality: calculated as the average of standardized values across selected ecosystem function variables.
Effective multifunctionality (Hill number–based): 
# Standardize ecosystem functions to unit scale
df_std <- data %>%
  mutate(across(all_of(function_vars), standardizeUnitScale, .names = "{.col}.std"))

# Compute mean-based multifunctionality
fvars.std <- paste0(function_vars, ".std")
df_std$meanFunction <- rowMeans(df_std[fvars.std])

# Compute effective multifunctionality (unadjusted and correlation-adjusted)
df_std <- df_std %>%
  mutate(n_eff_func_1 = eff_num_func(., fvars.std, q = 1),
         mf_eff_1 = n_eff_func_1 * meanFunction)

D <- cor_dist(df_std[fvars.std])        # Compute function correlation distance matrix
tau <- dmean(df_std[fvars.std], D)      # Compute average dissimilarity
df_std <- df_std %>%
  mutate(mf_eff_1_cor = getMF_eff(., fvars.std, q = 1, D = D, tau = tau))

Institutions
Lanzhou University
Categories

Functional Plant Ecology, Soil
