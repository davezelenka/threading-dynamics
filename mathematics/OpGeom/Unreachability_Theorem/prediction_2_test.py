import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Load the data
df = pd.read_csv('lmfdb_ec_curvedata_0128_1616.csv')

# 2. Clean the column names (Extract the text inside the second set of quotes)
def clean_header(col):
    match = re.search(r'""([^""]+)""\)$', col)
    return match.group(1) if match else col

df.columns = [clean_header(c) for c in df.columns]
print("Cleaned columns:", df.columns.tolist())

# 3. Filter for Rank 1 (where Regulator = Canonical Height)
# If the CSV values also have formula junk, we clean them:
if df['rank'].dtype == object:
    df['rank'] = df['rank'].str.extract(r'(\d+)').astype(int)

df_rank1 = df[df['rank'] == 1].copy()

# 4. Extract |Delta| and Height
# Ensure numeric conversion
df_rank1['abs_delta'] = pd.to_numeric(df_rank1['disc']).abs()
df_rank1['height'] = pd.to_numeric(df_rank1['regulator'])

# 5. Log-Log Transformation
# We use log10 to make the slope interpretation intuitive
x = np.log10(df_rank1['abs_delta'])
y = np.log10(df_rank1['height'])

# 6. Plotting the "Forbidden Zone"
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, s=1, label='LMFDB Data (Rank 1)')

# Theoretical Floor: h_min ~ |Delta|^(-1/12)
# We pick a constant C to align the line with the bottom of the data cloud
x_line = np.linspace(x.min(), x.max(), 100)
predicted_slope = -1/12  # -0.0833
intercept = np.percentile(y - (predicted_slope * x), 5) # Fit to the 5th percentile (the floor)
y_line = predicted_slope * x_line + intercept

plt.plot(x_line, y_line, color='red', linestyle='--', 
         label=f'Prediction 2 Floor (Slope: {predicted_slope:.4f})')

plt.xlabel('log10(|Discriminant|)')
plt.ylabel('log10(Minimal Height)')
plt.title('Validation of Prediction 2: Height Floor Scaling')
plt.legend()
plt.grid(True, alpha=0.2)

# 7. Calculate Empirical Slope of the Floor
# We fit only to the points near the bottom of the distribution
floor_indices = y < np.percentile(y, 10) 
if any(floor_indices):
    floor_slope, _ = np.polyfit(x[floor_indices], y[floor_indices], 1)
    print(f"Empirical Floor Slope: {floor_slope:.4f}")
    print(f"Predicted Slope: -0.0833")
    print(f"Difference: {abs(floor_slope - (-0.0833)):.4f}")

plt.show()