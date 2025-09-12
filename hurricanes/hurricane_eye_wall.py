import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib.pyplot as plt

# =====================================================
# Hurricane Eyewall Radius Prediction Model (Database Version)
# =====================================================

# Load data
df = pd.read_csv("hurricane_eye_radius_database.csv")

# Observed eyewall radius column
obs_col = "eye_radius_obs"

# Add default storm-top height if not present
if "H_km" not in df.columns:
    df["H_km"] = 12.0

# Earth's rotation rate (s^-1)
Omega = 7.2921150e-5

# Compute Coriolis parameter
df["f"] = 2 * Omega * np.sin(np.radians(df["latitude"]))
df["f_abs"] = np.abs(df["f"])
f_ref = Omega  # reference f at ~30° latitude
df["f_norm"] = df["f_abs"] / f_ref

# Geometric terms
df["V_cuberoot"] = (df["H_km"] * np.pi * (df["R_vortex_km"] ** 2)) ** (1 / 3)
df["H_over_R"] = df["H_km"] / df["R_vortex_km"]

# =====================================================
# Model definition
# =====================================================

def model_f(params, Vcuberoot, f_norm, H_over_R):
    """Coriolis-based model: r_eye = V^(1/3) * (phi_offset + phi_scale * f_norm) * (H/R)^gamma"""
    phi_scale, phi_offset, gamma = params
    return Vcuberoot * (phi_offset + phi_scale * f_norm) * (H_over_R ** gamma)

# =====================================================
# Fitting routine
# =====================================================

def fit_model(df):
    Vcuberoot = df["V_cuberoot"].values
    f_norm = df["f_norm"].values
    H_over_R = df["H_over_R"].values
    y = df[obs_col].values

    def residuals(params):
        pred = model_f(params, Vcuberoot, f_norm, H_over_R)
        return pred - y

    init = np.array([0.8, 0.3, 0.3])
    bounds = ([0, 0, 0], [5, 5, 2])
    res = optimize.least_squares(residuals, init, bounds=bounds)
    return res.x

# =====================================================
# Run fit
# =====================================================

params_f = fit_model(df)
print("F-based params:", params_f)

# Predictions
df["pred_eye_f"] = model_f(params_f, df["V_cuberoot"], df["f_norm"], df["H_over_R"])

# =====================================================
# Diagnostics
# =====================================================

def diagnostics(obs, pred, label="Model"):
    error = pred - obs
    metrics = {
        "mean_abs_error": np.abs(error).mean(),
        "median_abs_error": np.median(np.abs(error)),
        "rmse": np.sqrt((error ** 2).mean()),
        "r_squared": stats.pearsonr(obs, pred)[0] ** 2,
    }
    print(f"{label} performance:", metrics)
    return error, metrics

error_f, metrics_f = diagnostics(df[obs_col], df["pred_eye_f"], label="F-based")

# Save results
df.to_csv("hurricane_eye_wall_results_from_database.csv", index=False)

# =====================================================
# Plots
# =====================================================

# Scatter: observed vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(df[obs_col], df["pred_eye_f"], alpha=0.7, label="F-based")
plt.plot([0, 80], [0, 80], "r--")
plt.xlabel("Observed r_eye (km)")
plt.ylabel("Predicted r_eye (km)")
plt.legend()
plt.title("Observed vs Predicted Eyewall Radius (F-based)")
plt.tight_layout()
plt.savefig("eye_pred_vs_obs_f_based.png")

# Histogram: percent error
df["percent_error"] = 100 * (df["pred_eye_f"] - df[obs_col]) / df[obs_col]

plt.figure(figsize=(6, 4))
plt.hist(df["percent_error"], bins=30, edgecolor="black")
plt.xlabel("Percent Error (%)")
plt.ylabel("Count")
plt.title("Percent Error Distribution (F-based)")
plt.tight_layout()
plt.savefig("eye_percent_error_hist_f_based.png")
