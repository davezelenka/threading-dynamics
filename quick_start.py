import numpy as np

def calculate_threading_wavelength(volume_km3, k=1e4):
    """
    Calculate characteristic threading wavelength.
    volume_km3 : system volume in cubic kilometers
    k          : coherence constant (default planetary-scale)
    """
    return k * (volume_km3 ** (-1/3))

def predict_harmonics(lambda_km, radius_km, max_n=10):
    """
    Predict harmonic band/polygon latitudes.
    Returns list of (harmonic_number, latitude_deg).
    """
    harmonics = []
    for n in range(1, max_n+1):
        sin_value = n * lambda_km / (2 * np.pi * radius_km)
        if sin_value <= 1:  # physically possible
            theta = 90 - np.degrees(np.arcsin(sin_value))
            harmonics.append((n, theta))
    return harmonics

# Example: Saturn
volume_km3 = 8.27e14  # Saturn volume
lambda_km = calculate_threading_wavelength(volume_km3)
harmonics = predict_harmonics(lambda_km, radius_km=58232)

print("λ =", lambda_km, "km")
print("Harmonics (n, latitude):", harmonics)