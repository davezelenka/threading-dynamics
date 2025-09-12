# Threading Dynamics

This repository explores **universal threading dynamics**: a clean mathematical framework unifying rotational systems across scales — from **hurricanes** and **climate oscillations (ENSO, PDO, AMO)** to planetary polygons (e.g., Saturn’s hexagon) and galactic spirals.  

At the core is the hypothesis that **all observed spirals, rings, bands, and geometric structures** are surface expressions of deeper annular EM and rotational threading patterns. These emerge from a universal scaling law:

\[
\lambda \propto V^{-1/3}
\]

where  
- **λ** = characteristic threading wavelength (pattern scale),  
- **V** = system volume (geometric coupling),  
- stability is set by resonance conditions and coherence.  

---

## Repository Structure

- **ENSO_PDO_AMO/**  
  Analyses of climate oscillations (ENSO, PDO, AMO) as expressions of deep rotational threading. Includes spectral studies and harmonic modeling.  

- **hurricanes/**  
  Hurricane rotation, Coriolis imprint, and persistence explained by deep rotation coupling. Links to planetary-scale threading phenomena.  

---

## Visual Tools

Interactive Desmos models help visualize how threading generates geometric and spiral patterns:

- [Jupiter/Saturn octagon & hexagon patterns](https://www.desmos.com/calculator/ue1uufq4u0)
- [Hurricane vortex parameters and predicted](https://www.desmos.com/calculator/ycpmeded0h)
- [Topography & bathymetric spirals](https://www.desmos.com/calculator/etbwbnvafy)  

---

## Quick Start

Use these Python functions to calculate threading wavelengths and predict harmonic locations:

```python
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
