import math

def is_prime(n):
    """Standard optimized primality test."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_radical(n):
    """Product of distinct prime factors (Structural Core)."""
    if n == 0: return 0
    res = 1
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            res *= d
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        res *= temp
    return res

def test_iogt_stress(p_n, p_next):
    """The Stress Test: Measuring Manifold Tension vs. Actual Flow."""
    start = p_n**2
    end = p_next**2
    
    # Expansion Potential (Theoretical PNT Pressure)
    phi_exp = (end / math.log(end)) - (start / math.log(start))
    
    phi_sum = 0
    actual_primes = 0
    total_integers = 0
    
    for x in range(start + 1, end):
        # Calculate local Porosity Phi(x)
        rad = get_radical(x)
        phi_x = math.log(rad) / math.log(x)
        phi_sum += phi_x
        
        if is_prime(x):
            actual_primes += 1
        total_integers += 1
    
    eta = phi_sum / total_integers # Structural Viscosity
    tension = phi_exp / eta        # Tension Metric
    
    # "Turbulence" is the difference between predicted tension and actual primes
    turbulence = tension - actual_primes
    
    return {
        "Interval": f"{p_n}^2 to {p_next}^2",
        "Length": total_integers,
        "Viscosity (η)": round(eta, 6),
        "Predicted (T)": round(tension, 2),
        "Actual (π)": actual_primes,
        "Turbulence": round(turbulence, 4),
        "Error %": round((turbulence / actual_primes * 100), 2) if actual_primes > 0 else "N/A"
    }

# Run the test for your specific interval
print(test_iogt_stress(1009, 1013))