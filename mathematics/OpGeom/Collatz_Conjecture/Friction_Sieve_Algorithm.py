import math
from sympy import factorint, isprime

def get_rad(n):
    if n <= 1: return 1
    return math.prod(factorint(n).keys())

def opgeom_weather_report(mode="peaks"):
    print(f"--- OpGeom Weather Report: {mode.upper()} MODE ---")
    print(f"{'n':<20} | {'Re_arith':<10} | {'Friction':<10} | {'Regime'}")
    
    if mode == "peaks":
        # Search near potential peaks (powers of 2) where turbulence is highest
        test_range = [2**k - 1 for k in range(10, 60)]
    else:
        # Standard manifold scan
        test_range = range(10**6, 10**6 + 100)

    for n in test_range:
        if n % 2 == 0: continue
        
        V = math.log(n)
        rad_n = get_rad(n)
        S = math.log(rad_n)
        re = V / S if S > 0 else 0
        
        # Friction of the Collatz Bridge
        next_n = 3 * n + 1
        f = math.log(get_rad(next_n)) / math.log(next_n)
        
        regime = "TURBULENT" if re > 3 else "LAMINAR"
        print(f"{n:<20} | {re:<10.2f} | {f:<10.2f} | {regime}")

opgeom_weather_report(mode="peaks")