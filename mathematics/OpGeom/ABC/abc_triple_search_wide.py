import math
from sympy import factorint, gcd

def get_rad(n):
    if n <= 1: return 1
    return math.prod(factorint(n).keys())

def opgeom_wide_probe(exp_start=30, exp_end=80):
    print(f"Scanning Manifold Pockets: Exponents {exp_start} to {exp_end}...")
    
    # Generate a wider variety of 'Target Smooths' for c
    # We include 7 and 11 to catch more resonances
    c_targets = []
    for i in range(10):
        for j in range(6):
            for k in range(4):
                for l in range(2):
                    val = (2**i) * (3**j) * (5**k) * (7**l)
                    if 1 < val < 10000:
                        c_targets.append(val)
    
    for a_base in [2, 3, 5]:
        for exp in range(exp_start, exp_end):
            a = a_base**exp
            for c_smooth in c_targets:
                b = abs(a - c_smooth)
                if b <= 1 or gcd(a, b) != 1: continue
                
                # Relaxed filter: looking for q > 1.1 or 1.2
                rad_b = get_rad(b)
                rad_abc = a_base * get_rad(c_smooth) * rad_b
                
                c_val = max(a, b, c_smooth)
                q = math.log(c_val) / math.log(rad_abc)
                
                if q > 1.15:
                    print(f"RESONANCE: {a_base}^{exp} | b={b} | c={c_smooth} | q={q:.4f}")

opgeom_wide_probe()