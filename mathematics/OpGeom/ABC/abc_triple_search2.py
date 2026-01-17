import math
from sympy import factorint, gcd

def get_rad(n):
    if n <= 1: return 1
    f = factorint(n)
    r = 1
    for p in f: r *= p
    return r

def opgeom_tunneling_search():
    # Targets for c that are extremely light (smooth)
    # We look for: p^x + k = q^y  where k is very smooth
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    hits = []
    
    print("Beginning high-resonance tunneling...")
    
    for p in [2, 3, 5]: # Primary bases
        for x in range(10, 60): # High potential
            a = p**x
            # Check a small neighborhood around the peak
            for b in range(1, 10000): 
                if gcd(a, b) != 1: continue
                
                # Test both directions: a + b = c and a - b = c
                for c in [a + b, abs(a - b)]:
                    if c == 0: continue
                    
                    # OpGeom efficiency filter: Is c highly compressed?
                    rad_c = get_rad(c)
                    if rad_c < math.sqrt(c): # The "Lightness" threshold
                        rad_abc = get_rad(a) * get_rad(b) * rad_c
                        q = math.log(max(a, b, c)) / math.log(rad_abc)
                        
                        if q > 1.3:
                            hits.append((a, b, c, q))
                            print(f"RESONANCE FOUND: {a}, {b}, {c} | q = {q:.4f}")
    return hits

# This is optimized for laptop CPU/RAM
opgeom_tunneling_search()