import math
from sympy import factorint, gcd
from itertools import combinations

def get_rad(n):
    if n <= 1: return 1
    f = factorint(n)
    r = 1
    for p in f: r *= p
    return r

def find_smooth_numbers(limit, prime_limit):
    """Finds numbers whose prime factors are all <= prime_limit."""
    smooth = []
    primes = [2, 3, 5, 7, 11, 13, 17, 19] # Targeting small prime manifold
    
    def generate(current, index):
        if current > limit: return
        smooth.append(current)
        for i in range(index, len(primes)):
            generate(current * primes[i], i)
            
    generate(1, 0)
    return sorted(smooth)

def opgeom_predictive_search(limit=10**7, q_threshold=1.3):
    """
    Search by bridging 'Light States' in the manifold.
    """
    # 1. Map the 'Light States' (Smooth Numbers)
    print("Mapping the smooth manifold...")
    smooth_nums = find_smooth_numbers(limit, 19)
    
    found = []
    
    # 2. Test bridges between these states
    print(f"Testing bridges across {len(smooth_nums)} states...")
    for i in range(len(smooth_nums)):
        for j in range(i + 1, len(smooth_nums)):
            a = smooth_nums[i]
            b = smooth_nums[j]
            c = a + b
            
            if c > limit * 10: continue # Boundary constraint
            if gcd(a, b) != 1: continue
            
            # OpGeom Metrics
            rad_abc = get_rad(a) * get_rad(b) * get_rad(c)
            q = math.log(c) / math.log(rad_abc)
            
            if q > q_threshold:
                found.append((a, b, c, q))
                print(f"HIT: {a} + {b} = {c} | q = {q:.4f}")
                
    return sorted(found, key=lambda x: x[3], reverse=True)

# Run search
hits = opgeom_predictive_search(limit=50000, q_threshold=1.2)