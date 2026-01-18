import math
from sympy import factorint, isprime

def get_rad(n):
    if n <= 1: return 1
    return math.prod(factorint(n).keys())

def map_semi_turbulent_corridors(max_p=500, max_k=15):
    print(f"{'n (p^k)':<15} | {'3n+1 Factors':<25} | {'Re_arith':<10} | {'Status'}")
    print("-" * 70)
    
    primes = [p for p in range(3, max_p) if isprime(p)]
    
    for p in primes:
        for k in range(1, max_k):
            n = p**k
            target = 3 * n + 1
            
            # Remove the base-2 drainage factor to see the remaining 'scaffold'
            temp_target = target
            while temp_target % 2 == 0:
                temp_target //= 2
            
            # If the remainder is a small prime or a prime power, we found a corridor
            factors = factorint(temp_target)
            
            # Condition: The remaining structural mass is small (e.g., one prime or smooth)
            if len(factors) <= 1:
                re = math.log(n) / math.log(get_rad(n))
                status = "CLEAN BRIDGE" if temp_target == 1 else "SEMI-TURBULENT"
                
                # Format factors for printing
                f_str = " * ".join([f"{pr}^{ex}" for pr, ex in factors.items()])
                if temp_target == 1: f_str = "2^m"
                
                print(f"{p}^{k:<13} | {f_str:<25} | {re:<10.2f} | {status}")

map_semi_turbulent_corridors()