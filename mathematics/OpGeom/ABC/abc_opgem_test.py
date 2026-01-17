import numpy as np
from sympy import factorint
import pandas as pd

def get_prime_distribution(n):
    if n == 1:
        return {}, 0
    factors = factorint(n)
    omega_n = sum(factors.values())
    return {p: e / omega_n for p, e in factors.items()}, omega_n

def kl_divergence(pc_dist, pab_dist):
    if not pc_dist: return 0
    kl = 0
    for p, p_c in pc_dist.items():
        # Using a tiny epsilon to handle 'new' primes in c
        p_ab = pab_dist.get(p, 1e-9)
        kl += p_c * np.log(p_c / p_ab)
    return kl

def analyze_triple(a, b, c):
    dist_a, omega_a = get_prime_distribution(a)
    dist_b, omega_b = get_prime_distribution(b)
    dist_c, omega_c = get_prime_distribution(c)
    
    total_omega_ab = omega_a + omega_b
    dist_ab = {}
    if total_omega_ab > 0:
        all_primes_ab = set(dist_a.keys()) | set(dist_b.keys())
        for p in all_primes_ab:
            weight_a = omega_a * dist_a.get(p, 0)
            weight_b = omega_b * dist_b.get(p, 0)
            dist_ab[p] = (weight_a + weight_b) / total_omega_ab
    else:
        # Case where a=1, b=1 (impossible since gcd(1,1) != 1 and c=2)
        # Or just a=1, b=prime. Then dist_ab is dist_b.
        pass
        
    c_cost = kl_divergence(dist_c, dist_ab)
    
    def get_rad(n):
        if n == 1: return 1
        f = factorint(n)
        r = 1
        for p in f: r *= p
        return r
    
    rad_abc = get_rad(a) * get_rad(b) * get_rad(c)
    ln_c = np.log(c)
    ln_rad = np.log(rad_abc)
    q = ln_c / ln_rad
    
    return {
        "a": a, "b": b, "c": c,
        "q": q,
        "C_cost": c_cost,
        "ln_c": ln_c,
        "ln_rad": ln_rad,
        "omega_c": omega_c,
        "omega_ab": total_omega_ab
    }




triples = [
    (2, 3**10 * 109, 23**5),
    (3, 4, 7),
    (1, 8, 9),
    (1, 4374, 4375),
    (1, 2, 3),
    (1, 1024, 1025),
    (1, 2401, 2402),
    (1, 3**7, 1 + 3**7), # 1 + 2187 = 2188 = 4 * 547
]

results = []
for a, b, c in triples:
    results.append(analyze_triple(a, b, c))

df = pd.DataFrame(results)
df.to_csv("abc_test_results.csv", index=False)
print(df[["a", "b", "c", "q", "C_cost", "omega_c", "omega_ab"]])

def calculate_compression(n):
    if n == 1: return 1
    f = factorint(n)
    omega = sum(f.values())
    dist_omega = len(f)
    return omega / dist_omega

results_v2 = []
for a, b, c in triples:
    res = analyze_triple(a, b, c)
    res["comp_a"] = calculate_compression(a)
    res["comp_b"] = calculate_compression(b)
    res["comp_c"] = calculate_compression(c)
    res["avg_comp"] = (res["comp_a"] + res["comp_b"] + res["comp_c"]) / 3
    results_v2.append(res)

df_v2 = pd.DataFrame(results_v2)
df_v2.to_csv("abc_test_results2.csv", index=False)
print(df_v2[["a", "b", "c", "q", "comp_a", "comp_b", "comp_c", "avg_comp"]])