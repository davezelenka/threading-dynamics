
import math
import random
from dataclasses import dataclass
from typing import List, Set, Generator, Tuple
from multiprocessing import Pool, cpu_count
import time
import csv

def segmented_sieve(limit: int, segment_size: int = 10**6) -> Generator[int, None, None]:
    if limit < 2:
        return
    sqrt_limit = int(math.sqrt(limit)) + 1
    small_primes = []
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, sqrt_limit + 1):
        if is_prime[i]:
            small_primes.append(i)
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
    for p in small_primes:
        yield p
    low = sqrt_limit + 1
    high = min(low + segment_size, limit + 1)
    while low <= limit:
        high = min(low + segment_size, limit + 1)
        is_prime_segment = [True] * (high - low)
        for p in small_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high, p):
                is_prime_segment[j - low] = False
        for i in range(low, high):
            if is_prime_segment[i - low]:
                yield i
        low = high

def get_primes_up_to(limit: int) -> List[int]:
    return list(segmented_sieve(limit))

def get_radical(n: int) -> int:
    if n <= 1:
        return 1
    factors: Set[int] = set()
    temp = n
    i = 2
    while i * i <= temp:
        while temp % i == 0:
            factors.add(i)
            temp //= i
        i += 1
    if temp > 1:
        factors.add(temp)
    rad = 1
    for f in factors:
        rad *= f
    return rad

def get_phi(x: int) -> float:
    if x <= 1:
        return 0.0
    return math.log(get_radical(x)) / math.log(x)

def estimate_viscosity_sampling(start: int, end: int, sample_size: int = 10000) -> float:
    interval_size = end - start - 1
    if interval_size <= sample_size:
        total = sum(get_phi(x) for x in range(start + 1, end))
        return total / interval_size if interval_size > 0 else 0.0
    sample = random.sample(range(start + 1, end), min(sample_size, interval_size))
    sample_mean = sum(get_phi(x) for x in sample) / len(sample)
    return sample_mean

def estimate_viscosity_exact(start: int, end: int) -> float:
    interval_size = end - start - 1
    total = sum(get_phi(x) for x in range(start + 1, end))
    return total / interval_size if interval_size > 0 else 0.0

def count_primes_in_range(start: int, end: int, prime_set: Set[int]) -> int:
    return sum(1 for x in range(start + 1, end) if x in prime_set)

@dataclass
class IntervalResult:
    index: int
    p1: int
    p2: int
    start: int
    end: int
    length: int
    prime_count: int
    eta: float
    phi_exp: float
    tension: float
    error: float

def analyze_single_interval(args: Tuple) -> IntervalResult:
    index, p1, p2, prime_set, use_sampling = args
    start = p1 * p1
    end = p2 * p2
    length = end - start
    prime_count = count_primes_in_range(start, end, prime_set)
    if use_sampling and length > 100000:
        eta = estimate_viscosity_sampling(start, end, sample_size=10000)
    else:
        eta = estimate_viscosity_exact(start, end)
    phi_exp = (end / math.log(end)) - (start / math.log(start)) if start > 1 else 0.0
    tension = phi_exp / eta if eta > 0 else 0.0
    error = ((tension - prime_count) / prime_count * 100) if prime_count > 0 else 0.0
    return IntervalResult(index=index, p1=p1, p2=p2, start=start, end=end, length=length, prime_count=prime_count, eta=eta, phi_exp=phi_exp, tension=tension, error=error)

def analyze_intervals_parallel(max_prime: int = 10000, max_intervals: int = 1000, use_sampling: bool = True, num_workers: int = None) -> List[IntervalResult]:
    print(f"\n{'='*120}")
    print(f"PRIME DISTRIBUTION ANALYSIS - OPTIMIZED")
    print(f"{'='*120}\n")
    print(f"[1/4] Generating primes up to {max_prime:,}...")
    start_time = time.time()
    prime_list = get_primes_up_to(max_prime)
    prime_set = set(prime_list)
    prime_time = time.time() - start_time
    print(f"      Generated {len(prime_list):,} primes in {prime_time:.2f}s")
    relevant_primes = prime_list[:min(len(prime_list), max_intervals + 1)]
    args_list = [(i, relevant_primes[i], relevant_primes[i + 1], prime_set, use_sampling) for i in range(min(len(relevant_primes) - 1, max_intervals))]
    num_workers = num_workers or cpu_count()
    print(f"\n[2/4] Analyzing {len(args_list):,} intervals using {num_workers} workers...")
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(analyze_single_interval, args_list)
    analysis_time = time.time() - start_time
    print(f"      Analysis complete in {analysis_time:.2f}s")
    return results

def print_results_summary(results: List[IntervalResult], num_to_print: int = 20):
    print(f"\n[3/4] INTERVAL ANALYSIS SUMMARY")
    print(f"{'='*140}")
    print(f"{'n':<4} {'Interval':<15} {'Length':<12} {'Primes':<8} {'η':<12} {'T':<12} {'Error %':<10}")
    print(f"{'-'*140}")
    for result in results[:num_to_print]:
        interval_name = f"{result.p1}² → {result.p2}²"
        print(f"{result.index:<4} {interval_name:<15} {result.length:<12,} {result.prime_count:<8} {result.eta:<12.6f} {result.tension:<12.2f} {result.error:<10.2f}%")
    if len(results) > num_to_print:
        print(f"... ({len(results) - num_to_print} more intervals)")
    print()

def analyze_results(results: List[IntervalResult]):
    etas = [r.eta for r in results]
    errors = [r.error for r in results if r.prime_count > 0]
    print(f"\n[4/4] STATISTICAL ANALYSIS")
    print(f"{'='*100}")
    print(f"\nVISCOSITY SATURATION:")
    print(f"{'-'*100}")
    print(f"  Mean η:              {sum(etas)/len(etas):.6f}")
    print(f"  Min η:               {min(etas):.6f}")
    print(f"  Max η:               {max(etas):.6f}")
    print(f"  Final η (n={len(results)-1}):     {etas[-1]:.6f}")
    print(f"  Distance from 0.90:  {max(etas) - 0.90:+.6f}")
    print(f"\nTENSION METRIC ACCURACY:")
    print(f"{'-'*100}")
    print(f"  Mean error:          {sum(errors)/len(errors):.2f}%")
    print(f"  Max error:           {max(abs(e) for e in errors):.2f}%")
    print(f"  Min error:           {min(abs(e) for e in errors):.2f}%")
    brocard_count = sum(1 for r in results[1:] if r.prime_count >= 4)
    brocard_total = len(results) - 1
    print(f"\nBROCARD'S CONJECTURE VERIFICATION:")
    print(f"{'-'*100}")
    print(f"  Intervals with ≥ 4 primes:  {brocard_count:,}/{brocard_total:,}")
    print(f"  Success rate:                {brocard_count/brocard_total*100:.1f}%")
    if brocard_count == brocard_total:
        print(f"  ✓ BROCARD'S CONJECTURE HOLDS FOR ALL n ≥ 1")
    else:
        print(f"  ✗ BROCARD'S CONJECTURE FAILS FOR {brocard_total - brocard_count} INTERVALS")
    print(f"\n{'='*100}\n")

def export_to_csv(results: List[IntervalResult], filename: str = "analysis_results.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'p_n', 'p_{n+1}', 'start', 'end', 'length', 'primes', 'eta', 'phi_exp', 'tension', 'error_percent'])
        for r in results:
            writer.writerow([r.index, r.p1, r.p2, r.start, r.end, r.length, r.prime_count, f"{r.eta:.6f}", f"{r.phi_exp:.2f}", f"{r.tension:.2f}", f"{r.error:.2f}"])
    print(f"Results exported to {filename}")

if __name__ == "__main__":
    print("\n" + "="*120)
    print("EXAMPLE 1: SMALL ANALYSIS")
    print("="*120)
    results_small = analyze_intervals_parallel(max_prime=1000, max_intervals=100, use_sampling=False, num_workers=4)
    print_results_summary(results_small, num_to_print=15)
    analyze_results(results_small)
    
    print("\n" + "="*120)
    print("EXAMPLE 2: MEDIUM ANALYSIS")
    print("="*120)
    results_medium = analyze_intervals_parallel(max_prime=100000, max_intervals=500, use_sampling=True, num_workers=4)
    print_results_summary(results_medium, num_to_print=15)
    analyze_results(results_medium)
    
    print("\n" + "="*120)
    print("EXAMPLE 3: LARGE ANALYSIS")
    print("="*120)
    results_large = analyze_intervals_parallel(max_prime=1000000, max_intervals=1000, use_sampling=True, num_workers=4)
    print_results_summary(results_large, num_to_print=15)
    analyze_results(results_large)
    export_to_csv(results_large, "large_analysis_results.csv")
    
    print("\n" + "="*120)
    print("COMPARISON ACROSS SCALES")
    print("="*120)
    print(f"\n{'Scale':<20} {'Intervals':<15} {'Mean η':<15} {'Mean Error':<15} {'Brocard %':<15}")
    print(f"{'-'*80}")
    for name, results in [("Small", results_small), ("Medium", results_medium), ("Large", results_large)]:
        etas = [r.eta for r in results]
        errors = [r.error for r in results if r.prime_count > 0]
        brocard = sum(1 for r in results[1:] if r.prime_count >= 4) / (len(results) - 1) * 100
        print(f"{name:<20} {len(results):<15} {sum(etas)/len(etas):<15.6f} {sum(errors)/len(errors):<15.2f}% {brocard:<15.1f}%")
    print(f"\n{'='*120}\n")