
import math
from dataclasses import dataclass
from typing import List, Set


# ----------------------------
# Prime generation (sieve)
# ----------------------------

def generate_primes(limit: int = 1000000):
    sieve = [True] * limit
    sieve[0] = sieve[1] = False

    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit, i):
                sieve[j] = False

    prime_list = [i for i in range(2, limit) if sieve[i]]
    prime_set = set(prime_list)

    return prime_list, prime_set


# ----------------------------
# Arithmetic functions
# ----------------------------

def get_radical(n: int) -> int:
    """rad(n): product of distinct prime factors"""
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
    """Φ(x) = ln(rad(x)) / ln(x)"""
    if x <= 1:
        return 0.0
    return math.log(get_radical(x)) / math.log(x)


def li_approx(x: int) -> float:
    """Logarithmic integral approximation: x / ln(x)"""
    if x <= 1:
        return 0.0
    return x / math.log(x)


def li_integral_approx(x: int) -> float:
    """More accurate Li(x) with first-order correction"""
    if x <= 1:
        return 0.0
    ln_x = math.log(x)
    return x / ln_x * (1 + 1 / ln_x)


# ----------------------------
# Data container
# ----------------------------

@dataclass
class IntervalData:
    """Container for interval analysis results"""
    index: int
    p1: int
    p2: int
    start: int
    end: int
    length: int
    prime_count: int
    composite_count: int
    
    # Viscosity methods
    eta_arithmetic: float      # Mean of all Φ(x)
    eta_composites: float      # Mean of Φ(x) for composites only
    eta_ratio: float           # Σln(rad(x)) / Σln(x)
    
    # Expansion potential
    phi_exp_pnt: float         # Using PNT: x/ln(x)
    phi_exp_li: float          # Using Li(x) with correction
    
    # Tension (using arithmetic mean η)
    tension_pnt: float         # T = Φ_exp_pnt / η_arithmetic
    tension_li: float          # T = Φ_exp_li / η_arithmetic
    
    # Error metrics
    error_pnt: float           # (T_pnt - prime_count) / prime_count * 100
    error_li: float            # (T_li - prime_count) / prime_count * 100


# ----------------------------
# Interval analysis
# ----------------------------

def analyze_intervals(max_prime: int = 600, max_intervals: int = 100) -> List[IntervalData]:
    """
    Analyze intervals [p_n^2, p_{n+1}^2] for n up to max_intervals.
    
    Args:
        max_prime: Upper limit for prime generation
        max_intervals: Maximum number of intervals to analyze
    
    Returns:
        List of IntervalData objects
    """
    prime_list, prime_set = generate_primes(100000)
    relevant_primes = [p for p in prime_list if p <= max_prime]

    intervals: List[IntervalData] = []

    for i in range(min(len(relevant_primes) - 1, max_intervals)):
        p1 = relevant_primes[i]
        p2 = relevant_primes[i + 1]
        start = p1 * p1
        end = p2 * p2
        length = end - start

        # Accumulators
        sum_phi_all = 0.0
        sum_phi_composites = 0.0
        sum_ln_rad = 0.0
        sum_ln_x = 0.0
        prime_count = 0
        composite_count = 0

        # Iterate through interval
        for x in range(start + 1, end):
            phi_x = get_phi(x)
            ln_x = math.log(x)
            ln_rad_x = math.log(get_radical(x))

            sum_phi_all += phi_x
            sum_ln_rad += ln_rad_x
            sum_ln_x += ln_x

            if x in prime_set:
                prime_count += 1
            else:
                sum_phi_composites += phi_x
                composite_count += 1

        # Calculate viscosity using different methods
        total_count = length - 1
        eta_arithmetic = sum_phi_all / total_count if total_count > 0 else 0.0
        eta_composites = sum_phi_composites / composite_count if composite_count > 0 else 0.0
        eta_ratio = sum_ln_rad / sum_ln_x if sum_ln_x > 0 else 0.0

        # Calculate expansion potential
        phi_exp_pnt = li_approx(end) - li_approx(start)
        phi_exp_li = li_integral_approx(end) - li_integral_approx(start)

        # Calculate tension
        tension_pnt = phi_exp_pnt / eta_arithmetic if eta_arithmetic > 0 else 0.0
        tension_li = phi_exp_li / eta_arithmetic if eta_arithmetic > 0 else 0.0

        # Calculate errors
        error_pnt = ((tension_pnt - prime_count) / prime_count * 100) if prime_count > 0 else 0.0
        error_li = ((tension_li - prime_count) / prime_count * 100) if prime_count > 0 else 0.0

        intervals.append(
            IntervalData(
                index=i,
                p1=p1,
                p2=p2,
                start=start,
                end=end,
                length=length,
                prime_count=prime_count,
                composite_count=composite_count,
                eta_arithmetic=eta_arithmetic,
                eta_composites=eta_composites,
                eta_ratio=eta_ratio,
                phi_exp_pnt=phi_exp_pnt,
                phi_exp_li=phi_exp_li,
                tension_pnt=tension_pnt,
                tension_li=tension_li,
                error_pnt=error_pnt,
                error_li=error_li,
            )
        )

    return intervals


# ----------------------------
# Analysis and reporting
# ----------------------------

def print_summary(intervals: List[IntervalData], num_to_print: int = 15):
    """Print summary table of first N intervals"""
    print("\nINTERVAL ANALYSIS SUMMARY")
    print("="*140)
    print(f"{'n':<4} {'Interval':<15} {'Length':<8} {'Primes':<8} {'η (Arith)':<12} {'Φ_exp':<10} {'T (PNT)':<10} {'Error %':<10}")
    print("-"*140)

    for interval in intervals[:num_to_print]:
        interval_name = f"{interval.p1}² → {interval.p2}²"
        print(f"{interval.index:<4} {interval_name:<15} {interval.length:<8} {interval.prime_count:<8} "
              f"{interval.eta_arithmetic:<12.6f} {interval.phi_exp_pnt:<10.2f} {interval.tension_pnt:<10.2f} "
              f"{interval.error_pnt:<10.2f}%")

    print()


def analyze_viscosity_saturation(intervals: List[IntervalData]):
    """Test if η(n) → 0.90"""
    print("\nVISCOSITY SATURATION ANALYSIS")
    print("="*100)
    
    etas = [i.eta_arithmetic for i in intervals]
    
    print(f"Mean η: {sum(etas)/len(etas):.6f}")
    print(f"Min η: {min(etas):.6f}")
    print(f"Max η: {max(etas):.6f}")
    print(f"Final η (n={len(intervals)-1}): {etas[-1]:.6f}")
    print(f"Distance from 0.90: {0.90 - etas[-1]:.6f}")
    print()


def analyze_tension_correlation(intervals: List[IntervalData]):
    """Test T ≈ π correlation"""
    print("\nTENSION-TO-PRIME CORRELATION ANALYSIS")
    print("="*100)
    
    errors_pnt = [i.error_pnt for i in intervals if i.prime_count > 0]
    errors_li = [i.error_li for i in intervals if i.prime_count > 0]

    mean_error_pnt = sum(errors_pnt) / len(errors_pnt) if errors_pnt else 0
    mean_error_li = sum(errors_li) / len(errors_li) if errors_li else 0
    
    max_error_pnt = max(abs(e) for e in errors_pnt) if errors_pnt else 0
    max_error_li = max(abs(e) for e in errors_li) if errors_li else 0

    print(f"Using PNT approximation (Φ_exp = x/ln(x)):")
    print(f"  Mean absolute error: {mean_error_pnt:.2f}%")
    print(f"  Max absolute error: {max_error_pnt:.2f}%")
    print()
    
    print(f"Using Li(x) approximation (more accurate):")
    print(f"  Mean absolute error: {mean_error_li:.2f}%")
    print(f"  Max absolute error: {max_error_li:.2f}%")
    print()

    brocard = sum(1 for i in intervals[1:] if i.prime_count >= 4)
    brocard_total = len(intervals) - 1
    print(f"Brocard's Conjecture verification (n > 1):")
    print(f"  Intervals with ≥ 4 primes: {brocard}/{brocard_total}")
    print(f"  Conjecture holds: {'✓ YES' if brocard == brocard_total else '✗ NO'}")
    print()


# ----------------------------
# Main execution
# ----------------------------

if __name__ == "__main__":
    data = analyze_intervals(max_prime=600, max_intervals=100)

    print_summary(data, num_to_print=15)
    analyze_viscosity_saturation(data)
    analyze_tension_correlation(data)

    print(f"Analysis complete: {len(data)} intervals analyzed")
