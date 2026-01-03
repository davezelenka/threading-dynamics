import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def sieve_of_eratosthenes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def prime_omega(n):
    """Count total number of prime factors (with multiplicity)"""
    if n <= 1:
        return 0
    count = 0
    for p in range(2, int(np.sqrt(n)) + 1):
        while n % p == 0:
            count += 1
            n //= p
        if n == 1:
            break
    if n > 1:
        count += 1
    return count

def divisor_count(n):
    if n <= 1:
        return 1
    count = 0
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def find_twin_primes(primes):
    twins = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 2:
            twins.append((primes[i], primes[i+1]))
    return twins

def compute_window_profile(center, radius, prime_set):
    """
    Compute profile around a center point.
    For each position k, record whether it's prime or composite,
    and if composite, compute Ω(n) = number of prime factors.
    """
    profile = {}

    for k in range(-radius, radius + 1):
        n = center + k
        if n > 1 and n not in prime_set:
            omega_n = prime_omega(n)
            tau_n = divisor_count(n)
            profile[k] = {'Omega': omega_n, 'tau': tau_n, 'is_prime': False, 'n': n}
        else:
            profile[k] = {'Omega': None, 'tau': None, 'is_prime': True, 'n': n}

    return profile

def main():
    print("\n" + "="*70)
    print("SINGLE TWIN PRIME LANDSCAPE ANALYSIS")
    print("Sampling one twin prime far out on the number line")
    print("="*70)

    # ========== ADJUSTABLE PARAMETERS ==========
    RADIUS = 100                  # Window radius around twin prime
    ROLLING_WINDOW = 10           # Rolling window size (1 = just that point, 10 = average 10 points around it)
    TWIN_PRIME_INDEX = -3000          # Which twin prime to select (-1 = last, 0 = first, etc.)
    # ===========================================
    
    # Generate primes up to 1 million
    limit = 1000000
    print(f"\n[1/4] Generating primes up to {limit:,}...")
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    print(f"  Found {len(primes):,} primes")

    # Find twin primes
    print(f"[2/4] Finding twin primes...")
    twin_primes = find_twin_primes(primes)
    print(f"  Found {len(twin_primes):,} twin prime pairs")

    # Pick a twin prime based on TWIN_PRIME_INDEX
    selected_pair = twin_primes[TWIN_PRIME_INDEX]
    p_a, p_b = selected_pair
    center = p_a + (p_b - p_a) // 2

    print(f"\n[3/4] Selected twin prime pair (index {TWIN_PRIME_INDEX}): ({p_a}, {p_b})")
    print(f"  Center: {center}")
    print(f"  Gap: {p_b - p_a}")

    # Compute profile with adjustable radius
    print(f"\n[4/4] Computing profile with R={RADIUS}...")
    profile = compute_window_profile(center, RADIUS, prime_set)

    # Extract data for plotting - Ω(n) for composites, 0 for primes
    k_values = sorted([k for k in profile.keys()])
    omega_values = []
    is_prime_values = []

    for k in k_values:
        is_prime = profile[k]['is_prime']
        is_prime_values.append(is_prime)
        if is_prime:
            omega_values.append(0)  # Primes at 0
        else:
            omega_values.append(profile[k]['Omega'])  # Actual Ω(n) value

    # Count composites and primes
    num_composites = sum(1 for is_p in is_prime_values if not is_p)
    num_primes = len(k_values) - num_composites

    # Analyze Ω values
    omega_composites = [profile[k]['Omega'] for k in k_values if not profile[k]['is_prime']]

    print(f"\n{'='*70}")
    print("LANDSCAPE STATISTICS")
    print(f"{'='*70}")
    print(f"\nWindow: [{center - RADIUS}, {center + RADIUS}]")
    print(f"Total positions: {len(k_values)}")
    print(f"Composites: {num_composites}")
    print(f"Primes: {num_primes}")
    print(f"Composite density: {100*num_composites/len(k_values):.1f}%")

    if omega_composites:
        print(f"\nΩ(n) Statistics (composites only):")
        print(f"  Mean: {np.mean(omega_composites):.2f}")
        print(f"  Std:  {np.std(omega_composites):.2f}")
        print(f"  Min:  {np.min(omega_composites)}")
        print(f"  Max:  {np.max(omega_composites)}")

    # Analyze by region
    print(f"\n{'='*70}")
    print("REGIONAL ANALYSIS")
    print(f"{'='*70}")

    # Adjust regions based on RADIUS
    if RADIUS >= 1000:
        regions = [
            (0, 100, "Very Near (|k| ≤ 100)"),
            (100, 250, "Near (100 < |k| ≤ 250)"),
            (250, 500, "Mid (250 < |k| ≤ 500)"),
            (500, 1000, "Far (500 < |k| ≤ 1000)")
        ]
    else:
        # For smaller radii, use proportional regions
        regions = [
            (0, RADIUS//4, f"Very Near (|k| ≤ {RADIUS//4})"),
            (RADIUS//4, RADIUS//2, f"Near ({RADIUS//4} < |k| ≤ {RADIUS//2})"),
            (RADIUS//2, RADIUS, f"Far ({RADIUS//2} < |k| ≤ {RADIUS})")
        ]

    for r_min, r_max, label in regions:
        omega_region = [profile[k]['Omega'] for k in k_values 
                       if r_min < abs(k) <= r_max and not profile[k]['is_prime']]
        composites_region = sum(1 for k in k_values if r_min < abs(k) <= r_max and not profile[k]['is_prime'])
        primes_region = sum(1 for k in k_values if r_min < abs(k) <= r_max and profile[k]['is_prime'])
        total_region = composites_region + primes_region

        if total_region > 0:
            print(f"\n{label}:")
            print(f"  Composites: {composites_region}, Primes: {primes_region}")
            print(f"  Composite density: {100*composites_region/total_region:.1f}%")
            if omega_region:
                print(f"  Ω(n) mean: {np.mean(omega_region):.2f}")

    # Create visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Twin Prime Landscape: ({p_a}, {p_b}) with R={RADIUS}', fontsize=16, fontweight='bold')

    # Top plot: Ω(n) values - number of prime factors
    ax_top = axes[0]
    
    # Separate composites and primes for different markers
    k_composites = [k for k, is_p in zip(k_values, is_prime_values) if not is_p]
    omega_composites_plot = [omega for omega, is_p in zip(omega_values, is_prime_values) if not is_p]
    k_primes = [k for k, is_p in zip(k_values, is_prime_values) if is_p]
    omega_primes_plot = [0 for _ in k_primes]  # Primes at y=0
    
    # Plot composites with blue dots
    ax_top.scatter(k_composites, omega_composites_plot, c='blue', alpha=0.6, s=20, label='Composites (Ω value)', marker='o')
    # Plot primes with LARGER STAR markers at y=0
    ax_top.scatter(k_primes, omega_primes_plot, c='red', alpha=0.8, s=100, marker='*', 
                   label='Primes', zorder=5, edgecolors='darkred', linewidths=0.5)
    
    ax_top.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_top.axvline(0, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Twin prime center')
    ax_top.set_xlabel('Position k', fontsize=12)
    ax_top.set_ylabel('Ω(n) = Number of Prime Factors', fontsize=12)
    ax_top.set_title('Prime Factor Count Ω(n) - Blue dots=Composites, Red stars=Primes at y=0', fontsize=12, fontweight='bold')
    ax_top.grid(True, alpha=0.3)
    ax_top.set_xlim(-RADIUS, RADIUS)
    # Set y-axis limits AFTER plotting to ensure all points are visible
    if omega_composites:
        y_max = max(omega_composites) * 1.15
    else:
        y_max = 5
    ax_top.set_ylim(-0.5, y_max)
    ax_top.legend(loc='upper right')

    # Bottom plot: Rolling average of Ω(n) for composites in window
    ax_bottom = axes[1]
    
    # Compute rolling average of Ω(n) values
    k_centers = []
    rolling_omega_avg = []
    
    for k in range(-RADIUS, RADIUS + 1):
        # Get Ω values in window [k - ROLLING_WINDOW//2, k + ROLLING_WINDOW//2]
        window_start = max(-RADIUS, k - ROLLING_WINDOW//2)
        window_end = min(RADIUS, k + ROLLING_WINDOW//2)
        
        omega_in_window = [profile[kk]['Omega'] for kk in range(window_start, window_end + 1)
                          if kk in profile and not profile[kk]['is_prime']]
        
        # Average the Ω values (or 0 if no composites in window)
        avg_omega = np.mean(omega_in_window) if omega_in_window else 0
        
        k_centers.append(k)
        rolling_omega_avg.append(avg_omega)
    
    ax_bottom.plot(k_centers, rolling_omega_avg, color='green', linewidth=2, alpha=0.7, label=f'Avg Ω(n) in window (size={ROLLING_WINDOW})')
    ax_bottom.fill_between(k_centers, rolling_omega_avg, alpha=0.3, color='green')
    ax_bottom.axvline(0, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Twin prime center')
    ax_bottom.set_xlabel('Position k', fontsize=12)
    ax_bottom.set_ylabel('Average Ω(n)', fontsize=12)
    ax_bottom.set_title(f'Rolling Average Prime Factor Count (window={ROLLING_WINDOW})', fontsize=12, fontweight='bold')
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_xlim(-RADIUS, RADIUS)
    # Set y-axis limits AFTER plotting
    if rolling_omega_avg:
        y_max_bottom = max(rolling_omega_avg) * 1.15
    else:
        y_max_bottom = 5
    ax_bottom.set_ylim(0, y_max_bottom)
    ax_bottom.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('single_twin_prime_landscape.png', dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Saved to: single_twin_prime_landscape.png")
    plt.show()

    # Print sample of the landscape
    print(f"\n{'='*70}")
    print(f"SAMPLE OF LANDSCAPE (k = {max(-50, -RADIUS)} to {min(50, RADIUS)})")
    print(f"{'='*70}")
    print(f"\n{'k':>5} {'n':>8} {'Type':>12} {'Ω(n)':>6} {'τ(n)':>6}")
    print("-" * 45)

    sample_range = min(50, RADIUS)
    for k in range(-sample_range, sample_range + 1, max(1, sample_range//10)):
        n = profile[k]['n']
        is_p = profile[k]['is_prime']
        omega = profile[k]['Omega']
        tau = profile[k]['tau']

        if is_p:
            print(f"{k:>5} {n:>8} {'PRIME':>12} {'-':>6} {'-':>6}")
        else:
            print(f"{k:>5} {n:>8} {'composite':>12} {omega:>6} {tau:>6}")

    print(f"\n" + "="*70)
    print(f"\nADJUSTABLE PARAMETERS USED:")
    print(f"  RADIUS = {RADIUS}")
    print(f"  ROLLING_WINDOW = {ROLLING_WINDOW}")
    print(f"  TWIN_PRIME_INDEX = {TWIN_PRIME_INDEX} (selected pair: {selected_pair})")
    print(f"\nTo select different twin primes:")
    print(f"  Use TWIN_PRIME_INDEX = 0 for first twin prime (3, 5)")
    print(f"  Use TWIN_PRIME_INDEX = -1 for last twin prime")
    print(f"  Use TWIN_PRIME_INDEX = {len(twin_primes)//2} for middle twin prime")
    print(f"  Total twin primes available: {len(twin_primes)}")
    print("\nNote: Ω(n) = total number of prime factors with multiplicity")
    print("      e.g., Ω(12) = Ω(2²×3) = 3, Ω(30) = Ω(2×3×5) = 3")
    print("\nROLLING_WINDOW explanation:")
    print("      ROLLING_WINDOW = 1: shows just that point (like top graph but connected)")
    print("      ROLLING_WINDOW = 10: averages the Ω values of composites in a 10-point window")
    print("="*70)

if __name__ == "__main__":
    main()