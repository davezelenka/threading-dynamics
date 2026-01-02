import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def sieve_of_eratosthenes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def prime_omega(n, prime_set):
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

def compute_window_profile(center, radius, prime_set):
    c_0 = center
    composites = {}
    for k in range(-radius, radius + 1):
        n = c_0 + k
        if n > 1 and n not in prime_set:
            omega_n = prime_omega(n, prime_set)
            tau_n = divisor_count(n)
            composites[k] = {'Omega': omega_n, 'tau': tau_n}
    omega_values = [v['Omega'] for k, v in composites.items() if k != 0]
    tau_values = [v['tau'] for k, v in composites.items() if k != 0]
    if len(omega_values) == 0:
        return {}
    omega_mean = np.mean(omega_values)
    tau_mean = np.mean(tau_values)
    for k in composites:
        composites[k]['Omega_norm'] = composites[k]['Omega'] - omega_mean
        composites[k]['tau_norm'] = composites[k]['tau'] - tau_mean
    return composites

def aggregate_fingerprint(centers, radius, prime_set):
    aggregated = defaultdict(lambda: {'Omega_norm': [], 'tau_norm': []})
    for center in centers:
        profile = compute_window_profile(center, radius, prime_set)
        for k, values in profile.items():
            aggregated[k]['Omega_norm'].append(values['Omega_norm'])
            aggregated[k]['tau_norm'].append(values['tau_norm'])
    return aggregated

def plot_resonance_fingerprint(aggregated, title_str=""):
    offsets = sorted(aggregated.keys())
    omega_means = [np.mean(aggregated[k]['Omega_norm']) for k in offsets]
    omega_stderr = [np.std(aggregated[k]['Omega_norm']) / np.sqrt(len(aggregated[k]['Omega_norm'])) for k in offsets]
    tau_means = [np.mean(aggregated[k]['tau_norm']) for k in offsets]
    tau_stderr = [np.std(aggregated[k]['tau_norm']) / np.sqrt(len(aggregated[k]['tau_norm'])) for k in offsets]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.errorbar(offsets, omega_means, yerr=omega_stderr, fmt='o-', capsize=5, linewidth=2, markersize=8, color='#2E86AB', ecolor='#A23B72', label='Mean Omega*(k)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Central point (k=0)')
    ax1.fill_between(offsets, [m - s for m, s in zip(omega_means, omega_stderr)], [m + s for m, s in zip(omega_means, omega_stderr)], alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Offset k', fontsize=12)
    ax1.set_ylabel('Normalized Omega*(k)', fontsize=12)
    ax1.set_title(f'Resonance Profile: Omega(n) {title_str}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.errorbar(offsets, tau_means, yerr=tau_stderr, fmt='s-', capsize=5, linewidth=2, markersize=8, color='#F18F01', ecolor='#C73E1D', label='Mean tau*(k)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Central point (k=0)')
    ax2.fill_between(offsets, [m - s for m, s in zip(tau_means, tau_stderr)], [m + s for m, s in zip(tau_means, tau_stderr)], alpha=0.2, color='#F18F01')
    ax2.set_xlabel('Offset k', fontsize=12)
    ax2.set_ylabel('Normalized tau*(k)', fontsize=12)
    ax2.set_title(f'Resonance Profile: tau(n) {title_str}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def compute_statistics(aggregated):
    if 0 not in aggregated:
        return {}
    omega_0_mean = np.mean(aggregated[0]['Omega_norm'])
    omega_0_std = np.std(aggregated[0]['Omega_norm'])
    omega_neighbors = []
    if -2 in aggregated:
        omega_neighbors.extend(aggregated[-2]['Omega_norm'])
    if 2 in aggregated:
        omega_neighbors.extend(aggregated[2]['Omega_norm'])
    omega_neighbor_mean = np.mean(omega_neighbors) if omega_neighbors else 0
    if omega_neighbors:
        pooled_std = np.sqrt((omega_0_std**2 + np.std(omega_neighbors)**2) / 2)
        cohens_d = (omega_0_mean - omega_neighbor_mean) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0
    return {'central_mean': omega_0_mean, 'central_std': omega_0_std, 'neighbor_mean': omega_neighbor_mean, 'effect_size': cohens_d, 'n_samples': len(aggregated[0]['Omega_norm'])}

def main():
    print("=" * 70)
    print("CONTROL A: RANDOM COMPOSITE FINGERPRINT TEST")
    print("=" * 70)
    
    LIMIT = 100000
    RADIUS = 10
    N_SAMPLES = 1224
    
    print(f"\nParameters:")
    print(f"  Limit: {LIMIT:,}")
    print(f"  Radius: +/-{RADIUS}")
    print(f"  Sample size: {N_SAMPLES}")
    
    print(f"\n[1/4] Generating primes...")
    primes = sieve_of_eratosthenes(LIMIT)
    prime_set = set(primes)
    print(f"  Found {len(primes):,} primes")
    
    print(f"\n[2/4] Generating random composite centers...")
    random_centers = []
    attempts = 0
    max_attempts = 10000
    while len(random_centers) < N_SAMPLES and attempts < max_attempts:
        candidate = 2 * random.randint(2, LIMIT // 2 - 1)
        if candidate not in prime_set and candidate > 1:
            random_centers.append(candidate)
        attempts += 1
    print(f"  Generated {len(random_centers)} random composite centers")
    print(f"  Sample range: {min(random_centers)} to {max(random_centers)}")
    
    print(f"\n[3/4] Computing resonance profiles for random composites...")
    aggregated_random = aggregate_fingerprint(random_centers, RADIUS, prime_set)
    print(f"  Aggregated {len(aggregated_random)} offset positions")
    
    print(f"\n[4/4] Computing statistics...")
    stats_random = compute_statistics(aggregated_random)
    print(f"\n  RANDOM COMPOSITE STATISTICS:")
    print(f"  --------------------------------")
    print(f"  Central Omega*(0):     {stats_random['central_mean']:+.4f} +/- {stats_random['central_std']:.4f}")
    print(f"  Neighbor Omega*(+/-2): {stats_random['neighbor_mean']:+.4f}")
    print(f"  Effect size (d):       {stats_random['effect_size']:+.4f}")
    print(f"  Sample size:           {stats_random['n_samples']:,}")
    
    print(f"\n  INTERPRETATION:")
    if abs(stats_random['central_mean']) < 0.2 and abs(stats_random['effect_size']) < 0.2:
        print(f"  [FLAT PROFILE - EXPECTED]")
        print(f"     Random composites show NO resonance structure")
        print(f"     This CONFIRMS the twin prime signal is REAL")
    elif abs(stats_random['central_mean']) < 0.5:
        print(f"  [WEAK SIGNAL]")
        print(f"     Random composites show minimal structure")
        print(f"     Twin prime signal is SIGNIFICANTLY STRONGER")
    else:
        print(f"  [UNEXPECTED SIGNAL]")
        print(f"     Random composites show structure")
        print(f"     May indicate artifact in normalization")
    
    print(f"\n[5/4] Generating visualization...")
    fig = plot_resonance_fingerprint(aggregated_random, f"(Random Composites, n={N_SAMPLES})")
    plt.savefig('control_a_random_composites.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: control_a_random_composites.png")
    plt.show()
    
    print(f"\n" + "=" * 70)
    print("COMPARISON: TWIN PRIMES vs RANDOM COMPOSITES")
    print("=" * 70)
    print(f"\nTwin Primes:")
    print(f"  Central Omega*(0):     +1.6782")
    print(f"  Effect size (d):       +0.7815")
    print(f"\nRandom Composites:")
    print(f"  Central Omega*(0):     {stats_random['central_mean']:+.4f}")
    print(f"  Effect size (d):       {stats_random['effect_size']:+.4f}")
    print(f"\nDifference:")
    print(f"  Delta Omega*(0):       {1.6782 - stats_random['central_mean']:+.4f}")
    print(f"  Delta effect size:     {0.7815 - stats_random['effect_size']:+.4f}")
    
    if stats_random['central_mean'] < 0.3:
        print(f"\n✓ CONTROL A PASSED: Twin prime resonance is REAL, not an artifact")
    else:
        print(f"\n✗ CONTROL A FAILED: Random composites show unexpected structure")
    
    print(f"\n" + "=" * 70)

if __name__ == "__main__":
    main()