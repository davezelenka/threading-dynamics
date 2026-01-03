
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

def find_twin_primes(primes):
    twins = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 2: #2 for twins 4 for cousins
            twins.append((primes[i], primes[i+1]))
    return twins

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

def compute_window_profile(twin_pair, radius, prime_set):
    p_a, p_b = twin_pair
    c_0 = p_a + 1
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

def aggregate_fingerprint(twin_primes, radius, prime_set):
    aggregated = defaultdict(lambda: {'Omega_norm': [], 'tau_norm': []})
    for twin in twin_primes:
        profile = compute_window_profile(twin, radius, prime_set)
        for k, values in profile.items():
            aggregated[k]['Omega_norm'].append(values['Omega_norm'])
            aggregated[k]['tau_norm'].append(values['tau_norm'])
    return aggregated

def plot_resonance_fingerprint(aggregated, radius, title_suffix=""):
    offsets = sorted(aggregated.keys())
    omega_means = [np.mean(aggregated[k]['Omega_norm']) for k in offsets]
    omega_stderr = [np.std(aggregated[k]['Omega_norm']) / np.sqrt(len(aggregated[k]['Omega_norm'])) for k in offsets]
    tau_means = [np.mean(aggregated[k]['tau_norm']) for k in offsets]
    tau_stderr = [np.std(aggregated[k]['tau_norm']) / np.sqrt(len(aggregated[k]['tau_norm'])) for k in offsets]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.errorbar(offsets, omega_means, yerr=omega_stderr, fmt='o-', capsize=5, linewidth=2, markersize=8, color='#2E86AB', ecolor='#A23B72', label='Mean Omega*(k)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Central barrier (k=0)')
    ax1.fill_between(offsets, [m - s for m, s in zip(omega_means, omega_stderr)], [m + s for m, s in zip(omega_means, omega_stderr)], alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Offset k from central composite (p+1)', fontsize=12)
    ax1.set_ylabel('Normalized Omega*(k)', fontsize=12)
    ax1.set_title(f'Operational Resonance: Omega(n) Profile{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.errorbar(offsets, tau_means, yerr=tau_stderr, fmt='s-', capsize=5, linewidth=2, markersize=8, color='#F18F01', ecolor='#C73E1D', label='Mean tau*(k)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Central barrier (k=0)')
    ax2.fill_between(offsets, [m - s for m, s in zip(tau_means, tau_stderr)], [m + s for m, s in zip(tau_means, tau_stderr)], alpha=0.2, color='#F18F01')
    ax2.set_xlabel('Offset k from central composite (p+1)', fontsize=12)
    ax2.set_ylabel('Normalized tau*(k)', fontsize=12)
    ax2.set_title(f'Operational Resonance: tau(n) Profile{title_suffix}', fontsize=14, fontweight='bold')
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
    print("TWIN PRIME OPERATIONAL RESONANCE FINGERPRINT")
    print("=" * 70)
    LIMIT = 100000
    RADIUS = 10
    print(f"\nParameters: Limit={LIMIT:,}, Radius=+/-{RADIUS}")
    print(f"\n[1/5] Generating primes...")
    primes = sieve_of_eratosthenes(LIMIT)
    prime_set = set(primes)
    print(f"  Found {len(primes):,} primes")
    print(f"\n[2/5] Finding twin primes...")
    twin_primes = find_twin_primes(primes)
    print(f"  Found {len(twin_primes):,} twin prime pairs")
    print(f"\n[3/5] Computing resonance profiles...")
    aggregated = aggregate_fingerprint(twin_primes, RADIUS, prime_set)
    print(f"  Aggregated {len(aggregated)} offset positions")
    print(f"\n[4/5] Computing statistics...")
    stats = compute_statistics(aggregated)
    print(f"\n  RESONANCE SIGNATURE:")
    print(f"  Central Omega*(0):     {stats['central_mean']:+.4f} +/- {stats['central_std']:.4f}")
    print(f"  Neighbor Omega*(+/-1): {stats['neighbor_mean']:+.4f}")
    print(f"  Effect size (d):       {stats['effect_size']:+.4f}")
    print(f"  Sample size:           {stats['n_samples']:,}")
    if stats['central_mean'] > 0.3 and stats['effect_size'] > 0.3:
        print(f"\n  STRONG RESONANCE DETECTED")
    elif stats['central_mean'] > 0.1:
        print(f"\n  WEAK RESONANCE")
    else:
        print(f"\n  NO CLEAR RESONANCE")
    print(f"\n[5/5] Generating visualization...")
    fig = plot_resonance_fingerprint(aggregated, RADIUS, f"\n(n={len(twin_primes):,} twins)")
    plt.savefig('twin_prime_resonance.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: twin_prime_resonance.png")
    plt.show()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
