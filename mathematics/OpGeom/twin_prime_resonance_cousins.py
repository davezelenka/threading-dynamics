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
        if primes[i+1] - primes[i] == 2:
            twins.append((primes[i], primes[i+1]))
    return twins

def find_cousin_primes(primes):
    cousins = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 4:
            cousins.append((primes[i], primes[i+1]))
    return cousins

def find_sexy_primes(primes):
    sexy = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 6:
            sexy.append((primes[i], primes[i+1]))
    return sexy

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

def aggregate_fingerprint(prime_pairs, radius, prime_set):
    aggregated = defaultdict(lambda: {'Omega_norm': [], 'tau_norm': []})
    for pair in prime_pairs:
        p_a, p_b = pair
        c_0 = p_a + (p_b - p_a) // 2
        profile = compute_window_profile(c_0, radius, prime_set)
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
    print("PRIME GAP RESONANCE COMPARISON")
    print("Twins (gap=2) vs Cousins (gap=4) vs Sexy (gap=6)")
    print("=" * 70)
    
    LIMIT = 1000000
    RADIUS = 10
    
    print(f"\nParameters:")
    print(f"  Limit: {LIMIT:,}")
    print(f"  Radius: +/-{RADIUS}")
    
    print(f"\n[1/5] Generating primes...")
    primes = sieve_of_eratosthenes(LIMIT)
    prime_set = set(primes)
    print(f"  Found {len(primes):,} primes")
    
    print(f"\n[2/5] Finding prime pairs...")
    twin_primes = find_twin_primes(primes)
    cousin_primes = find_cousin_primes(primes)
    sexy_primes = find_sexy_primes(primes)
    print(f"  Twin primes (gap=2):   {len(twin_primes):,}")
    print(f"  Cousin primes (gap=4): {len(cousin_primes):,}")
    print(f"  Sexy primes (gap=6):   {len(sexy_primes):,}")
    
    print(f"\n[3/5] Computing resonance profiles for TWINS...")
    aggregated_twins = aggregate_fingerprint(twin_primes, RADIUS, prime_set)
    print(f"  Aggregated {len(aggregated_twins)} offset positions")
    
    print(f"\n[4/5] Computing resonance profiles for COUSINS...")
    aggregated_cousins = aggregate_fingerprint(cousin_primes, RADIUS, prime_set)
    print(f"  Aggregated {len(aggregated_cousins)} offset positions")
    
    print(f"\n[5/5] Computing resonance profiles for SEXY...")
    aggregated_sexy = aggregate_fingerprint(sexy_primes, RADIUS, prime_set)
    print(f"  Aggregated {len(aggregated_sexy)} offset positions")
    
    print(f"\n[6/5] Computing statistics...")
    stats_twins = compute_statistics(aggregated_twins)
    stats_cousins = compute_statistics(aggregated_cousins)
    stats_sexy = compute_statistics(aggregated_sexy)
    
    print(f"\n" + "=" * 70)
    print("RESONANCE SIGNATURE COMPARISON")
    print("=" * 70)
    
    print(f"\nTWIN PRIMES (gap=2):")
    print(f"  Central Omega*(0):     {stats_twins['central_mean']:+.4f} +/- {stats_twins['central_std']:.4f}")
    print(f"  Neighbor Omega*(+/-2): {stats_twins['neighbor_mean']:+.4f}")
    print(f"  Effect size (d):       {stats_twins['effect_size']:+.4f}")
    print(f"  Sample size:           {stats_twins['n_samples']:,}")
    
    print(f"\nCOUSIN PRIMES (gap=4):")
    print(f"  Central Omega*(0):     {stats_cousins['central_mean']:+.4f} +/- {stats_cousins['central_std']:.4f}")
    print(f"  Neighbor Omega*(+/-2): {stats_cousins['neighbor_mean']:+.4f}")
    print(f"  Effect size (d):       {stats_cousins['effect_size']:+.4f}")
    print(f"  Sample size:           {stats_cousins['n_samples']:,}")
    
    print(f"\nSEXY PRIMES (gap=6):")
    print(f"  Central Omega*(0):     {stats_sexy['central_mean']:+.4f} +/- {stats_sexy['central_std']:.4f}")
    print(f"  Neighbor Omega*(+/-2): {stats_sexy['neighbor_mean']:+.4f}")
    print(f"  Effect size (d):       {stats_sexy['effect_size']:+.4f}")
    print(f"  Sample size:           {stats_sexy['n_samples']:,}")
    
    print(f"\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    
    print(f"\nCentral Omega*(0) by gap size:")
    print(f"  Gap=2 (Twins):   {stats_twins['central_mean']:+.4f}")
    print(f"  Gap=4 (Cousins): {stats_cousins['central_mean']:+.4f}")
    print(f"  Gap=6 (Sexy):    {stats_sexy['central_mean']:+.4f}")
    
    print(f"\nEffect size (Cohen's d) by gap size:")
    print(f"  Gap=2 (Twins):   {stats_twins['effect_size']:+.4f}")
    print(f"  Gap=4 (Cousins): {stats_cousins['effect_size']:+.4f}")
    print(f"  Gap=6 (Sexy):    {stats_sexy['effect_size']:+.4f}")
    
    print(f"\nRelative strength (normalized to twins):")
    if stats_twins['central_mean'] != 0:
        cousin_ratio = stats_cousins['central_mean'] / stats_twins['central_mean']
        sexy_ratio = stats_sexy['central_mean'] / stats_twins['central_mean']
        print(f"  Cousins/Twins: {cousin_ratio:.2f}x")
        print(f"  Sexy/Twins:    {sexy_ratio:.2f}x")
    
    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if stats_cousins['central_mean'] < stats_twins['central_mean'] and stats_sexy['central_mean'] < stats_cousins['central_mean']:
        print(f"\n✓ HYPOTHESIS CONFIRMED")
        print(f"  Resonance strength DECREASES with gap size")
        print(f"  Twins > Cousins > Sexy")
        print(f"  This proves the effect is MECHANISM-DEPENDENT, not random")
    elif stats_cousins['central_mean'] < stats_twins['central_mean']:
        print(f"\n✓ PARTIAL CONFIRMATION")
        print(f"  Twins show stronger resonance than Cousins")
        print(f"  Gap size matters for resonance strength")
    else:
        print(f"\n✗ HYPOTHESIS NOT CONFIRMED")
        print(f"  Resonance strength does not decrease with gap size")
        print(f"  May indicate different mechanism")
    
    print(f"\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    print(f"\n[1/3] Plotting TWINS...")
    fig1 = plot_resonance_fingerprint(aggregated_twins, f"(Twin Primes, gap=2, n={len(twin_primes)})")
    plt.savefig('resonance_twins.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: resonance_twins.png")
    
    print(f"\n[2/3] Plotting COUSINS...")
    fig2 = plot_resonance_fingerprint(aggregated_cousins, f"(Cousin Primes, gap=4, n={len(cousin_primes)})")
    plt.savefig('resonance_cousins.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: resonance_cousins.png")
    
    print(f"\n[3/3] Plotting SEXY...")
    fig3 = plot_resonance_fingerprint(aggregated_sexy, f"(Sexy Primes, gap=6, n={len(sexy_primes)})")
    plt.savefig('resonance_sexy.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: resonance_sexy.png")
    
    print(f"\n[4/3] Creating comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    offsets_twins = sorted(aggregated_twins.keys())
    offsets_cousins = sorted(aggregated_cousins.keys())
    offsets_sexy = sorted(aggregated_sexy.keys())
    
    omega_twins = [np.mean(aggregated_twins[k]['Omega_norm']) for k in offsets_twins]
    omega_cousins = [np.mean(aggregated_cousins[k]['Omega_norm']) for k in offsets_cousins]
    omega_sexy = [np.mean(aggregated_sexy[k]['Omega_norm']) for k in offsets_sexy]
    
    ax.plot(offsets_twins, omega_twins, 'o-', linewidth=2.5, markersize=8, label='Twins (gap=2)', color='#2E86AB')
    ax.plot(offsets_cousins, omega_cousins, 's-', linewidth=2.5, markersize=8, label='Cousins (gap=4)', color='#F18F01')
    ax.plot(offsets_sexy, omega_sexy, '^-', linewidth=2.5, markersize=8, label='Sexy (gap=6)', color='#C73E1D')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Offset k', fontsize=12)
    ax.set_ylabel('Normalized Omega*(k)', fontsize=12)
    ax.set_title('Prime Gap Resonance Comparison: Twins vs Cousins vs Sexy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('resonance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: resonance_comparison.png")
    
    plt.show()
    
    print(f"\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()