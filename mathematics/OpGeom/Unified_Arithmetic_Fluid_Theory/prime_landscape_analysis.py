import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

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

def prime_omega(n):
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
            omega_n = prime_omega(n)
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

def compute_landscape_metrics(aggregated):
    """
    Compute gradient, curvature, and asymmetry of the complexity landscape.
    Returns: dict with landscape features
    """
    k_values = sorted([k for k in aggregated.keys()])
    if len(k_values) < 3:
        return {}
    
    omega_means = {k: np.mean(aggregated[k]['Omega_norm']) for k in k_values}
    
    # Gradient at each position (central difference)
    gradients = {}
    for i in range(1, len(k_values) - 1):
        k = k_values[i]
        k_prev, k_next = k_values[i-1], k_values[i+1]
        gradient = (omega_means[k_next] - omega_means[k_prev]) / (k_next - k_prev)
        gradients[k] = gradient
    
    # Curvature at each position (second derivative)
    curvatures = {}
    for i in range(1, len(k_values) - 1):
        k = k_values[i]
        k_prev, k_next = k_values[i-1], k_values[i+1]
        # Assuming roughly uniform spacing
        curvature = omega_means[k_next] - 2*omega_means[k] + omega_means[k_prev]
        curvatures[k] = curvature
    
    # Curvature at center (k=0)
    curvature_center = curvatures.get(0, 0)
    gradient_at_center = gradients.get(0, 0)
    
    # Asymmetry (left vs right)
    left_omega = [omega_means[k] for k in k_values if k < 0]
    right_omega = [omega_means[k] for k in k_values if k > 0]
    asymmetry = np.mean(right_omega) - np.mean(left_omega) if left_omega and right_omega else 0
    
    # Mean absolute gradient (measure of "steepness")
    mean_abs_gradient = np.mean([abs(g) for g in gradients.values()]) if gradients else 0
    
    return {
        'omega_profile': omega_means,
        'gradients': gradients,
        'curvatures': curvatures,
        'curvature_center': curvature_center,
        'gradient_at_center': gradient_at_center,
        'asymmetry': asymmetry,
        'mean_abs_gradient': mean_abs_gradient
    }

def plot_landscape(aggregated_twins, aggregated_cousins, aggregated_sexy, limit, radius, save_path=None):
    """
    Create comprehensive landscape visualization showing:
    - Ω*(k) profile
    - Gradient ΔΩ/Δk
    - Curvature Δ²Ω/Δk²
    For twins, cousins, and sexy primes
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Prime Pair Complexity Landscape (N={limit:,}, R={radius})', fontsize=16, fontweight='bold')
    
    datasets = [
        (aggregated_twins, 'Twin Primes', '#e74c3c'),
        (aggregated_cousins, 'Cousin Primes', '#3498db'),
        (aggregated_sexy, 'Sexy Primes', '#2ecc71')
    ]
    
    for col, (aggregated, label, color) in enumerate(datasets):
        metrics = compute_landscape_metrics(aggregated)
        
        if not metrics:
            continue
        
        k_values = sorted(metrics['omega_profile'].keys())
        omega_values = [metrics['omega_profile'][k] for k in k_values]
        
        # Row 0: Ω*(k) profile
        ax0 = axes[0, col]
        ax0.plot(k_values, omega_values, 'o-', color=color, linewidth=2, markersize=4, alpha=0.7)
        ax0.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax0.axvline(0, color='red', linestyle='--', alpha=0.5, label='Center')
        ax0.set_xlabel('Position k', fontsize=10)
        ax0.set_ylabel('Ω*(k)', fontsize=10)
        ax0.set_title(f'{label}\nΩ*(0) = {metrics["omega_profile"][0]:.3f}', fontsize=11, fontweight='bold')
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8)
        
        # Row 1: Gradient ΔΩ/Δk
        ax1 = axes[1, col]
        grad_k = sorted(metrics['gradients'].keys())
        grad_values = [metrics['gradients'][k] for k in grad_k]
        ax1.plot(grad_k, grad_values, 's-', color=color, linewidth=2, markersize=4, alpha=0.7)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Position k', fontsize=10)
        ax1.set_ylabel('ΔΩ/Δk', fontsize=10)
        ax1.set_title(f'Gradient\n(ΔΩ/Δk)|₀ = {metrics["gradient_at_center"]:.4f}', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Row 2: Curvature Δ²Ω/Δk²
        ax2 = axes[2, col]
        curv_k = sorted(metrics['curvatures'].keys())
        curv_values = [metrics['curvatures'][k] for k in curv_k]
        ax2.plot(curv_k, curv_values, '^-', color=color, linewidth=2, markersize=4, alpha=0.7)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Position k', fontsize=10)
        ax2.set_ylabel('Δ²Ω/Δk²', fontsize=10)
        ax2.set_title(f'Curvature\n(Δ²Ω/Δk²)|₀ = {metrics["curvature_center"]:.4f}', fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[PLOT] Saved landscape visualization to: {save_path}")
    
    plt.show()
    return fig

def adaptive_radius(limit, scale_factor=2.5):
    """
    Compute adaptive radius based on prime density.
    At limit n, average gap between primes is approximately ln(n).
    We want radius to be scale_factor * ln(n) to capture local structure.
    """
    radius = int(scale_factor * np.log(limit))
    return max(radius, 5)  # Minimum radius of 5

def run_checkpoint(limit, checkpoint_name, use_adaptive=True):
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {checkpoint_name} (N = {limit:,})")
    print(f"{'='*70}")

    start_time = time.time()

    print(f"[1/6] Generating primes...")
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    print(f"  Found {len(primes):,} primes")

    print(f"[2/6] Finding prime pairs...")
    twin_primes = find_twin_primes(primes)
    cousin_primes = find_cousin_primes(primes)
    sexy_primes = find_sexy_primes(primes)
    print(f"  Twins: {len(twin_primes):,}")
    print(f"  Cousins: {len(cousin_primes):,}")
    print(f"  Sexy: {len(sexy_primes):,}")

    # Determine radius
    if use_adaptive:
        RADIUS = adaptive_radius(limit, scale_factor=2.5)
        print(f"\n[3/6] Computing resonance profiles (adaptive radius = {RADIUS})...")
    else:
        RADIUS = 10
        print(f"\n[3/6] Computing resonance profiles (fixed radius = {RADIUS})...")

    agg_twins = aggregate_fingerprint(twin_primes, RADIUS, prime_set)
    agg_cousins = aggregate_fingerprint(cousin_primes, RADIUS, prime_set)
    agg_sexy = aggregate_fingerprint(sexy_primes, RADIUS, prime_set)

    print(f"[4/6] Computing statistics...")
    stats_twins = compute_statistics(agg_twins)
    stats_cousins = compute_statistics(agg_cousins)
    stats_sexy = compute_statistics(agg_sexy)

    print(f"[5/6] Computing landscape metrics...")
    landscape_twins = compute_landscape_metrics(agg_twins)
    landscape_cousins = compute_landscape_metrics(agg_cousins)
    landscape_sexy = compute_landscape_metrics(agg_sexy)

    elapsed = time.time() - start_time

    print(f"\nRESULTS:")
    print(f"  Twins:   Ω*(0) = {stats_twins['central_mean']:+.4f}, d = {stats_twins['effect_size']:+.4f}")
    print(f"           Curvature = {landscape_twins['curvature_center']:+.4f}, |∇Ω| = {landscape_twins['mean_abs_gradient']:.4f}")
    print(f"  Cousins: Ω*(0) = {stats_cousins['central_mean']:+.4f}, d = {stats_cousins['effect_size']:+.4f}")
    print(f"           Curvature = {landscape_cousins['curvature_center']:+.4f}, |∇Ω| = {landscape_cousins['mean_abs_gradient']:.4f}")
    print(f"  Sexy:    Ω*(0) = {stats_sexy['central_mean']:+.4f}, d = {stats_sexy['effect_size']:+.4f}")
    print(f"           Curvature = {landscape_sexy['curvature_center']:+.4f}, |∇Ω| = {landscape_sexy['mean_abs_gradient']:.4f}")
    print(f"\nTime: {elapsed:.1f} seconds")

    return {
        'limit': limit,
        'name': checkpoint_name,
        'radius': RADIUS,
        'twins': stats_twins,
        'cousins': stats_cousins,
        'sexy': stats_sexy,
        'landscape_twins': landscape_twins,
        'landscape_cousins': landscape_cousins,
        'landscape_sexy': landscape_sexy,
        'agg_twins': agg_twins,
        'agg_cousins': agg_cousins,
        'agg_sexy': agg_sexy,
        'time': elapsed
    }

def main():
    print("\n" + "="*70)
    print("OPERATIONAL RESONANCE: LANDSCAPE ANALYSIS")
    print("Gradient and Curvature Analysis of Prime Pair Complexity")
    print("="*70)

    checkpoints = [
        (100000, "100K"),
        (1000000, "1M"),
    ]

    print("\n" + "="*70)
    print("ADAPTIVE RADIUS ANALYSIS (R = 2.5 * ln(N))")
    print("="*70)

    results_adaptive = []
    for limit, name in checkpoints:
        result = run_checkpoint(limit, name, use_adaptive=True)
        results_adaptive.append(result)
        
        # Generate landscape plot for this checkpoint
        print(f"\n[6/6] Generating landscape visualization...")
        plot_landscape(
            result['agg_twins'],
            result['agg_cousins'],
            result['agg_sexy'],
            limit,
            result['radius'],
            save_path=f'landscape_{name}.png'
        )

    # Print comparison
    print(f"\n\n" + "="*70)
    print("LANDSCAPE COMPARISON ACROSS SCALES")
    print("="*70)

    print(f"\n{'Checkpoint':<12} {'Ω*(0)':<12} {'Curvature':<12} {'|∇Ω|':<12} {'d':<12}")
    print("-" * 60)

    for r in results_adaptive:
        print(f"\n{r['name']} (R={r['radius']})")
        print(f"  Twins:   {r['twins']['central_mean']:+.4f}      {r['landscape_twins']['curvature_center']:+.4f}       {r['landscape_twins']['mean_abs_gradient']:.4f}      {r['twins']['effect_size']:+.4f}")
        print(f"  Cousins: {r['cousins']['central_mean']:+.4f}      {r['landscape_cousins']['curvature_center']:+.4f}       {r['landscape_cousins']['mean_abs_gradient']:.4f}      {r['cousins']['effect_size']:+.4f}")
        print(f"  Sexy:    {r['sexy']['central_mean']:+.4f}      {r['landscape_sexy']['curvature_center']:+.4f}       {r['landscape_sexy']['mean_abs_gradient']:.4f}      {r['sexy']['effect_size']:+.4f}")

    print(f"\n\n" + "="*70)
    print("TREND ANALYSIS: LANDSCAPE FEATURES")
    print("="*70)

    twin_curvatures = [r['landscape_twins']['curvature_center'] for r in results_adaptive]
    twin_gradients = [r['landscape_twins']['mean_abs_gradient'] for r in results_adaptive]

    print(f"\nTwin Prime Curvature trend:")
    for i, r in enumerate(results_adaptive):
        if i == 0:
            print(f"  {r['name']}: {r['landscape_twins']['curvature_center']:+.4f} (baseline)")
        else:
            prev = results_adaptive[i-1]['landscape_twins']['curvature_center']
            change = r['landscape_twins']['curvature_center'] - prev
            pct_change = (change / abs(prev)) * 100 if prev != 0 else 0
            print(f"  {r['name']}: {r['landscape_twins']['curvature_center']:+.4f} (change: {change:+.4f}, {pct_change:+.1f}%)")

    print(f"\nTwin Prime Mean Gradient Magnitude trend:")
    for i, r in enumerate(results_adaptive):
        if i == 0:
            print(f"  {r['name']}: {r['landscape_twins']['mean_abs_gradient']:.4f} (baseline)")
        else:
            prev = results_adaptive[i-1]['landscape_twins']['mean_abs_gradient']
            change = r['landscape_twins']['mean_abs_gradient'] - prev
            pct_change = (change / abs(prev)) * 100 if prev != 0 else 0
            print(f"  {r['name']}: {r['landscape_twins']['mean_abs_gradient']:.4f} (change: {change:+.4f}, {pct_change:+.1f}%)")

    print(f"\n\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Curvature interpretation
    if twin_curvatures[-1] < 0:
        print(f"\n✓ POTENTIAL WELL: Twin primes sit in a complexity well (negative curvature)")
        if abs(twin_curvatures[-1]) > abs(twin_curvatures[0]):
            print(f"  Well is DEEPENING with scale → True threading attractor")
        else:
            print(f"  Well depth is stable/decreasing → May be local feature")
    else:
        print(f"\n✗ NO WELL: Twin primes do not show negative curvature")

    # Gradient interpretation
    if twin_gradients[-1] > twin_gradients[0]:
        print(f"\n✓ STEEPENING: Gradient magnitude INCREASES with scale")
        print(f"  Complexity flows more strongly toward twins at larger N")
    else:
        print(f"\n→ STABLE/WEAKENING: Gradient magnitude stable or decreasing")

    # Comparison with cousins
    cousin_curvatures = [r['landscape_cousins']['curvature_center'] for r in results_adaptive]
    if twin_curvatures[-1] < cousin_curvatures[-1]:
        print(f"\n✓ TWINS ARE SPECIAL: Twin curvature ({twin_curvatures[-1]:+.4f}) < Cousin curvature ({cousin_curvatures[-1]:+.4f})")
        print(f"  Twins are stronger attractors than cousins")

    print(f"\n" + "="*70)

if __name__ == "__main__":
    main()