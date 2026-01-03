import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from scipy import signal
from scipy.fft import fft, fftfreq

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

def aggregate_fingerprint_with_counts(prime_pairs, radius, prime_set):
    """
    Aggregate fingerprints and track composite counts at each k.
    """
    aggregated = defaultdict(lambda: {'Omega_norm': [], 'tau_norm': [], 'count': 0})
    for pair in prime_pairs:
        p_a, p_b = pair
        c_0 = p_a + (p_b - p_a) // 2
        profile = compute_window_profile(c_0, radius, prime_set)
        for k, values in profile.items():
            aggregated[k]['Omega_norm'].append(values['Omega_norm'])
            aggregated[k]['tau_norm'].append(values['tau_norm'])
            aggregated[k]['count'] += 1
    return aggregated

def plot_boundary_analysis(aggregated_twins, aggregated_cousins, aggregated_sexy, limit, radius, save_path=None):
    """
    Create visualization showing Ω*(k) and composite count across extended radius.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Boundary Analysis: Extended Radius (N={limit:,}, R={radius})', fontsize=16, fontweight='bold')
    
    datasets = [
        (aggregated_twins, 'Twin Primes', '#e74c3c'),
        (aggregated_cousins, 'Cousin Primes', '#3498db'),
        (aggregated_sexy, 'Sexy Primes', '#2ecc71')
    ]
    
    for col, (aggregated, label, color) in enumerate(datasets):
        k_values = sorted([k for k in aggregated.keys()])
        omega_means = [np.mean(aggregated[k]['Omega_norm']) for k in k_values]
        counts = [aggregated[k]['count'] for k in k_values]
        
        # Top row: Ω*(k) profile
        ax_top = axes[0, col]
        ax_top.plot(k_values, omega_means, 'o-', color=color, linewidth=2, markersize=4, alpha=0.7)
        ax_top.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_top.axvline(0, color='red', linestyle='-', alpha=0.7, linewidth=2)
        ax_top.set_xlabel('Position k', fontsize=10)
        ax_top.set_ylabel('Ω*(k)', fontsize=10)
        ax_top.set_title(f'{label}\nΩ* Profile', fontsize=11, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        
        # Bottom row: Composite count at each k
        ax_bottom = axes[1, col]
        ax_bottom.bar(k_values, counts, color=color, alpha=0.6, edgecolor='black', width=0.8)
        ax_bottom.set_xlabel('Position k', fontsize=10)
        ax_bottom.set_ylabel('Composite Count', fontsize=10)
        ax_bottom.set_title(f'{label}\nData Sparsity', fontsize=11, fontweight='bold')
        ax_bottom.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[PLOT] Saved boundary analysis to: {save_path}")
    
    plt.show()
    return fig

def adaptive_radius(limit, scale_factor=2.5):
    radius = int(scale_factor * np.log(limit))
    return max(radius, 5)

def main():
    print("\n" + "="*70)
    print("BOUNDARY ANALYSIS: EXTENDED RADIUS")
    print("Testing if Ω* decays or remains constant at distance")
    print("="*70)

    limit = 100000
    print(f"\n[1/5] Generating primes up to {limit:,}...")
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    print(f"  Found {len(primes):,} primes")

    print(f"[2/5] Finding prime pairs...")
    twin_primes = find_twin_primes(primes)
    cousin_primes = find_cousin_primes(primes)
    sexy_primes = find_sexy_primes(primes)
    print(f"  Twins: {len(twin_primes):,}")
    print(f"  Cousins: {len(cousin_primes):,}")
    print(f"  Sexy: {len(sexy_primes):,}")

    # Use extended radius
    RADIUS = 80  # Extended from adaptive ~28 to 80
    print(f"\n[3/5] Computing profiles with extended radius R={RADIUS}...")
    agg_twins = aggregate_fingerprint_with_counts(twin_primes, RADIUS, prime_set)
    agg_cousins = aggregate_fingerprint_with_counts(cousin_primes, RADIUS, prime_set)
    agg_sexy = aggregate_fingerprint_with_counts(sexy_primes, RADIUS, prime_set)

    print(f"[4/5] Analyzing boundary behavior...")
    
    # Analyze twins
    print(f"\n{'='*70}")
    print("TWIN PRIMES: Ω* AND COMPOSITE COUNT BY DISTANCE")
    print(f"{'='*70}")
    
    k_values = sorted([k for k in agg_twins.keys()])
    print(f"\n{'k':>4} {'Ω*(k)':>10} {'Count':>8} {'Avg Count':>10}")
    print("-" * 40)
    
    total_count = sum([agg_twins[k]['count'] for k in k_values])
    avg_count = total_count / len(k_values) if k_values else 0
    
    for k in k_values:
        omega = np.mean(agg_twins[k]['Omega_norm'])
        count = agg_twins[k]['count']
        print(f"{k:>4} {omega:>10.4f} {count:>8} {avg_count:>10.1f}")
    
    # Check for decay
    print(f"\n{'='*70}")
    print("DECAY ANALYSIS")
    print(f"{'='*70}")
    
    # Split into regions
    near = [np.mean(agg_twins[k]['Omega_norm']) for k in k_values if abs(k) <= 15]
    mid = [np.mean(agg_twins[k]['Omega_norm']) for k in k_values if 15 < abs(k) <= 40]
    far = [np.mean(agg_twins[k]['Omega_norm']) for k in k_values if abs(k) > 40]
    
    near_mean = np.mean(near) if near else 0
    mid_mean = np.mean(mid) if mid else 0
    far_mean = np.mean(far) if far else 0
    
    print(f"\nTwin Primes:")
    print(f"  Near (|k| ≤ 15):  Ω* = {near_mean:+.4f} (n={len(near)})")
    print(f"  Mid  (15 < |k| ≤ 40): Ω* = {mid_mean:+.4f} (n={len(mid)})")
    print(f"  Far  (|k| > 40):  Ω* = {far_mean:+.4f} (n={len(far)})")
    
    decay_near_to_mid = ((mid_mean - near_mean) / abs(near_mean)) * 100 if near_mean != 0 else 0
    decay_mid_to_far = ((far_mean - mid_mean) / abs(mid_mean)) * 100 if mid_mean != 0 else 0
    
    print(f"\n  Decay (near → mid): {decay_near_to_mid:+.1f}%")
    print(f"  Decay (mid → far):  {decay_mid_to_far:+.1f}%")
    
    if abs(decay_near_to_mid) < 5 and abs(decay_mid_to_far) < 5:
        print(f"\n  ✓ NO DECAY: Ω* is CONSTANT across all distances")
        print(f"    This suggests a GLOBAL field perturbation, not a localized resonance")
    elif decay_near_to_mid < -5 or decay_mid_to_far < -5:
        print(f"\n  ✓ DECAY DETECTED: Ω* decreases with distance")
        print(f"    This suggests a LOCALIZED resonance with finite influence zone")
    else:
        print(f"\n  → COMPLEX BEHAVIOR: Mixed decay pattern")
    
    # Same for cousins
    print(f"\n{'='*70}")
    print("COUSIN PRIMES: DECAY ANALYSIS")
    print(f"{'='*70}")
    
    near_c = [np.mean(agg_cousins[k]['Omega_norm']) for k in k_values if abs(k) <= 15]
    mid_c = [np.mean(agg_cousins[k]['Omega_norm']) for k in k_values if 15 < abs(k) <= 40]
    far_c = [np.mean(agg_cousins[k]['Omega_norm']) for k in k_values if abs(k) > 40]
    
    near_mean_c = np.mean(near_c) if near_c else 0
    mid_mean_c = np.mean(mid_c) if mid_c else 0
    far_mean_c = np.mean(far_c) if far_c else 0
    
    print(f"\nCousin Primes:")
    print(f"  Near (|k| ≤ 15):  Ω* = {near_mean_c:+.4f} (n={len(near_c)})")
    print(f"  Mid  (15 < |k| ≤ 40): Ω* = {mid_mean_c:+.4f} (n={len(mid_c)})")
    print(f"  Far  (|k| > 40):  Ω* = {far_mean_c:+.4f} (n={len(far_c)})")
    
    decay_near_to_mid_c = ((mid_mean_c - near_mean_c) / abs(near_mean_c)) * 100 if near_mean_c != 0 else 0
    decay_mid_to_far_c = ((far_mean_c - mid_mean_c) / abs(mid_mean_c)) * 100 if mid_mean_c != 0 else 0
    
    print(f"\n  Decay (near → mid): {decay_near_to_mid_c:+.1f}%")
    print(f"  Decay (mid → far):  {decay_mid_to_far_c:+.1f}%")
    
    # Data sparsity check
    print(f"\n{'='*70}")
    print("DATA SPARSITY CHECK")
    print(f"{'='*70}")
    
    print(f"\nComposite counts at key distances:")
    print(f"  k=0:   {agg_twins[0]['count']:>4} twins, {agg_cousins[0]['count']:>4} cousins")
    print(f"  k=±15: {agg_twins[15]['count']:>4} twins, {agg_cousins[15]['count']:>4} cousins")
    print(f"  k=±30: {agg_twins[30]['count']:>4} twins, {agg_cousins[30]['count']:>4} cousins")
    print(f"  k=±50: {agg_twins[50]['count']:>4} twins, {agg_cousins[50]['count']:>4} cousins")
    print(f"  k=±80: {agg_twins[80]['count']:>4} twins, {agg_cousins[80]['count']:>4} cousins")
    
    min_count_twins = min([agg_twins[k]['count'] for k in k_values])
    min_count_cousins = min([agg_cousins[k]['count'] for k in k_values])
    
    print(f"\nMinimum composite count:")
    print(f"  Twins:   {min_count_twins} (at edges)")
    print(f"  Cousins: {min_count_cousins} (at edges)")
    
    if min_count_twins >= 10:
        print(f"\n  ✓ SUFFICIENT DATA: Even at edges, {min_count_twins}+ composites per position")
        print(f"    Ω* values are statistically reliable")
    else:
        print(f"\n  ⚠ SPARSE DATA: Only {min_count_twins} composites at edges")
        print(f"    Ω* values may be noisy at extreme distances")
    
    print(f"\n[5/5] Generating visualization...")
    plot_boundary_analysis(
        agg_twins, agg_cousins, agg_sexy,
        limit, RADIUS,
        save_path='boundary_analysis.png'
    )
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    if abs(decay_near_to_mid) < 5 and abs(decay_mid_to_far) < 5:
        print(f"\nTwin primes show NO DECAY across R={RADIUS}")
        print(f"The complexity enhancement is GLOBAL, not localized.")
        print(f"\nThis suggests:")
        print(f"  • Twin primes perturb the entire complexity field")
        print(f"  • The effect is intrinsic to the gap=2 structure")
        print(f"  • Not a wave phenomenon (would decay)")
    else:
        print(f"\nTwin primes show DECAY with distance")
        print(f"The influence zone extends to approximately k ≈ {RADIUS}")
    
    print(f"\n" + "="*70)

if __name__ == "__main__":
    main()