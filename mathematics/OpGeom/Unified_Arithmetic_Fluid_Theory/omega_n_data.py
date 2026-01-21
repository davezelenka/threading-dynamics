import csv

MAX_N = 100_000
MODULI = [2, 3, 5, 7, 9, 11]

def compute_omega_sieve(limit: int):
    """
    Returns an array omega[n] = Ω(n) for 0 <= n <= limit
    Uses a modified sieve for prime factor multiplicity.
    """
    omega = [0] * (limit + 1)
    temp = list(range(limit + 1))

    for p in range(2, limit + 1):
        if temp[p] == p:  # p is prime
            for k in range(p, limit + 1, p):
                while temp[k] % p == 0:
                    omega[k] += 1
                    temp[k] //= p
    return omega

def omega_contribution(n: int, base: int):
    """
    Contribution of a specific modulus to Ω(n).
    - For primes: counts multiplicity
    - For 9: counts powers of 3^2 explicitly
    """
    count = 0

    if base == 9:
        while n % 9 == 0:
            count += 2
            n //= 9
    else:
        while n % base == 0:
            count += 1
            n //= base

    return count

def generate_csv(filename="omega_field_100k.csv"):
    omega = compute_omega_sieve(MAX_N)

    headers = (
        ["n"]
        + [f"Omega_{m}" for m in MODULI]
        + ["Omega_total"]
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for n in range(1, MAX_N + 1):
            row = [n]

            for m in MODULI:
                row.append(omega_contribution(n, m))

            row.append(omega[n])
            writer.writerow(row)

    print(f"CSV written to: {filename}")

if __name__ == "__main__":
    generate_csv()
