# Fabric Mathematics

**A mathematical framework where operations are fundamental**

## Overview

This repository contains computational tools, visualizations, and examples for **Operational Geometry**: a category-theoretic framework in which operations (rather than points, sets, or manifolds) are the primitive entities of mathematics.

**Key ideas:**

- Objects are equivalence classes of operational sequences
- Constants like , , emerge as fixed points of operations
- Primes index an infinite hierarchy of constants
- Complex structures (helices, knots, lattices) arise from simple generating operations

## Paper

The foundational paper is available on Zenodo:

**Operational Geometry: Foundations of a Geometry of Operations**  
David D. Zelenka (2025)  
\[DOI link will be added after Zenodo publication\]

## Repository Status

🚧 **Under Construction** 🚧

This repository is a placeholder for future computational work. Planned content includes:

- **Core library**: Python implementation of operational sequences and primitive operations
- **Constant computation**: Numerical calculation of prime-indexed constants
- **Visualizations**: Interactive plots of operational sequences generating geometric structures
- **Examples**: Jupyter notebooks demonstrating key constructions (golden ratio, double helix, trefoil knot)
- **Invariant calculators**: Tools for computing operational invariants (, , , )

## The Five Primitive Operations

All structure emerges from compositions of:

- Thread(τ) - advance through threading depth
- Divide(n) - partition into parts
- Rotate(θ) - rotate by angle
- Scale(λ) - scale by factor
- Iterate(k) - apply operation times

## Example: Deriving the Golden Ratio

The golden ratio emerges as the fixed point of:

Operationally: Scale(+1) ∘ Divide(x) ∘ Scale(x)

Iterating from converges to .

## Example: Prime-Indexed Constants

Each prime generates a constant from the fixed point:

| Prime | Polynomial | Constant | Name |
| --- | --- | --- | --- |
| 2   |     | 1.618... | Golden Ratio |
| 3   |     | 1.466... | Plastic Number |
| 5   |     | 1.221... | Pentic Constant |
| 137 |     | ≈ 1.007? | Fine-structure? |



## License

Creative Commons Attribution 4.0 International. All mathematical content is freely available for research and educational purposes.

##
_"Every form is merely the memory of an act."_






