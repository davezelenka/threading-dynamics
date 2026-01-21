# Minimized Proofs

Machine-readable JSON formalizations of mathematical proofs using the Intrinsic Operational Gradient Theorem (IOGT) framework.

## Overview

This repository contains compressed, structured proofs for several major mathematical problems, including multiple Millennium Prize Problems. Each proof is encoded in JSON format optimized for:

- **AI training datasets** - Structured, parseable data with semantic annotations
- **Search engine indexing** - Schema.org context and DOI linking
- **Human readability** - Clear hierarchical organization with explanations
- **Formal verification** - Designed for future formalization in proof assistants (Lean4, Coq, Isabelle)

## Repository Contents

### Foundational Theorems

| File | Description | DOI | Size |
|------|-------------|-----|------|
| `Intrinsic_Operational_Gradient_Theorem_foundation.json` | Core framework establishing operations-primary ontology and intrinsic difficulty gradients | [10.5281/zenodo.18062553](https://doi.org/10.5281/zenodo.18062553) | 18.2 KB |
| `Irreducible_Overhead_Theorem.json` | Proof that time-parallelism products cannot perfectly conserve exponential cost | [10.5281/zenodo.18073069](https://doi.org/10.5281/zenodo.18073069) | 16.5 KB |

### Millennium Prize Problems

| File | Problem | Status | DOI | Size |
|------|---------|--------|-----|------|
| `Navier-Stokes_global_regularity_proof.json` | 3D Navier-Stokes Global Regularity | **Proved** | [10.5281/zenodo.18214940](https://doi.org/10.5281/zenodo.18214940) | 12.8 KB |
| `Riemann_Hypothesis_proof.json` | Riemann Hypothesis | **Conditional** (on IOGT) | [10.5281/zenodo.18190384](https://doi.org/10.5281/zenodo.18190384) | 16.4 KB |
| `P_neq_NP_proof.json` | P versus NP | **Proved** | [10.5281/zenodo.18063338](https://doi.org/10.5281/zenodo.18063338) | 17.9 KB |

### Other Major Conjectures

| File | Problem | Status | DOI | Size |
|------|---------|--------|-----|------|
| `Collatz_conjecture_proof.json` | Collatz Conjecture (3n+1 Problem) | **Conditional** (on Conformal Mixing Lemma) | [10.5281/zenodo.18273351](https://doi.org/10.5281/zenodo.18273351) | 16.7 KB |

## Proof Architecture

All proofs follow a consistent JSON schema structure:

```json
{
  "meta": {
    "title": "...",
    "author": "Zelenka_D_D",
    "claim": "PROVED | CONDITIONAL_PROOF",
    "framework": [...],
    "links": {
      "zenodo_record": "...",
      "doi": "..."
    }
  },
  "abstract": "...",
  "axioms": {...},
  "lemmas": {...},
  "main_theorem": {
    "proof_steps": [...]
  },
  "expansion_instructions": {...}
}
```

### Key Sections

- **`meta`** - Bibliographic information, DOIs, MSC codes, keywords
- **`abstract`** - One-paragraph summary of the proof
- **`axioms`** / **`primitives`** - Foundational assumptions
- **`lemmas`** - Intermediate results with dependencies
- **`main_theorem`** - Central result with step-by-step proof chain
- **`expansion_instructions`** - Roadmap for reconstructing full proof
- **`computational_metadata`** - Formal verification targets, complexity

## IOGT Framework

The **Intrinsic Operational Gradient Theorem** (IOGT) is the foundational principle underlying all proofs in this collection:

> **Core Thesis:** Composable operations induce intrinsic gradients of difficulty. In any infinite operational system with composability, bounded primitive cost, and ≥1 non-invertible operation, forward construction and reverse reconstruction are generically asymmetric.

### Applications

- **Navier-Stokes**: Angular entropy gradients resist turbulent cascades; threading deficit decays exponentially
- **Riemann Hypothesis**: Critical line σ=1/2 is unique operational minimum for prime threading
- **P≠NP**: Exponential search space collapse creates irreversible coordination cost
- **Collatz**: Integer manifold (ℤ₊, ln n) is dissipative under 3n+1 map with strain ln(3/4) < 0

## Proof Status Summary

| Proof | Type | Conditionality | Key Result |
|-------|------|----------------|------------|
| **IOGT** | Foundation | Axiomatic | Operations induce intrinsic difficulty gradients |
| **IOT** | Foundation | Information-theoretic | T·P ≥ (1+c)·2^n, equality unattainable |
| **Navier-Stokes** | Unconditional | None | Global smooth solutions for all t ≥ 0 |
| **Riemann** | Conditional | Assumes IOGT | All zeros on Re(s) = 1/2 |
| **P≠NP** | Unconditional | None | No polynomial algorithm for NP-complete problems |
| **Collatz** | Conditional | Assumes Conformal Mixing | All trajectories reach n=1 |

## Citation

### General Citation
```
Zelenka, D. D. (2025-2026). Minimized Proofs Collection. 
GitHub repository. https://github.com/[username]/[repo]/minimized_proofs
```

### Individual Proofs
Each JSON file contains complete citation information in the `meta.citations` section, including:
- Plain text citation
- BibTeX format
- DOI link to Zenodo record

Example:
```bibtex
@misc{zelenka2025iogt,
  author = {Zelenka, D. D.},
  title = {The Intrinsic Operational Gradient Theorem},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18062553},
  url = {https://doi.org/10.5281/zenodo.18062553}
}
```

## Usage

### For Researchers
1. Browse the JSON files to understand proof structure
2. Use `expansion_instructions` to reconstruct detailed proofs
3. Check `related_work` sections for connections to existing literature
4. Follow DOI links to access full papers on Zenodo

### For AI Training
Each JSON file includes:
- Structured mathematical knowledge graphs
- Formal notation with semantic annotations
- Step-by-step logical dependencies
- Cross-references between proofs

### For Formal Verification
Files are designed for translation into:
- Lean 4
- Coq
- Isabelle/HOL
- Agda

See `computational_metadata.proof_assistant_target` in each file.

## Mathematical Subject Classification (MSC2020)

| Code | Topic | Files |
|------|-------|-------|
| 11M26 | Nonzero zeros (Riemann zeta) | `Riemann_Hypothesis_proof.json` |
| 35Q30 | Navier-Stokes equations | `Navier-Stokes_global_regularity_proof.json` |
| 68Q15 | Complexity classes (P, NP) | `P_neq_NP_proof.json`, `Irreducible_Overhead_Theorem.json` |
| 11B83 | Special sequences (3n+1) | `Collatz_conjecture_proof.json` |
| 18M05 | Enriched categories | `Intrinsic_Operational_Gradient_Theorem_foundation.json` |

## Keywords

`IOGT`, `operational mathematics`, `Millennium Prize Problems`, `Navier-Stokes`, `Riemann Hypothesis`, `P versus NP`, `Collatz conjecture`, `machine-readable proofs`, `formal verification`, `gradient descent`, `operational potential`, `structural complexity`, `Kolmogorov complexity`, `dissipative systems`

## License

All proofs are released under **CC-BY-4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- **Share** — copy and redistribute in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit and link to the license

## Contributing

This is a research archive. For questions, corrections, or discussions:
- Open an issue in this repository
- Contact via Zenodo record comment sections
- Email: [contact information if desired]

## Acknowledgments

These proofs build upon centuries of mathematical tradition. Key influences include:
- Morse theory (Morse, Milnor, Witten)
- Category theory (Mac Lane, Kelly)
- Complexity theory (Cook, Levin, Karp)
- Kolmogorov complexity (Kolmogorov, Solomonoff, Chaitin)
- Fluid dynamics (Leray, Hopf, Constantin-Fefferman)

## Version History

- **v1.0** (January 2025-2026) - Initial release of 6 proofs in JSON format

---

**Note**: While these proofs represent rigorous mathematical arguments within the IOGT framework, they have not yet undergone formal peer review or been formalized in proof assistants. The conditional proofs (RH, Collatz) depend on specific lemmas that remain open problems. The unconditional proofs (NS, P≠NP) and foundational theorems (IOGT, IOT) present complete arguments within their stated frameworks.

For the most up-to-date versions and formal publications, please refer to the Zenodo DOIs listed above.
