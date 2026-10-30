# vqe-h2 — quantum chemistry on a laptop: VQE for H₂ and QAOA for MaxCut

**The full H₂/STO-3G pipeline written from scratch**: Gaussian integrals →
molecular orbitals → exact CI → the classic reduced 2-qubit Pauli
Hamiltonian → VQE with Qiskit. No quantum-chemistry package was used to
produce any number below; the only inputs are the STO-3G contraction
constants and the carbon hopping distance. Plus QAOA for MaxCut with an
exact classical benchmark.

## The pipeline

```
STO-3G basis (3 Gaussians/H)
  -> closed-form Gaussian integrals (overlap, kinetic, nuclear, ERI)
  -> symmetry-adapted MOs g = (A+B)/+, u = (A-B)/-
  -> RHF energy, orbital energies, MO integrals          (Szabo & Ostlund)
  -> 4x4 full CI in the Sz=0 determinant space            (Slater-Condon)
  -> 2-qubit Pauli Hamiltonian  H = c0 II + c1(Z0+Z1) + c0 Z0Z1 + c4(XX-YY)
  -> VQE (hardware-efficient ansatz, COBYLA, Qiskit estimator)
```

## Results (all measured by this code, nothing hard-coded)

| Quantity | Reference | This code |
|---|---|---|
| Benzene-level integral anchors at R = 1.4 bohr (S12, T11, (11\|11)) | Szabo table | **exact to 4 decimals** ✓ |
| RHF energy at 1.4 bohr | −1.1167 Ha (famous textbook SCF) | **−1.1167** ✓ |
| MO energies at 1.4 bohr | −0.578 / +0.670 Ha | ✓ |
| (gg\|gg) in MO basis | 0.6757 Ha | ✓ |
| Full-CI energy at R = 0.735 Å | −1.1373 Ha (canonical) | **−1.1373** ✓ |
| Correlation energy at equilibrium | −0.0206 Ha | −0.0210 ✓ |
| Dissociation limit | 2 × E(H, STO-3G) = −0.9332 Ha | **−0.9332** ✓ |
| Equilibrium bond length | ≈ 0.735 Å | 0.750 Å (minimal basis) |
| **VQE vs exact** (6 geometries) | — | **max error 2 × 10⁻⁹ Ha** ✓ |
| QAOA p=1 on K4 MaxCut | — | ratio 0.924 vs brute force ✓ |
| QAOA p=1 on C4 MaxCut | — | ratio 0.750, P(best cut) = 0.53 |

| | |
|---|---|
| ![dissociation](figures/h2_dissociation.png) | The dissociation curve: mean-field RHF fails at large R (ionic contamination), the reduced 2-qubit CI follows, and VQE (black dots) sits exactly on its target at machine precision. The plateau is the *correct* STO-3G limit 2 × (−0.4666) = −0.9332 Ha — a minimal basis is not exact even for one hydrogen atom. |
| ![qaoa](figures/qaoa_maxcut.png) | QAOA p=1 on C4: the optimized state concentrates 53% of the probability on the optimal cuts (red). |

## Things learned the hard way (documented in code/tests)

- **YY sign in the reduced Hamiltonian**: Y⊗Y|00⟩ = −|11⟩, so the
  |00⟩↔|11⟩ coupling is (XX − YY)/2, not (XX + YY)/2. Getting this wrong
  shifts the ground state by 20 mHa — the test suite catches it.
- **SCF symmetry breaking**: an iterated SCF collapses to localized orbitals
  at large R, silently corrupting the CI. We use fixed symmetry-adapted MOs
  instead — deterministic at every geometry.
- **The exchange integral does not vanish at dissociation** (it tends to
  (AA|AA)/2 = 0.387): this is exactly what lets the {g², u²} CI dissociate
  correctly.
- **STO-3G dissociation limit ≠ −1.0 Ha**: a minimal basis leaves each
  hydrogen at −0.4666 Ha (not −0.5), so the correct limit is −0.9332 Ha.

## Using it

```python
from vqeh2.hamiltonian import exact_energy, qubit_hamiltonian
from vqeh2.vqe import run_vqe

operator = qubit_hamiltonian(0.735)     # 2-qubit Pauli Hamiltonian at R = 0.735 A
result = run_vqe(operator)              # -> E = -1.137306 Ha (exact: -1.137306)
```

Swapping `StatevectorEstimator` for a real `QiskitRuntimeService` session is
the only change needed to run on actual IBM Quantum hardware (free tier).

## Repository layout

```
vqe-h2/
├── src/vqeh2/
│   ├── integrals.py    # STO-3G Gaussian integrals from closed forms
│   ├── hamiltonian.py  # MOs, RHF, 4x4 full CI, 2-qubit Pauli operator
│   ├── vqe.py          # variational eigensolver (Qiskit primitives + scipy)
│   └── qaoa.py         # QAOA for MaxCut + brute-force benchmark
├── scripts/run_dissociation.py   # the curve + VQE points + figures/CSV
├── scripts/run_qaoa.py           # C4 and K4 benchmarks
├── tests/              # 18 physics tests (pytest)
└── figures/, results/  # generated plots and CSV data (committed)
```

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest                          # 18 physics tests, ~4 s
python scripts/run_dissociation.py
python scripts/run_qaoa.py
```

## Roadmap

- [ ] Run on real IBM Quantum hardware (free tier) with shot-noise-aware VQE
- [ ] UCCSD ansatz and comparison against the hardware-efficient one
- [ ] Noise models (depolarizing) and error mitigation (ZNE)
- [ ] HeH⁺ and H₂ in 6-31G (bigger basis, more qubits)

## References

- A. Szabo & N. S. Ostlund, *Modern Quantum Chemistry* (integral tables, ch. 3)
- Peruzzo et al., *A variational eigenvalue solver on a photonic quantum
  processor*, Nat. Commun. 5, 4213 (2014) — the original VQE, on H₂
- O'Malley et al., PRL 97, 220502 (2006) — the reduced 2-qubit H₂ Hamiltonian
- Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm* (2014)
- Kandala et al., Nature 549, 242 (2017) — hardware-efficient VQE

## License

MIT — see [LICENSE](LICENSE).
