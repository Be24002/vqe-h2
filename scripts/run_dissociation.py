"""H2 dissociation curve: RHF vs exact FCI vs VQE on the 2-qubit Hamiltonian.

Produces figures/h2_dissociation.png and results/h2_curve.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vqeh2.hamiltonian import dissociation_curve, exact_energy, qubit_hamiltonian, solve_rhf
from vqeh2.vqe import run_vqe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

r_grid = np.linspace(0.35, 3.0, 54)
print(f"Scanning {len(r_grid)} geometries (symmetry MOs + CI) ...")
rows = dissociation_curve(r_grid)

r_eq_row = min(rows, key=lambda row: row["e_fci"])
szabo = solve_rhf(0.529177 * 1.4)

vqe_points = []
for r in [0.5, 0.735, 1.0, 1.5, 2.0, 3.0]:
    operator = qubit_hamiltonian(r)
    target = float(np.linalg.eigvalsh(operator.to_matrix()).min())
    result = run_vqe(operator)
    error = result.energy - target
    vqe_points.append((r, result.energy, error))
    print(
        f"VQE at R = {r:.3f} A: E = {result.energy:.6f} Ha "
        f"(exact {target:.6f}, error {error:+.2e}, "
        f"{result.n_evaluations} evaluations)"
    )

print(f"E_HF at 1.4 bohr          : {szabo.e_hf:.4f} Ha   (Szabo: -1.1167)")
print(f"E_FCI at R = 0.735 A      : {exact_energy(0.735):.4f} Ha   (canonical: -1.1373)")
print(f"Equilibrium bond length   : {r_eq_row['r_angstrom']:.3f} A   (STO-3G ref: ~0.735)")
print(f"Correlation at eq.        : {r_eq_row['correlation']:+.4f} Ha   (canonical: -0.0206)")
print(f"Dissociation limit E(3 A) : {rows[-1]['e_fci']:.4f} Ha   (2 x H in STO-3G: -0.9332)")

# ---- Figure ---------------------------------------------------------------
r = np.array([row["r_angstrom"] for row in rows])
e_hf = np.array([row["e_hf"] for row in rows])
e_reduced = np.array([row["e_reduced"] for row in rows])
e_fci = np.array([row["e_fci"] for row in rows])

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(r, e_hf, "-", color="C0", label="RHF (mean field)")
ax.plot(r, e_reduced, "--", color="C1", label="2-qubit reduced CI (VQE target)")
ax.plot(r, e_fci, "-", color="C3", label="full CI (4 determinants)")
vqe_r = [point[0] for point in vqe_points]
vqe_e = [point[1] for point in vqe_points]
ax.plot(vqe_r, vqe_e, "k.", ms=10, label="VQE (2 qubits, this work)")
ax.axhline(-0.9332, color="gray", ls=":", lw=1,
           label="2 H atoms in STO-3G ($-0.9332$ Ha)")
ax.set_xlabel("bond length $R$ ($\\AA$)")
ax.set_ylabel("total energy (Ha)")
ax.set_title(
    f"H$_2$/STO-3G: VQE on the reduced 2-qubit Hamiltonian "
    f"($E_{{min}}$ = {r_eq_row['e_fci']:.4f} Ha at {r_eq_row['r_angstrom']:.3f} $\\AA$)"
)
ax.set_ylim(-1.25, 0.1)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "h2_dissociation.png", dpi=150)
plt.close(fig)

np.savetxt(
    RESULTS / "h2_curve.csv",
    np.column_stack([r, e_hf, e_reduced, e_fci]),
    delimiter=",",
    header="r_angstrom,e_hf_hartree,e_reduced_ci_hartree,e_full_ci_hartree",
    comments="",
)
print("Wrote figures/h2_dissociation.png, results/h2_curve.csv")
