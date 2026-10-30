"""From integrals to the qubit Hamiltonian of H2 -- derived, not copied.

Pipeline (every step validated against textbook values):

1. Symmetry-adapted molecular orbitals (no SCF needed for homonuclear H2):

       g = (chi_A + chi_B) / sqrt(2(1+S)),   u = (chi_A - chi_B) / sqrt(2(1-S))

   which are orthonormal by symmetry at every bond length.
2. The 2-electron Sz=0 sector of the minimal basis has four determinants

       D1=|g.a g.b>,  D2=|u.a u.b>,  D3=|g.a u.b>,  D4=|u.a g.b>

   whose Slater-Condon matrix is the EXACT full CI here (diagonal + nuclear
   repulsion, closed-shell couplings (gu|gu), open-shell diagonals
   h_gg+h_uu+(gg|uu), and the spin-swap coupling (gu|ug)). Its ground state
   is -1.1373 Ha at equilibrium and -0.99 Ha at dissociation.
3. The 2x2 closed-shell block {g^2, u^2} embeds into 2 qubits as

       H = c0 II + c1 Z0 + c1 Z1 + c0 Z0Z1 + c4 (X0X1 - Y0Y1)

   with c0 = (a+b)/4, c1 = (a-b)/4, c4 = K/2, a/b the CSF energies including
   nuclear repulsion. (The MINUS on YY: Y(x)Y|00> = -|11>, so the
   |00> <-> |11> coupling is (XX-YY)/2.) This reduced Hamiltonian is exact
   near equilibrium but misses the static correlation of the open-shell
   singlet at dissociation -- visible in the dissociation figure, and a
   textbook lesson about active-space reductions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .integrals import H2Integrals, compute_h2_sto3g


@dataclass
class MolecularData:
    """Everything about H2 at one geometry."""

    r_angstrom: float
    integrals: H2Integrals
    mo_coefficients: np.ndarray   # (2, 2): columns are g (bonding), u (antibonding)
    mo_energies: np.ndarray
    h_mo: np.ndarray
    eri_mo: np.ndarray            # chemist notation, MO basis
    e_hf: float                   # RHF energy with the g^2 configuration
    ci_matrix: np.ndarray         # 2x2 closed-shell CI (total energies)
    full_ci_matrix: np.ndarray    # 4x4 Sz=0 full CI (total energies)
    exchange_coupling: float      # K = (gu|gu)


def solve(r_angstrom: float) -> MolecularData:
    """Symmetry-adapted MOs + RHF + closed-shell CI + full 4x4 CI."""
    ints = compute_h2_sto3g(r_angstrom)
    s12 = ints.s12
    c_g = np.array([1.0, 1.0]) / np.sqrt(2.0 * (1.0 + s12))
    c_u = np.array([1.0, -1.0]) / np.sqrt(2.0 * (1.0 - s12))
    mo = np.column_stack([c_g, c_u])

    h_mo = mo.T @ ints.h1 @ mo
    eri_mo = np.einsum("ap,bq,cr,ds,abcd->pqrs", mo, mo, mo, mo, ints.eri,
                       optimize=True)

    g, u = 0, 1
    # RHF with the doubly occupied bonding MO (converged by symmetry).
    e_hf = float(2.0 * h_mo[g, g] + eri_mo[g, g, g, g] + ints.e_nuc)

    # RHF Fock in the MO basis -> orbital energies (canonical at equilibrium).
    fock = np.zeros((2, 2))
    for p in range(2):
        for q in range(2):
            fock[p, q] = h_mo[p, q] + 2.0 * eri_mo[p, q, g, g] - eri_mo[p, g, q, g]
    mo_energies = np.linalg.eigvalsh(fock)

    e_gg = 2.0 * h_mo[g, g] + eri_mo[g, g, g, g] + ints.e_nuc
    e_uu = 2.0 * h_mo[u, u] + eri_mo[u, u, u, u] + ints.e_nuc
    coupling = float(eri_mo[g, u, g, u])          # K = (gu|gu)
    ci_matrix = np.array([[e_gg, coupling], [coupling, e_uu]])

    # --- full CI in the Sz=0 determinant basis (Slater-Condon) -----------
    single_g = float(h_mo[g, u] + eri_mo[u, g, g, g])   # <g^2|H|g.u> spin-summed
    single_u = float(h_mo[g, u] + eri_mo[g, u, u, u])   # <u^2|H|g.u>
    e_open = h_mo[g, g] + h_mo[u, u] + eri_mo[g, g, u, u] + ints.e_nuc
    full = np.array(
        [
            [e_gg, coupling, single_g, single_g],
            [coupling, e_uu, single_u, single_u],
            [single_g, single_u, e_open, coupling],
            [single_g, single_u, coupling, e_open],
        ]
    )
    return MolecularData(
        r_angstrom=r_angstrom,
        integrals=ints,
        mo_coefficients=mo,
        mo_energies=mo_energies,
        h_mo=h_mo,
        eri_mo=eri_mo,
        e_hf=e_hf,
        ci_matrix=ci_matrix,
        full_ci_matrix=full,
        exchange_coupling=coupling,
    )


# Backwards-compatible alias used by older scripts/tests.
solve_rhf = solve


def exact_energy(r_angstrom: float) -> float:
    """Full-CI (minimal basis) total energy at bond length R."""
    data = solve(r_angstrom)
    return float(np.linalg.eigvalsh(data.full_ci_matrix).min())


def reduced_energy(r_angstrom: float) -> float:
    """Ground energy of the 2x2 closed-shell (2-qubit) problem."""
    data = solve(r_angstrom)
    return float(np.linalg.eigvalsh(data.ci_matrix).min())


def qubit_hamiltonian(r_angstrom: float) -> SparsePauliOp:
    """The classic reduced 2-qubit Pauli Hamiltonian of H2, from our integrals."""
    data = solve(r_angstrom)
    a = float(data.ci_matrix[0, 0])
    b = float(data.ci_matrix[1, 1])
    k = data.exchange_coupling

    c0 = 0.25 * (a + b)
    c1 = 0.25 * (a - b)
    c4 = 0.5 * k

    return SparsePauliOp.from_list(
        [
            ("II", c0),
            ("ZI", c1),
            ("IZ", c1),
            ("ZZ", c0),
            ("XX", c4),
            ("YY", -c4),
        ]
    )


def dissociation_curve(r_values) -> list[dict]:
    """HF, reduced CI, full CI and coupling along the bond-length scan."""
    rows = []
    for r in r_values:
        data = solve(float(r))
        e_ci = float(np.linalg.eigvalsh(data.ci_matrix).min())
        e_fci = float(np.linalg.eigvalsh(data.full_ci_matrix).min())
        rows.append(
            {
                "r_angstrom": float(r),
                "e_hf": data.e_hf,
                "e_reduced": e_ci,
                "e_fci": e_fci,
                "correlation": e_fci - data.e_hf,
                "coupling_k": data.exchange_coupling,
                "e_nuc": data.integrals.e_nuc,
            }
        )
    return rows
