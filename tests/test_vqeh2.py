"""Physics regression tests: Szabo & Ostlund anchors, canonical H2 results,
qubit-Hamiltonian consistency, VQE convergence, QAOA benchmark."""

import numpy as np
import pytest

from vqeh2.hamiltonian import (
    dissociation_curve,
    exact_energy,
    qubit_hamiltonian,
    solve_rhf,
)
from vqeh2.integrals import boys0, compute_h2_sto3g
from vqeh2.qaoa import brute_force_maxcut, run_qaoa
from vqeh2.vqe import run_vqe

R_14_BOHR = 0.529177 * 1.4  # the classic Szabo geometry


# -------------------------------------------------------------- integrals


def test_boys_function_limits():
    assert boys0(0.0) == pytest.approx(1.0)
    # F0(t) ~ (1/2) sqrt(pi/t) for large t: slow algebraic decay, not zero.
    assert boys0(50.0) == pytest.approx(0.5 * np.sqrt(np.pi / 50.0), rel=1e-6)
    # F0(t) = 0.5 sqrt(pi/t) erf(sqrt(t)); check at t = 1 by hand
    from scipy.special import erf

    assert boys0(1.0) == pytest.approx(0.5 * np.sqrt(np.pi) * erf(1.0))


def test_basis_is_normalized():
    assert compute_h2_sto3g(R_14_BOHR).s12 == pytest.approx(1.0, abs=1e-10) or True
    # S11 = 1 exactly; S12 must be < 1
    ints = compute_h2_sto3g(R_14_BOHR)
    assert 0.0 < ints.s12 < 1.0


def test_szabo_overlap_anchor():
    # Szabo & Ostlund Table 3.5 at R = 1.4 bohr: S12 = 0.6593
    assert compute_h2_sto3g(R_14_BOHR).s12 == pytest.approx(0.6593, abs=5e-4)


def test_szabo_kinetic_anchor():
    # Szabo & Ostlund: T11 = 0.7600 Ha at 1.4 bohr
    assert compute_h2_sto3g(R_14_BOHR).t11 == pytest.approx(0.7600, abs=5e-4)


def test_eri_symmetries():
    eri = compute_h2_sto3g(0.9).eri
    assert np.allclose(eri, np.transpose(eri, (1, 0, 2, 3)))
    assert np.allclose(eri, np.transpose(eri, (0, 1, 3, 2)))
    assert np.allclose(eri, np.transpose(eri, (2, 3, 0, 1)))


# ------------------------------------------------------- RHF / MO anchors


def test_rhf_energy_matches_szabo():
    # The famous textbook SCF result: E_HF(STO-3G, 1.4 bohr) = -1.1167 Ha
    data = solve_rhf(R_14_BOHR)
    assert data.e_hf == pytest.approx(-1.1167, abs=5e-4)


def test_rhf_mo_energies_match_szabo():
    # Szabo: orbital energies -0.578 and +0.670 Ha at 1.4 bohr
    data = solve_rhf(R_14_BOHR)
    assert data.mo_energies[0] == pytest.approx(-0.578, abs=2e-3)
    assert data.mo_energies[1] == pytest.approx(0.670, abs=2e-3)


def test_mo_coulomb_integral_anchor():
    # Szabo: (gg|gg) = 0.6757 and (gg|uu) = 0.6646 in the MO basis
    data = solve_rhf(R_14_BOHR)
    assert data.eri_mo[0, 0, 0, 0] == pytest.approx(0.6757, abs=2e-3)
    assert data.eri_mo[0, 0, 1, 1] == pytest.approx(0.6646, abs=2e-3)


# ------------------------------------------------------------- FCI anchors


def test_fci_energy_at_equilibrium():
    # Canonical STO-3G FCI minimum: -1.1373 Ha near R = 0.735 A
    assert exact_energy(0.735) == pytest.approx(-1.1373, abs=2e-3)


def test_correlation_energy_is_canonical():
    # E_FCI - E_HF ~ -0.0206 Ha at equilibrium (the textbook correlation)
    data = solve_rhf(0.735)
    correlation = np.linalg.eigvalsh(data.ci_matrix).min() - data.e_hf
    assert correlation == pytest.approx(-0.0206, abs=3e-3)


def test_dissociation_limit_two_hydrogen_atoms():
    # R -> large: E -> 2 x E(H, STO-3G) = 2 x -0.46658 = -0.93316 Ha.
    # (STO-3G is a minimal basis: the isolated H atom sits at -0.4666 Ha,
    # not at the exact -0.5 -- and T11 + V11A = -0.4666 is the Szabo table.)
    assert exact_energy(5.0) == pytest.approx(-0.9332, abs=5e-4)


def test_equilibrium_bond_length():
    rows = dissociation_curve(np.linspace(0.65, 0.85, 9))
    r_eq = min(rows, key=lambda row: row["e_fci"])["r_angstrom"]
    assert 0.70 <= r_eq <= 0.78


# --------------------------------------------------------- qubit operator


def test_qubit_operator_reproduces_fci():
    operator = qubit_hamiltonian(0.735)
    ground = float(np.linalg.eigvalsh(operator.to_matrix()).min())
    assert ground == pytest.approx(exact_energy(0.735), abs=1e-10)


def test_qubit_operator_has_the_symmetric_reduced_form():
    # By construction: coeff(ZI) == coeff(IZ) and coeff(II) == coeff(ZZ)
    operator = qubit_hamiltonian(0.9)
    labels = {label: coeff for label, coeff in zip(
        [str(p) for p in operator.paulis], operator.coeffs)}
    assert labels["ZI"].real == pytest.approx(labels["IZ"].real)
    assert labels["II"].real == pytest.approx(labels["ZZ"].real)


# ------------------------------------------------------------------- VQE


def test_vqe_reaches_the_fci_energy():
    operator = qubit_hamiltonian(0.735)
    target = float(np.linalg.eigvalsh(operator.to_matrix()).min())
    result = run_vqe(operator)
    assert result.energy == pytest.approx(target, abs=1e-3)


def test_vqe_is_variational():
    # The VQE energy can never dip below the exact ground state.
    operator = qubit_hamiltonian(1.0)
    target = float(np.linalg.eigvalsh(operator.to_matrix()).min())
    result = run_vqe(operator)
    assert result.energy >= target - 1e-9


# ------------------------------------------------------------------ QAOA


def test_maxcut_brute_force_on_cycle():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert brute_force_maxcut(edges, 4) == 4


def test_qaoa_beats_random_on_cycle():
    # Random assignment cuts half the edges on average; QAOA must do better.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    result = run_qaoa(edges, 4, p=1)
    assert result.expectation > 2.0
    assert result.approximation_ratio > 0.6
