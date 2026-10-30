"""STO-3G contracted Gaussian integrals for the H2 molecule, from scratch.

Every quantum-chemistry calculation starts by evaluating integrals over
basis functions. We use the standard minimal basis STO-3G: each hydrogen 1s
orbital is a contraction of 3 primitive Gaussians (Hehre, Stewart & Pople,
1969). For s-type Gaussians all integrals have closed forms (Szabo &
Ostlund, *Modern Quantum Chemistry*, ch. 3 / appendix B):

    overlap:            (a|b) = (pi/p)^(3/2) exp(-mu R_AB^2)
    kinetic:            <a|T|b> = mu (3 - 2 mu R_AB^2) (a|b)
    nuclear attraction: <a|-Z_C/r_C|b> = -2 pi Z_C / p (a|b) F0(p|P-C|^2)
    electron repulsion: (ab|cd) = 2 pi^2 / (p q) sqrt(pi/(p+q))
                              x exp(-mu R_AB^2 - nu R_CD^2)
                              x F0(pq/(p+q) |P-Q|^2)

with p = a+b, q = c+d, mu = ab/p, nu = cd/q, P (Q) the Gaussian product
centers and F0(t) = (1/2) sqrt(pi/t) erf(sqrt(t)) the Boys function.

Units: bohr (1 bohr = 0.529177 A); energies in Hartree.
Anchors: at R = 1.4 bohr this engine reproduces the classic Szabo & Ostlund
STO-3G H2 integral table (S12 = 0.6593, T11 = 0.7600, ...), and the FCI
energy of -1.1373 Ha near R = 0.735 A -- see the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOHR_ANGSTROM = 0.529177

# STO-3G contraction for hydrogen 1s (Hehre, Stewart, Pople 1969, Table I).
STO3G_H_EXPONENTS = np.array([3.42525091, 0.62391373, 0.16885540])
STO3G_H_COEFFICIENTS = np.array([0.15432897, 0.53532814, 0.44463454])


def boys0(t: float) -> float:
    """Boys function F0(t) = (1/2) sqrt(pi/t) erf(sqrt(t)); F0(0) = 1."""
    if t < 1e-12:
        return 1.0
    return _boys(t)


def _boys(t: float) -> float:
    from scipy.special import erf

    return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))


@dataclass
class H2Integrals:
    """One- and two-electron integrals plus nuclear repulsion for H2 STO-3G."""

    h1: np.ndarray        # (2, 2) core Hamiltonian, Hartree
    eri: np.ndarray       # (2, 2, 2, 2) chemist notation (mu nu | lam sig)
    e_nuc: float          # 1/R nuclear repulsion, Hartree
    r_bohr: float

    @property
    def s12(self) -> float:
        """Basis overlap S12 = <phi_A | phi_B> (also stored inside h1 block)."""
        return float(self._s12)

    _s12: float = 0.0
    _t11: float = 0.0

    @property
    def t11(self) -> float:
        return float(self._t11)


def _norm(alpha: float) -> float:
    """Normalization of a primitive s-Gaussian: N = (2a/pi)^(3/4)."""
    return (2.0 * alpha / np.pi) ** 0.75


def _overlap_primitive(alpha: float, beta: float, rab2: float) -> float:
    p = alpha + beta
    mu = alpha * beta / p
    return _norm(alpha) * _norm(beta) * (np.pi / p) ** 1.5 * np.exp(-mu * rab2)


def _kinetic_primitive(alpha: float, beta: float, rab2: float) -> float:
    p = alpha + beta
    mu = alpha * beta / p
    return mu * (3.0 - 2.0 * mu * rab2) * _overlap_primitive(alpha, beta, rab2)


def _nuclear_primitive(alpha: float, beta: float, a: np.ndarray, b: np.ndarray,
                       center: np.ndarray, z: float) -> float:
    """<a | -Z_C / |r - C| | b> for normalized s primitives."""
    p = alpha + beta
    mu = alpha * beta / p
    product_center = (alpha * a + beta * b) / p
    rab2 = float(np.dot(a - b, a - b))
    rpc2 = float(np.dot(product_center - center, product_center - center))
    return (
        -2.0 * np.pi * z / p
        * _norm(alpha) * _norm(beta)
        * np.exp(-mu * rab2)
        * boys0(p * rpc2)
    )


def _eri_primitive(alpha, beta, gamma, delta, a, b, c, d) -> float:
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rab2 = float(np.dot(a - b, a - b))
    rcd2 = float(np.dot(c - d, c - d))
    big_p = (alpha * a + beta * b) / p
    big_q = (gamma * c + delta * d) / q
    rpq2 = float(np.dot(big_p - big_q, big_p - big_q))
    prefactor = (
        _norm(alpha) * _norm(beta) * _norm(gamma) * _norm(delta)
        * 2.0 * np.pi**2 / (p * q) * np.sqrt(np.pi / (p + q))
    )
    return (
        prefactor
        * np.exp(-mu * rab2 - nu * rcd2)
        * boys0(p * q / (p + q) * rpq2)
    )


def compute_h2_sto3g(r_angstrom: float) -> H2Integrals:
    """All STO-3G integrals for H2 at bond length ``r_angstrom`` (Angstrom)."""
    r = r_angstrom / BOHR_ANGSTROM
    centers = [np.array([0.0, 0.0, -r / 2]), np.array([0.0, 0.0, r / 2])]
    exponents = STO3G_H_EXPONENTS
    coefficients = STO3G_H_COEFFICIENTS
    n_basis = 2

    # --- one-electron integrals (contracted) -----------------------------
    h1 = np.zeros((n_basis, n_basis))
    s_matrix = np.zeros((n_basis, n_basis))
    kinetic = np.zeros((n_basis, n_basis))
    for mu in range(n_basis):
        for nu in range(n_basis):
            rab2 = float(np.dot(centers[mu] - centers[nu], centers[mu] - centers[nu]))
            s = k = v = 0.0
            for i, ci in enumerate(coefficients):
                for j, cj in enumerate(coefficients):
                    a, b = exponents[i], exponents[j]
                    primitive_overlap = _overlap_primitive(a, b, rab2)
                    s += ci * cj * primitive_overlap
                    k += ci * cj * _kinetic_primitive(a, b, rab2)
                    for atom, center in enumerate(centers):
                        v += ci * cj * _nuclear_primitive(
                            a, b, centers[mu], centers[nu], center, z=1.0
                        )
            s_matrix[mu, nu] = s
            kinetic[mu, nu] = k
            h1[mu, nu] = k + v

    # --- two-electron integrals, chemist notation ------------------------
    eri = np.zeros((n_basis,) * 4)
    for mu in range(n_basis):
        for nu in range(n_basis):
            for lam in range(n_basis):
                for sig in range(n_basis):
                    value = 0.0
                    for (i, ci), (j, cj), (m, cm), (n, cn) in _quad(coefficients):
                        value += ci * cj * cm * cn * _eri_primitive(
                            exponents[i], exponents[j], exponents[m], exponents[n],
                            centers[mu], centers[nu], centers[lam], centers[sig],
                        )
                    eri[mu, nu, lam, sig] = value

    return H2Integrals(
        h1=h1, eri=eri, e_nuc=1.0 / r, r_bohr=r,
        _s12=float(s_matrix[0, 1]), _t11=float(kinetic[0, 0]),
    )


def _quad(coefficients):
    """All index/coefficient quadruples for the contraction loops."""
    for i, ci in enumerate(coefficients):
        for j, cj in enumerate(coefficients):
            for m, cm in enumerate(coefficients):
                for n, cn in enumerate(coefficients):
                    yield (i, ci), (j, cj), (m, cm), (n, cn)
