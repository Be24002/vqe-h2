"""vqeh2: variational quantum algorithms for real chemistry, from scratch.

Modules
-------
integrals
    STO-3G Gaussian-integral engine for H2 (overlap, kinetic, nuclear
    attraction, electron repulsion) -- closed forms, no quantum-chemistry
    package required.
hamiltonian
    Second quantization, qubit mapping (parity with 2-qubit reduction) and
    exact diagonalization via qiskit-nature.
vqe
    Variational Quantum Eigensolver with Qiskit primitives + scipy optimizers.
qaoa
    QAOA for MaxCut with brute-force classical benchmark.
"""

__version__ = "0.1.0"
