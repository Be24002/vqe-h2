"""Variational Quantum Eigensolver on the reduced H2 Hamiltonian.

A hardware-efficient 2-qubit ansatz (Ry Ry + CNOT entangler + Ry Ry) is
optimized with scipy against the energy expectation value computed with a
Qiskit estimator primitive (exact statevector evaluation by default --
swap in a real backend session and nothing else changes). Because the
ansatz includes the exact ground state's entanglement structure, VQE
reaches the FCI energy to optimizer tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


def build_ansatz(reps: int = 1) -> tuple[QuantumCircuit, ParameterVector]:
    """Hardware-efficient ansatz: Ry layer(s) interleaved with CNOT entanglers."""
    theta = ParameterVector("t", 2 * (reps + 1))
    circuit = QuantumCircuit(2)
    index = 0
    for layer in range(reps + 1):
        circuit.ry(theta[index], 0)
        circuit.ry(theta[index + 1], 1)
        index += 2
        if layer < reps:
            circuit.cx(0, 1)
    return circuit, theta


@dataclass
class VQEResult:
    energy: float
    optimal_parameters: np.ndarray
    history: list[float] = field(default_factory=list)
    n_evaluations: int = 0


def run_vqe(
    hamiltonian: SparsePauliOp,
    reps: int = 1,
    maxiter: int = 200,
    seed: int = 7,
    method: str = "COBYLA",
) -> VQEResult:
    """Minimize <psi(theta)|H|psi(theta)> over the ansatz parameters."""
    circuit, theta = build_ansatz(reps)
    estimator = StatevectorEstimator(seed=seed)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-0.1, 0.1, size=len(theta))

    history: list[float] = []

    def cost(x: np.ndarray) -> float:
        job = estimator.run([(circuit, hamiltonian, x)])
        value = float(job.result()[0].data.evs.item())
        history.append(value)
        return value

    result = minimize(cost, x0, method=method,
                      options={"maxiter": maxiter} if method == "COBYLA" else {})
    return VQEResult(
        energy=float(result.fun),
        optimal_parameters=np.asarray(result.x),
        history=history,
        n_evaluations=len(history),
    )
