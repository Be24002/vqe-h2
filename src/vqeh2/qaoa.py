"""QAOA for MaxCut with an exact classical benchmark.

MaxCut on a graph G: assign +/-1 spins to vertices to maximize cut edges.
The QAOA cost Hamiltonian is

    C = sum_(i,j) in E (I - Z_i Z_j) / 2

and one QAOA layer alternates evolving under C (angle gamma) and under a
mixer sum_i X_i (angle beta). We evaluate the final state exactly
(statevector) so the only approximation left is the variational
optimization itself -- a clean measure of what p=1 QAOA can and cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


def maxcut_hamiltonian(edges: list[tuple[int, int]], n_nodes: int) -> SparsePauliOp:
    """C = sum_edges (I - Z_i Z_j)/2 as a Pauli operator."""
    terms = []
    for i, j in edges:
        chars = ["I"] * n_nodes
        chars[i] = "Z"
        chars[j] = "Z"
        label = "".join(reversed(chars))  # Qiskit little-endian
        terms.append(("I" * n_nodes, 0.5))
        terms.append((label, -0.5))
    return SparsePauliOp.from_list(terms).simplify()


def cut_value(bits: np.ndarray, edges: list[tuple[int, int]]) -> int:
    """Classical cut of a spin assignment bits in {0,1}."""
    return int(sum(1 for i, j in edges if bits[i] != bits[j]))


def brute_force_maxcut(edges: list[tuple[int, int]], n_nodes: int) -> int:
    """Exhaustive best cut (only for small graphs)."""
    best = 0
    for mask in range(2**n_nodes):
        bits = np.array([(mask >> k) & 1 for k in range(n_nodes)])
        best = max(best, cut_value(bits, edges))
    return best


def _qaoa_circuit(hamiltonian: SparsePauliOp, n_nodes: int, p: int):
    gamma = [Parameter(f"g{k}") for k in range(p)]
    beta = [Parameter(f"b{k}") for k in range(p)]
    circuit = QuantumCircuit(n_nodes)
    circuit.h(range(n_nodes))
    for layer in range(p):
        # exp(-i gamma C) via the Pauli evolution gate
        gate = PauliEvolutionGate(hamiltonian, time=gamma[layer])
        circuit.append(gate, range(n_nodes))
        for node in range(n_nodes):
            circuit.rx(2.0 * beta[layer], node)
    return circuit, gamma + beta


@dataclass
class QAOAResult:
    expectation: float
    best_cut: int
    best_cut_probability: float
    approximation_ratio: float
    parameters: np.ndarray


def run_qaoa(edges: list[tuple[int, int]], n_nodes: int, p: int = 1,
             seed: int = 3) -> QAOAResult:
    """Optimize p-layer QAOA on the exact statevector."""
    hamiltonian = maxcut_hamiltonian(edges, n_nodes)
    circuit, params = _qaoa_circuit(hamiltonian, n_nodes, p)
    best_cut_classical = brute_force_maxcut(edges, n_nodes)

    rng = np.random.default_rng(seed)

    def expectation(x: np.ndarray) -> float:
        bound = circuit.assign_parameters(dict(zip(params, x)))
        state = Statevector(bound)
        value = float(np.real(state.expectation_value(hamiltonian)))
        return -value  # maximize cut -> minimize negative

    result = minimize(
        expectation,
        x0=rng.uniform(0.1, 0.6, size=len(params)),
        method="COBYLA",
        options={"maxiter": 400},
    )

    bound = circuit.assign_parameters(dict(zip(params, result.x)))
    probabilities = np.abs(Statevector(bound).data) ** 2
    cuts = np.array(
        [
            cut_value(np.array([(k >> node) & 1 for node in range(n_nodes)]), edges)
            for k in range(2**n_nodes)
        ]
    )
    mean_cut = float(np.sum(probabilities * cuts))
    best_probability = float(np.sum(probabilities[cuts == best_cut_classical]))
    return QAOAResult(
        expectation=mean_cut,
        best_cut=best_cut_classical,
        best_cut_probability=best_probability,
        approximation_ratio=mean_cut / best_cut_classical,
        parameters=np.asarray(result.x),
    )
