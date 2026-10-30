"""QAOA for MaxCut on two small graphs, benchmarked against brute force.

Produces figures/qaoa_maxcut.png and results/qaoa.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from vqeh2.qaoa import maxcut_hamiltonian, run_qaoa

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

graphs = {
    "square (C4)": ([(0, 1), (1, 2), (2, 3), (3, 0)], 4),
    "complete (K4)": ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4),
}

rows = []
for name, (edges, n) in graphs.items():
    result = run_qaoa(edges, n, p=1)
    print(
        f"{name:>14}: <cut> = {result.expectation:.3f} / best {result.best_cut} "
        f"(ratio {result.approximation_ratio:.3f}), "
        f"P(best cut) = {result.best_cut_probability:.3f}"
    )
    rows.append((name, result.expectation, result.best_cut,
                 result.approximation_ratio, result.best_cut_probability))

# ---- Figure: exact cut distribution of the optimized QAOA state ----------
edges, n = graphs["square (C4)"]
hamiltonian = maxcut_hamiltonian(edges, n)
result = run_qaoa(edges, n, p=1)

# rebuild the optimized state for plotting
gamma = Parameter("g")
beta = Parameter("b")
circuit = QuantumCircuit(n)
circuit.h(range(n))
gate = PauliEvolutionGate(hamiltonian, time=gamma)
circuit.append(gate, range(n))
for node in range(n):
    circuit.rx(2.0 * beta, node)
bound = circuit.assign_parameters([result.parameters[0], result.parameters[1]])
probabilities = np.abs(Statevector(bound).data) ** 2

cuts = []
labels = []
for k in range(2**n):
    bits = [(k >> node) & 1 for node in range(n)]
    cut = sum(1 for i, j in edges if bits[i] != bits[j])
    cuts.append(cut)
    labels.append("".join(str(b) for b in reversed(bits)))

fig, ax = plt.subplots(figsize=(8.5, 4.5))
colors = ["C3" if cuts[k] == 4 else "C0" for k in range(2**n)]
ax.bar(labels, probabilities, color=colors)
ax.set_xlabel("basis state (bitstring)")
ax.set_ylabel("probability")
ax.set_title(
    f"QAOA p=1 on C4: optimal cuts (red) get "
    f"{np.sum(probabilities[np.array(cuts) == 4]):.0%} of the probability"
)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIGURES / "qaoa_maxcut.png", dpi=150)
plt.close(fig)

with (RESULTS / "qaoa.csv").open("w") as handle:
    handle.write("graph,mean_cut,best_cut,approximation_ratio,p_best_cut\n")
    for name, mean, best, ratio, p_best in rows:
        handle.write(f"{name},{mean:.4f},{best},{ratio:.4f},{p_best:.4f}\n")
print("Wrote figures/qaoa_maxcut.png, results/qaoa.csv")
