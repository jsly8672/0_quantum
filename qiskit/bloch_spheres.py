from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_vector
import numpy as np
import matplotlib.pyplot as plt

def apply_gate_and_get_bloch_vector(gate):
    qc = QuantumCircuit(1)
    if gate == 'x':
        qc.x(0)
    elif gate == 'y':
        qc.y(0)
    elif gate == 'z':
        qc.z(0)

    qc.save_statevector(label='final_state')
    backend = Aer.get_backend('statevector_simulator')
    compiled = transpile(qc, backend)
    result = backend.run(compiled).result()
    statevector = result.data(0)['final_state']

    # Convert to Bloch vector
    bloch_vector = [
        2 * np.real(np.conj(statevector[0]) * statevector[1]),
        2 * np.imag(np.conj(statevector[0]) * statevector[1]),
        np.abs(statevector[0])**2 - np.abs(statevector[1])**2
    ]
    return bloch_vector

# Gates and labels
gates = ['i', 'x', 'y', 'z']
titles = {
    'i': 'Identity (|0⟩)',
    'x': 'Pauli-X Gate',
    'y': 'Pauli-Y Gate',
    'z': 'Pauli-Z Gate'
}

# Plot with visible title
for gate in gates:
    vec = apply_gate_and_get_bloch_vector(gate)
    plot_bloch_vector(vec)
    plt.gcf().suptitle(titles[gate], fontsize=14)  # Sets the title on the figure
    plt.show()
