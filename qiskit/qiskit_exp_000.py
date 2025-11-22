from qiskit_aer import Aer
from qiskit import QuantumCircuit, execute
import matplotlib.pyplot as plt

# Create and simulate a basic circuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

sim = Aer.get_backend('qasm_simulator')
job = execute(qc, backend=sim, shots=1000)
counts = job.result().get_counts()

print(counts)
plot_histogram(counts)
plt.show()
from qiskit_aer import Aer
from qiskit import QuantumCircuit, execute
