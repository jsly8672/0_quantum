# Qiskit Example: Using a Noisy/Restricted Qubit (Q2)
# Q2: Can only do H, X, Z (no T, no CNOT)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer  # Updated import due to namespace issue
from qiskit.circuit.library import HGate, XGate, ZGate, TGate, CXGate

# Define a 3-qubit quantum circuit
qc = QuantumCircuit(3, 3)

# Q2 is restricted: only H, X, Z
qc.h(2)          # OK on Q2
qc.z(2)          # OK on Q2

# Q1: clean qubit, T gate allowed
qc.t(1)          # T gate on Q1

# Q0 and Q1: clean, can use CNOT
qc.cx(0, 1)      # Control: Q0, Target: Q1

# Q2 cannot be target of CX, so we avoid something like: qc.cx(0, 2)
# Instead, we apply a single X as a placeholder
qc.x(2)          # Still respecting Q2's constraints

# Measurement
qc.measure([0, 1, 2], [0, 1, 2])

# Transpile for Aer simulator (preserves gate-level view)
backend = Aer.get_backend('qasm_simulator')
t_qc = transpile(qc, backend)

# Run and get results (modern Qiskit 1.0+ style)
job = backend.run(t_qc, shots=1024)
result = job.result()
counts = result.get_counts()

print("Result counts:", counts)

# To draw the circuit
print("\nCircuit:")
print(qc.draw())
