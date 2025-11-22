from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

qc = QuantumCircuit(1)
qc.x(0)  # Apply Pauli-X gate

qc.save_statevector(label='final_state')  # name it explicitly

backend = Aer.get_backend('statevector_simulator')
compiled = transpile(qc, backend)
result = backend.run(compiled).result()

# Access by index and key
statevector = result.data(0)['final_state']
print(statevector)

