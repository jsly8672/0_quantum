from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram

# ------------------------------
# Quantum Teleportation Protocol
# ------------------------------
# Goal: Transfer an *unknown* quantum state from Alice (qubit 0) to Bob (qubit 2)
# using:
#   - Entanglement
#   - Measurement
#   - Classical communication
#
# This is a foundational quantum networking protocol, showing how information
# can be transmitted without physically moving the qubit.
# ------------------------------

# Step 1: Create a quantum circuit
# - 3 qubits: [0] is the unknown state |ψ>, [1] Alice's half of entangled pair, [2] Bob's half
# - 2 classical bits: store Alice's measurement results
qc = QuantumCircuit(3, 2)

# Step 2: Prepare the unknown state |ψ> on qubit 0
# For this demo, we'll construct |ψ> = H + Phase(0.5)|0>
# This gives a state that's not just |0>, |1>, or |+>, but something more general.
qc.h(0)        # Hadamard puts qubit 0 into superposition (|0> + |1>)/√2
qc.p(0.5, 0)   # Add a small phase rotation → now qubit 0 holds |ψ>

# Step 3: Create entanglement between Alice and Bob
# - Alice has qubit 1, Bob has qubit 2
# - Use H + CNOT to generate a Bell pair (|00> + |11>)/√2
qc.h(1)        # Put qubit 1 into superposition
qc.cx(1, 2)    # Entangle qubit 1 with qubit 2

# Step 4: Alice performs a Bell measurement on qubits 0 and 1
# This entangles the unknown state with her half of the Bell pair,
# and projects them into one of 4 possible Bell states.
qc.cx(0, 1)    # Entangle qubit 0 (unknown state) with qubit 1
qc.h(0)        # Put qubit 0 into superposition → completes Bell basis transform

# Step 5: Alice measures her two qubits
# The results (00, 01, 10, 11) determine which correction Bob must apply.
qc.barrier()   # Visual separator in the circuit diagram
qc.measure([0, 1], [0, 1])

# Step 6: Bob applies conditional corrections
# - If Alice measures 01 → Bob applies X
# - If Alice measures 10 → Bob applies Z
# - If Alice measures 11 → Bob applies X and Z
# - If Alice measures 00 → no correction needed
qc.x(2).c_if(0, 1)  # Apply X if classical bit 0 == 1
qc.z(2).c_if(1, 1)  # Apply Z if classical bit 1 == 1

# At this point, qubit 2 holds the original |ψ>, even though it was never sent!

# ------------------------------
# Visualization & Simulation
# ------------------------------
print(qc.draw())  # Print the circuit diagram in text form

# Use the Aer simulator to examine the final statevector
sim = Aer.get_backend('aer_simulator')
qc.save_statevector()  # Save final quantum state for inspection
result = sim.run(qc).result()
statevector = result.get_statevector()

# Print the statevector (mostly to confirm Bob's qubit == original |ψ>)
print("Final statevector:")
print(statevector)

# Optionally: plot the Bloch sphere of the final state
# (requires matplotlib to display)
# plot_bloch_multivector(statevector)

# Optionally: run multiple shots to see measurement statistics
# transpiled = transpile(qc, sim)
# qobj = assemble(transpiled, shots=1024)
# result = sim.run(qobj).result()
# counts = result.get_counts()
# plot_histogram(counts).show()
