# Canvas: IBM Quantum Bell State Connection Test (Direct Execution Mode with Transpilation)
# ============================================================================
# PURPOSE
# -------
# This script demonstrates how to securely connect to IBM Quantum, list all
# available backends (both real QPUs and simulators), prompt the user for a
# selection, build and transpile a simple Bell-state circuit, and execute it in
# direct mode. It includes extensive commentary for educational clarity.
# ============================================================================

import os
import json
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile

# ----------------------------------------------------------------------------
# 1. Load API key securely
# ----------------------------------------------------------------------------
# This section checks for two possible API key files in the current directory:
#   • .ibm_api — a plaintext file containing only the API key
#   • ibm_apikey.json — the JSON file IBM provides when exporting credentials
# If neither file is found, execution halts with a clear error.
# ----------------------------------------------------------------------------
api_key = None
if os.path.exists(".ibm_api"):
    # Load from plaintext key file
    with open(".ibm_api", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
elif os.path.exists("ibm_apikey.json"):
    # Load from IBM JSON key file
    with open("ibm_apikey.json", "r", encoding="utf-8") as f:
        api_key = json.load(f).get("apikey")

# Fail if key not found
if not api_key:
    raise FileNotFoundError("No IBM Quantum API key found (.ibm_api or ibm_apikey.json)")

# ----------------------------------------------------------------------------
# 2. Connect to IBM Quantum Platform
# ----------------------------------------------------------------------------
# Create an authenticated QiskitRuntimeService object using your API key.
# The 'channel' argument must be 'ibm_quantum_platform' for modern accounts.
# This object manages all communication with IBM's Quantum Cloud.
# ----------------------------------------------------------------------------
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
print("Connected successfully to IBM Quantum Platform.")

# ----------------------------------------------------------------------------
# 3. List available backends and build name list
# ----------------------------------------------------------------------------
# Retrieve a list of all devices visible to your account. These may include
# hardware backends (actual QPUs) and sometimes software simulators. The list
# of backends depends on your plan tier (Open, Lite, Standard, etc.).
# ----------------------------------------------------------------------------
backends = service.backends()
print("\nAvailable backends:")
backend_names = [b.name for b in backends]
for i, b in enumerate(backends, start=1):
    try:
        status = b.status().status  # Retrieve operational status if available
    except Exception:
        status = "Unknown"  # Some backends do not expose status data
    print(f"  {i}. {b.name} ({b.num_qubits} qubits, status: {status})")
print(f"  {len(backends)+1}. simulator_mps (cloud simulator)")  # Add manual simulator option

# ----------------------------------------------------------------------------
# 4. Prompt user for backend selection
# ----------------------------------------------------------------------------
# Ask the user which backend to run on. If no input is given, default to the
# cloud simulator. This allows manual testing without consuming QPU queue time.
# ----------------------------------------------------------------------------
choice = input("\nSelect a backend by number (press Enter for simulator): ").strip()
if not choice:
    backend_name = "simulator_mps"
else:
    try:
        # Convert selection to integer index
        choice_index = int(choice) - 1
        if choice_index == len(backends):
            backend_name = "simulator_mps"
        elif 0 <= choice_index < len(backends):
            backend_name = backend_names[choice_index]
        else:
            print("Invalid selection, defaulting to simulator.")
            backend_name = "simulator_mps"
    except ValueError:
        print("Invalid input, defaulting to simulator.")
        backend_name = "simulator_mps"

print(f"\nRunning Bell-state circuit on backend: {backend_name}")

# ----------------------------------------------------------------------------
# 5. Resolve backend for real devices or simulator
# ----------------------------------------------------------------------------
# Try to retrieve the backend object corresponding to the chosen name.
# If this fails (common for Open Plan simulators), fallback to the local
# Aer simulator included with Qiskit. This guarantees the script can always run.
# ----------------------------------------------------------------------------
backend_obj = None
try:
    backend_obj = service.backend(backend_name)
except Exception:
    print("Warning: Could not resolve backend object; using local Aer simulator.")
    try:
        from qiskit.providers.aer import AerSimulator
        backend_obj = AerSimulator()
    except Exception:
        backend_obj = None

# ----------------------------------------------------------------------------
# 6. Create and transpile a Bell-state circuit
# ----------------------------------------------------------------------------
# The Bell-state circuit creates quantum entanglement:
#   • Start with |00>
#   • Apply a Hadamard gate on qubit 0 to create superposition → (|00> + |10>)/√2
#   • Apply a CNOT with control=0, target=1 to entangle → (|00> + |11>)/√2
#   • Measure both qubits in the computational basis.
# Transpilation converts unsupported gates (like 'h') into a valid sequence of
# the backend’s native basis gates (e.g., 'rz', 'sx', 'cx') so real hardware can
# execute them. Simulators don’t require this, but hardware backends do.
# ----------------------------------------------------------------------------
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

if backend_obj:
    qc = transpile(qc, backend_obj)  # Convert circuit into hardware-compatible form

# ----------------------------------------------------------------------------
# 7. Run the circuit in DIRECT EXECUTION mode using mode=backend_obj
# ----------------------------------------------------------------------------
# Qiskit Runtime’s Sampler primitive is the modern way to execute circuits. It
# provides shot-based measurement results and abstracts away backend details.
# The 'mode' argument must receive the backend object to define where execution
# will occur. Circuits must be wrapped in a list, even if only one circuit is run.
# ----------------------------------------------------------------------------

def run_bell_state_direct(backend_obj, qc):
    """Run a Bell-state circuit using direct Sampler execution with transpilation."""
    # Ensure a backend exists before attempting execution
    if backend_obj is None:
        raise RuntimeError("No valid backend found for execution.")

    # Instantiate Sampler with backend context
    sampler = Sampler(mode=backend_obj)

    # Submit the transpiled circuit for execution. The list wrapper is required.
    job = sampler.run([qc])

    # Block until results return. In large jobs, this can take minutes to hours.
    result = job.result()

    # Return full result object (contains quasi-distributions, metadata, etc.)
    return result

# ----------------------------------------------------------------------------
# 8. Execute the circuit and display results
# ----------------------------------------------------------------------------
# The output of 'result' depends on backend and API version. It generally
# includes counts or quasi-probability distributions of measurement outcomes.
# ----------------------------------------------------------------------------
result = run_bell_state_direct(backend_obj, qc)
print("\nBell-state job complete. Raw result:")
print(result)

# ----------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------
# Notes:
# • This script uses the most current Qiskit Runtime v2 API style.
# • It supports both real QPU execution and local simulation.
# • Transpilation is mandatory for hardware; optional for simulators.
# • You can verify execution progress on the IBM Quantum dashboard.
# ============================================================================
