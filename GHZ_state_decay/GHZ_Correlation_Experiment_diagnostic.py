"""GHZ Correlation Experiment
================================

This script prepares an N-qubit GHZ state on an IBM backend (e.g., ibm_torino),
measures it, and computes pairwise Z–Z correlations between qubit 0 and qubit i
for i = 1..N-1. It then plots ⟨Z0 Zi⟩ vs distance i, similar in spirit to
IBM's 100-qubit GHZ demo plot.

Phase 1: raw correlations only (no mitigation). We are looking directly at
what the hardware does without correction. Readout error mitigation and ZNE
could be added later, but this file is deliberately kept to the "bare metal"
experiment.
"""

import os
import json
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile

# ---------------------------------------------------------------------------
# 1. API KEY LOADING AND SERVICE SETUP
# ---------------------------------------------------------------------------
# This section mirrors the pattern used in the Bell / error-mitigation
# projects: we try to load an API key from local files first, and only if that
# fails do we rely on a previously saved Qiskit account configuration.
#
# This keeps credentials:
#   * out of the source code,
#   * under your control in the project directory,
#   * and compatible with other scripts that use the same helper.


def load_api_key() -> str | None:
    """Load IBM Quantum API key from local files, if present.

    Search order:
      1. .ibm_api          (plaintext file with just the token)
      2. ibm_apikey.json   (JSON file with an "apikey" field)

    Returns:
        The API key string if found, otherwise None. Returning None means
        "let QiskitRuntimeService() discover any saved account on disk".
    """
    api_key = None

    # Preferred: simple plaintext file that contains *only* the token.
    if os.path.exists(".ibm_api"):
        with open(".ibm_api", "r", encoding="utf-8") as f:
            api_key = f.read().strip()

    # Fallback: JSON file exported from IBM Cloud, containing an "apikey".
    elif os.path.exists("ibm_apikey.json"):
        with open("ibm_apikey.json", "r", encoding="utf-8") as f:
            api_key = json.load(f).get("apikey")

    # If neither file exists, api_key remains None.
    return api_key


def get_service() -> QiskitRuntimeService:
    """Create a QiskitRuntimeService instance.

    If we have an explicit token from load_api_key(), we feed it directly to
    QiskitRuntimeService with the "ibm_quantum_platform" channel (the modern
    IBM Quantum API). Otherwise, we assume the user has previously saved an
    account via QiskitRuntimeService.save_account(), and the no-argument
    constructor will pick that up from ~/.qiskit.
    """
    key = load_api_key()
    if key:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=key)
    return QiskitRuntimeService()


def get_backend(service: QiskitRuntimeService, backend_name: str = "ibm_torino"):
    """Select a backend to run on (default: ibm_torino).

    Args:
        service:      A QiskitRuntimeService object.
        backend_name: String name of the backend (e.g., "ibm_torino").

    Returns:
        A backend object compatible with the Sampler primitive.
    """
    return service.backend(backend_name)


# ---------------------------------------------------------------------------
# 2. GHZ CIRCUIT CONSTRUCTION
# ---------------------------------------------------------------------------
# Here we build the logical N-qubit GHZ state circuit:
#   |GHZ_N> = (|0...0> + |1...1>)/√2
# using the standard recipe:
#   * Start in |0...0>.
#   * Apply H to qubit 0 → (|0> + |1>)/√2 ⊗ |0...0>.
#   * Apply a CNOT chain: 0→1, 1→2, ..., (N-2)→(N-1) to propagate the
#     superposition across the line.
#   * Measure every qubit in the Z basis.


def build_ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """Build an N-qubit GHZ circuit with measurements on all qubits.

    Logical structure:
      * Start in |0...0>.
      * Apply H on qubit 0 → (|0> + |1>)/√2 ⊗ |0...0>.
      * Apply a chain of CX gates: 0→1, 1→2, ..., (N-2)→(N-1).
        This entangles all qubits into (|0...0> + |1...1>)/√2.
      * Measure each qubit i into classical bit i.
    """
    # Quantum and classical registers have the same size; each qubit has a
    # dedicated classical bit for its measurement result.
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Create a superposition on the first qubit.
    qc.h(0)

    # Step 2: Entangle the rest via a CX chain.
    # After this loop, qubits are ideally in (|0...0> + |1...1>)/√2.
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Step 3: Measure each qubit into the matching classical bit index.
    # Qubit i → classical bit i.
    for i in range(n_qubits):
        qc.measure(i, i)

    return qc


# ---------------------------------------------------------------------------
# 3. UNIVERSAL COUNT EXTRACTION FOR SAMPLER V2
# ---------------------------------------------------------------------------
# IBM's Runtime API has evolved, and the way measurement results are wrapped
# inside the "pub_result.data" object can vary slightly across versions.
# Instead of assuming a specific layout, we:
#   * print the DataBin structure (for debugging), and
#   * try multiple known access patterns to get a counts dictionary.
#
# This mirrors the defensive approach used in the Bell-state project.


def extract_counts(pub_result) -> Dict[str, int]:
    """Extract bitstring counts from a SamplerV2 pub_result.

    The SamplerV2 result structure wraps measurement data in a DataBin-like
    object. Depending on the exact qiskit_ibm_runtime version, counts may be
    accessible via:
      * data.meas.get_counts(), or
      * data.get_counts(), or
      * some other attribute that exposes a get_counts() method.

    This function inspects the structure and picks whatever path is available.
    """
    data = pub_result.data
    print("Inspecting DataBin:", data)
    print("Attributes:", dir(data))

    # Case 1: data.meas.get_counts()
    if hasattr(data, "meas"):
        meas = data.meas
        if hasattr(meas, "get_counts"):
            return meas.get_counts()

    # Case 2: data.get_counts()
    if hasattr(data, "get_counts"):
        return data.get_counts()

    # Case 3: scan for something with get_counts()
    for name in dir(data):
        if not name.startswith("_"):
            obj = getattr(data, name)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()

    # If we get here, we do not recognize the structure.
    raise RuntimeError("Unable to locate counts in DataBin structure.")


# ---------------------------------------------------------------------------
# 4. CORRELATION CALCULATION ⟨Z0 Zi⟩
# ---------------------------------------------------------------------------
# We want to compute the correlation function:
#       ⟨Z0 Zi⟩   for i = 1..N-1
# where Zi is the Pauli-Z observable on qubit i.
#
# For a bitstring outcome s (e.g., '01011') we map each measured bit to a
# Z eigenvalue:
#       bit '0' → +1
#       bit '1' → -1
# so that Z acting on |0> gives +1 and on |1> gives -1.
#
# Then, for each bitstring s with probability p_s, the contribution to
# ⟨Z0 Zi⟩ is:
#       Z0(s) * Zi(s) * p_s
# where Zk(s) is the eigenvalue (+1 or -1) read off from the k-th qubit in s.


def z_value(bit: str) -> int:
    """Map a classical bit ('0' or '1') to its Z eigenvalue (+1 or -1)."""
    return +1 if bit == "0" else -1


def compute_z0_zi_correlations(counts: Dict[str, int], n_qubits: int) -> List[float]:
    """Compute ⟨Z0 Zi⟩ for i = 1..N-1 from raw bitstring counts.

    Bitstring convention in Qiskit:
        * For an N-bit classical register, bitstrings are reported as
          c_{N-1} c_{N-2} ... c_0 (most significant bit is highest index).
        * We measured qubit i into classical bit i, so the mapping from
          qubit index → string index is:
              qubit 0 → c_0  → bitstring[-1]
              qubit i → c_i  → bitstring[-1 - i]

    For each observed bitstring s with count n_s:
        z0 = (+1 if s[-1]     == '0' else -1)
        zi = (+1 if s[-1 - i] == '0' else -1)
        contribution to ⟨Z0 Zi⟩ is:
            z0 * zi * (n_s / total_shots).

    The function returns a list [⟨Z0 Z1⟩, ⟨Z0 Z2⟩, ..., ⟨Z0 Z_{N-1}⟩].
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        # If this happens, something is fundamentally wrong with the run.
        raise ValueError("No counts to compute correlations from.")

    correlations = []

    # Precompute per-bitstring Z eigenvalues for all qubits to avoid recomputing
    # the mapping inside the nested loop. z_vals[s][i] = Z eigenvalue for qubit
    # i given outcome s.
    z_vals: Dict[str, List[int]] = {}
    for bitstring in counts.keys():
        if len(bitstring) != n_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != n_qubits {n_qubits}")
        # Build list of Z eigenvalues for qubits 0..N-1.
        # Remember: qubit i corresponds to string index -1-i.
        z_list = []
        for i in range(n_qubits):
            bit = bitstring[-1 - i]
            z_list.append(z_value(bit))
        z_vals[bitstring] = z_list

    # Now compute ⟨Z0 Zi⟩ for each i.
    for i in range(1, n_qubits):
        acc = 0.0
        for bitstring, n_s in counts.items():
            z0 = z_vals[bitstring][0]   # eigenvalue for qubit 0
            zi = z_vals[bitstring][i]   # eigenvalue for qubit i
            acc += z0 * zi * (n_s / total_shots)
        correlations.append(acc)

    return correlations


# ---------------------------------------------------------------------------
# 5. EXECUTION AND PLOTTING
# ---------------------------------------------------------------------------
# This is the orchestration layer:
#   * Connect to IBM Quantum.
#   * Select a backend (ibm_torino by default).
#   * Build and transpile an N-qubit GHZ circuit.
#   * Execute it with the Sampler primitive.
#   * Extract counts and compute ⟨Z0 Zi⟩.
#   * Print a small table and plot the correlation vs distance.


def run_ghz_experiment(n_qubits: int = 5, shots: int = 4096):
    """Run an N-qubit GHZ experiment and plot ⟨Z0 Zi⟩ vs distance i.

    Args:
        n_qubits: Number of qubits in the GHZ chain. For an open-plan backend,
                  small values like 3–7 are a good starting point: deep GHZ
                  chains get very noisy very quickly.
        shots:    Number of measurement shots to collect.
    """
    print(f"Preparing {n_qubits}-qubit GHZ experiment...")

    # Connect to IBM Quantum and select backend.
    service = get_service()
    print("Connected to IBM Quantum.")
    backend = get_backend(service, backend_name="ibm_fez")
    print("Using backend:", backend.name)

    # Construct and transpile the GHZ circuit for this backend. Transpilation
    # maps the logical H+CX chain into the device's native gate set and
    # connectivity, potentially reordering qubits and inserting additional
    # operations as needed.
    qc = build_ghz_circuit(n_qubits)
    t_qc = transpile(qc, backend)
    print("Transpiled GHZ circuit:")
    print(t_qc)

    # Run with the Sampler primitive. We pass a list of circuits (just one in
    # this case) and the number of shots.
    sampler = Sampler(mode=backend)
    job = sampler.run([t_qc], shots=shots)
    print("Job submitted. Waiting for result...")
    result = job.result()

    print("\n=== SINGLE-TEST RESULT DEBUG START ===")
    print("RAW RESULT STRUCTURE:", dir(result))
    print("RAW RESULT DICT:", getattr(result, "__dict__", "<no __dict__>"))

    try:
        pub = result[0]
        print("PUB:", pub)
        print("PUB.DATA:", pub.data)
        print("PUB.DATA DIR:", dir(pub.data))
        print("PUB.DATA DICT:", getattr(pub.data, "__dict__", "<no __dict__>"))

        if hasattr(pub.data, "c"):
            print("DATA.c TYPE:", type(pub.data.c))
            try:
                items = list(pub.data.c.items())
                print("DATA.c ITEMS SAMPLE:", items[:8])
                print("DATA.c TOTAL ITEMS:", len(items))
            except Exception as e:
                print("Could not iterate DATA.c items:", e)

    except Exception as e:
        print("Could not inspect pub in single-test script:", e)

    print("=== SINGLE-TEST RESULT DEBUG END ===\n")


    # Extract raw bitstring counts from the Sampler result.
    counts = extract_counts(result[0])
    print("Raw counts:", counts)

    # Compute Z0–Zi correlations from the raw counts.
    correlations = compute_z0_zi_correlations(counts, n_qubits)

    print("Distance  i   <Z0 Zi>")
    for i, c in enumerate(correlations, start=1):
        print(f"  {i:3d}        {c:+.6f}")

    # Plot ⟨Z0 Zi⟩ vs distance i.
    distances = list(range(1, n_qubits))
    plt.figure(figsize=(6, 4))
    plt.plot(distances, correlations, marker="o", label=f"{n_qubits}-qubit GHZ state")
    plt.xlabel("Distance between qubits i and 0 (index difference)")
    plt.ylabel("⟨Z₀Zᵢ⟩")
    plt.title(f"GHZ Correlations on {backend.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Entry point: you can tweak n_qubits here. Start small (e.g., 3–5) to
    # keep the circuit shallow and the noise manageable, then gradually
    # increase if you are curious how quickly correlations collapse.
    run_ghz_experiment(n_qubits=20, shots=4096)
