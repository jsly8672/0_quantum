# Error-Mitigated Bell State Project
# ==================================
# Integrated Phases 1 and 2: Baseline Bell-State + Readout Error Mitigation
#
# This script does two main things:
#   1. Prepares and runs a 2-qubit Bell-state circuit on a real IBM backend
#      (e.g., ibm_torino), then computes the raw expectation value <Z ⊗ Z>.
#   2. Calibrates the readout (by preparing |00>, |01>, |10>, |11>), builds a
#      4×4 confusion matrix, inverts it, and uses it to mitigate readout
#      errors in the Bell-state measurement results.
#
# The actual *code* (function calls, logic, behavior) is unchanged; what has
# been added here is extensive commentary explaining **why** each part exists
# and **what** it is doing.

import os
import json
from typing import Dict, Tuple
import numpy as _np

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile

# ---------------------------------------------------------------------------
# 1. API KEY LOADING
# ---------------------------------------------------------------------------
# QiskitRuntimeService needs an API key to authenticate against IBM Quantum.
# We *never* hard-code it; instead, we look for it in local files that are
# ignored by version control (e.g., .gitignore), so your credentials stay
# private.
#
# Strategy:
#   * If a file named ".ibm_api" exists, treat it as a plaintext token.
#   * Else if "ibm_apikey.json" exists, read the "apikey" field from it.
#   * Else, fall back to a previously saved account configuration.


def load_api_key() -> str | None:
    """Attempt to load the IBM Quantum API key from local files.

    Returns:
        The API key string if found, otherwise None. Returning None signals
        that the code should fall back to an already-saved Qiskit account
        (e.g., via QiskitRuntimeService.save_account()).
    """
    api_key = None

    # Preferred: a simple plaintext file with just the token.
    if os.path.exists(".ibm_api"):
        with open(".ibm_api", "r", encoding="utf-8") as f:
            api_key = f.read().strip()

    # Fallback: JSON file exported from IBM Cloud that includes the apikey.
    elif os.path.exists("ibm_apikey.json"):
        with open("ibm_apikey.json", "r", encoding="utf-8") as f:
            api_key = json.load(f).get("apikey")

    # If neither file exists, api_key remains None and the caller must decide
    # what to do (we choose to rely on a saved Qiskit account).
    return api_key


# ---------------------------------------------------------------------------
# 2. SERVICE AND BACKEND
# ---------------------------------------------------------------------------
# These helpers encapsulate the logic for creating the IBM Quantum service
# object and selecting a backend to run on. Keeping this separate makes it
# easier to swap backends later (e.g., ibm_fez, simulator_mps, etc.).


def get_service() -> QiskitRuntimeService:
    """Create a QiskitRuntimeService instance using local credentials.

    If an API key is found via load_api_key(), it is passed explicitly to
    QiskitRuntimeService. Otherwise, we assume that a saved account exists
    on this machine and let QiskitRuntimeService() resolve it from disk.
    """
    key = load_api_key()
    if key:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=key)
    return QiskitRuntimeService()


def get_backend(service, backend_name="ibm_torino"):
    """Return a backend object from the IBM Quantum service.

    Args:
        service: The QiskitRuntimeService instance returned by get_service().
        backend_name: Name of the backend to use (default: "ibm_torino").
    """
    return service.backend(backend_name)


# ---------------------------------------------------------------------------
# 3. BELL CIRCUIT
# ---------------------------------------------------------------------------
# This is the logical 2-qubit Bell-state circuit. It uses high-level gates
# (H and CX). After we construct it, we later **transpile** it for the real
# hardware, which expands H and CX into the device's basis gates.
#
# Logical behavior:
#   * Start in |00>.
#   * Apply H to qubit 0 → (|0> + |1>)/√2 ⊗ |0>.
#   * Apply CX (control=0, target=1) → (|00> + |11>)/√2, a Bell state.
#   * Measure both qubits into classical bits 0 and 1.


def build_bell_circuit() -> QuantumCircuit:
    """Construct a 2-qubit Bell-state circuit with measurements."""
    qc = QuantumCircuit(2, 2)      # 2 qubits, 2 classical bits
    qc.h(0)                        # Hadamard on qubit 0: create superposition
    qc.cx(0, 1)                    # CNOT: entangle qubit 1 with qubit 0
    qc.measure(0, 0)               # Measure qubit 0 into classical bit 0
    qc.measure(1, 1)               # Measure qubit 1 into classical bit 1
    return qc


# ---------------------------------------------------------------------------
# 4. EXPECTATION <Z ⊗ Z>
# ---------------------------------------------------------------------------
# We want a quantitative measure of how "good" the Bell state is. For the
# ideal state |Φ+> = (|00> + |11>)/√2, the operator Z ⊗ Z has expectation
# value +1.
#
# Z ⊗ Z eigenvalues in the computational basis:
#   |00> → +1
#   |01> → -1
#   |10> → -1
#   |11> → +1
#
# Given counts for each bitstring, we compute:
#   <Z⊗Z> = (N00 + N11 - N01 - N10) / (N00 + N01 + N10 + N11)


def compute_zz_expectation(counts: Dict[str, int]) -> float:
    """Compute <Z ⊗ Z> from 2-qubit bitstring counts.

    Args:
        counts: A dictionary like {"00": n00, "01": n01, ...}.

    Returns:
        The expectation value <Z ⊗ Z> as a float in [-1, 1].
    """
    n00, n01 = counts.get("00",0), counts.get("01",0)
    n10, n11 = counts.get("10",0), counts.get("11",0)
    total = n00 + n01 + n10 + n11
    if total == 0:
        # If this happens, something went very wrong with the measurement.
        raise ValueError("No counts available.")
    # +1 for 00 and 11, -1 for 01 and 10.
    return (n00 + n11 - n01 - n10) / total


# ---------------------------------------------------------------------------
# 5. UNIVERSAL COUNT EXTRACTION
# ---------------------------------------------------------------------------
# IBM's Runtime Sampler API has evolved over time, and the object that stores
# measurement results (pub_result.data) can expose its data in slightly
# different ways depending on version.
#
# This helper function inspects the DataBin and tries several known access
# paths to extract a counts dictionary. It also prints out the structure so
# you can see exactly what Torino returned.


def extract_counts(pub_result) -> Dict[str,int]:
    """Extract bitstring counts from a SamplerV2 pub_result object.

    This function is intentionally defensive: it prints the DataBin structure
    and tries multiple known layouts (data.meas.get_counts(), direct
    data.get_counts(), or scanning attributes for an object that supports
    get_counts()).
    """
    data = pub_result.data
    print("Inspecting DataBin: ", data)
    print("Attributes:", dir(data))

    # Case 1: data.meas.get_counts()
    if hasattr(data, "meas"):
        meas = data.meas
        if hasattr(meas, "get_counts"):
            return meas.get_counts()

    # Case 2: data.get_counts()
    if hasattr(data, "get_counts"):
        return data.get_counts()

    # Case 3: search attributes for a BitArray with get_counts()
    for name in dir(data):
        if not name.startswith("_"):
            obj = getattr(data, name)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()

    # If we reach this point, the DataBin structure is unfamiliar.
    raise RuntimeError("Unable to locate counts in DataBin.")


# ---------------------------------------------------------------------------
# 6. RUN BASELINE BELL EXPERIMENT
# ---------------------------------------------------------------------------
# This function wires everything together for Phase 1:
#   * build the logical Bell circuit,
#   * transpile it for the backend (so it uses native gates/topology),
#   * call the Sampler primitive to execute it with a given number of shots,
#   * extract the measurement counts,
#   * compute <Z ⊗ Z> from those counts.


def run_bell_experiment(backend, shots=4096) -> Tuple[Dict[str,int], float]:
    """Run the Bell-state circuit on the given backend and compute <Z ⊗ Z>.

    Args:
        backend: The IBM Quantum backend object (e.g., ibm_torino).
        shots:   Number of repeated measurements to perform.

    Returns:
        A tuple (counts, zz) where `counts` is a bitstring→count dictionary and
        `zz` is the raw <Z ⊗ Z> expectation value.
    """
    qc = build_bell_circuit()          # Logical Bell circuit
    t_qc = transpile(qc, backend)      # Hardware-specific version
    print("Transpiled circuit:", t_qc)


    # Bind the Sampler primitive to this backend. The Sampler handles circuit
    # execution and returns results in a standardized format.
    sampler = Sampler(mode=backend)
    job = sampler.run([t_qc], shots=shots)
    print("Job submitted. Waiting...")
    result = job.result()

    # There is a single circuit, so result[0] is the corresponding pub_result.
    counts = extract_counts(result[0])
    print("Counts extracted:", counts)

    # Compute and return <Z ⊗ Z> from those counts.
    return counts, compute_zz_expectation(counts)


# ---------------------------------------------------------------------------
# 7. PHASE 2 — READOUT ERROR MITIGATION
# ---------------------------------------------------------------------------
# The second phase characterizes measurement errors by preparing each of the
# four computational basis states (|00>, |01>, |10>, |11>), measuring them,
# and observing how often the backend *mislabels* them.
#
# From these runs we build a 4×4 confusion matrix M, where rows correspond to
# prepared states and columns to measured states:
#     M[i, j] = P(measured_state_j | prepared_state_i)
#
# We then invert M to approximate the "true" probability vector from the
# observed one:
#     p_meas ≈ M · p_true  →  p_true ≈ M^{-1} · p_meas
#
# This corrects biases in the readout channel (e.g., a tendency to flip 1→0).

_basis_states = ["00","01","10","11"]


def build_basis_circuit(bitstring: str) -> QuantumCircuit:
    """Build a circuit that prepares a specific 2-qubit basis state.

    For example:
      * "00" → do nothing (start from |00>)
      * "01" → X on qubit 1
      * "10" → X on qubit 0
      * "11" → X on both qubits

    Then measure both qubits in the Z basis.
    """
    qc = QuantumCircuit(2,2)
    if bitstring[0]=="1": qc.x(0)
    if bitstring[1]=="1": qc.x(1)
    qc.measure(0,0); qc.measure(1,1)
    return qc


def run_basis_calibrations(backend, shots=4096):
    """Run calibration circuits for |00>, |01>, |10>, |11> on the backend.

    Returns:
        A dictionary mapping prepared-state labels ("00", "01", ...) to the
        measured counts dictionary obtained for each.
    """
    sampler = Sampler(mode=backend)
    results = {}
    for state in _basis_states:
        qc = build_basis_circuit(state)
        t_qc = transpile(qc, backend)
        job = sampler.run([t_qc], shots=shots)
        r = job.result()[0]
        results[state] = extract_counts(r)
    return results


def build_confusion_matrix(cal: dict) -> _np.ndarray:
    """Construct the 4×4 confusion matrix from calibration results.

    Args:
        cal: Dictionary mapping prepared bitstrings to measured counts.

    Returns:
        A 4×4 NumPy array M where:
          M[i, j] = P(measured_state_j | prepared_state_i).
    """
    M = _np.zeros((4,4))
    for i, prep in enumerate(_basis_states):
        total = sum(cal[prep].values())
        for j, meas in enumerate(_basis_states):
            M[i,j] = cal[prep].get(meas,0) / total
    return M


def apply_readout_mitigation(M, counts):
    """Apply readout error mitigation using the confusion matrix.

    Args:
        M:      4×4 confusion matrix from build_confusion_matrix().
        counts: Raw counts from the Bell-state run.

    Returns:
        A dictionary mapping bitstrings to **mitigated** probabilities, which
        we then scale back up to counts in main().
    """
    total = sum(counts.values())

    # Measured probability vector ordered as [P(00), P(01), P(10), P(11)].
    p_meas = _np.array([counts.get(s,0)/total for s in _basis_states])

    # Invert the confusion matrix: p_true ≈ M^{-1} · p_meas.
    M_inv = _np.linalg.inv(M)
    p_true = M_inv @ p_meas

    # Clamp negatives to zero (numerical noise) and renormalize to 1.
    p_true = _np.clip(p_true, 0, None)
    p_true = p_true / p_true.sum()

    # Return as a dict keyed by bitstring.
    return {s:p_true[i] for i,s in enumerate(_basis_states)}


# ---------------------------------------------------------------------------
# 8. MAIN ORCHESTRATION (INTEGRATED PHASES 1 + 2)
# ---------------------------------------------------------------------------
# The main() function coordinates everything:
#   1. Connect to IBM Quantum.
#   2. Select a backend (default: ibm_torino).
#   3. Run the Bell experiment and compute raw <Z ⊗ Z>.
#   4. Run readout calibration and build the confusion matrix.
#   5. Apply readout mitigation to the raw Bell counts.
#   6. Compute mitigated <Z ⊗ Z> and print both results.


def main():
    """Entry point: run baseline and readout-mitigated Bell experiments."""
    print("Connecting to IBM Quantum service...")
    service = get_service()
    print("Connected.")

    backend = get_backend(service)
    print("Using backend:", backend.name)

    shots = 4096

    # ---- Phase 1: Baseline (no mitigation) ----
    raw_counts, raw_zz = run_bell_experiment(backend, shots)

    # ---- Phase 2: Calibration for readout mitigation ----
    print("Running readout calibration...")
    cal = run_basis_calibrations(backend, shots)
    M = build_confusion_matrix(cal)

    # ---- Apply readout mitigation to the Bell-state counts ----
    mitig_probs = apply_readout_mitigation(M, raw_counts)

    # Convert mitigated probabilities back to a counts-like dictionary by
    # scaling with the same number of shots (for convenience).
    mitig_counts = {s:int(mitig_probs[s]*shots) for s in _basis_states}
    mitig_zz = compute_zz_expectation(mitig_counts)

    # ---- Output summary ----
    print("=== Baseline Results ===")
    print("Counts:", raw_counts)
    print("<Z⊗Z> (raw):", raw_zz)

    print("=== Readout-Mitigated Results ===")
    print("Counts:", mitig_counts)
    print("<Z⊗Z> (mitigated):", mitig_zz)


if __name__ == '__main__':
    main()
