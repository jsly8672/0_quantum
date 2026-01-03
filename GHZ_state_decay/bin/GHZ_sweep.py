#!/usr/bin/env python3
"""
GHZ Sweep Experiment
--------------------
Runs GHZ_N experiments for a range of N (e.g., 3–8), on a chosen backend,
logs all console chatter to a log file, stores correlations in CSV,
and generates composite plots including:
  • correlation-vs-distance curves for each N
  • entanglement-horizon curve (Z0Z_{N-1} vs N)

This script is intentionally verbose and prints everything to console AND logs.
"""
import os
import csv
import json
import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import matplotlib
matplotlib.use("Agg")  # ensure non-interactive
import matplotlib.pyplot as plt

LOGFILE = "ghz_sweep_log.txt"
CSVFILE = f"ghz_sweep_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------
def load_api_key():
    if os.path.exists(".ibm_api"):
        return open(".ibm_api").read().strip()
    if os.path.exists("ibm_apikey.json"):
        return json.load(open("ibm_apikey.json"))['apikey']
    raise FileNotFoundError("No API key found.")

# ---------------------------------------------------------------------------
# Connect to IBM
# ---------------------------------------------------------------------------
def get_service():
    token = load_api_key()
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return svc

# ---------------------------------------------------------------------------
# Build GHZ circuit
# ---------------------------------------------------------------------------
def build_ghz(n):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    qc.measure(range(n), range(n))
    return qc

# ---------------------------------------------------------------------------
# Extract counts from DataBin or other structures
# ---------------------------------------------------------------------------
def extract_counts(result):
    """
    Fully adaptive extractor for Sampler results across all IBM formats:
    - New Runtime V2 (data.meas.get_counts())
    - Transitional (data.get_counts())
    - DataBin legacy (_data list)
    - Fez-style BitArray direct access (data.c)
    - Quasi-distributions (data.meas.get_dist)
    """

    # ------------------------------
    # Newest Qiskit Runtime V2
    # ------------------------------
    try:
        pub = result[0]
        data = pub.data

        # Modern: data.meas.get_counts()
        if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
            counts = data.meas.get_counts()
            if counts:
                return counts

        # Modern quasi-distribution: data.meas.get_dist()
        if hasattr(data, "meas") and hasattr(data.meas, "get_dist"):
            dist = data.meas.get_dist()
            if dist:
                # Convert probabilities→pseudo-counts (scale by shots)
                shots = getattr(result.metadata, "shots", 4096)
                return {k: int(v * shots) for k, v in dist.items()}

        # Transitional format: data.get_counts()
        if hasattr(data, "get_counts"):
            counts = data.get_counts()
            if counts:
                return counts

        # Fez-style direct BitArray inside 'c'
        if hasattr(data, "c"):
            try:
                bitarray = data.c
                # BitArray items(): list of (bitstring, count)
                return dict(data.c)
            except Exception:
                pass

    except Exception:
        pass

    # ------------------------------
    # Legacy DataBin format
    # ------------------------------
    try:
        data_list = getattr(result, "_data", None)
        if data_list:
            data = data_list[0]

            # Legacy: data.meas.get_counts()
            if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
                return data.meas.get_counts()

            # Legacy: data.get_counts()
            if hasattr(data, "get_counts"):
                return data.get_counts()

            # Search for any attribute with get_counts()
            for name in dir(data):
                attr = getattr(data, name)
                if hasattr(attr, "get_counts"):
                    return attr.get_counts()
    except Exception:
        pass

    # ------------------------------
    # Fallback
    # ------------------------------
    raise RuntimeError("Could not extract measurement counts from Sampler result.")

# ---------------------------------------------------------------------------
# Compute Z0Zi correlations
# ---------------------------------------------------------------------------
def compute_correlations(counts, n):
    total = sum(counts.values())
    corr = []
    for i in range(1, n):
        val = 0.0
        for bitstring, ct in counts.items():
            # Map bitstring: index -1-i = classical bit i
            b0 = 1 if bitstring[-1] == '0' else -1
            bi = 1 if bitstring[-1 - i] == '0' else -1
            val += b0 * bi * (ct / total)
        corr.append(val)
    return corr

# ---------------------------------------------------------------------------
# GHZ experiment for one N
# ---------------------------------------------------------------------------
def run_ghz_n(backend, n, shots):
    log(f"Running GHZ_{n} on backend {backend.name}")
    qc = build_ghz(n)
    log(f"Building circuit for N={n}")
    tqc = transpile(qc, backend=backend, optimization_level=1)
    log(f"Transpiled depth={tqc.depth()}")

    sampler = Sampler(mode=backend)
    job = sampler.run([tqc], shots=shots)
    log(f"Submitted job: {job.job_id()}")
    res = job.result()
    # DEBUG structural inspection
    log(f"DEBUG: Sampler result raw structure: {dir(res)}")
    try:
        log(f"DEBUG: First pub: {res[0]}")
        log(f"DEBUG: First pub data attrs: {dir(res[0].data)}")
    except Exception as e:
        log(f"DEBUG: Could not inspect pub structure: {e}")
    counts = extract_counts(res)

    log(f"Raw counts for N={n}: {counts}")
    correlations = compute_correlations(counts, n)
    log(f"Correlations Z0Zi for N={n}: {correlations}")

    return {
        "N": n,
        "counts": counts,
        "correlations": correlations,
        "job_id": job.job_id(),
        "timestamp": datetime.datetime.now().isoformat()
    }

# ---------------------------------------------------------------------------
# Sweep across N
# ---------------------------------------------------------------------------
def sweep_ghz(backend_name, sizes, shots):
    svc = get_service()
    backend = svc.backend(backend_name)
    log(f"Connected. Using backend: {backend.name}")

    results = []
    for n in sizes:
        out = run_ghz_n(backend, n, shots)
        results.append(out)
    return results

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
def save_csv(results):
    exists = os.path.exists(CSVFILE)
    with open(CSVFILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["backend", "N", "i", "correlation"])
        for out in results:
            N = out["N"]
            corr = out["correlations"]
            for i, val in enumerate(corr, start=1):
                w.writerow([out.get('backend', 'unknown'), N, i, val])

# ---------------------------------------------------------------------------
# Composite plots
# ---------------------------------------------------------------------------
# Generate timestamp for file outputs
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# ---------------------------------------------------------------------------
def plot_composite(results, backend_name):
    plt.figure(figsize=(10,6))
    for out in results:
        N = out['N']
        corr = out['correlations']
        x = list(range(1, len(corr)+1))
        plt.plot(x, corr, marker='o', label=f"GHZ_{N}")
    plt.title(f"GHZ Correlation Curves on {backend_name}")
    plt.xlabel("Distance i (Z0Zi)")
    plt.ylabel("<Z0Zi>")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ghz_composite_{backend_name}_{TIMESTAMP}.png", dpi=150)
    plt.close()

    # horizon plot
    plt.figure(figsize=(8,5))
    Ns = [out['N'] for out in results]
    last_corr = [out['correlations'][-1] for out in results]
    plt.plot(Ns, last_corr, marker='o')
    plt.title(f"Entanglement Horizon on {backend_name}")
    plt.xlabel("GHZ size N")
    plt.ylabel("<Z0Z_{N-1}>")
    plt.grid(True)
    plt.savefig(f"ghz_horizon_{backend_name}_{TIMESTAMP}.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    # Allow backend choice from command-line, defaulting to ibm_fez
    backend = sys.argv[1] if len(sys.argv) > 1 else "ibm_fez"
    sizes = [3,4,5,6,7,8]
    shots = 4096
    log(f"Starting GHZ sweep on {backend}")
    results = sweep_ghz(backend, sizes, shots)
    for r in results:
        r['backend'] = backend
    save_csv(results)
    plot_composite(results, backend)
    log("Sweep complete.")