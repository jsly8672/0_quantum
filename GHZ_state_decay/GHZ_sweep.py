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
CSVFILE = "ghz_sweep_results.csv"

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
def extract_counts(pub_result):
    data_list = getattr(pub_result, "_data", None)
    if not data_list:
        raise RuntimeError("Sampler returned empty _data list.")
    data = data_list[0]
    # Try the modern path
    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        return data.meas.get_counts()
    # Try legacy
    if hasattr(data, "get_counts"):
        return data.get_counts()
    # Try scanning attributes
    for name in dir(data):
        attr = getattr(data, name)
        if hasattr(attr, "get_counts"):
            return attr.get_counts()
    raise RuntimeError("Could not extract counts.")

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
        out['backend'] = backend_name
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
                w.writerow([out["backend"], N, i, val])

# ---------------------------------------------------------------------------
# Composite plots
# ---------------------------------------------------------------------------
def plot_composite(results, backend_name):
    # correlation curves
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
    plt.savefig(f"ghz_composite_{backend_name}.png", dpi=150)
    plt.close()

    # entanglement horizon curve
    plt.figure(figsize=(8,5))
    Ns = [out['N'] for out in results]
    last_corr = [out['correlations'][-1] for out in results]
    plt.plot(Ns, last_corr, marker='o')
    plt.title(f"Entanglement Horizon on {backend_name}")
    plt.xlabel("GHZ size N")
    plt.ylabel("<Z0Z_{N-1}>")
    plt.grid(True)
    plt.savefig(f"ghz_horizon_{backend_name}.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    backend = "ibm_torino"
    sizes = [3,4,5,6,7,8]
    shots = 4096
    log(f"Starting GHZ sweep on {backend}")
    results = sweep_ghz(backend, sizes, shots)
    save_csv(results)
    plot_composite(results, backend)
    log("Sweep complete.")
