import os
import json
from qiskit_ibm_runtime import QiskitRuntimeService

# -------------------------------
# Load API key (existing logic)
# -------------------------------
api_key = None

if os.path.exists(".ibm_api"):
    with open(".ibm_api", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

elif os.path.exists("ibm_apikey.json"):
    with open("ibm_apikey.json", "r", encoding="utf-8") as f:
        api_key = json.load(f).get("apikey")

if not api_key:
    raise FileNotFoundError("No API key found (.ibm_api or ibm_apikey.json)")

# -------------------------------
# Connect to IBM Quantum
# -------------------------------
try:
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=api_key
    )
    backends = service.backends()
    print("API verification successful. Key is valid.")
    print(f"{len(backends)} backends visible.\n")

except Exception as e:
    print("API verification failed:", e)
    raise


# --------------------------------------------------------
# New routine: lookup and print queue lengths per backend
# --------------------------------------------------------
def print_backend_queue_lengths(backends):
    print("Backend Queue Lengths:")
    print("-" * 60)
    print(f"{'Backend':25s} {'Qubits':>6s} {'Queue':>8s}  {'Status'}")
    print("-" * 60)

    for b in backends:
        try:
            status = b.status()
            queue = getattr(status, "pending_jobs", "N/A")
            state = getattr(status, "status", "Unknown")
            print(f"{b.name:25s} {b.num_qubits:6d} {str(queue):>8s}  {state}")
        except Exception:
            print(f"{b.name:25s} {b.num_qubits:6d} {'N/A':>8s}  {'Unknown'}")

    print("-" * 60)


# Run the new routine
print_backend_queue_lengths(backends)
