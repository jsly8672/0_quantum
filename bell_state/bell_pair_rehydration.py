# IBM Quantum Job Retrieval and Visualization Tool
# ============================================================================
# PURPOSE
# -------
# This utility script allows a user to reconnect to a completed IBM Quantum job
# using its job ID and visualize the measurement results as a histogram.
# It works for any past job, provided the user has a valid API key and instance.
# ============================================================================

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import os, json

# ----------------------------------------------------------------------------
# 1. Load API key securely from local files if not already saved
# ----------------------------------------------------------------------------
api_key = None
if os.path.exists(".ibm_api"):
    with open(".ibm_api", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
elif os.path.exists("ibm_apikey.json"):
    with open("ibm_apikey.json", "r", encoding="utf-8") as f:
        api_key = json.load(f).get("apikey")

# ----------------------------------------------------------------------------
# 2. Prompt user for instance and job ID
# ----------------------------------------------------------------------------
print("\n=== IBM Quantum Job Retrieval Utility ===")
instance = input("Enter your IBM Quantum instance (press Enter for none): ").strip()
job_id = input("Enter the job ID to retrieve: ").strip()
if not job_id:
    raise ValueError("A job ID is required to retrieve results.")

# ----------------------------------------------------------------------------
# 3. Connect to IBM Quantum Service
# ----------------------------------------------------------------------------
print("\nConnecting to IBM Quantum service...")
if api_key:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key, instance=instance or None)
else:
    # If user previously saved account credentials, token may not be needed
    service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance or None)

print("Connected successfully.")

# ----------------------------------------------------------------------------
# 4. Retrieve the job and its results
# ----------------------------------------------------------------------------
print(f"Retrieving job {job_id}...")
job = service.job(job_id)
result = job.result()
print("Job retrieved successfully.")

# ----------------------------------------------------------------------------
# 5. Extract measurement counts
# ----------------------------------------------------------------------------
# Most jobs contain a single 'pub' result at index 0.
# Some may have multiple, depending on the circuit batch.
# ----------------------------------------------------------------------------
try:
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
except Exception as e:
    print("\nError: Unable to extract counts. Check job data format.")
    raise e

print("\nMeasurement counts retrieved:")
print(counts)

# ----------------------------------------------------------------------------
# 6. Visualize results using Qiskit’s built-in histogram plotter
# ----------------------------------------------------------------------------
print("\nRendering histogram...")
plot_histogram(counts)
plt.title(f"IBM Quantum Job {job_id} Results")
plt.show()

# ----------------------------------------------------------------------------
# END OF SCRIPT
# ----------------------------------------------------------------------------
