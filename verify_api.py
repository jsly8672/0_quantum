# IBM Quantum API Verification Script (Updated)
# ---------------------------------------------
# This short program checks whether the locally stored API key (either .ibm_api
# or ibm_apikey.json) is valid and accepted by IBM Quantum's authentication
# service. It does not run circuits or modify the account; it simply attempts
# to connect and confirm that your credentials work.

import os
import json
from qiskit_ibm_runtime import QiskitRuntimeService

api_key = None

# Attempt to load from .ibm_api first
if os.path.exists(".ibm_api"):
    with open(".ibm_api", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

# If not found, try the IBM-generated JSON file
elif os.path.exists("ibm_apikey.json"):
    with open("ibm_apikey.json", "r", encoding="utf-8") as f:
        api_key = json.load(f).get("apikey")

# Stop if no key found
if not api_key:
    raise FileNotFoundError("No API key found (.ibm_api or ibm_apikey.json)")

# Attempt authentication
try:
    # Use the correct channel name per latest QiskitRuntimeService API
    # Valid options are 'ibm_cloud' or 'ibm_quantum_platform'.
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    # Try listing backends to verify the token is actually accepted
    backends = service.backends()
    print("API verification successful. Key is valid.")
    print(f"{len(backends)} backends visible to this account.")
except Exception as e:
    print("API verification failed:", e)
