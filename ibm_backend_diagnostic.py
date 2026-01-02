#!/usr/bin/env python3
"""
IBM Backend Deep Diagnostic Tool
--------------------------------
A detailed backend health inspector designed to help understand exactly what a
QPU is doing: operational state, queue health, calibration times, qubit-level
properties, gate errors, routing-map validity, and actual sampler execution via
ping circuits.

This script DOES NOT modify anything on IBM's side; it only queries and reports.
"""
import datetime
import json
import os
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile


def load_api_key():
    if os.path.exists(".ibm_api"):
        return open(".ibm_api").read().strip()
    if os.path.exists("ibm_apikey.json"):
        return json.load(open("ibm_apikey.json"))['apikey']
    raise FileNotFoundError("No API key found.")


def connect():
    token = load_api_key()
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)


def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)


def backend_core_info(backend):
    print_header(f"Backend: {backend.name}")
    cfg = backend.configuration()
    print(f"• Backend name:          {cfg.backend_name}")
    print(f"• Backend version:       {cfg.backend_version}")
    print(f"• Qubits:                {cfg.num_qubits}")
    print(f"• Basis gates:           {cfg.basis_gates}")
    print(f"• Simulation backend?:   {cfg.simulator}")
    print(f"• Coupling map:          {len(cfg.coupling_map)} edges")


def backend_status_info(backend):
    st = backend.status()
    print_header("Operational Status")
    print(f"• Operational:           {st.operational}")
    print(f"• Pending jobs:          {st.pending_jobs}")
    print(f"• Status message:        {st.status_msg}")


def backend_calibration_info(backend):
    print_header("Calibration Data (if available)")
    try:
        props = backend.properties()
        print(f"• Last calibration:      {props.last_update_date}")
        # basic qubit properties
        for i, qubit in enumerate(props.qubits):
            T1 = qubit[0].value if qubit and qubit[0].name == 'T1' else None
            T2 = qubit[1].value if qubit and qubit[1].name == 'T2' else None
            print(f"  Q[{i}]  T1={T1}  T2={T2}")
        print("• Gate errors:")
        for gate in props.gates:
            name = gate.gate
            qubits = gate.qubits
            for param in gate.parameters:
                if param.name == 'gate_error':
                    print(f"  {name}{qubits}: error={param.value}")
    except Exception as e:
        print(f"No calibration data available: {e}")


def backend_config_dump(backend):
    print_header("Full Backend Configuration Snapshot")
    cfg = backend.configuration()
    for k, v in cfg.to_dict().items():
        print(f"• {k}: {v}")


def backend_gateway_map(backend):
    print_header("Connectivity Map")
    cfg = backend.configuration()
    cmap = cfg.coupling_map
    for edge in cmap:
        print(f"  {edge[0]} ↔ {edge[1]}")


def ping_sampler(backend, shots=64):
    print_header("Sampler Execution Test (Ping Circuit)")
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    try:
        tqc = transpile(qc, backend=backend)
        sampler = Sampler(mode=backend)
        job = sampler.run([tqc], shots=shots)
        res = job.result()
        print(f"• Job ID: {job.job_id()}")
        # Try reading raw structure
        print(f"• Raw structure: {dir(res)}")
        try:
            pub = res[0]
            print(f"• Pub: {pub}")
            print(f"• Pub.data fields: {dir(pub.data)}")
        except Exception as e:
            print(f"Cannot inspect pub: {e}")
    except Exception as e:
        print(f"Sampler test failed: {e}")


if __name__ == "__main__":
    import sys
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "ibm_fez"
    svc = connect()
    backend = svc.backend(backend_name)

    backend_core_info(backend)
    backend_status_info(backend)
    backend_calibration_info(backend)
    backend_gateway_map(backend)
    ping_sampler(backend)

    print("\nDiagnostic complete.\n")
