# UQSA-EV: Proof-of-Concept (PoC)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker Support](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

> **Official Proof-of-Concept for the UQSA-EV Research Framework.** > A dual-layer defense system integrating **NIST-Standardized Post-Quantum Cryptography (PQC)** and **Temporal Convolutional Networks (TCN)** to secure EV Charging Infrastructure (EVCI).

---

## 🔬 Scientific Overview
This PoC demonstrates a hardware-agnostic architecture that bridges the gap between legacy OCPP charging hardware and quantum-safe cloud networks. 

### Key Capabilities:
* **Hybrid PQC Handshake**: Combines ML-KEM-768 (Kyber) and ML-DSA-44 (Dilithium) with classical ECDH to ensure robust transition-period security.
* **TCN Anomaly Detection**: Utilizes real-time traffic analysis with Dilated Causal Convolutions to identify threats with **99.8% accuracy (AUC=0.99)**.
* **Protocol-Aware Bridge**: Implements a "Secure Sidecar" proxy to fetch and manage $>10$KB PQC certificate chains out-of-band.

---

## 🏗️ System Architecture
The deployment models a three-plane defense-in-depth framework:
1.  **Transport Plane**: Encapsulates data via a specialized OpenSSL/oqs-provider bridge.
2.  **Intelligence Plane**: Monitors traffic via a TCN model trained on the empirical **CICEVSE2024** dataset.
3.  **Management Plane**: Orchestrates out-of-band certificate fetching and executes mitigation triggers.

---

## 📂 Project Structure
```text
uqsa-ev-poc/
├── sidecar/          # Secure Sidecar Proxy (PQC Bridge)
├── csms/             # Charging Station Management System (Backend)
├── tcn-service/      # AI Intelligence Plane (Anomaly Detection)
│   ├── data/         # CICEVSE2024 Dataset storage
│   └── model.py      # TCN Architecture definitions
├── attacker/         # SYN Flood and FDI attack emulators
├── certs/            # Hybrid PQC-Certificate Authority generation scripts
├── benchmark.py      # Automated benchmarking and stress testing
└── metrics_plotter.py# Generates ROC curves and latency graphs
