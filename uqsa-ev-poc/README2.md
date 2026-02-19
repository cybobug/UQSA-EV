# UQSA-EV: Unified Quantum-Resilient Security Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker Support](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **Official Proof-of-Concept (PoC) for the UQSA-EV Research Framework.**  
> A dual-layer defense system integrating **NIST-Standardized Post-Quantum Cryptography (PQC)** and **Temporal Convolutional Networks (TCN)** to secure EV Charging Infrastructure (EVCI).

---

## 🔬 Scientific Overview

UQSA-EV addresses the convergence of quantum threats (**"Harvest Now, Decrypt Later"**) and AI-driven runtime attacks in Electric Vehicle Charging Networks. This PoC demonstrates a hardware-agnostic architecture that bridges the gap between legacy OCPP hardware and quantum-safe cloud systems.

### Key Contributions:
*   **Hybrid PQC Handshake**: Combines ML-KEM-768 (Kyber) and ML-DSA-44 (Dilithium) with classical ECDH for transition-period security.
*   **TCN Anomaly Detection**: Real-time traffic analysis using Dilated Causal Convolutions to identify DoS and FDI attacks with **99.8% accuracy (AUC=1.00)**.
*   **Protocol-Aware Bridge**: A "Secure Sidecar" proxy pattern that overcomes the 2KB certificate limit of the OCPP 2.0.1 specification.

---

## 🏗️ System Architecture

The PoC implements a three-plane defense-in-depth model:

1.  **Transport Plane (PQC)**: Implemented via a specialized OpenSSL/oqs-provider bridge.
2.  **Intelligence Plane (AI)**: A TCN-based monitor trained on the **CICEVSE2024** real-world dataset.
3.  **Management Plane (Orchestration)**: Handles out-of-band certificate fetching and mitigation triggers.

---

## 🚀 Getting Started

### 1. Prerequisites
*   **OS**: Linux (Ubuntu 22.04+), Mac, or WSL2.
*   **Engine**: Docker & Docker Compose v2.x.
*   **Dataset**: Real data integration requires placing `CICEVSE2024` CSV files in `tcn-service/data/` (optional, falls back to synthetic data).

### 2. Environment Setup
Build the quantum-safe containers (includes compiling PQC-enabled OpenSSL):
```bash
docker compose build
```

### 3. Initialize PKI
Generate the hybrid PQC certificate authority and signed identity chains:
```bash
chmod +x ./certs/gen_certs.sh
./certs/gen_certs.sh
```

### 4. Deploy Infrastructure
Start the CSMS, Secure Sidecar, EVSE Simulator, and TCN Monitor:
```bash
docker compose up -d
```

---

## 📊 Evaluation Outputs
- `performance_comparison.png`: Latency profile under DoS stress.
- `roc_curve_final.png`: Real AI detection efficacy from CICEVSE2024 (AUC=1.00).
- `experiment_results.csv`: Raw data points for academic verification.

### Running Benchmarks:
```bash
# 1. Run latency & stress simulation
python benchmark.py

# 2. Generate visualization plots
python metrics_plotter.py
```

### Key Metrics (RPi 4 Baseline):
| Metric | Legacy (RSA) | UQSA-EV (Hybrid) |
| :--- | :--- | :--- |
| **Handshake Latency** | ~45ms | +7.2ms overhead |
| **Attack Detection** | 0.0% | **99.8% (TCN)** |
| **PQC Certificate Size** | ~1.2KB | ~10KB (Managed via Sidecar) |
| **Resilience (DoS)** | Collapse @ 3k req/s | Stable @ 5k+ req/s |

---

## 📂 Project Structure

```text
├── sidecar/          # Secure Sidecar Proxy (PQC Bridge)
├── csms/             # Charging Station Management System (Backend)
├── tcn-service/      # AI Intelligence Plane (Anomaly Detection)
│   ├── data/         # CICEVSE2024 Dataset storage
│   └── model.py      # TCN Architecture
├── attacker/         # SYN Flood / FDI attack emulators
├── certs/            # Hybrid PQC-Certificate Authority scripts
└── paper.tex         # LaTeX source for the research paper
```

---

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{haswani2025uqsa,
  title={UQSA-EV: A Unified Quantum-Resilient Security Architecture for Intelligent EV Charging Networks},
  author={Haswani, Garvit and Kandpal, Mahi},
  journal={arXiv preprint arXiv:2602.xxxxx},
  year={2025}
}
```

---
**Disclaimer**: This is a Research PoC. For production deployment, ensure hardware-level side-channel protection for PQC primitives.
