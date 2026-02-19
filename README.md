# UQSA-EV: A Unified Quantum-Resilient Security Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and research artifacts for the **UQSA-EV** framework, as presented in the paper: *"UQSA-EV: A Unified Quantum-Resilient Security Architecture for Intelligent EV Charging Networks"*.

## 📑 Contents
* **`paper.tex`**: The LaTeX source code for the research paper.
* **`uqsa-ev-poc/`**: The complete software-defined Proof-of-Concept (PoC).
* **`experimental_results/`**: Pre-generated ROC curves, stress test logs, and performance plots.

## 🌟 Research Highlights
* **First-of-its-kind Fusion**: Integrates NIST-standardized Post-Quantum Cryptography (ML-KEM/ML-DSA) with Temporal Convolutional Network (TCN) anomaly detection.
* **OCPP Protocol Resilience**: Bypasses the strict 2KB certificate limit in the OCPP 2.0.1 specification using a novel "Secure Sidecar" proxy pattern.
* **State-of-the-Art Detection**: Achieves **99.8% attack detection accuracy** against Denial of Service (DoS) and False Data Injection (FDI) attacks.

## 🛠️ Performance Summary (Raspberry Pi 4 Baseline)

| Feature / Metric | Legacy Environment (RSA) | UQSA-EV Framework (Ours) |
| :--- | :--- | :--- |
| **Handshake Security** | Classical (Vulnerable to HNDL) | **Quantum-Resilient (Hybrid)** |
| **Anomaly Detection** | None | **Autonomous (TCN)** |
| **Handshake Latency** | 45.0 ms | **52.2 ms (+7.2 ms overhead)** |
| **Detection AUC Score**| N/A | **0.99** |
| **DoS Resilience** | Collapses @ ~3,000 req/s | **Stable @ 5,000+ req/s** |

## 🚀 Quick Start
For detailed instructions on deploying the PoC, running the benchmarks, and generating your own evaluation metrics, please navigate to the [**UQSA-EV PoC Documentation**](./uqsa-ev-poc/README.md).

## 📄 Citation
If you utilize this framework or PoC in your research, please cite:
```bibtex
@article{haswani2025uqsa,
  title={UQSA-EV: A Unified Quantum-Resilient Security Architecture for Intelligent EV Charging Networks},
  author={Haswani, Garvit and Kandpal, Mahi},
  journal={arXiv preprint arXiv:2602.xxxxx},
  year={2026}
}
