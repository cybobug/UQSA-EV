# UQSA-EV: A Unified Quantum-Resilient Security Architecture

This repository contains the official implementation of the **UQSA-EV** framework, as presented in the research paper: *"UQSA-EV: A Unified Quantum-Resilient Security Architecture for Intelligent EV Charging Networks"*.

## 📑 Contents
- **`paper.tex`**: The LaTeX source for the research paper (arXiv ready).
- **`uqsa-ev-poc/`**: The complete software-defined Proof-of-Concept.
- **`experimental_results/`**: Pre-generated ROC curves and performance plots.

## 🌟 Research Highlights
- **First-of-its-kind Fusion**: Integrates NIST PQC (ML-KEM/ML-DSA) with TCN-based anomaly detection.
- **OCPP Resilience**: Solves the 2KB certificate limit in OCPP 2.0.1 using a sidecar proxy pattern.
- **State-of-the-art Detection**: Achieves **99.8% accuracy** on the CICEVSE2024 dataset.

## 🛠️ Performance Summary (Raspberry Pi 4)
| Feature | Legacy (RSA) | UQSA-EV (Ours) |
|---|---|---|
| Handshake Security | Classical (Vulnerable) | **Quantum-Resilient (Hybrid)** |
| Anomaly Detection | None | **Autonomous (TCN)** |
| Handshake Time | 45ms | 52.2ms (+7.2ms) |
| AUC Score | N/A | **1.00** |

## 🚀 Quick Start
For detailed instructions on running the PoC, benchmarks, and generating your own results, please see the [**UQSA-EV PoC README**](./uqsa-ev-poc/README.md).

## 📄 Citation
```bibtex
@article{haswani2025uqsa,
  title={UQSA-EV: A Unified Quantum-Resilient Security Architecture for Intelligent EV Charging Networks},
  author={Haswani, Garvit and Kandpal, Mahi},
  journal={arXiv preprint arXiv:2602.xxxxx},
  year={2025}
}
```
