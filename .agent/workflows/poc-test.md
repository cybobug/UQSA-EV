---
description: Test the UQSA-EV Proof of Concept (POC) for readiness
---

This workflow will guide you through verifying that all components (PQC Handshake, TCN Anomaly Detection, and Reporting) are fully operational for a POC demonstration.

### 1. Verify Infrastructure Status
Ensure all core services are running.
// turbo
```powershell
docker compose -f ./uqsa-ev-poc/docker-compose.yml ps
```

### 2. Verify PQC-TLS Handshake
Check the sidecar logs to confirm the hybrid PQC-TLS handshake is successful between the EVSE and CSMS.
// turbo
```powershell
docker compose -f ./uqsa-ev-poc/docker-compose.yml logs --tail 20 sidecar
```
*Expected: Look for `INFO: PQC-TLS Handshake Successful.`*

### 3. Simulate and Detect Attack
Run the attacker tool to trigger a DoS event and verify the TCN service identifies the anomaly.
// turbo
```powershell
docker compose -f ./uqsa-ev-poc/docker-compose.yml run --rm attacker
```
Wait for about 10-20 seconds, then check TCN logs:
// turbo
```powershell
docker compose -f ./uqsa-ev-poc/docker-compose.yml logs --tail 20 tcn-service
```
*Expected: Look for `🚨 [CRITICAL ALERT] Anomaly Detected!`*

### 4. Generate Performance Metrics
Generate the final charts and data for the POC report.
// turbo
```powershell
# Generate raw data (Simulation mode)
python ./uqsa-ev-poc/benchmark.py

# Generate visualization plots
python ./uqsa-ev-poc/metrics_plotter.py
```

### 5. Final Checklist
- [ ] `performance_comparison.png` is generated and shows PQC resilience.
- [ ] `roc_curve_poc.png` shows high detection efficacy.
- [ ] Sidecar logs show active PQC fingerprints.
