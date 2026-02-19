#!/bin/bash
set -e

# UQSA-EV Certificate Generation Script
# Requires OpenSSL 3.x with oqs-provider

CERT_DIR=$(dirname "$0")
cd "$CERT_DIR"

echo "--- Generating Quantum-Resilient CA (ML-DSA-44) ---"
openssl genpkey -algorithm mldsa44 -out ca.key
openssl req -x509 -new -key ca.key -out ca.crt -nodes -subj "/CN=UQSA-EV-CA" -days 365

echo "--- Generating CSMS Server Key (ML-DSA-44) ---"
openssl genpkey -algorithm mldsa44 -out server.key

echo "--- Generating CSMS CSR ---"
openssl req -new -key server.key -out server.csr -nodes -subj "/CN=csms"

echo "--- Signing CSMS Certificate with ML-DSA-44 CA ---"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365

echo "--- Generating Sidecar Client Key (ML-DSA-44) ---"
openssl genpkey -algorithm mldsa44 -out client.key
openssl req -new -key client.key -out client.csr -nodes -subj "/CN=sidecar"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365

echo "--- PQC Artifacts Generated ---"
ls -l .
