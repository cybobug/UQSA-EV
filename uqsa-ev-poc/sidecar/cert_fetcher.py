import aiohttp
import logging
import hashlib

logger = logging.getLogger("CertFetcher")

async def fetch_pqc_certificate(fingerprint, csms_host="csms"):
    """
    Implements the 'Hash-and-Fetch' workaround for oversized PQC certificates.
    Retrieves the full ML-DSA certificate chain via an OOB HTTPS channel.
    This bypasses OCPP 2.0.1 string length limits (< 2KB).
    """
    # In this PoC, we assume the CSMS hosts a certificate repository at /certs/
    url = f"https://{csms_host}:443/certs/{fingerprint}.pem"
    logger.info(f"Oversized Certificate Caught: Fetching full PQC chain from {url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Note: ssl=False used here as this is the side-channel for the cert itself
            async with session.get(url, ssl=False) as response:
                if response.status == 200:
                    cert_data = await response.text()
                    # Verify integrity via SHA-256 fingerprint
                    actual_hash = hashlib.sha256(cert_data.encode()).hexdigest()
                    if actual_hash == fingerprint:
                        logger.info("Full ML-DSA certificate chain verified and loaded.")
                        return cert_data
                    else:
                        logger.error("Integrity check failed: Certificate fingerprint mismatch.")
                else:
                    logger.error(f"Hash-and-Fetch failed. HTTP Status: {response.status}")
        except Exception as e:
            logger.error(f"Network error during certificate retrieval: {e}")
    return None
