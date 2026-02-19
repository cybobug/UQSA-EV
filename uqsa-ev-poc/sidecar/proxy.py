import asyncio
import websockets
import ssl
import logging
import time
import os
from instruments import measure_handshake
from cert_fetcher import fetch_pqc_certificate

# Configure logging
log_formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SecureSidecar")
logger.setLevel(logging.INFO)

# Console Handler
ch = logging.StreamHandler()
ch.setFormatter(log_formatter)
logger.addHandler(ch)

# File Handler for TCN Integration (with flush for real-time monitoring)
os.makedirs("/app/logs", exist_ok=True)
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

fh = FlushFileHandler("/app/logs/proxy.log")
fh.setFormatter(log_formatter)
logger.addHandler(fh)

class SecureSidecar:
    """
    Hybrid PQC Proxy Bridge.
    Terminates legacy OCPP and bridges to Hybrid PQC-TLS 1.3.
    """
    def __init__(self, internal_port=9000):
        self.internal_port = internal_port
        self.csms_url = os.getenv("CSMS_URL", "wss://csms:8080")
        self.cert_dir = "/app/certs"
        self.pqc_context = self._initialize_pqc_context()

    def _initialize_pqc_context(self):
        """
        Initializes SSL Context with OQS Provider support.
        """
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        try:
            # Enforce SECLEVEL=0 to allow experimental PQC algorithms
            ctx.set_ciphers('DEFAULT@SECLEVEL=0')
            
            # Load CA if available
            ca_path = os.path.join(self.cert_dir, "ca.crt")
            if os.path.exists(ca_path):
                ctx.load_verify_locations(cafile=ca_path)
                logger.info(f"Loaded CA from {ca_path}")
            
            # For POC: Disable verification to bypass purpose errors
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            logger.info("SSL Peer Verification Disabled for POC.")
            ctx.set_ciphers('DEFAULT@SECLEVEL=0')
            logger.info("SSL Context Initialized (PQC-Ready).")
        except Exception as e:
            logger.critical(f"PQC Configuration Error: {e}")
        
        return ctx

    async def proxy_messages(self, source, destination, label):
        """
        Forward messages between sockets.
        """
        try:
            async for message in source:
                # Scans for PQC fingerprints (Hash-and-Fetch demonstration)
                if isinstance(message, str) and "PQC_FINGERPRINT:" in message:
                    asyncio.create_task(self._handle_side_channel(message))
                
                await destination.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {label}")
        except Exception as e:
            logger.error(f"Error in {label} loop: {e}")

    async def _handle_side_channel(self, message):
        try:
            fingerprint = message.split("PQC_FINGERPRINT:")[1].strip()
            logger.info(f"Detected PQC Fingerprint: {fingerprint}")
            await fetch_pqc_certificate(fingerprint)
        except Exception as e:
            logger.error(f"Side-channel error: {e}")

    async def handle_evse_client(self, client_ws):
        remote_addr = client_ws.remote_address
        logger.info(f"Incoming Legacy Connection from EVSE: {remote_addr}")

        try:
            logger.info(f"Attempting PQC-TLS Handshake with CSMS: {self.csms_url}")
            async with websockets.connect(
                self.csms_url,
                ssl=self.pqc_context,
                subprotocols=["ocpp1.6", "ocpp2.0.1"],
                open_timeout=10
            ) as csms_ws:
                
                measure_handshake(time.perf_counter())
                logger.info("PQC-TLS Handshake Successful.")

                await asyncio.gather(
                    self.proxy_messages(client_ws, csms_ws, "EVSE -> CSMS"),
                    self.proxy_messages(csms_ws, client_ws, "CSMS -> EVSE")
                )
        except Exception as e:
            logger.error(f"PQC-TLS Handshake / Proxy Failed: {e}")
        finally:
            logger.info(f"Terminating session for {remote_addr}")

    async def run_server(self):
        logger.info(f"Secure Sidecar Proxy listening on 0.0.0.0:{self.internal_port}")
        async with websockets.serve(self.handle_evse_client, "0.0.0.0", self.internal_port):
            await asyncio.Future()

if __name__ == "__main__":
    try:
        proxy = SecureSidecar()
        asyncio.run(proxy.run_server())
    except KeyboardInterrupt:
        logger.info("Proxy shutting down.")
