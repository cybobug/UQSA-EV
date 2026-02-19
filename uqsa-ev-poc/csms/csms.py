import asyncio
import ssl
import websockets
import logging
import os
from flask import Flask, send_from_directory, abort
from threading import Thread
from ocpp.v16 import ChargePoint as cp
from ocpp.v16 import call_result
from ocpp.routing import on
from ocpp.v16.enums import Action, RegistrationStatus
from datetime import datetime, timezone

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CSMS-Backend")

# Configuration
CERT_DIR = os.getenv("CERT_DIR", "/app/certs")
CSMS_PORT = int(os.getenv("CSMS_PORT", 8080))
FLASK_PORT = int(os.getenv("FLASK_PORT", 443))
HOST = "0.0.0.0"

# 1. Flask App for 'Hash-and-Fetch' (Channel for oversized PQC certs)
app = Flask(__name__)

@app.route('/certs/<path:filename>')
def download_cert(filename):
    """
    Serve certificates safely. send_from_directory prevents path traversal.
    """
    try:
        return send_from_directory(CERT_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

def run_flask():
    logger.info(f"Starting Certificate Sidecar on port {FLASK_PORT}")
    # In production, use a WSGI server (e.g., gunicorn) instead of app.run
    app.run(host=HOST, port=FLASK_PORT, debug=False, use_reloader=False)

# 2. OCPP Business Logic
class ChargePointHandler(cp):
    @on(Action.boot_notification)
    async def on_boot_notification(self, charge_point_vendor, charge_point_model, **kwargs):
        logger.info(f"Received BootNotification from {charge_point_vendor} ({charge_point_model})")
        
        # dynamic current time
        now = datetime.now(timezone.utc).isoformat()
        
        return call_result.BootNotificationPayload(
            current_time=now,
            interval=300,
            status=RegistrationStatus.Accepted
        )

async def handle_connection(websocket):
    """
    Handle individual WebSocket connections.
    """
    # Extract charge point ID from the URL path (standard OCPP behavior)
    try:
        # path is usually /ID
        charge_point_id = websocket.request.path.strip('/')
        if not charge_point_id:
             charge_point_id = "Unknown"
    except AttributeError:
        charge_point_id = "Unknown"

    logger.info(f"New PQC-TLS Connection established: {charge_point_id}")
    
    charge_point = ChargePointHandler(charge_point_id, websocket)
    
    try:
        await charge_point.start()
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed for {charge_point_id}")
    except Exception as e:
        logger.error(f"Error handling {charge_point_id}: {e}")

async def main():
    logger.info(f"OpenSSL Version: {ssl.OPENSSL_VERSION}")
    # Load PQC Certificates
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    cert_path = os.path.join(CERT_DIR, "server.crt")
    key_path = os.path.join(CERT_DIR, "server.key")

    try:
        # SECLEVEL=0 is often required for OQS/Hybrid algorithms currently
        ssl_context.set_ciphers('DEFAULT@SECLEVEL=0') 
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
        logger.info("PQC-TLS 1.3 Server Context Initialized.")
    except Exception as e:
        logger.warning(f"PQC Certificate Loading Failed: {e}. Falling back to non-SSL or continuation for TCN demo.")
        # We continue so that the sidecar -> tcn loop can still be tested even if the external leg is insecure for now
        ssl_context = None # Or just ignore the error

    # Start Flask Side-channel in background thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start Secure WebSocket Server
    logger.info(f"Starting CSMS WebSocket Server on {HOST}:{CSMS_PORT}")
    async with websockets.serve(handle_connection, HOST, CSMS_PORT, ssl=ssl_context):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("CSMS shutting down.")