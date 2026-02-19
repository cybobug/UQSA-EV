import asyncio
import websockets
import logging
import os
from ocpp.v16 import ChargePoint as cp
from ocpp.v16 import call
from ocpp.v16.enums import RegistrationStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVSE")

class ChargePoint(cp):
    async def send_boot_notification(self):
        # Compatibility fix for newer ocpp library versions
        try:
            request = call.BootNotificationPayload(
                charge_point_vendor="UQSA-EV-Vendor",
                charge_point_model="Quantum-Ready-V1"
            )
        except AttributeError:
            request = call.BootNotification(
                charge_point_vendor="UQSA-EV-Vendor",
                charge_point_model="Quantum-Ready-V1"
            )
        response = await self.call(request)

        status = getattr(response, 'status', None)
        if status == RegistrationStatus.accepted or str(status).lower() == "accepted":
            logger.info("Connected to Sidecar/CSMS: Registration Accepted.")
            # Start a dummy heartbeat loop to keep connection alive
            while True:
                await asyncio.sleep(10)
                logger.info("EVSE Heartbeat ...")

async def main():
    proxy_url = os.getenv("PROXY_URL", "ws://sidecar:9000")
    
    while True:
        logger.info(f"Connecting to Sidecar at {proxy_url}...")
        try:
            async with websockets.connect(
                proxy_url,
                subprotocols=['ocpp1.6']
            ) as ws:
                charge_point = ChargePoint("EVSE-01", ws)
                await asyncio.gather(
                    charge_point.start(),
                    charge_point.send_boot_notification()
                )
        except Exception as e:
            logger.error(f"Connection Failed: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
