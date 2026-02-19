import time
import logging

def measure_handshake(start_time):
    """
    Calculates and logs the handshake overhead.
    """
    duration = (time.perf_counter() - start_time) * 1000
    logging.info(f"[INSTRUMENTATION] PQC Handshake Overhead: {duration:.4f} ms")
    return duration
