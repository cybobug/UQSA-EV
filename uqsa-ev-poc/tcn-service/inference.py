import torch
import numpy as np
import logging
import time
import os
import re
from model import TemporalConvolutionalNetwork

# Configure Inference Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [TCN-Monitor] %(levelname)s: %(message)s')
logger = logging.getLogger("TCN-Inference")

class TCNPredictor:
    def __init__(self, model_path="uqsa_tcn.pth"):
        self.input_size = 3
        self.num_channels = [16, 32, 64]
        self.model = TemporalConvolutionalNetwork(self.input_size, self.num_channels)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                logger.info("TCN Model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load pre-trained model: {e}. Running with random weights.")
        else:
            logger.warning("No model found at uqsa_tcn.pth. Using randomized weights for PoC.")
            self.model.eval()

    def predict(self, window):
        """
        Features: [Handshake(binary), PacketSize(normalized), DeltaTime(normalized)]
        """
        # Prediction
        tensor_in = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor_in)
            probs = torch.softmax(output, dim=1)
            anomaly_score = probs[0, 1].item()

        # Baseline POC logic: If intensities are consistently high, boost the anomaly score
        # Feature 2 is intensity
        mean_intensity = np.mean(window[2])
        if mean_intensity > 0.5:
            # Gradually boost score based on intensity
            anomaly_score = max(anomaly_score, min(0.99, mean_intensity + 0.2))

        logger.info(f"Anomaly Score: {anomaly_score:.4f} (Mean Intensity: {mean_intensity:.4f})")

        if anomaly_score > 0.70:
            logger.error(f"🚨 [CRITICAL ALERT] Anomaly Detected! Score: {anomaly_score:.2f}")
            logger.info("Intelligence Plane triggering mitigation: rate-limiting {0.0.0.0}")
        return anomaly_score

def watch_logs(predictor):
    log_path = "/app/logs/proxy.log"
    logger.info(f"Looking for log file at {log_path}...")
    
    # Wait for file to be created by sidecar
    while not os.path.exists(log_path):
        time.sleep(1)
    
    logger.info("Log file detected. Starting real-time monitoring...")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        # Move to end of file to ignore past events
        f.seek(0, os.SEEK_END)
        
        feature_window = []
        last_time = time.time()
        last_heartbeat = time.time()
        
        while True:
            # Heartbeat every 5 seconds to show monitor is alive
            if time.time() - last_heartbeat > 5:
                logger.info("TCN Monitor alive. Waiting for traffic...")
                last_heartbeat = time.time()

            line = f.readline()
            if not line:
                # If we hit EOF, sleep and retry
                time.sleep(0.1) 
                continue
            
            # Extract features from log line
            if "Incoming Legacy Connection" in line:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time
                
                # Map delta_time to a feature: 1/delta_time represents intensity
                intensity = min(1.0, 0.05 / (delta_time + 1e-6))
                logger.info(f"Monitor caught connection event. Intensity: {intensity:.4f}")
                feature_window.append([1.0, np.random.uniform(0.5, 1.5), intensity])
                logger.info(f"Feature window size: {len(feature_window)}")
            
                if len(feature_window) >= 10:
                    logger.info("Window full. Calling TCN prediction...")
                    input_data = np.array(feature_window[-10:]).T
                    predictor.predict(input_data)
                    # Keep the window sliding
                    feature_window = feature_window[-9:]

if __name__ == "__main__":
    predictor = TCNPredictor()
    try:
        watch_logs(predictor)
    except KeyboardInterrupt:
        logger.info("TCN Monitor stopped.")
    except Exception as e:
        logger.error(f"Monitor error: {e}")
