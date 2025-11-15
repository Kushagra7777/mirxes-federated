from fastapi import FastAPI
from typing import List, Dict
import os
import glob
from datetime import datetime

app = FastAPI(title="MiRXES Federated Platform API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/clients/logs")
def get_client_logs():
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return {}
    
    client_logs = {}
    for log_file in glob.glob(os.path.join(logs_dir, "*.log")):
        hospital_id = os.path.basename(log_file).replace(".log", "")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                # Get last 10 entries
                recent_lines = lines[-10:] if len(lines) > 10 else lines
                client_logs[hospital_id] = [line.strip() for line in recent_lines]
        except Exception as e:
            client_logs[hospital_id] = [f"Error reading log: {str(e)}"]
    
    return client_logs

@app.get("/metrics")
def get_metrics():
    # In a real implementation, this would return actual training metrics
    # For MVP, we'll return a placeholder
    return {
        "global_round": 0,
        "total_clients": 3,
        "active_clients": 3,
        "last_updated": datetime.utcnow().isoformat()
    }