import os
from datetime import datetime

def log_event(hospital_id, event_name):
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    log_entry = f"{timestamp} | {hospital_id} | {event_name}\n"
    
    log_file_path = f"logs/{hospital_id}.log"
    with open(log_file_path, "a") as log_file:
        log_file.write(log_entry)