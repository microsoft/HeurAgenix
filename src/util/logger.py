import os
import psutil
import platform
from datetime import datetime

def build_logger(log_file_path: str, context: str):
    def log(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"[{timestamp}, {context}] {message}"
        # print(full_msg, flush=True)  # Still keeping print commented out as per request
        if log_file_path:
            try:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(full_msg + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass
    return log

def log_system_status(context: str, logger=None):
    if logger is None:
        return
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        disk_info = f"Disk R/W: {disk.read_bytes>>20}MB/{disk.write_bytes>>20}MB" if disk else "Disk: N/A"
        load_avg = "N/A"
        if hasattr(os, 'getloadavg'):
            load_avg = f"{os.getloadavg()}"
            
        logger(f"[System Status - {context}] Host: {platform.node()} | CPU: {cpu_percent}% | Load: {load_avg} | "
              f"Mem: {mem.percent}% (Used: {mem.used>>20}MB, Avail: {mem.available>>20}MB) | {disk_info}")
    except Exception as e:
        logger(f"Failed to log system status: {e}")
