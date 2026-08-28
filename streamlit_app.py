"""Single Streamlit entry point for the assignment sandbox.

Starts the local FastAPI backend, then imports the original Streamlit UI as a
normal package so app-relative imports work from both CLI and IDE execution.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

def port_open(port: int = 8000) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0

def start_backend() -> None:
    if port_open():
        return
    log_path = ROOT / "fastapi_backend.log"
    log = open(log_path, "a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT),
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    for _ in range(60):
        if port_open():
            return
        time.sleep(0.25)
    raise RuntimeError(f"FastAPI backend failed to start. Check {log_path.name}.")

start_backend()
os.environ.setdefault("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
from app.main import *  # noqa: F401,F403,E402
