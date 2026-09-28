"""Report what is ready for the workshop without changing anything."""

import importlib.metadata
import subprocess
from urllib.request import urlopen

for package in ["httpx", "openai", "opentelemetry-sdk", "deepeval"]:
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: not installed")
try:
    result = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=5)
    print(f"Docker daemon: {result.stdout.strip() if result.returncode == 0 else 'unavailable'}")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("Docker daemon: unavailable")
try:
    with urlopen("http://127.0.0.1:3000/api/public/health", timeout=2) as response:
        print(f"Langfuse: HTTP {response.status}")
except Exception:
    print("Langfuse: unavailable")
