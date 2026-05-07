# Example: query VLMMM server status and run one step
import requests
import time

SERVER = "http://localhost:8000"

# health check
r = requests.get(f"{SERVER}/status")
print("status:", r.json())

# run one MM step
payload = {"debug": True}  # optional
start = time.perf_counter()
r = requests.post(f"{SERVER}/run_step", json=payload, timeout=300)
elapsed = time.perf_counter() - start
resp = r.json()
if r.ok:
    print(f"Time taken: {elapsed:.3f}s")
    print("run_step outputs:", resp["outputs"])
else:
    print("error:", resp)