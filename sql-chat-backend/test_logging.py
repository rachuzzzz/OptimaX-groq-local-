"""
Simple test script to verify logging works correctly.
Run this directly: python test_logging.py
"""
import sys
import os

# Force unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 60)
print("LOGGING TEST - This should appear immediately")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"PYTHONUNBUFFERED: {os.environ.get('PYTHONUNBUFFERED', 'NOT SET')}")
print()

# Test 1: Direct print
print("[TEST 1] Direct print() - should appear", flush=True)

# Test 2: sys.stdout.write
sys.stdout.write("[TEST 2] sys.stdout.write() - should appear\n")
sys.stdout.flush()

# Test 3: Logging
import logging

class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

root = logging.getLogger()
root.setLevel(logging.INFO)
for h in root.handlers[:]:
    root.removeHandler(h)
handler = FlushingStreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root.addHandler(handler)

logging.info("[TEST 3] logging.info() - should appear")

# Test 4: Test in a simple FastAPI request simulation
print()
print("[TEST 4] Starting uvicorn with test endpoint...")
print("         Send a request to http://localhost:8001/test")
print("         Press Ctrl+C to stop")
print()

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/test")
def test_endpoint():
    print("\n" + "="*60, flush=True)
    print("[REQUEST] /test endpoint called!", flush=True)
    print("="*60, flush=True)
    logging.info("[REQUEST] logging.info from /test endpoint")
    return {"status": "ok", "message": "Check terminal for logs!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
