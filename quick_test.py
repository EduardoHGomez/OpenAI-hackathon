"""
Quick test to verify /optimize endpoint works
"""
import requests
import json

# Simple test code
code = """
import torch
import torch.nn as nn

def get_metrics():
    return {
        "tokens_per_second": 1000.0,
        "validation_accuracy": 0.95,
        "peak_gpu_memory_mb": 100.0,
        "run_variance": 0.01,
        "throughput_per_gb": 10.0
    }
"""

print("🧪 Testing /optimize endpoint...")
print(f"Code length: {len(code)} chars\n")

# Test the API
try:
    response = requests.post(
        'http://localhost:8000/optimize',
        json={
            'code': code,
            'objective': 'balanced',
            'max_iterations': 2
        },
        timeout=300
    )

    print(f"Status Code: {response.status_code}\n")

    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS!\n")
        print(json.dumps(result, indent=2))
    else:
        print("❌ ERROR!")
        print(response.text)

except Exception as e:
    print(f"❌ Request failed: {e}")
