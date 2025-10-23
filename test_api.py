"""
Quick test script for the FastAPI endpoint
Run this after starting the server to verify it works
"""

import requests
import json

# Test code - simple PyTorch training loop
TEST_CODE = """
import torch
import torch.nn as nn

def train():
    model = nn.Linear(1000, 1000).cuda()
    x = torch.randn(32, 1000).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(100):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    return {"loss": loss.item(), "samples": 3200}

def validate():
    return 0.95

def get_metrics():
    result = train()
    acc = validate()
    import torch
    import time

    # Measure speed
    start = time.time()
    train()
    elapsed = time.time() - start

    return {
        "tokens_per_second": 3200 / elapsed,
        "validation_accuracy": acc,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "run_variance": 0.01,
        "throughput_per_gb": 100.0
    }
"""

def test_api(api_url="http://localhost:8000"):
    print("🧪 Testing PyTorch Optimizer API\n")

    # Test 1: Health check
    print("1. Health check...")
    try:
        response = requests.get(f"{api_url}/")
        print(f"   ✅ Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return

    # Test 2: Optimize endpoint
    print("2. Testing /optimize endpoint...")
    print("   (This may take 30-90 seconds...)\n")

    payload = {
        "code": TEST_CODE,
        "objective": "balanced",
        "max_iterations": 2  # Quick test with 2 iterations
    }

    try:
        response = requests.post(
            f"{api_url}/optimize",
            json=payload,
            timeout=180  # 3 minutes timeout
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success!\n")
            print(f"   Experiment ID: {result['experiment_id']}")
            print(f"\n   📊 Native Metrics:")
            for key, value in result['native'].items():
                print(f"      {key}: {value:.2f}")

            print(f"\n   🔥 Optimized Metrics:")
            for key, value in result['optimized'].items():
                print(f"      {key}: {value:.2f}")

            print(f"\n   📈 Improvements:")
            for key, value in result['improvement_percent'].items():
                print(f"      {key}: {value:+.1f}%")

            print(f"\n   ✅ Test PASSED - API is working!")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    # Test locally
    test_api("http://localhost:8000")

    # Or test on AWS (replace with your instance IP)
    # test_api("http://149.36.1.201:8000")
