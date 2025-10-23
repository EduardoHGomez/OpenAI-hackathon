#!/bin/bash

# Quick API Test - Use this to verify your API is working

# Your AWS IP
IP="149.36.1.201"
PORT="8000"

echo "🧪 Testing PyTorch Optimizer API"
echo "=================================="
echo ""

# Test 1: Health check
echo "1️⃣  Health Check (GET /)"
echo "   curl http://$IP:$PORT/"
echo ""
curl -s http://$IP:$PORT/ | python3 -m json.tool
echo ""
echo ""

# Test 2: Detailed health
echo "2️⃣  Detailed Health (GET /health)"
echo "   curl http://$IP:$PORT/health"
echo ""
curl -s http://$IP:$PORT/health | python3 -m json.tool
echo ""
echo ""

# Test 3: Full optimization (optional - takes 30-90s)
read -p "Run full optimization test? (takes 30-90s) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "3️⃣  Optimization Test (POST /optimize)"
    echo "   This will take 30-90 seconds..."
    echo ""

    curl -X POST http://$IP:$PORT/optimize \
      -H "Content-Type: application/json" \
      -d '{
        "code": "import torch\nimport torch.nn as nn\n\ndef get_metrics():\n    return {\n        \"tokens_per_second\": 100.0,\n        \"validation_accuracy\": 0.95,\n        \"peak_gpu_memory_mb\": 1024.0,\n        \"run_variance\": 0.01,\n        \"throughput_per_gb\": 10.0\n    }",
        "objective": "balanced",
        "max_iterations": 2
      }' | python3 -m json.tool
fi

echo ""
echo "✅ Done!"
