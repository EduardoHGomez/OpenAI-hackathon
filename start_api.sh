#!/bin/bash

echo "🚀 Starting PyTorch Optimizer API..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Create it with: echo 'OPENAI_API_KEY=your-key' > .env"
    exit 1
fi

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: CUDA not available. Will run on CPU (slower)"
fi

# Check if dependencies are installed
python3 -c "import fastapi, torch, openai, langgraph" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Start API
echo "✅ Starting API on http://0.0.0.0:8888"
echo ""
echo "Test it locally with:"
echo "  python3 test_api.py"
echo ""
echo "Or from frontend:"
echo "  fetch('http://YOUR-IP:8888/optimize', {...})"
echo ""

python3 api.py
