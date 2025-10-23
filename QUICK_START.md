# ⚡ Quick Start - PyTorch Optimizer API

## 🎯 What This Does

**Receives:** PyTorch code (via POST request)
**Returns:** Performance metrics for both native and optimized versions
**Use case:** Your Vercel frontend sends code → AWS processes it → Returns metrics for graphs

---

## 🚀 Local Testing (5 minutes)

```bash
# 1. Set API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python3 api.py
# Server runs on http://localhost:8000

# 4. Test it (in another terminal)
python3 test_api.py
```

---

## ☁️ AWS Deployment (10 minutes)

```bash
# 1. SSH into your GPU instance
ssh -i your-key.pem ubuntu@149.36.1.201:17136
# 2. Upload files
scp -i your-key.pem -r ./* ubuntu@149.36.1.201:~/openai-hackathon/

# 3. On AWS instance:
cd openai-hackathon
echo "OPENAI_API_KEY=your-key" > .env
pip3 install -r requirements.txt
chmod +x start_api.sh
./start_api.sh

# 4. Open port 8000 in AWS Security Groups
# EC2 Console → Security Groups → Add inbound rule → Port 8000

# 5. Test from anywhere
curl http://149.36.1.201:8000/
```

---

## 📡 API Endpoints

### `GET /`
Health check

```bash
curl http://149.36.1.201:8000/
# Response: {"status": "online", "gpu_available": true}
```

### `POST /optimize`
Main endpoint - optimize PyTorch code

```bash
curl -X POST http://149.36.1.201:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch\n...",
    "objective": "balanced",
    "max_iterations": 3
  }'
```

**Request:**
```json
{
  "code": "string (required) - Your PyTorch code",
  "objective": "balanced | speed | memory | accuracy (default: balanced)",
  "max_iterations": "int (default: 3) - Number of optimization iterations"
}
```

**Response:**
```json
{
  "experiment_id": "exp-abc123",
  "timestamp": "2025-01-23T10:30:00Z",
  "native": {
    "tokensPerSecond": 150.5,
    "validationAccuracy": 0.95,
    "peakGpuMemoryMb": 1024.0,
    "runToRunVariance": 0.05,
    "throughputPerDollar": 50.2
  },
  "optimized": {
    "tokensPerSecond": 245.8,
    "validationAccuracy": 0.96,
    "peakGpuMemoryMb": 768.0,
    "runToRunVariance": 0.02,
    "throughputPerDollar": 85.4
  },
  "improvement_percent": {
    "tokens_per_second": 63.3,
    "validation_accuracy": 1.1,
    "peak_gpu_memory_mb": 25.0,
    "run_variance": 60.0,
    "throughput_per_gb": 70.1
  },
  "optimized_code": "import torch\n# Optimized version..."
}
```

---

## 🎨 Frontend Integration

```typescript
const response = await fetch("http://149.36.1.201:8000/optimize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    code: userCode,
    objective: "balanced",
    max_iterations: 3
  })
})

const result = await response.json()

// Use result.native and result.optimized for your graphs
```

See `FRONTEND_INTEGRATION.md` for full example.

---

## 📊 Metrics Explained

| Metric | Description | Higher/Lower is better |
|--------|-------------|------------------------|
| **tokensPerSecond** | Processing speed | Higher ✅ |
| **validationAccuracy** | Model accuracy (0-1) | Higher ✅ |
| **peakGpuMemoryMb** | GPU memory usage | Lower ✅ |
| **runToRunVariance** | Stability across runs | Lower ✅ |
| **throughputPerDollar** | Cost efficiency | Higher ✅ |

---

## 🧪 Test Code Examples

### Simple Linear Model
```python
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

def get_metrics():
    import time
    start = time.time()
    train()
    elapsed = time.time() - start

    return {
        "tokens_per_second": 3200 / elapsed,
        "validation_accuracy": 0.95,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "run_variance": 0.01,
        "throughput_per_gb": 100.0
    }
```

---

## 🐛 Troubleshooting

**"Connection refused"**
- Check if API is running: `curl http://localhost:8000/`
- Check AWS security group allows port 8000

**"CUDA not available"**
- Verify GPU: `nvidia-smi`
- Check PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA drivers if needed

**"OpenAI API error"**
- Check `.env` file exists with valid `OPENAI_API_KEY`
- Verify key: `cat .env`

**"Optimization takes too long"**
- Reduce `max_iterations` to 2
- Use simpler test code
- Check network latency to OpenAI

**"Out of memory"**
- Your test code uses too much GPU memory
- Reduce batch size in your code
- Use smaller model

---

## 📁 File Structure

```
openai-hackathon/
├── api.py                      # FastAPI server ⭐
├── pytorch_optimizer.py        # Core optimization logic
├── test_api.py                 # Test script
├── start_api.sh                # Startup script
├── requirements.txt            # Dependencies
├── .env                        # API keys (create this!)
├── AWS_SETUP.md               # Detailed AWS guide
├── FRONTEND_INTEGRATION.md     # Frontend code examples
└── QUICK_START.md             # This file
```

---

## 🎉 That's It!

**Next Steps:**
1. ✅ Start API: `./start_api.sh`
2. ✅ Test locally: `python3 test_api.py`
3. ✅ Deploy to AWS (see AWS_SETUP.md)
4. ✅ Update frontend URL
5. ✅ Ship it! 🚀

Questions? Check the detailed guides:
- AWS deployment → `AWS_SETUP.md`
- Frontend integration → `FRONTEND_INTEGRATION.md`
