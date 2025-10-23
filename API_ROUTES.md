# 🚀 API Routes - Quick Reference

## Your API Details

**AWS IP:** `149.36.1.201`
**Port:** `8000`
**Base URL:** `http://149.36.1.201:8000`

---

## 📡 Available Routes

### 1. `GET /` - Basic Health Check

**Test it:**
```bash
curl http://149.36.1.201:8000/
```

**Response:**
```json
{
  "status": "online",
  "service": "PyTorch Optimizer API",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_count": 1,
  "gpu_name": "Tesla T4"
}
```

---

### 2. `GET /health` - Detailed Health Check

**Test it:**
```bash
curl http://149.36.1.201:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu": {
    "available": true,
    "count": 1,
    "name": "Tesla T4"
  },
  "api_key_set": true,
  "endpoints": {
    "health": "GET /health",
    "optimize": "POST /optimize"
  }
}
```

---

### 3. `POST /optimize` - Main Optimization Endpoint

**Test it:**
```bash
curl -X POST http://149.36.1.201:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch\n...",
    "objective": "balanced",
    "max_iterations": 3
  }'
```

**Request Body:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `code` | string | Yes | - | PyTorch code to optimize |
| `objective` | string | No | "balanced" | "speed", "memory", "accuracy", or "balanced" |
| `max_iterations` | int | No | 3 | Number of optimization iterations (1-5) |

**Response (30-90 seconds):**
```json
{
  "experiment_id": "exp-abc123",
  "timestamp": "2025-01-23T10:30:00.000Z",
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
  "optimized_code": "import torch\n..."
}
```

---

## ⚡ Quick Test Commands

**Health check:**
```bash
curl http://149.36.1.201:8000/
```

**Full test:**
```bash
./quick_test.sh
```

**From frontend:**
```javascript
const response = await fetch("http://149.36.1.201:8000/optimize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    code: userCode,
    objective: "balanced",
    max_iterations: 3
  })
})
const data = await response.json()
```

---

## 🔥 Start the API

```bash
# Local
python3 api.py

# AWS
ssh -i your-key.pem ubuntu@149.36.1.201
cd openai-hackathon
./start_api.sh
```

---

## 🐛 Troubleshooting

**Can't connect?**
- Check API is running: `ps aux | grep api.py`
- Check port 8000 is open in AWS Security Groups
- Test locally first: `curl http://localhost:8000/`

**"Connection refused"**
- API not running → Start it: `./start_api.sh`
- Firewall blocking → Open port 8000 in AWS Console

**"Internal Server Error"**
- Check logs: `tail -f /var/log/pytorch-api.log` (if using systemd)
- Or: `pm2 logs pytorch-api` (if using PM2)
- Verify `.env` has `OPENAI_API_KEY`

---

## 📊 Route Summary

| Route | Method | Purpose | Time |
|-------|--------|---------|------|
| `/` | GET | Basic health check | <1s |
| `/health` | GET | Detailed status | <1s |
| `/optimize` | POST | Optimize PyTorch code | 30-90s |

---

**Your IP: 149.36.1.201:8000** ✅
