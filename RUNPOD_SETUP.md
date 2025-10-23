# 🚀 Runpod Deployment Guide

**Your Runpod IP:** `149.36.1.201:8000`

---

## ⚡ Quick Setup (5 minutes)

### 1. Start Your Runpod Instance
- Go to Runpod dashboard
- Start your GPU pod
- Note the exposed port for 8000 (usually auto-mapped)

### 2. Connect via SSH
```bash
# Get SSH command from Runpod dashboard, usually:
ssh root@149.36.1.201 -p 22 -i ~/.ssh/id_ed25519
```

### 3. Upload Your Code
```bash
# From your local machine
scp -r /Users/eduardo/Documents/openai-hackathon/* root@149.36.1.201:~/app/

# Or use Runpod's web terminal and git clone
```

### 4. Install Dependencies
```bash
# On Runpod instance
cd ~/app
pip install -r requirements.txt
```

### 5. Set API Key
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 6. Start API
```bash
chmod +x start_api.sh
./start_api.sh

# Or directly
python3 api.py
```

---

## 🌐 Port Mapping

Runpod automatically exposes ports. Your API runs on:
- **Internal:** `0.0.0.0:8000` (inside the pod)
- **External:** `149.36.1.201:8000` (public access)

No firewall/security group needed! ✅

---

## 🧪 Test It

```bash
# From your local machine
curl http://149.36.1.201:8000/

# Should return:
# {"status":"online","service":"PyTorch Optimizer API",...}
```

Or use the test script:
```bash
./quick_test.sh
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
```

---

## 🔄 Keep API Running (Auto-restart)

### Option 1: Using `screen` (simplest)
```bash
screen -S pytorch-api
python3 api.py
# Press Ctrl+A, then D to detach

# Reconnect later:
screen -r pytorch-api
```

### Option 2: Using PM2
```bash
pip install pm2
pm2 start api.py --name pytorch-api --interpreter python3
pm2 save
```

---

## 💰 Runpod vs AWS

| Feature | Runpod | AWS |
|---------|--------|-----|
| **Setup** | ✅ 5 min | ❌ 15-30 min |
| **Cost** | ✅ $0.2-0.4/hr | ❌ $0.5-1.5/hr |
| **Ports** | ✅ Auto-exposed | ❌ Manual security groups |
| **GPU** | ✅ Instant | ❌ May need quotas |
| **Billing** | ✅ Per-second | ❌ Per-hour |

---

## 🐛 Troubleshooting

**Can't connect to API?**
```bash
# 1. Check API is running inside pod
curl http://localhost:8000/

# 2. Check port mapping in Runpod dashboard
# Should show: 8000 -> external port

# 3. Test from local machine
curl http://149.36.1.201:8000/
```

**CUDA not found?**
```bash
# Check GPU
nvidia-smi

# Most Runpod templates have CUDA pre-installed ✅
```

**Out of memory?**
- Reduce `max_iterations` to 2
- Use smaller test code
- Or upgrade to higher-tier GPU pod

---

## 📁 Runpod File Locations

```
~/app/                          # Your project folder
├── api.py                      # FastAPI server
├── pytorch_optimizer.py        # Optimizer logic
├── .env                        # API keys (create this!)
├── requirements.txt            # Dependencies
└── start_api.sh               # Startup script
```

---

## 🎯 Your API Endpoints

- **Health:** `http://149.36.1.201:8000/`
- **Detailed Health:** `http://149.36.1.201:8000/health`
- **Optimize:** `POST http://149.36.1.201:8000/optimize`

---

## ⏸️ Pause/Resume

**Pause pod (save money):**
- Stop API: `Ctrl+C`
- Runpod dashboard → Stop pod
- ✅ Your files persist!

**Resume:**
- Start pod in dashboard
- SSH in
- `cd ~/app && ./start_api.sh`

---

## 🚀 Quick Commands

```bash
# Start API
./start_api.sh

# Test API
curl http://149.36.1.201:8000/

# Full test
./quick_test.sh

# Check logs
tail -f logs.txt  # if you redirect output
```

---

**Runpod is perfect for hackathons - cheap, fast, easy!** 💪

Your API: `http://149.36.1.201:8000` ✅
