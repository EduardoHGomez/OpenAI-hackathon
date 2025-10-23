# 🚀 AWS Deployment Guide - PyTorch Optimizer API

## Prerequisites
- AWS EC2 instance with GPU (e.g., `g4dn.xlarge` or `p3.2xlarge`)
- Ubuntu 20.04+ or Amazon Linux 2
- CUDA installed
- OpenAI API key

---

## Step 1: Instance Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3-pip -y

# Install CUDA drivers (if not already installed)
# Follow NVIDIA's official guide for your instance type
```

---

## Step 2: Upload Project Files

```bash
# From your local machine, copy files to AWS
scp -i your-key.pem -r /path/to/openai-hackathon ubuntu@149.36.1.201:~/

# Or clone from git
# git clone your-repo.git
```

---

## Step 3: Install Dependencies

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@149.36.1.201:17136
# Navigate to project
cd openai-hackathon

# Install dependencies
pip3 install -r requirements.txt

# Set environment variable
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

---

## Step 4: Test Locally First

```bash
# Start the API server
python3 api.py

# In another terminal, test it
python3 test_api.py
```

---

## Step 5: Run in Production (with PM2 or systemd)

### Option A: Using PM2 (Recommended)

```bash
# Install PM2
sudo npm install -g pm2

# Start API
pm2 start api.py --name pytorch-api --interpreter python3

# Enable auto-restart on reboot
pm2 startup
pm2 save

# View logs
pm2 logs pytorch-api
```

### Option B: Using systemd

```bash
# Create service file
sudo nano /etc/systemd/system/pytorch-api.service
```

Paste this:
```ini
[Unit]
Description=PyTorch Optimizer API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/openai-hackathon
Environment="PATH=/usr/bin:/usr/local/bin"
EnvironmentFile=/home/ubuntu/openai-hackathon/.env
ExecStart=/usr/bin/python3 /home/ubuntu/openai-hackathon/api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable pytorch-api
sudo systemctl start pytorch-api
sudo systemctl status pytorch-api
```

---

## Step 6: Configure Security Group

In AWS Console:
1. Go to EC2 → Security Groups
2. Add inbound rule:
   - Type: Custom TCP
   - Port: 8000
   - Source: `0.0.0.0/0` (or restrict to your Vercel IP)

---

## Step 7: Update Frontend URL

In your Vercel frontend, update the API URL:

```typescript
const API_URL = "http://YOUR-AWS-PUBLIC-IP:8000"

// Or use HTTPS with nginx reverse proxy (recommended)
const API_URL = "https://api.yourdomain.com"
```

---

## 🔐 Production Hardening (Recommended)

### Add Nginx Reverse Proxy + SSL

```bash
# Install nginx
sudo apt install nginx certbot python3-certbot-nginx -y

# Configure nginx
sudo nano /etc/nginx/sites-available/pytorch-api
```

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;  # Important for SSE if you add it later
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/pytorch-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d api.yourdomain.com
```

Now your API is at `https://api.yourdomain.com`

---

## 🧪 Test from Frontend

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

const data = await response.json()
console.log(data.native)      // Native metrics
console.log(data.optimized)   // Optimized metrics
```

---

## 📊 Monitoring

```bash
# View API logs
pm2 logs pytorch-api

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

---

## 🐛 Troubleshooting

**API not accessible?**
- Check security group port 8000 is open
- Verify API is running: `pm2 status` or `sudo systemctl status pytorch-api`

**CUDA errors?**
- Verify CUDA: `nvidia-smi`
- Check PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

**Slow optimization?**
- Reduce `max_iterations` to 2
- Use smaller test code first
- Check OpenAI API rate limits

---

## 💰 Cost Optimization

- Use Spot Instances (save 70%)
- Auto-stop instance when not in use
- Monitor costs in AWS Cost Explorer
