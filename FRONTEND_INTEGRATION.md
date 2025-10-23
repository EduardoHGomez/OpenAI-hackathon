# 🎨 Frontend Integration Guide

## How to integrate the FastAPI endpoint with your Vercel app

---

## 1. Update your BenchmarkForm component

Replace the mock data in your `handleSubmit` function with a real API call:

```typescript
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault()

  if (!prompt.trim()) {
    toast({
      title: "Error",
      description: "Please enter a prompt to benchmark",
      variant: "destructive",
    })
    return
  }

  setIsRunning(true)

  try {
    // Call your AWS API endpoint
    const response = await fetch("http://149.36.1.201:8888/optimize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        code: prompt,  // The code from textarea
        objective: "balanced",  // or "speed", "memory", "accuracy"
        max_iterations: 3
      }),
      // Important: optimization takes 30-90 seconds
      signal: AbortSignal.timeout(188880)  // 3 min timeout
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const result = await response.json()

    // result structure:
    // {
    //   experiment_id: "exp-abc123",
    //   timestamp: "2025-01-23T...",
    //   native: { tokensPerSecond, validationAccuracy, ... },
    //   optimized: { tokensPerSecond, validationAccuracy, ... },
    //   improvement_percent: { ... },
    //   optimized_code: "..."
    // }

    const timestamp = new Date().toISOString()
    const experimentId = result.experiment_id

    // Create two separate run entries
    const nativeRun = {
      id: Date.now(),
      experimentId,
      prompt,
      timestamp,
      kernelType: "Native",
      tokensPerSecond: result.native.tokensPerSecond,
      validationAccuracy: result.native.validationAccuracy,
      peakGpuMemoryMb: result.native.peakGpuMemoryMb,
      runToRunVariance: result.native.runToRunVariance,
      throughputPerDollar: result.native.throughputPerDollar,
    }

    const optimizedRun = {
      id: Date.now() + 1,
      experimentId,
      prompt,
      timestamp,
      kernelType: "Optimized",
      tokensPerSecond: result.optimized.tokensPerSecond,
      validationAccuracy: result.optimized.validationAccuracy,
      peakGpuMemoryMb: result.optimized.peakGpuMemoryMb,
      runToRunVariance: result.optimized.runToRunVariance,
      throughputPerDollar: result.optimized.throughputPerDollar,
    }

    // Store both runs
    const existingResults = JSON.parse(localStorage.getItem("benchmark-runs") || "[]")
    existingResults.push(nativeRun, optimizedRun)
    localStorage.setItem("benchmark-runs", JSON.stringify(existingResults))

    // Store optimized code separately if you want to show it later
    localStorage.setItem(`optimized-code-${experimentId}`, result.optimized_code)

    setIsRunning(false)

    toast({
      title: "Benchmark Complete",
      description: `Speed improved by ${result.improvement_percent.tokens_per_second?.toFixed(1)}%!`,
    })

    // Navigate to dashboard
    setTimeout(() => {
      router.push("/dashboard")
    }, 500)

  } catch (error) {
    setIsRunning(false)
    toast({
      title: "Optimization Failed",
      description: error.message || "Server error - check if API is running",
      variant: "destructive",
    })
  }
}
```

---

## 2. Add environment variable (recommended)

Create `.env.local` in your Vercel project:

```bash
NEXT_PUBLIC_API_URL=http://149.36.1.201:8888
```

Then use it:

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888"

const response = await fetch(`${API_URL}/optimize`, {
  // ...
})
```

---

## 3. Add loading states and progress

```typescript
const [isRunning, setIsRunning] = useState(false)
const [progress, setProgress] = useState("")

// Update progress message
setProgress("Measuring native code performance...")
// ... API call ...
setProgress("Running optimization (may take 60 seconds)...")
```

Display in UI:

```tsx
{isRunning && (
  <div className="py-4 text-sm text-muted-foreground">
    <Loader2 className="inline h-4 w-4 animate-spin mr-2" />
    {progress}
  </div>
)}
```

---

## 4. Error Handling

```typescript
try {
  const response = await fetch(...)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || "Optimization failed")
  }

  const result = await response.json()
  // ...
} catch (error) {
  if (error.name === 'TimeoutError') {
    toast({
      title: "Request Timeout",
      description: "Optimization took too long. Try simpler code or reduce iterations.",
      variant: "destructive",
    })
  } else if (error.name === 'TypeError') {
    toast({
      title: "Connection Error",
      description: "Cannot reach API server. Is it running?",
      variant: "destructive",
    })
  } else {
    toast({
      title: "Error",
      description: error.message,
      variant: "destructive",
    })
  }
}
```

---

## 5. (Optional) Show Optimized Code

Add a button to view the optimized code:

```typescript
const viewOptimizedCode = (experimentId: string) => {
  const code = localStorage.getItem(`optimized-code-${experimentId}`)
  if (code) {
    // Show in modal or navigate to code viewer
    setOptimizedCode(code)
    setShowCodeModal(true)
  }
}
```

---

## 6. Testing Locally

Before deploying to Vercel, test locally:

```bash
# Terminal 1: Start your FastAPI server
cd openai-hackathon
python3 api.py

# Terminal 2: Start your Next.js app
cd your-frontend-project
npm run dev
```

Then try submitting a simple PyTorch code snippet through your form.

---

## 7. CORS Issues?

If you get CORS errors, the FastAPI server already has CORS enabled for all origins (`allow_origins=["*"]`).

For production, restrict to your domain:

```python
# In api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.vercel.app",
        "http://localhost:3000"  # for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 8. Example Test Code for Users

Provide users with example PyTorch snippets they can test:

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

def validate():
    return 0.95

def get_metrics():
    import time
    start = time.time()
    result = train()
    elapsed = time.time() - start

    return {
        "tokens_per_second": 3200 / elapsed,
        "validation_accuracy": validate(),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "run_variance": 0.01,
        "throughput_per_gb": 100.0
    }
```

---

## 📊 Expected Response Format

```json
{
  "experiment_id": "exp-a1b2c3d4",
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

## 🚀 Deploy to Production

1. Deploy FastAPI to AWS (see AWS_SETUP.md)
2. Update `NEXT_PUBLIC_API_URL` in Vercel settings
3. Deploy frontend: `vercel --prod`

Done! 🎉
