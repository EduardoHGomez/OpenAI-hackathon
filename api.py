"""
FastAPI Server for PyTorch Optimization
Runs on Runpod GPU Instance - Receives code, returns metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pytorch_optimizer import optimize_pytorch_code, execute_and_measure
import traceback
from datetime import datetime
import uuid

app = FastAPI(title="PyTorch Optimizer API")

# CORS - Allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (wildcard)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


class OptimizeRequest(BaseModel):
    code: str
    objective: str = "balanced"  # "speed", "memory", "accuracy", or "balanced"
    max_iterations: int = 3


class MetricsResponse(BaseModel):
    experiment_id: str
    timestamp: str
    native: dict
    optimized: dict
    improvement_percent: dict
    optimized_code: str


@app.get("/")
def root():
    """Health check"""
    import torch
    return {
        "status": "online",
        "service": "PyTorch Optimizer API",
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }


@app.get("/health")
def health():
    """Detailed health check"""
    import torch
    import os
    return {
        "status": "healthy",
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        },
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "endpoints": {
            "health": "GET /health",
            "optimize": "POST /optimize"
        }
    }


@app.post("/optimize", response_model=MetricsResponse)
async def optimize_code(request: OptimizeRequest):
    """
    Main endpoint - Receives PyTorch code, runs optimization, returns metrics

    Returns both native (original) and optimized metrics for comparison graphs
    """
    try:
        print(f"\n{'='*70}")
        print(f"🚀 NEW OPTIMIZATION REQUEST")
        print(f"{'='*70}")
        print(f"Code length: {len(request.code)} chars")
        print(f"Objective: {request.objective}")
        print(f"Max iterations: {request.max_iterations}")

        experiment_id = f"exp-{uuid.uuid4().hex[:8]}"

        # Step 1: Measure NATIVE (original) code metrics
        print(f"\n📊 Step 1/2: Measuring NATIVE code performance...")
        try:
            native_metrics = execute_and_measure(request.code, {})
            print(f"✅ Native metrics captured")
        except Exception as e:
            print(f"⚠️  Native execution failed: {e}")
            # Return dummy metrics if native code fails
            native_metrics = {
                "tokens_per_second": 0.0,
                "validation_accuracy": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "run_variance": 999.0,
                "throughput_per_gb": 0.0
            }

        # Step 2: Run OPTIMIZATION
        print(f"\n🔥 Step 2/2: Running optimization (may take 30-90 seconds)...")
        optimization_result = optimize_pytorch_code(
            code=request.code,
            objective=request.objective,
            max_iterations=request.max_iterations
        )

        if not optimization_result or not optimization_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail="Optimization failed - check server logs"
            )

        optimized_metrics = optimization_result["final_metrics"]
        optimized_code = optimization_result["best_code"]

        print(f"✅ Optimization complete!")

        # Step 3: Calculate improvements
        improvements = {}
        for key in native_metrics.keys():
            if key in optimized_metrics and native_metrics[key] != 0:
                # For metrics where LOWER is better (memory, variance)
                if "memory" in key or "variance" in key:
                    improvements[key] = ((native_metrics[key] - optimized_metrics[key]) / native_metrics[key]) * 100
                # For metrics where HIGHER is better (speed, accuracy, throughput)
                else:
                    improvements[key] = ((optimized_metrics[key] - native_metrics[key]) / native_metrics[key]) * 100

        # Step 4: Format response for frontend
        response = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "native": {
                "tokensPerSecond": native_metrics.get("tokens_per_second", 0.0),
                "validationAccuracy": native_metrics.get("validation_accuracy", 0.0),
                "peakGpuMemoryMb": native_metrics.get("peak_gpu_memory_mb", 0.0),
                "runToRunVariance": native_metrics.get("run_variance", 0.0),
                "throughputPerDollar": native_metrics.get("throughput_per_gb", 0.0),
            },
            "optimized": {
                "tokensPerSecond": optimized_metrics.get("tokens_per_second", 0.0),
                "validationAccuracy": optimized_metrics.get("validation_accuracy", 0.0),
                "peakGpuMemoryMb": optimized_metrics.get("peak_gpu_memory_mb", 0.0),
                "runToRunVariance": optimized_metrics.get("run_variance", 0.0),
                "throughputPerDollar": optimized_metrics.get("throughput_per_gb", 0.0),
            },
            "improvement_percent": improvements,
            "optimized_code": optimized_code
        }

        print(f"\n🎉 SUCCESS - Returning metrics to frontend")
        print(f"   Speed improvement: {improvements.get('tokens_per_second', 0):+.1f}%")
        print(f"   Memory improvement: {improvements.get('peak_gpu_memory_mb', 0):+.1f}%")

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Optimization error: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR:\n{error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run on all interfaces (0.0.0.0) so AWS instance can receive external requests
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )

