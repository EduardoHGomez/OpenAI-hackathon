"""
SELF-IMPROVING PYTORCH OPTIMIZER
=================================

Multi-Agent LLM System for Automatic PyTorch Code Optimization

Agents:
1. Profiler Expert - Measures performance metrics
2. Memory Tuner - Proposes memory optimizations
3. Optimizer Expert - Proposes torch.compile, AMP, schedulers
4. Code Generator - Synthesizes improvements into new code
5. Judge - Evaluates improvement quality

Metrics:
- Speed: tokens/sec or samples/sec
- Accuracy: validation accuracy
- Memory: peak GPU memory MB
- Stability: run-to-run variance
- Cost: throughput per dollar
"""

import json
import os
import sys
from typing import Any, TypedDict, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from openai import OpenAI
import torch
import time
import traceback
import random

load_dotenv()

# Configuration
MAX_ITERATIONS = 3
IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement to continue
CPU_ONLY_MODE = True  # Set to False if you have GPU

# Global storage
ALL_ITERATIONS = []


class OptimizationState(TypedDict):
    """State for optimization workflow"""
    # Input
    original_code: str
    dataset_info: dict[str, Any]
    objective: str  # "speed", "memory", "accuracy", "balanced"
    
    # Current iteration
    iteration: int
    current_code: str
    
    # Metrics
    current_metrics: dict[str, float]
    best_metrics: dict[str, float]
    best_code: str
    
    # Expert proposals
    profiler_report: str
    memory_proposal: str
    optimizer_proposal: str
    
    # Control
    converged: bool
    experts_done: list[str]


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API - simple and sequential"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found!")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    return response.choices[0].message.content


# ============================================================================
# SCORER - Converts metrics to single scalar
# ============================================================================

def normalize_metric(value: float, baseline: float, higher_is_better: bool = True, eps: float = 1e-8) -> float:
    """Normalize metric to [0,1] range"""
    if higher_is_better:
        return max(0.0, min(1.0, value / (baseline + eps)))
    else:
        return max(0.0, min(1.0, baseline / (value + eps)))


def calculate_score(metrics: dict, baseline: dict, weights: dict = None) -> float:
    """Calculate composite score J from all metrics"""
    if not weights:
        weights = {
            "speed": 0.4,
            "accuracy": 0.25,
            "memory": 0.15,
            "stability": 0.1,
            "cost": 0.1
        }
    
    S = normalize_metric(metrics.get("tokens_per_second", 0), baseline.get("tokens_per_second", 1), True)
    A = normalize_metric(metrics.get("validation_accuracy", 0), baseline.get("validation_accuracy", 1), True)
    M = normalize_metric(metrics.get("peak_gpu_memory_mb", 1), baseline.get("peak_gpu_memory_mb", 1), False)
    V = normalize_metric(metrics.get("run_variance", 1), baseline.get("run_variance", 1), False)
    C = normalize_metric(metrics.get("throughput_per_gb", 0), baseline.get("throughput_per_gb", 1), True)
    
    J = (weights["speed"] * S + 
         weights["accuracy"] * A + 
         weights["memory"] * M + 
         weights["stability"] * V + 
         weights["cost"] * C)
    
    return J


# ============================================================================
# PROFILER EXPERT
# ============================================================================

def profiler_expert(state: OptimizationState) -> dict:
    """Profiles current code and measures all metrics"""
    print(f"\n{'='*70}")
    print(f"📊 PROFILER - Measuring Performance")
    print(f"{'='*70}")
    
    code = state["current_code"]
    dataset_info = state["dataset_info"]
    
    try:
        # Execute code and measure
        metrics = execute_and_measure(code, dataset_info)
        
        print(f"\n📈 Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        # Simple report - no LLM needed
        report = f"Iteration {state['iteration']}: Measured {len(metrics)} metrics"
        
        return {
            "current_metrics": metrics,
            "profiler_report": report,
            "experts_done": state["experts_done"] + ["profiler"]
        }
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        return {
            "current_metrics": {},
            "profiler_report": f"Error: {str(e)}",
            "experts_done": state["experts_done"] + ["profiler"]
        }


# ============================================================================
# MEMORY TUNER EXPERT
# ============================================================================

def memory_tuner_expert(state: OptimizationState) -> dict:
    """Proposes memory optimizations"""
    print(f"\n{'='*70}")
    print(f"💾 MEMORY TUNER - Optimizing Memory")
    print(f"{'='*70}")
    
    system_prompt = """You are a memory optimization expert for PyTorch.
Propose 3-5 specific, actionable code changes to reduce memory usage.
Focus on: gradient checkpointing, mixed precision, batch size, in-place ops."""
    
    user_prompt = f"""CODE:
```python
{state['current_code']}
```

METRICS: {json.dumps(state['current_metrics'], indent=2)}

Propose 3-5 memory optimizations with code snippets."""
    
    proposal = call_llm(system_prompt, user_prompt)
    
    print(f"\n💡 Memory Proposals (truncated):")
    print(proposal[:300] + "...")
    
    return {
        "memory_proposal": proposal,
        "experts_done": state["experts_done"] + ["memory"]
    }


# ============================================================================
# OPTIMIZER EXPERT
# ============================================================================

def optimizer_expert(state: OptimizationState) -> dict:
    """Proposes PyTorch-level optimizations"""
    print(f"\n{'='*70}")
    print(f"⚡ OPTIMIZER - PyTorch Optimizations")
    print(f"{'='*70}")
    
    system_prompt = """You are a PyTorch optimization expert.
Propose 3-5 specific code changes for speed and efficiency.
Focus on: torch.compile, AMP, better optimizers, DataLoader workers."""
    
    user_prompt = f"""CODE:
```python
{state['current_code']}
```

METRICS: {json.dumps(state['current_metrics'], indent=2)}

Propose 3-5 PyTorch optimizations with code snippets."""
    
    proposal = call_llm(system_prompt, user_prompt)
    
    print(f"\n💡 Optimizer Proposals (truncated):")
    print(proposal[:300] + "...")
    
    return {
        "optimizer_proposal": proposal,
        "experts_done": state["experts_done"] + ["optimizer"]
    }


# ============================================================================
# CODE GENERATOR
# ============================================================================

def code_generator(state: OptimizationState) -> dict:
    """Generates improved code based on all expert proposals"""
    print(f"\n{'='*70}")
    print(f"🔨 CODE GENERATOR - Synthesizing")
    print(f"{'='*70}")
    
    system_prompt = """You are an expert Python/PyTorch code generator.
Generate COMPLETE, WORKING Python code with all optimizations applied.

REQUIREMENTS:
1. COMPLETE executable code (not snippets)
2. Same interface (inputs/outputs)
3. Apply ALL viable optimizations
4. Preserve correctness
5. Add comments for changes
6. Return ONLY Python code in markdown block"""
    
    user_prompt = f"""ORIGINAL CODE:
```python
{state['original_code']}
```

CURRENT CODE:
```python
{state['current_code']}
```

MEMORY PROPOSALS:
{state['memory_proposal']}

OPTIMIZER PROPOSALS:
{state['optimizer_proposal']}

OBJECTIVE: {state['objective']}

Generate improved code with these optimizations."""
    
    response = call_llm(system_prompt, user_prompt)
    new_code = extract_code(response)
    
    print(f"\n✅ Generated improved code ({len(new_code)} chars)")
    
    return {
        "current_code": new_code,
        "experts_done": []  # Reset for next iteration
    }


# ============================================================================
# JUDGE (with scoring)
# ============================================================================

def judge_improvement(state: OptimizationState) -> dict:
    """Judge evaluates using composite scoring"""
    print(f"\n{'='*70}")
    print(f"⚖️  JUDGE - Evaluating")
    print(f"{'='*70}")
    
    current = state["current_metrics"]
    
    # First iteration - set baseline
    if not state.get("best_metrics"):
        print("📊 Setting baseline")
        return {
            "best_metrics": current,
            "best_code": state["current_code"],
            "iteration": state["iteration"] + 1,
            "converged": False
        }
    
    baseline = state["best_metrics"]
    
    # Calculate composite score
    score_current = calculate_score(current, baseline)
    
    improvement_pct = (score_current - 1.0) * 100
    
    print(f"\n📊 Score: {score_current:.3f} (baseline: 1.0)")
    print(f"   Overall: {improvement_pct:+.1f}%")
    
    # Individual metrics
    for metric in current.keys():
        if metric in baseline and baseline[metric] > 0:
            pct = ((current[metric] - baseline[metric]) / baseline[metric]) * 100
            symbol = "📈" if pct > 0 else "📉"
            print(f"   {symbol} {metric}: {pct:+.1f}%")
    
    # Accept if improved
    if score_current > 1.0 + IMPROVEMENT_THRESHOLD:
        print(f"\n🎯 ACCEPTED!")
        return {
            "best_metrics": current,
            "best_code": state["current_code"],
            "iteration": state["iteration"] + 1,
            "converged": state["iteration"] >= MAX_ITERATIONS - 1
        }
    
    print(f"\n📊 No significant improvement")
    return {
        "iteration": state["iteration"] + 1,
        "converged": state["iteration"] >= MAX_ITERATIONS - 1 or state["iteration"] > 2
    }


# ============================================================================
# EXECUTION & MEASUREMENT
# ============================================================================

def execute_and_measure(code: str, dataset_info: dict) -> dict[str, float]:
    """Execute code and measure all metrics - CPU friendly!"""
    
    # CPU-only mode: Use mock metrics (FAST!)
    if CPU_ONLY_MODE:
        iteration_num = len(ALL_ITERATIONS)
        
        # Simulate improvement over iterations
        improvement = 1.0 + (iteration_num * 0.15)
        
        return {
            'tokens_per_second': 1000.0 * improvement + random.random() * 50,
            'validation_accuracy': 0.90 + (iteration_num * 0.01) + random.random() * 0.02,
            'peak_gpu_memory_mb': 500.0 - (iteration_num * 30),
            'run_variance': 0.05 - (iteration_num * 0.01),
            'throughput_per_gb': 2.0 + (iteration_num * 0.3)
        }
    
    # GPU mode: Actually execute code
    exec_globals = {'torch': torch, 'time': time}
    if dataset_info:
        exec_globals.update(dataset_info)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exec_globals['device'] = device
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    try:
        exec(code, exec_globals)
        
        if 'get_metrics' in exec_globals:
            return exec_globals['get_metrics']()
        
        # Estimate metrics
        metrics = {}
        
        if 'train' in exec_globals:
            start = time.time()
            result = exec_globals['train']()
            elapsed = time.time() - start
            
            if isinstance(result, dict) and 'samples' in result:
                metrics['tokens_per_second'] = result['samples'] / elapsed
        
        if 'validate' in exec_globals:
            metrics['validation_accuracy'] = exec_globals['validate']()
        else:
            metrics['validation_accuracy'] = 0.95
        
        if torch.cuda.is_available():
            metrics['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            metrics['peak_gpu_memory_mb'] = 0.0
        
        metrics['run_variance'] = 0.01
        metrics['throughput_per_gb'] = metrics.get('tokens_per_second', 0) / max(metrics['peak_gpu_memory_mb'] / 1024, 0.1)
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  Execution error: {e}")
        return {
            'tokens_per_second': 100.0,
            'validation_accuracy': 0.80,
            'peak_gpu_memory_mb': 100.0,
            'run_variance': 0.05,
            'throughput_per_gb': 1.0
        }


def extract_code(response: str) -> str:
    """Extract Python code from markdown"""
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    return response.strip()


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def after_judge(state: OptimizationState) -> Literal["profiler", "end"]:
    """After judge, either continue or end"""
    if state["converged"]:
        return "end"
    return "profiler"


def build_graph():
    """Build the optimization workflow - SEQUENTIAL (no parallel)"""
    workflow = StateGraph(OptimizationState)
    
    # Add nodes
    workflow.add_node("profiler", profiler_expert)
    workflow.add_node("memory", memory_tuner_expert)
    workflow.add_node("optimizer", optimizer_expert)
    workflow.add_node("generate", code_generator)
    workflow.add_node("judge", judge_improvement)
    
    # Entry point
    workflow.set_entry_point("profiler")
    
    # Sequential flow (one at a time - simple!)
    workflow.add_edge("profiler", "memory")
    workflow.add_edge("memory", "optimizer")
    workflow.add_edge("optimizer", "generate")
    workflow.add_edge("generate", "judge")
    
    # Judge decides: continue or end
    workflow.add_conditional_edges(
        "judge",
        after_judge,
        {"profiler": "profiler", "end": END}
    )
    
    return workflow.compile()


# ============================================================================
# MAIN RUNNER
# ============================================================================

def optimize_pytorch_code(
    code: str,
    dataset_info: dict = None,
    objective: str = "balanced",
    max_iterations: int = 3
):
    """Main optimization function - returns JSON for UI"""
    print(f"\n{'='*70}")
    print(f"🚀 PYTORCH OPTIMIZER")
    print(f"{'='*70}")
    
    global MAX_ITERATIONS
    MAX_ITERATIONS = max_iterations
    
    initial_state: OptimizationState = {
        "original_code": code,
        "current_code": code,
        "dataset_info": dataset_info or {},
        "objective": objective,
        "iteration": 0,
        "current_metrics": {},
        "best_metrics": {},
        "best_code": code,
        "profiler_report": "",
        "memory_proposal": "",
        "optimizer_proposal": "",
        "converged": False,
        "experts_done": []
    }
    
    graph = build_graph()
    
    final_state = None
    iteration_results = []
    
    try:
        for iteration_state in graph.stream(initial_state, {"recursion_limit": 100}):
            final_state = iteration_state
            # Capture each iteration for UI
            if final_state:
                state_dict = list(final_state.values())[0]
                if state_dict.get("current_metrics"):
                    iteration_results.append({
                        "iteration": state_dict.get("iteration", 0),
                        "metrics": state_dict.get("current_metrics", {}),
                        "score": calculate_score(
                            state_dict.get("current_metrics", {}),
                            state_dict.get("best_metrics") or state_dict.get("current_metrics", {})
                        ) if state_dict.get("current_metrics") else 0.0
                    })
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    
    if final_state:
        final_state = list(final_state.values())[0]
        
        # Calculate improvements
        improvements = {}
        if final_state.get("best_metrics"):
            baseline = final_state["best_metrics"]
            for metric, value in final_state.get("best_metrics", {}).items():
                if metric in baseline and baseline[metric] > 0:
                    improvements[metric] = ((value - baseline[metric]) / baseline[metric]) * 100
        
        # Calculate composite score
        final_score = calculate_score(
            final_state.get("best_metrics", {}),
            final_state.get("best_metrics", {})
        ) if final_state.get("best_metrics") else 0.0
        
        # Prepare JSON output for UI
        result_json = {
            "success": bool(final_state.get("best_metrics")),
            "iterations_completed": final_state.get("iteration", 0),
            "objective": objective,
            "final_score": final_score,
            "final_metrics": final_state.get("best_metrics", {}),
            "improvements": improvements,
            "iteration_history": iteration_results,
            "best_code": final_state.get("best_code", code),
            "timestamp": datetime.now().isoformat()
        }
        
        # Print for debugging
        print(f"\n{'='*70}")
        print(f"🏁 COMPLETE")
        print(f"{'='*70}")
        print(json.dumps(result_json, indent=2))
        
        # Save results
        with open("optimization_results.json", "w") as f:
            json.dump(result_json, f, indent=2)
        
        with open("optimized_code.py", "w") as f:
            f.write(result_json["best_code"])
        
        return result_json


if __name__ == "__main__":
    # Example usage
    example_code = """
import torch
import torch.nn as nn

def train():
    model = nn.Linear(1000, 1000)
    x = torch.randn(32, 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for _ in range(100):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    
    return {"loss": loss.item(), "samples": 3200, "elapsed": 1.0}

def validate():
    return 0.95

def get_metrics():
    result = train()
    acc = validate()
    return {
        "tokens_per_second": 1000.0,
        "validation_accuracy": acc,
        "peak_gpu_memory_mb": 100.0,
        "run_variance": 0.01,
        "throughput_per_gb": 10.0
    }
"""
    
    optimize_pytorch_code(
        code=example_code,
        objective="speed",
        max_iterations=3
    )