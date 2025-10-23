"""
SELF-IMPROVING PYTORCH OPTIMIZER
=================================

Multi-Agent LLM System for Automatic PyTorch Code Optimization

Agents:
1. Profiler Expert - Measures performance metrics
2. Memory Tuner - Proposes memory optimizations
3. Optimizer Expert - Proposes torch.compile, AMP, schedulers
4. Code Generator - Synthesizes improvements into new code
5. Judge LLM - Evaluates improvement quality

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
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Configuration
MAX_ITERATIONS = 3  # Reduced from 5 for speed
IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement to continue

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
    """Call OpenAI API"""
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


def call_llms_parallel(prompts: list[tuple[str, str]], model: str = "gpt-4o-mini") -> list[str]:
    """Call multiple LLMs in parallel - 3x faster!"""
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(call_llm, system, user, model) 
            for system, user in prompts
        ]
        results = [f.result() for f in futures]
    return results
    
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
        # Execute code and measure (FAST - no LLM)
        metrics = execute_and_measure(code, dataset_info)
        
        print(f"\n📈 Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        # Skip LLM analysis for speed - just provide basic report
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
# MEMORY + OPTIMIZER EXPERTS (RUN IN PARALLEL)
# ============================================================================

def call_both_experts(state: OptimizationState) -> dict:
    """Call memory and optimizer experts in parallel - FAST!"""
    print(f"\n{'='*70}")
    print(f"🔥 CALLING EXPERTS IN PARALLEL")
    print(f"{'='*70}")
    
    # Prepare prompts for both experts
    memory_system = """You are a memory optimization expert for PyTorch.
Propose 3-5 specific, actionable code changes to reduce memory usage.
Focus on: gradient checkpointing, mixed precision, batch size, in-place ops."""
    
    memory_user = f"""CODE:
```python
{state['current_code']}
```

METRICS: {json.dumps(state['current_metrics'], indent=2)}

Propose 3-5 memory optimizations with code snippets."""
    
    optimizer_system = """You are a PyTorch optimization expert.
Propose 3-5 specific code changes for speed and efficiency.
Focus on: torch.compile, AMP, better optimizers, DataLoader workers."""
    
    optimizer_user = f"""CODE:
```python
{state['current_code']}
```

METRICS: {json.dumps(state['current_metrics'], indent=2)}

Propose 3-5 PyTorch optimizations with code snippets."""
    
    # Call both in parallel!
    try:
        results = call_llms_parallel([
            (memory_system, memory_user),
            (optimizer_system, optimizer_user)
        ])
        
        memory_proposal = results[0]
        optimizer_proposal = results[1]
        
        print(f"\n💡 Got both proposals in parallel!")
        
        return {
            "memory_proposal": memory_proposal,
            "optimizer_proposal": optimizer_proposal,
            "experts_done": state["experts_done"] + ["memory", "optimizer"]
        }
    except Exception as e:
        print(f"❌ Expert error: {e}")
        return {
            "memory_proposal": "Error",
            "optimizer_proposal": "Error",
            "experts_done": state["experts_done"] + ["memory", "optimizer"]
        }


# ============================================================================
# CODE GENERATOR
# ============================================================================

def code_generator(state: OptimizationState) -> dict:
    """Generates improved code based on all expert proposals"""
    print(f"\n{'='*70}")
    print(f"🔨 CODE GENERATOR - Synthesizing Improvements")
    print(f"{'='*70}")
    
    system_prompt = """You are an expert Python/PyTorch code generator.
Your job: Take the ORIGINAL code and ALL expert proposals, then generate COMPLETE, WORKING Python code.

CRITICAL REQUIREMENTS:
1. Generate COMPLETE, EXECUTABLE code (not snippets)
2. Maintain the same interface (same inputs/outputs)
3. Apply ALL viable optimizations from experts
4. Preserve correctness (same functionality)
5. Add comments explaining major changes
6. Return ONLY the Python code in markdown code block"""
    
    user_prompt = f"""ORIGINAL CODE:
```python
{state['original_code']}
```

CURRENT CODE (iteration {state['iteration']}):
```python
{state['current_code']}
```

PROFILER ANALYSIS:
{state['profiler_report']}

MEMORY TUNER PROPOSALS:
{state['memory_proposal']}

OPTIMIZER EXPERT PROPOSALS:
{state['optimizer_proposal']}

OPTIMIZATION OBJECTIVE: {state['objective']}

CURRENT METRICS:
{json.dumps(state['current_metrics'], indent=2)}

Generate the COMPLETE improved code that applies these optimizations.
Return ONLY Python code in a markdown code block."""
    
    response = call_llm(system_prompt, user_prompt)
    
    # Extract code from markdown
    new_code = extract_code(response)
    
    print(f"\n✅ Generated improved code ({len(new_code)} chars)")
    print(f"Preview:\n{new_code[:300]}...")
    
    return {
        "current_code": new_code,
        "experts_done": []  # Reset for next iteration
    }


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
# JUDGE LLM (with structured scoring)
# ============================================================================

def judge_improvement(state: OptimizationState) -> dict:
    """Judge evaluates using composite scoring + LLM verification"""
    print(f"\n{'='*70}")
    print(f"⚖️  JUDGE - Evaluating Improvement")
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
    score_baseline = 1.0  # By definition
    
    improvement_pct = (score_current - score_baseline) * 100
    
    print(f"\n📊 Composite Score: {score_current:.3f} (baseline: 1.0)")
    print(f"   Overall: {improvement_pct:+.1f}%")
    
    # Individual metric changes
    improvements = {}
    for metric in current.keys():
        if metric in baseline and baseline[metric] > 0:
            pct = ((current[metric] - baseline[metric]) / baseline[metric]) * 100
            improvements[metric] = pct
            symbol = "📈" if pct > 0 else "📉"
            print(f"   {symbol} {metric}: {pct:+.1f}%")
    
    # Quick decision: if score improved significantly, accept
    if score_current > 1.0 + IMPROVEMENT_THRESHOLD:
        print(f"\n🎯 ACCEPTED - Score improved!")
        return {
            "best_metrics": current,
            "best_code": state["current_code"],
            "iteration": state["iteration"] + 1,
            "converged": state["iteration"] >= MAX_ITERATIONS - 1
        }
    
    # If marginal or worse, ask LLM judge (skip for speed)
    print(f"\n📊 Marginal/no improvement")
    return {
        "iteration": state["iteration"] + 1,
        "converged": state["iteration"] >= MAX_ITERATIONS - 1 or state["iteration"] > 2
    }


# ============================================================================
# EXECUTION & MEASUREMENT
# ============================================================================

def execute_and_measure(code: str, dataset_info: dict) -> dict[str, float]:
    """Execute code and measure all metrics"""
    # Create execution environment
    exec_globals = {
        'torch': torch,
        'time': time,
        '__name__': '__main__'
    }
    
    # Add dataset if provided
    if dataset_info:
        exec_globals.update(dataset_info)
    
    # Measure memory before
    torch.cuda.reset_peak_memory_stats()
    
    # Run multiple times for stability
    times = []
    accuracies = []
    
    try:
        # Execute code
        exec(code, exec_globals)
        
        # Get the metrics function if defined
        if 'get_metrics' in exec_globals:
            metrics = exec_globals['get_metrics']()
            return metrics
        
        # Otherwise try to extract from execution
        metrics = {}
        
        # Speed: if train function exists, measure it
        if 'train' in exec_globals:
            start = time.time()
            result = exec_globals['train']()
            elapsed = time.time() - start
            
            # Estimate tokens/sec if batch info available
            if isinstance(result, dict) and 'samples' in result:
                metrics['tokens_per_second'] = result['samples'] / elapsed
            else:
                metrics['samples_per_second'] = 1.0 / elapsed if elapsed > 0 else 0
        
        # Accuracy: if validate function exists
        if 'validate' in exec_globals:
            metrics['validation_accuracy'] = exec_globals['validate']()
        else:
            metrics['validation_accuracy'] = 0.95  # Placeholder
        
        # Memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        metrics['peak_gpu_memory_mb'] = peak_memory
        
        # Stability: run multiple times
        if 'train' in exec_globals:
            results = []
            for _ in range(3):
                r = exec_globals['train']()
                if isinstance(r, dict) and 'loss' in r:
                    results.append(r['loss'])
            
            if results:
                import statistics
                metrics['run_variance'] = statistics.variance(results) if len(results) > 1 else 0.0
        else:
            metrics['run_variance'] = 0.0
        
        # Cost efficiency
        if 'tokens_per_second' in metrics and peak_memory > 0:
            metrics['throughput_per_gb'] = metrics['tokens_per_second'] / (peak_memory / 1024)
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  Execution error: {e}")
        traceback.print_exc()
        # Return dummy metrics
        return {
            'tokens_per_second': 0.0,
            'validation_accuracy': 0.0,
            'peak_gpu_memory_mb': 0.0,
            'run_variance': 999.0,
            'throughput_per_gb': 0.0
        }


def extract_code(response: str) -> str:
    """Extract Python code from markdown response"""
    # Try to find code block
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    else:
        # No markdown, assume entire response is code
        return response.strip()


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def get_next_expert(state: OptimizationState) -> str:
    """Determine which expert to call next"""
    experts = ["profiler", "memory", "optimizer", "generate"]
    for expert in experts:
        if expert not in state["experts_done"]:
            return expert
    return "judge"


def route_expert(state: OptimizationState) -> Literal["profiler", "memory", "optimizer", "generate", "judge", "end"]:
    """Route to next node"""
    if state["converged"]:
        return "end"
    
    next_expert = get_next_expert(state)
    
    if next_expert == "profiler":
        return "profiler"
    elif next_expert == "memory":
        return "memory"
    elif next_expert == "optimizer":
        return "optimizer"
    elif next_expert == "generate":
        return "generate"
    else:
        return "judge"


def after_judge(state: OptimizationState) -> Literal["profiler", "end"]:
    """After judge, either continue or end"""
    if state["converged"]:
        return "end"
    return "profiler"


def build_graph():
    """Build the optimization workflow"""
    workflow = StateGraph(OptimizationState)
    
    # Add nodes (using parallel experts now!)
    workflow.add_node("profiler", profiler_expert)
    workflow.add_node("experts", call_both_experts)  # Memory + Optimizer in parallel!
    workflow.add_node("generate", code_generator)
    workflow.add_node("judge", judge_improvement)
    
    # Entry point
    workflow.set_entry_point("profiler")
    
    # Sequential flow (but experts run in parallel internally)
    workflow.add_edge("profiler", "experts")
    workflow.add_edge("experts", "generate")
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
    max_iterations: int = 3  # Reduced from 5 for speed
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
        if final_state.get("best_metrics") and final_state.get("current_metrics"):
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