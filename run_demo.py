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

load_dotenv()

# Configuration
MAX_ITERATIONS = 5
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


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4") -> str:
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


# ============================================================================
# PROFILER EXPERT
# ============================================================================

def profiler_expert(state: OptimizationState) -> dict:
    """Profiles current code and measures all metrics"""
    print(f"\n{'='*70}")
    print(f"📊 PROFILER EXPERT - Measuring Performance")
    print(f"{'='*70}")
    
    code = state["current_code"]
    dataset_info = state["dataset_info"]
    
    try:
        # Execute code and measure
        metrics = execute_and_measure(code, dataset_info)
        
        print(f"\n📈 Measured Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        # Generate expert analysis
        system_prompt = """You are a performance profiling expert for PyTorch models.
Analyze the code and metrics, then provide detailed insights on performance bottlenecks."""
        
        user_prompt = f"""CODE:
```python
{code}
```

METRICS:
{json.dumps(metrics, indent=2)}

Analyze the performance and identify:
1. Speed bottlenecks (data loading, forward pass, backward pass)
2. Memory bottlenecks (large tensors, unnecessary copies)
3. Accuracy issues (loss not decreasing, overfitting)
4. Stability issues (NaNs, exploding gradients)

Provide specific observations about THIS code."""
        
        report = call_llm(system_prompt, user_prompt)
        
        print(f"\n🔍 Profiler Analysis:")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        return {
            "current_metrics": metrics,
            "profiler_report": report,
            "experts_done": state["experts_done"] + ["profiler"]
        }
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        traceback.print_exc()
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
    print(f"💾 MEMORY TUNER EXPERT - Optimizing Memory Usage")
    print(f"{'='*70}")
    
    system_prompt = """You are a memory optimization expert for PyTorch.
Propose specific, actionable code changes to reduce memory usage while maintaining or improving performance.

Focus on:
- Gradient checkpointing
- Mixed precision (FP16/BF16)
- Batch size optimization
- In-place operations
- Memory-efficient attention
- Clearing cache strategically"""
    
    user_prompt = f"""CURRENT CODE:
```python
{state['current_code']}
```

PROFILER ANALYSIS:
{state['profiler_report']}

CURRENT METRICS:
{json.dumps(state['current_metrics'], indent=2)}

Propose 3-5 SPECIFIC memory optimizations with code snippets.
Format as a list of changes."""
    
    proposal = call_llm(system_prompt, user_prompt)
    
    print(f"\n💡 Memory Optimization Proposals:")
    print(proposal[:500] + "..." if len(proposal) > 500 else proposal)
    
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
    print(f"⚡ OPTIMIZER EXPERT - PyTorch Optimizations")
    print(f"{'='*70}")
    
    system_prompt = """You are a PyTorch optimization expert.
Propose specific code changes for speed and efficiency.

Focus on:
- torch.compile() with appropriate backend
- Automatic Mixed Precision (AMP)
- Learning rate schedulers
- Optimizer selection (AdamW, Lion, etc.)
- DataLoader workers and pin_memory
- Gradient accumulation
- Better forward pass structure"""
    
    user_prompt = f"""CURRENT CODE:
```python
{state['current_code']}
```

PROFILER ANALYSIS:
{state['profiler_report']}

MEMORY PROPOSALS:
{state['memory_proposal']}

CURRENT METRICS:
{json.dumps(state['current_metrics'], indent=2)}

Propose 3-5 SPECIFIC PyTorch optimizations with code snippets.
Consider what was already proposed by memory tuner."""
    
    proposal = call_llm(system_prompt, user_prompt)
    
    print(f"\n💡 Optimizer Proposals:")
    print(proposal[:500] + "..." if len(proposal) > 500 else proposal)
    
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
# JUDGE LLM
# ============================================================================

def judge_improvement(state: OptimizationState) -> dict:
    """Judge evaluates if improvement was made"""
    print(f"\n{'='*70}")
    print(f"⚖️  JUDGE LLM - Evaluating Improvement")
    print(f"{'='*70}")
    
    # Check if we have previous metrics
    if not state.get("best_metrics"):
        # First iteration - set as baseline
        print("📊 Setting baseline metrics")
        return {
            "best_metrics": state["current_metrics"],
            "best_code": state["current_code"],
            "iteration": state["iteration"] + 1,
            "converged": False
        }
    
    # Calculate improvement
    improvements = {}
    objective = state["objective"]
    
    current = state["current_metrics"]
    best = state["best_metrics"]
    
    # Calculate improvements for each metric
    for metric in current.keys():
        if metric in best and best[metric] > 0:
            improvement = ((current[metric] - best[metric]) / best[metric]) * 100
            improvements[metric] = improvement
    
    print(f"\n📊 Improvement Analysis:")
    for metric, improvement in improvements.items():
        symbol = "📈" if improvement > 0 else "📉"
        print(f"   {symbol} {metric}: {improvement:+.2f}%")
    
    # Determine if this is better based on objective
    is_better = False
    if objective == "speed":
        is_better = improvements.get("tokens_per_second", -999) > IMPROVEMENT_THRESHOLD * 100
    elif objective == "memory":
        is_better = improvements.get("peak_gpu_memory_mb", 999) < -IMPROVEMENT_THRESHOLD * 100
    elif objective == "accuracy":
        is_better = improvements.get("validation_accuracy", -999) > IMPROVEMENT_THRESHOLD * 100
    else:  # balanced
        # Weighted score
        score = (
            improvements.get("tokens_per_second", 0) * 0.3 +
            improvements.get("validation_accuracy", 0) * 0.3 +
            -improvements.get("peak_gpu_memory_mb", 0) * 0.2 +
            -improvements.get("run_variance", 0) * 0.2
        )
        is_better = score > IMPROVEMENT_THRESHOLD * 100
    
    # Call LLM Judge for qualitative assessment
    system_prompt = """You are an expert judge evaluating PyTorch optimization improvements.
Provide a brief assessment of whether the changes represent genuine improvement."""
    
    user_prompt = f"""OBJECTIVE: {objective}

PREVIOUS METRICS:
{json.dumps(best, indent=2)}

CURRENT METRICS:
{json.dumps(current, indent=2)}

IMPROVEMENTS:
{json.dumps(improvements, indent=2)}

Is this a meaningful improvement? Consider:
1. Metrics aligned with objective
2. No significant regression in other metrics
3. Trade-offs are reasonable

Respond with: ACCEPT or REJECT and brief reason."""
    
    judge_verdict = call_llm(system_prompt, user_prompt)
    
    print(f"\n⚖️  Judge Verdict:")
    print(judge_verdict)
    
    # Update if better
    if is_better or "ACCEPT" in judge_verdict.upper():
        print(f"\n🎯 NEW BEST! Updating best metrics")
        return {
            "best_metrics": current,
            "best_code": state["current_code"],
            "iteration": state["iteration"] + 1,
            "converged": state["iteration"] >= MAX_ITERATIONS - 1
        }
    else:
        print(f"\n📊 No significant improvement")
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
    
    # Add nodes
    workflow.add_node("profiler", profiler_expert)
    workflow.add_node("memory", memory_tuner_expert)
    workflow.add_node("optimizer", optimizer_expert)
    workflow.add_node("generate", code_generator)
    workflow.add_node("judge", judge_improvement)
    
    # Entry point
    workflow.set_entry_point("profiler")
    
    # Sequential flow
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
    max_iterations: int = 5
):
    """Main optimization function"""
    print(f"\n{'='*70}")
    print(f"🚀 SELF-IMPROVING PYTORCH OPTIMIZER")
    print(f"{'='*70}")
    print(f"Objective: {objective}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'='*70}\n")
    
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
    try:
        for iteration_state in graph.stream(initial_state, {"recursion_limit": 100}):
            final_state = iteration_state
    except Exception as e:
        print(f"\n❌ Optimization error: {e}")
        traceback.print_exc()
        if final_state:
            final_state = list(final_state.values())[0]
    
    if final_state:
        final_state = list(final_state.values())[0]
        
        print(f"\n{'='*70}")
        print(f"🏁 OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        
        # Generate final report
        report = {
            "success": bool(final_state.get("best_metrics")),
            "iterations_completed": final_state.get("iteration", 0),
            "objective": objective,
            "initial_metrics": {},  # First iteration metrics
            "final_metrics": final_state.get("best_metrics", {}),
            "improvements": {},
            "best_code": final_state.get("best_code", code)
        }
        
        # Calculate improvements
        if report["final_metrics"]:
            # Find initial metrics from first iteration
            # (In real implementation, store this)
            for metric, final_val in report["final_metrics"].items():
                # Placeholder - would need to store initial
                report["improvements"][metric] = "N/A (need initial)"
        
        print(f"\n📊 Final Metrics:")
        for metric, value in report["final_metrics"].items():
            print(f"   {metric}: {value:.4f}")
        
        print(f"\n💾 Saving results...")
        with open("optimization_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        with open("optimized_code.py", "w") as f:
            f.write(report["best_code"])
        
        print(f"\n✅ Results saved to:")
        print(f"   - optimization_results.json")
        print(f"   - optimized_code.py")
        
        return report


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