"""
GPU Kernel Autotuner - SIMPLE SEQUENTIAL FLOW
==============================================

FIXED: Sequential execution - one expert at a time
Easier to debug, understand, and works correctly with LangGraph
"""

import json
import os
import sys
from typing import Any, Literal, TypedDict
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'triton-optimizer'))
from triton_matmul import triton_matmul
from metrics_evaluator import MetricsEvaluator
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import torch
import numpy as np


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file!")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content


class TuningState(TypedDict):
    """State - NO operator.add needed for sequential flow"""
    kernel: str
    shape: dict[str, int]
    knobs: list[dict[str, Any]]
    history: list[dict[str, Any]]
    symptoms: list[str]
    
    # Current expert being evaluated
    current_expert: str
    proposal: dict[str, Any] | None
    
    # Results
    results: list[dict[str, Any]]
    best_config: dict[str, Any] | None
    best_latency: float
    
    # Control
    iteration: int
    max_iterations: int
    experts_done: list[str]  # Track which experts ran this iteration
    converged: bool


def validate_config(config: dict[str, Any], knobs: list[dict[str, Any]]) -> tuple[bool, str]:
    knob_map = {k["name"]: k for k in knobs}
    for knob in knobs:
        if knob["name"] not in config:
            return False, f"Missing knob: {knob['name']}"
    for key in config:
        if key not in knob_map:
            return False, f"Unknown knob: {key}"
    for key, value in config.items():
        knob = knob_map[key]
        if knob["type"] == "int" and not isinstance(value, int):
            return False, f"{key} must be int"
        if "allowed" in knob and value not in knob["allowed"]:
            return False, f"{key}={value} not in allowed: {knob['allowed']}"
        if knob.get("multiple_of") and value % knob["multiple_of"] != 0:
            return False, f"{key}={value} must be multiple of {knob['multiple_of']}"
    return True, ""


def estimate_shared_memory(config: dict[str, Any], kernel: str) -> int:
    if kernel == "matmul":
        tile_a = config.get("block_m", 128) * config.get("block_k", 32)
        tile_b = config.get("block_k", 32) * config.get("block_n", 128)
        stages = config.get("num_stages", 2)
        return (tile_a + tile_b) * 2 * stages
    return 0


SYSTEM_PROMPT = """You propose kernel tuning parameters for GPU performance.
Return ONLY valid JSON matching the knob schema.
Never use values outside the allowed set."""


def create_expert_prompt(state: TuningState, bias: str) -> str:
    bias_instructions = {
        "throughput": "Maximize arithmetic intensity. Push block sizes and stages up.",
        "memory": "Minimize register and shared memory pressure. Shrink tiles.",
        "robustness": "Conservative values. Tensor-core multiples."
    }
    
    return f"""KERNEL: {state['kernel']}
SHAPE: {json.dumps(state['shape'])}
KNOB_SCHEMA: {json.dumps(state['knobs'], indent=2)}
HISTORY: {json.dumps(state['history'][-3:], indent=2) if state['history'] else "[]"}

EXPERT: {bias.upper()}
{bias_instructions[bias]}

Return JSON only: {{"block_m": 128, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 2}}"""


def get_next_expert(state: TuningState) -> str:
    """Determine which expert to call next"""
    experts = ["throughput", "memory", "robustness"]
    for expert in experts:
        if expert not in state["experts_done"]:
            return expert
    return "done"


def call_expert(state: TuningState) -> dict:
    """Call the next expert"""
    expert = get_next_expert(state)
    
    if expert == "done":
        return {"current_expert": "done"}
    
    print(f"\n🔍 Calling {expert} expert...")
    prompt = create_expert_prompt(state, expert)
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            config = json.loads(json_str)
            print(f"📊 {expert.capitalize()}: {config}")
            return {
                "current_expert": expert,
                "proposal": config
            }
    except Exception as e:
        print(f"❌ {expert} error: {e}")
        return {
            "current_expert": expert,
            "proposal": None
        }
    
    return {"current_expert": expert, "proposal": None}


def validate_and_execute(state: TuningState) -> dict:
    """Validate proposal and execute if valid"""
    expert = state["current_expert"]
    config = state["proposal"]
    
    if config is None:
        print(f"❌ {expert}: No valid proposal")
        experts_done = state["experts_done"] + [expert]
        return {"experts_done": experts_done}
    
    # Validate
    is_valid, error = validate_config(config, state["knobs"])
    if not is_valid:
        print(f"❌ {expert}: {error}")
        experts_done = state["experts_done"] + [expert]
        return {"experts_done": experts_done}
    
    # SMEM check
    smem = estimate_shared_memory(config, state["kernel"])
    if smem > 96 * 1024:
        print(f"❌ {expert}: SMEM {smem/1024:.1f}KB > 96KB")
        experts_done = state["experts_done"] + [expert]
        return {"experts_done": experts_done}
    
    print(f"✅ {expert}: Valid (SMEM: {smem/1024:.1f}KB)")
    
    # Execute
    print(f"⚡ Benchmarking {expert}...")
    
    M, K, N = state["shape"]["M"], state["shape"]["K"], state["shape"]["N"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    C_ref = (A.float() @ B.float()).half()
    
    try:
        # Warmup
        for _ in range(10):
            _ = triton_matmul(A, B, **config)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            C = triton_matmul(A, B, **config)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        
        median_ms = float(np.median(times))
        p90_ms = float(np.percentile(times, 90))
        max_abs_diff = (C - C_ref).abs().max().item()
        correct = max_abs_diff < 1e-2
        
        result = {
            "expert": expert,
            "config": config,
            "median_ms": median_ms,
            "p90_ms": p90_ms,
            "correct": correct,
            "max_abs_diff": max_abs_diff
        }
        
        status = "✅" if correct else "❌"
        print(f"{status} {expert}: {median_ms:.2f}ms (p90: {p90_ms:.2f}ms, diff: {max_abs_diff:.3e})")
        
        results = state["results"] + [result]
        experts_done = state["experts_done"] + [expert]
        
        return {
            "results": results,
            "experts_done": experts_done
        }
        
    except Exception as e:
        print(f"❌ {expert}: ERROR - {e}")
        experts_done = state["experts_done"] + [expert]
        return {"experts_done": experts_done}


def analyze_and_continue(state: TuningState) -> dict:
    """Analyze results and decide next action"""
    # Check if all experts done this iteration
    if len(state["experts_done"]) < 3:
        return {}  # Continue to next expert
    
    print("\n📊 Iteration complete, analyzing results...")
    
    # Filter valid results
    valid_results = [r for r in state["results"] if r["correct"]]
    
    if not valid_results:
        print("⚠️  No valid results")
        return {
            "results": [],
            "experts_done": [],
            "iteration": state["iteration"] + 1,
            "converged": state["iteration"] + 1 >= state["max_iterations"]
        }
    
    # Find best
    best = min(valid_results, key=lambda r: r["median_ms"])
    
    # Update history
    history = state["history"] + [{
        "cfg": best["config"],
        "median_ms": best["median_ms"],
        "iteration": state["iteration"]
    }]
    
    # Check improvement
    improvement = False
    if state["best_config"] is None:
        improvement = True
        best_config = best["config"]
        best_latency = best["median_ms"]
        print(f"🎯 First best: {best_latency:.2f}ms from {best['expert']}")
    elif best["median_ms"] < state["best_latency"] * 0.98:
        improvement = True
        old = state["best_latency"]
        best_config = best["config"]
        best_latency = best["median_ms"]
        speedup = ((old / best_latency) - 1) * 100
        print(f"🎯 New best: {best_latency:.2f}ms (+{speedup:.1f}% speedup)")
    else:
        best_config = state["best_config"]
        best_latency = state["best_latency"]
        print(f"📊 No improvement (best still {best_latency:.2f}ms)")
    
    # Convergence
    converged = (
        state["iteration"] + 1 >= state["max_iterations"] or
        (not improvement and state["iteration"] > 1)
    )
    
    if converged:
        print("✅ CONVERGED!")
    
    return {
        "history": history,
        "results": [],
        "experts_done": [],
        "best_config": best_config,
        "best_latency": best_latency,
        "iteration": state["iteration"] + 1,
        "converged": converged
    }


def should_continue(state: TuningState) -> Literal["expert", "end"]:
    """Routing: continue or stop"""
    if state["converged"]:
        return "end"
    
    # Check if need to call next expert in current iteration
    if len(state["experts_done"]) < 3:
        return "expert"
    
    # All experts done, analyzed, check if need another iteration
    return "expert" if not state["converged"] else "end"


def build_graph():
    """SIMPLE SEQUENTIAL GRAPH"""
    workflow = StateGraph(TuningState)
    
    # Only 3 nodes: expert → execute → analyze
    workflow.add_node("expert", call_expert)
    workflow.add_node("execute", validate_and_execute)
    workflow.add_node("analyze", analyze_and_continue)
    
    # Sequential flow
    workflow.set_entry_point("expert")
    workflow.add_edge("expert", "execute")
    workflow.add_edge("execute", "analyze")
    
    # Conditional: analyze decides if continue or end
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "expert": "expert",  # Call next expert
            "end": END
        }
    )
    
    return workflow.compile()


MATMUL_SCHEMA = {
    "kernel": "matmul",
    "shape": {"M": 2048, "K": 2048, "N": 2048},
    "knobs": [
        {"name": "block_m", "type": "int", "allowed": [64, 128, 256], "multiple_of": 16},
        {"name": "block_n", "type": "int", "allowed": [64, 128, 256], "multiple_of": 16},
        {"name": "block_k", "type": "int", "allowed": [16, 32, 64]},
        {"name": "num_warps", "type": "int", "allowed": [2, 4, 8]},
        {"name": "num_stages", "type": "int", "allowed": [1, 2, 3, 4]}
    ]
}


def run_tuning(schema: dict, max_iterations: int = 3):
    print(f"\n{'='*70}")
    print(f"🚀 SEQUENTIAL FLOW Autotuner")
    print(f"   Kernel: {schema['kernel'].upper()}")
    print(f"   Shape: {schema['shape']}")
    print(f"{'='*70}")
    
    initial_state: TuningState = {
        "kernel": schema["kernel"],
        "shape": schema["shape"],
        "knobs": schema["knobs"],
        "history": [],
        "symptoms": [],
        "current_expert": "",
        "proposal": None,
        "results": [],
        "best_config": None,
        "best_latency": float('inf'),
        "iteration": 0,
        "max_iterations": max_iterations,
        "experts_done": [],
        "converged": False
    }
    
    graph = build_graph()
    
    final_state = None
    for iteration_state in graph.stream(initial_state):
        final_state = iteration_state
    
    if final_state:
        final_state = list(final_state.values())[0]
        
        print(f"\n{'='*70}")
        print(f"🏁 COMPLETE")
        print(f"{'='*70}")
        print(f"Best Config: {json.dumps(final_state['best_config'], indent=2)}")
        print(f"Best Latency: {final_state['best_latency']:.2f}ms")
        print(f"Iterations: {final_state['iteration']}")
        print(f"{'='*70}\n")
        
        return final_state


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        sys.exit(1)
    
    result = run_tuning(MATMUL_SCHEMA, max_iterations=3)