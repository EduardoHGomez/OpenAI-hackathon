"""
GPU Kernel Autotuner - INTEGRATED with OpenAI + Triton Matmul
==============================================================

Fully integrated with:
- OpenAI API (reads from .env)
- Your triton_matmul.py for benchmarking
- Your metrics_evaluator.py for evaluation
- Complex flow (parallel experts)
"""

import json
import operator
import os
import sys
from typing import Annotated, Any, Literal, TypedDict
from langgraph.graph import StateGraph, END

# Add triton-optimizer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'triton-optimizer'))

# Import your modules
from triton_matmul import triton_matmul
from metrics_evaluator import MetricsEvaluator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# OpenAI imports
from openai import OpenAI


# ============================================================================
# OPENAI LLM CALL
# ============================================================================

def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4") -> str:
    """
    Real OpenAI API call.
    Reads OPENAI_API_KEY from .env file.
    """
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
        temperature=0.7,  # Some exploration
        max_tokens=1024
    )
    
    return response.choices[0].message.content


# ============================================================================
# STATE DEFINITION
# ============================================================================

class TuningState(TypedDict):
    """State that flows through the graph."""
    # Kernel specification
    kernel: str
    shape: dict[str, int]
    knobs: list[dict[str, Any]]
    
    # History & symptoms
    history: list[dict[str, Any]]
    symptoms: list[str]
    
    # Proposals from experts
    proposals: Annotated[list[dict[str, Any]], operator.add]
    
    # Validated & deduplicated configs
    valid_configs: list[dict[str, Any]]
    
    # Execution results
    results: list[dict[str, Any]]
    
    # Best config so far
    best_config: dict[str, Any] | None
    best_latency: float
    
    # Iteration control
    iteration: int
    max_iterations: int
    converged: bool


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_config(config: dict[str, Any], knobs: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    Validate a config against the knob schema.
    Returns (is_valid, error_message)
    """
    knob_map = {k["name"]: k for k in knobs}
    
    # Check all required knobs present
    for knob in knobs:
        if knob["name"] not in config:
            return False, f"Missing knob: {knob['name']}"
    
    # Check no extra knobs
    for key in config:
        if key not in knob_map:
            return False, f"Unknown knob: {key}"
    
    # Validate each knob
    for key, value in config.items():
        knob = knob_map[key]
        
        # Type check
        if knob["type"] == "int" and not isinstance(value, int):
            return False, f"{key} must be int, got {type(value).__name__}"
        
        # Range check
        if "allowed" in knob and value not in knob["allowed"]:
            return False, f"{key}={value} not in allowed: {knob['allowed']}"
        
        # Multiple-of constraint (for tensor cores)
        if knob.get("multiple_of"):
            if value % knob["multiple_of"] != 0:
                return False, f"{key}={value} must be multiple of {knob['multiple_of']}"
    
    return True, ""


def estimate_shared_memory(config: dict[str, Any], kernel: str) -> int:
    """
    Estimate shared memory usage in bytes.
    """
    if kernel == "matmul":
        # Rough estimate: block_m*block_k + block_k*block_n, 2 bytes (fp16)
        tile_a = config.get("block_m", 128) * config.get("block_k", 32)
        tile_b = config.get("block_k", 32) * config.get("block_n", 128)
        stages = config.get("num_stages", 2)
        return (tile_a + tile_b) * 2 * stages
    
    return 0  # Conservative: assume safe


# ============================================================================
# EXPERT NODES
# ============================================================================

SYSTEM_PROMPT = """You propose kernel tuning parameters for GPU performance.
You must return ONLY valid JSON matching the provided knob schema.
Never use values outside the allowed set.
Prefer tensor-core friendly multiples (16, 32, 64, 128) for fp16/bf16 where applicable.
"""

def create_expert_prompt(state: TuningState, bias: str) -> str:
    """Generate the expert-specific prompt."""
    bias_instructions = {
        "throughput": "Maximize arithmetic intensity and latency hiding. Push block sizes and stages up. Watch occupancy.",
        "memory": "Minimize register and shared memory pressure to raise occupancy. Shrink tiles, moderate stages.",
        "robustness": "Choose conservative values that rarely spill or go OOB. Keep tensor-core multiples."
    }
    
    prompt = f"""KERNEL: {state['kernel']}
SHAPE: {json.dumps(state['shape'])}
KNOB_SCHEMA: {json.dumps(state['knobs'], indent=2)}
HISTORY (last 5): {json.dumps(state['history'][-5:], indent=2) if state['history'] else "[]"}
SYMPTOMS: {state['symptoms']}

EXPERT BIAS: {bias.upper()}
{bias_instructions[bias]}

TASK: Propose a config that improves median latency while respecting allowed values.
RETURN: JSON with exactly the knob names. No prose, no explanation.

Example output format:
{{"block_m": 128, "block_n": 64, "block_k": 32, "num_warps": 8, "num_stages": 2}}
"""
    return prompt


def throughput_expert(state: TuningState) -> dict:
    """Throughput-focused expert."""
    prompt = create_expert_prompt(state, "throughput")
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    try:
        # Extract JSON from response (LLM might add explanation)
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            config = json.loads(json_str)
            print(f"📊 Throughput Expert: {config}")
            return {"proposals": [{"expert": "throughput", "config": config}]}
    except Exception as e:
        print(f"❌ Throughput Expert error: {e}")
    
    return {"proposals": []}


def memory_expert(state: TuningState) -> dict:
    """Memory/occupancy-focused expert."""
    prompt = create_expert_prompt(state, "memory")
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            config = json.loads(json_str)
            print(f"🧠 Memory Expert: {config}")
            return {"proposals": [{"expert": "memory", "config": config}]}
    except Exception as e:
        print(f"❌ Memory Expert error: {e}")
    
    return {"proposals": []}


def robustness_expert(state: TuningState) -> dict:
    """Conservative/safety-focused expert."""
    prompt = create_expert_prompt(state, "robustness")
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            config = json.loads(json_str)
            print(f"🛡️  Robustness Expert: {config}")
            return {"proposals": [{"expert": "robustness", "config": config}]}
    except Exception as e:
        print(f"❌ Robustness Expert error: {e}")
    
    return {"proposals": []}


# ============================================================================
# VALIDATION NODE
# ============================================================================

def validate_proposals(state: TuningState) -> dict:
    """
    Validate all proposals against schema and resource constraints.
    Deduplicate identical configs.
    """
    valid_configs = []
    seen = set()
    
    for proposal in state["proposals"]:
        config = proposal["config"]
        
        # Schema validation
        is_valid, error = validate_config(config, state["knobs"])
        if not is_valid:
            print(f"❌ {proposal['expert']}: {error}")
            continue
        
        # Resource guard (shared memory)
        smem = estimate_shared_memory(config, state["kernel"])
        MAX_SMEM = 96 * 1024  # 96KB conservative limit
        if smem > MAX_SMEM:
            print(f"❌ {proposal['expert']}: SMEM {smem/1024:.1f}KB > {MAX_SMEM/1024}KB")
            continue
        
        # Deduplicate
        config_key = json.dumps(config, sort_keys=True)
        if config_key in seen:
            continue
        seen.add(config_key)
        
        valid_configs.append({
            "expert": proposal["expert"],
            "config": config,
            "smem_kb": smem / 1024
        })
        print(f"✅ {proposal['expert']}: {config} (SMEM: {smem/1024:.1f}KB)")
    
    return {"valid_configs": valid_configs, "proposals": []}


# ============================================================================
# EXECUTION NODE - INTEGRATED WITH YOUR TRITON CODE
# ============================================================================

import torch
import numpy as np

def execute_configs(state: TuningState) -> dict:
    """
    Execute configs using YOUR triton_matmul.py and metrics_evaluator.py
    """
    results = []
    
    # Initialize metrics evaluator
    evaluator = MetricsEvaluator()
    
    # Prepare inputs based on shape
    M, K, N = state["shape"]["M"], state["shape"]["K"], state["shape"]["N"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("⚠️  WARNING: CUDA not available, using CPU (slow!)")
    
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # Reference output for correctness check
    with torch.no_grad():
        C_ref = (A.float() @ B.float()).half()
    
    for item in state["valid_configs"]:
        config = item["config"]
        expert = item["expert"]
        
        try:
            # Extract config parameters
            block_m = config["block_m"]
            block_n = config["block_n"]
            block_k = config["block_k"]
            num_warps = config["num_warps"]
            num_stages = config["num_stages"]
            
            print(f"\n⚡ Benchmarking {expert}: block_m={block_m}, block_n={block_n}, block_k={block_k}, warps={num_warps}, stages={num_stages}")
            
            # Warmup
            for _ in range(10):
                _ = triton_matmul(
                    A, B,
                    block_m=block_m, block_n=block_n, block_k=block_k,
                    num_warps=num_warps, num_stages=num_stages
                )
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            num_trials = 100
            
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                C = triton_matmul(
                    A, B,
                    block_m=block_m, block_n=block_n, block_k=block_k,
                    num_warps=num_warps, num_stages=num_stages
                )
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # milliseconds
            
            median_ms = float(np.median(times))
            p90_ms = float(np.percentile(times, 90))
            
            # Correctness check
            max_abs_diff = (C - C_ref).abs().max().item()
            correct = max_abs_diff < 1e-2  # 1% tolerance
            
            # Use metrics evaluator to get detailed metrics
            metrics = evaluator.evaluate({
                "config": config,
                "median_ms": median_ms,
                "p90_ms": p90_ms,
                "max_abs_diff": max_abs_diff,
                "correct": correct
            })
            
            results.append({
                "expert": expert,
                "config": config,
                "median_ms": median_ms,
                "p90_ms": p90_ms,
                "correct": correct,
                "max_abs_diff": max_abs_diff,
                "metrics": metrics
            })
            
            status = "✅" if correct else "❌"
            print(f"{status} {expert:12s}: {median_ms:6.2f}ms (p90: {p90_ms:6.2f}ms, diff: {max_abs_diff:.3e})")
        
        except Exception as e:
            print(f"❌ {expert:12s}: ERROR - {e}")
            results.append({
                "expert": expert,
                "config": config,
                "median_ms": float('inf'),
                "p90_ms": float('inf'),
                "correct": False,
                "max_abs_diff": float('inf'),
                "metrics": {}
            })
    
    return {"results": results, "valid_configs": []}


# ============================================================================
# ANALYSIS NODE
# ============================================================================

def analyze_results(state: TuningState) -> dict:
    """
    Pick the best config and update history.
    Check for convergence.
    """
    # Filter for correct results
    valid_results = [r for r in state["results"] if r["correct"]]
    
    if not valid_results:
        print("⚠️  No valid results this iteration")
        return {
            "results": [],
            "iteration": state["iteration"] + 1,
            "converged": state["iteration"] + 1 >= state["max_iterations"]
        }
    
    # Find best
    best = min(valid_results, key=lambda r: r["median_ms"])
    
    # Update history
    history = state["history"] + [{
        "cfg": best["config"],
        "median_ms": best["median_ms"],
        "p90_ms": best["p90_ms"],
        "correct": best["correct"]
    }]
    
    # Check if improvement
    improvement = False
    if state["best_config"] is None:
        improvement = True
        best_config = best["config"]
        best_latency = best["median_ms"]
        print(f"\n🎯 First best: {best_latency:.2f}ms")
    elif best["median_ms"] < state["best_latency"] * 0.98:  # 2% threshold
        improvement = True
        old_latency = state["best_latency"]
        best_config = best["config"]
        best_latency = best["median_ms"]
        speedup = ((old_latency / best_latency) - 1) * 100
        print(f"\n🎯 New best: {best_latency:.2f}ms (was {old_latency:.2f}ms, +{speedup:.1f}% speedup)")
    else:
        best_config = state["best_config"]
        best_latency = state["best_latency"]
        print(f"\n📊 No improvement (best still {best_latency:.2f}ms)")
    
    # Convergence check
    converged = (
        state["iteration"] + 1 >= state["max_iterations"] or
        (not improvement and state["iteration"] > 2)
    )
    
    if converged:
        print(f"\n✅ CONVERGED!")
    
    return {
        "history": history,
        "results": [],
        "best_config": best_config,
        "best_latency": best_latency,
        "iteration": state["iteration"] + 1,
        "converged": converged
    }


# ============================================================================
# ROUTING
# ============================================================================

def should_continue(state: TuningState) -> Literal["propose", "end"]:
    """Route: continue tuning or stop."""
    if state["converged"]:
        return "end"
    return "propose"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """Build the LangGraph autotuning workflow."""
    workflow = StateGraph(TuningState)
    
    # Add nodes
    workflow.add_node("throughput_expert", throughput_expert)
    workflow.add_node("memory_expert", memory_expert)
    workflow.add_node("robustness_expert", robustness_expert)
    workflow.add_node("validate", validate_proposals)
    workflow.add_node("execute", execute_configs)
    workflow.add_node("analyze", analyze_results)
    
    # Entry point: call all experts in parallel
    workflow.set_entry_point("throughput_expert")
    workflow.add_edge("throughput_expert", "validate")
    workflow.add_edge("memory_expert", "validate")
    workflow.add_edge("robustness_expert", "validate")
    
    # Expert fan-out (all run in parallel before validate)
    workflow.add_edge("throughput_expert", "memory_expert")
    workflow.add_edge("throughput_expert", "robustness_expert")
    
    # Sequential flow after experts
    workflow.add_edge("validate", "execute")
    workflow.add_edge("execute", "analyze")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "propose": "throughput_expert",
            "end": END
        }
    )
    
    return workflow.compile()


# ============================================================================
# KERNEL SCHEMA
# ============================================================================

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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_tuning(schema: dict, max_iterations: int = 5):
    """Run the autotuning workflow."""
    print(f"\n{'='*70}")
    print(f"🚀 Starting autotuning for {schema['kernel'].upper()}")
    print(f"   Shape: {schema['shape']}")
    print(f"   Using OpenAI API")
    print(f"{'='*70}\n")
    
    # Initialize state
    initial_state: TuningState = {
        "kernel": schema["kernel"],
        "shape": schema["shape"],
        "knobs": schema["knobs"],
        "history": [],
        "symptoms": ["initial run", "no baseline"],
        "proposals": [],
        "valid_configs": [],
        "results": [],
        "best_config": None,
        "best_latency": float('inf'),
        "iteration": 0,
        "max_iterations": max_iterations,
        "converged": False
    }
    
    # Build and run graph
    graph = build_graph()
    
    final_state = None
    for iteration_state in graph.stream(initial_state):
        final_state = iteration_state
        print()  # Blank line between steps
    
    # Extract final state
    if final_state:
        final_state = list(final_state.values())[0]
        
        print(f"\n{'='*70}")
        print(f"🏁 TUNING COMPLETE")
        print(f"{'='*70}")
        print(f"Best Config: {json.dumps(final_state['best_config'], indent=2)}")
        print(f"Best Latency: {final_state['best_latency']:.2f}ms")
        print(f"Iterations: {final_state['iteration']}")
        
        if final_state['history']:
            initial = final_state['history'][0]['median_ms']
            final_lat = final_state['best_latency']
            speedup = ((initial / final_lat) - 1) * 100
            print(f"Speedup: {speedup:.1f}%")
        
        print(f"{'='*70}\n")
        
        return final_state


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GPU KERNEL AUTOTUNER - OpenAI + Triton Integration")
    print("="*70)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
        print("Create a .env file with: OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available!")
    
    # Run tuning
    result = run_tuning(MATMUL_SCHEMA, max_iterations=3)