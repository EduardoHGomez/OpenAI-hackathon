"""
GPU Kernel Autotuner with LangGraph + Multi-Expert LLM Orchestration
====================================================================

Three expert LLMs propose kernel configurations:
- Throughput Expert: maximize arithmetic intensity
- Memory Expert: minimize register/shared memory pressure
- Robustness Expert: conservative, tensor-core friendly choices

Works across kernels: matmul, softmax, layernorm, conv2d, etc.
"""

import json
import operator
from typing import Annotated, Any, Literal, TypedDict
from langgraph.graph import StateGraph, END

# Mock LLM call - replace with actual Anthropic/OpenAI client
def call_llm(system_prompt: str, user_prompt: str, model: str = "claude-sonnet-4") -> str:
    """
    Replace this with actual LLM API call.
    For now, returns mock JSON configs.
    """
    # Mock responses for demonstration
    import random
    if "throughput" in user_prompt.lower():
        return json.dumps({
            "block_m": 128, "block_n": 128, "block_k": 64,
            "num_warps": 8, "num_stages": 3
        })
    elif "memory" in user_prompt.lower():
        return json.dumps({
            "block_m": 64, "block_n": 64, "block_k": 32,
            "num_warps": 4, "num_stages": 2
        })
    else:  # robustness
        return json.dumps({
            "block_m": 128, "block_n": 64, "block_k": 32,
            "num_warps": 4, "num_stages": 2
        })


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
    Cheap pre-check to avoid OOM proposals.
    """
    if kernel == "matmul":
        # Rough estimate: block_m*block_k + block_k*block_n, 2 bytes (fp16)
        tile_a = config.get("block_m", 128) * config.get("block_k", 32)
        tile_b = config.get("block_k", 32) * config.get("block_n", 128)
        stages = config.get("num_stages", 2)
        return (tile_a + tile_b) * 2 * stages
    
    elif kernel == "softmax":
        # block_rows * vector_width * sizeof(float)
        return config.get("block_rows", 512) * config.get("vector_width", 4) * 4
    
    elif kernel == "layernorm":
        # block_hidden * vector_width * sizeof(float)
        return config.get("block_hidden", 1024) * config.get("vector_width", 4) * 4
    
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
HISTORY (last 5): {json.dumps(state['history'][-5:], indent=2)}
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
        config = json.loads(response)
        return {"proposals": [{"expert": "throughput", "config": config}]}
    except json.JSONDecodeError:
        return {"proposals": []}


def memory_expert(state: TuningState) -> dict:
    """Memory/occupancy-focused expert."""
    prompt = create_expert_prompt(state, "memory")
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    try:
        config = json.loads(response)
        return {"proposals": [{"expert": "memory", "config": config}]}
    except json.JSONDecodeError:
        return {"proposals": []}


def robustness_expert(state: TuningState) -> dict:
    """Conservative/safety-focused expert."""
    prompt = create_expert_prompt(state, "robustness")
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    try:
        config = json.loads(response)
        return {"proposals": [{"expert": "robustness", "config": config}]}
    except json.JSONDecodeError:
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
    
    return {"valid_configs": valid_configs, "proposals": []}  # Clear proposals


# ============================================================================
# EXECUTION NODE
# ============================================================================

def execute_configs(state: TuningState) -> dict:
    """
    Execute validated configs and measure performance.
    Replace with actual kernel benchmarking.
    """
    import random
    
    results = []
    for item in state["valid_configs"]:
        config = item["config"]
        
        # Mock execution - replace with real kernel benchmark
        median_ms = random.uniform(40, 80)
        p90_ms = median_ms * 1.1
        correct = random.random() > 0.1  # 90% correctness rate
        
        results.append({
            "expert": item["expert"],
            "config": config,
            "median_ms": median_ms,
            "p90_ms": p90_ms,
            "correct": correct
        })
        
        status = "✅" if correct else "❌"
        print(f"{status} {item['expert']}: {median_ms:.2f}ms (p90: {p90_ms:.2f}ms)")
    
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
    elif best["median_ms"] < state["best_latency"] * 0.98:  # 2% threshold
        improvement = True
        best_config = best["config"]
        best_latency = best["median_ms"]
        print(f"🎯 New best: {best_latency:.2f}ms (from {state['best_latency']:.2f}ms)")
    else:
        best_config = state["best_config"]
        best_latency = state["best_latency"]
    
    # Convergence check
    converged = (
        state["iteration"] + 1 >= state["max_iterations"] or
        (not improvement and state["iteration"] > 2)
    )
    
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
# KERNEL SCHEMAS
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

SOFTMAX_SCHEMA = {
    "kernel": "softmax",
    "shape": {"batch": 32, "seq_len": 2048, "hidden": 768},
    "knobs": [
        {"name": "block_rows", "type": "int", "allowed": [256, 512, 1024]},
        {"name": "vector_width", "type": "int", "allowed": [1, 2, 4, 8]},
        {"name": "num_warps", "type": "int", "allowed": [2, 4, 8]},
        {"name": "num_stages", "type": "int", "allowed": [1, 2, 3]}
    ]
}

LAYERNORM_SCHEMA = {
    "kernel": "layernorm",
    "shape": {"batch": 32, "seq_len": 2048, "hidden": 1024},
    "knobs": [
        {"name": "block_hidden", "type": "int", "allowed": [256, 512, 1024]},
        {"name": "vector_width", "type": "int", "allowed": [1, 2, 4, 8]},
        {"name": "num_warps", "type": "int", "allowed": [2, 4, 8]},
        {"name": "num_stages", "type": "int", "allowed": [1, 2, 3]}
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
        print(f"{'='*70}\n")
        
        return final_state


if __name__ == "__main__":
    # Example: tune different kernels
    print("\n" + "="*70)
    print("GPU KERNEL AUTOTUNER - Multi-Expert LLM Orchestration")
    print("="*70)
    
    # Tune matmul
    matmul_result = run_tuning(MATMUL_SCHEMA, max_iterations=3)
    
    # Tune softmax
    softmax_result = run_tuning(SOFTMAX_SCHEMA, max_iterations=3)
    
    # Tune layernorm
    layernorm_result = run_tuning(LAYERNORM_SCHEMA, max_iterations=3)