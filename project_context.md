# GPU Kernel Autotuner - Complete Project Context for Debugging

This document provides complete context about the GPU Kernel Autotuner project for AI-assisted debugging.

## 🎯 PROJECT OVERVIEW

**Purpose**: Automatically tune GPU kernel parameters using multiple LLM experts competing in parallel to find optimal configurations.

**Key Innovation**: Uses LangGraph to orchestrate 3 specialized LLM "experts" that propose different kernel configurations, which are then benchmarked on real hardware to find the best performer.

**Target Use Case**: Optimizing Triton GPU kernels (initially matmul, extensible to any kernel type).

---

## 📁 PROJECT STRUCTURE

```
project/
├── .env                                    # Contains OPENAI_API_KEY
├── gpu_kernel_tuner_integrated.py         # MAIN FILE - Fully integrated version
├── gpu_kernel_tuner_simple.py             # Simple flow (sequential experts)
├── gpu_kernel_turner.py                   # Original version (typo in name)
├── README.md
└── triton-optimizer/
    ├── metrics_evaluator.py               # Evaluates kernel performance metrics
    ├── optimize_simple.py                 # (Not used currently)
    └── triton_matmul.py                   # Triton matmul kernel implementation
```

---

## 🏗️ ARCHITECTURE

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              PARALLEL: Three Expert LLM Calls                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Throughput   │  │   Memory     │  │  Robustness  │         │
│  │   Expert     │  │   Expert     │  │   Expert     │         │
│  │ (OpenAI GPT) │  │ (OpenAI GPT) │  │ (OpenAI GPT) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                 │                  │                  │
│         └─────────────────┴──────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│            Each proposes a kernel config JSON:                  │
│            {block_m: 128, block_n: 64, ...}                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Validation Node                              │
│  • Check configs against schema (allowed values)                │
│  • Estimate shared memory usage                                 │
│  • Deduplicate identical configs                                │
│  • Filter out invalid proposals                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Node                               │
│  FOR EACH valid config:                                         │
│    1. Compile Triton kernel with config                         │
│    2. Prepare input tensors (A, B matrices)                     │
│    3. Warmup (10 runs)                                          │
│    4. Benchmark (100 trials with CUDA events)                   │
│    5. Calculate median & p90 latency                            │
│    6. Verify correctness vs. PyTorch reference                  │
│    7. Use MetricsEvaluator for detailed metrics                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Node                                │
│  • Find best performing config (lowest median latency)          │
│  • Check if better than previous best (>2% improvement)         │
│  • Update history with results                                  │
│  • Check convergence criteria                                   │
│  • Decide: continue iterating or stop                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Converged?           │
                    │  YES → END            │
                    │  NO  → Loop back to   │
                    │        Expert Calls   │
                    └───────────────────────┘
```

---

## 🔧 COMPONENT DETAILS

### 1. State Management (TuningState)

The entire workflow state is managed via a TypedDict that flows through the graph:

```python
class TuningState(TypedDict):
    # Kernel specification
    kernel: str                              # "matmul"
    shape: dict[str, int]                    # {"M": 2048, "K": 2048, "N": 2048}
    knobs: list[dict[str, Any]]              # Tunable parameters schema
    
    # History & symptoms
    history: list[dict[str, Any]]            # Past configs and results
    symptoms: list[str]                       # Issues to guide experts
    
    # Proposals from experts
    proposals: Annotated[list[dict], operator.add]  # Merged from all experts
    
    # Validated configs
    valid_configs: list[dict[str, Any]]      # After validation
    
    # Execution results
    results: list[dict[str, Any]]            # Benchmark results
    
    # Best config tracking
    best_config: dict[str, Any] | None       # Current best
    best_latency: float                       # Current best latency (ms)
    
    # Iteration control
    iteration: int                            # Current iteration
    max_iterations: int                       # Stop after this many
    converged: bool                           # Should we stop?
```

**Key Pattern**: Uses `Annotated[list, operator.add]` for proposals so LangGraph automatically merges lists from parallel expert nodes.

---

### 2. Expert Nodes (Parallel Execution)

Three expert LLM calls run **in parallel** via LangGraph:

#### Throughput Expert
```python
def throughput_expert(state: TuningState) -> dict:
    """
    Bias: Maximize arithmetic intensity and latency hiding
    Strategy: Push block sizes up, increase stages, maximize warps
    Goal: Highest compute throughput
    """
    prompt = create_expert_prompt(state, "throughput")
    response = call_llm(SYSTEM_PROMPT, prompt)
    
    # Extract JSON from response (LLM might wrap in prose)
    config = extract_json(response)
    
    return {"proposals": [{"expert": "throughput", "config": config}]}
```

#### Memory Expert
```python
def memory_expert(state: TuningState) -> dict:
    """
    Bias: Minimize register and shared memory pressure
    Strategy: Shrink tile sizes, moderate stages, lower warps
    Goal: Higher occupancy via lower resource usage
    """
    prompt = create_expert_prompt(state, "memory")
    response = call_llm(SYSTEM_PROMPT, prompt)
    config = extract_json(response)
    
    return {"proposals": [{"expert": "memory", "config": config}]}
```

#### Robustness Expert
```python
def robustness_expert(state: TuningState) -> dict:
    """
    Bias: Conservative, safe choices
    Strategy: Tensor-core friendly multiples, avoid edge cases
    Goal: Reliability and correctness
    """
    prompt = create_expert_prompt(state, "robustness")
    response = call_llm(SYSTEM_PROMPT, prompt)
    config = extract_json(response)
    
    return {"proposals": [{"expert": "robustness", "config": config}]}
```

**Expert Prompt Structure**:
```
KERNEL: matmul
SHAPE: {"M": 2048, "K": 2048, "N": 2048}
KNOB_SCHEMA: [
  {"name": "block_m", "type": "int", "allowed": [64, 128, 256], ...},
  ...
]
HISTORY (last 5): [
  {"cfg": {...}, "median_ms": 45.2, "correct": true},
  ...
]
SYMPTOMS: ["high register pressure", "spilling detected"]

EXPERT BIAS: THROUGHPUT
Maximize arithmetic intensity and latency hiding. Push block sizes and stages up.

TASK: Propose a config that improves median latency while respecting allowed values.
RETURN: JSON with exactly the knob names. No prose, no explanation.

Example output format:
{"block_m": 128, "block_n": 64, "block_k": 32, "num_warps": 8, "num_stages": 2}
```

---

### 3. Validation Node

```python
def validate_proposals(state: TuningState) -> dict:
    """
    Validates proposals from experts.
    
    Checks:
    1. Schema validation (all knobs present, correct types, allowed values)
    2. Tensor-core alignment (multiple_of constraints)
    3. Shared memory estimation (must fit in 96KB limit)
    4. Deduplication (same config from multiple experts)
    
    Returns:
        valid_configs: List of configs that passed all checks
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
        
        # Resource check
        smem = estimate_shared_memory(config, state["kernel"])
        if smem > 96 * 1024:  # 96KB limit
            print(f"❌ {proposal['expert']}: SMEM too large")
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
    
    return {"valid_configs": valid_configs, "proposals": []}
```

---

### 4. Execution Node (Benchmarking)

```python
def execute_configs(state: TuningState) -> dict:
    """
    Benchmarks each valid config using real Triton kernel.
    
    For each config:
    1. Compile Triton kernel with parameters
    2. Create input tensors (fp16)
    3. Warmup (10 runs)
    4. Benchmark (100 trials)
    5. Measure with CUDA events
    6. Verify correctness vs PyTorch
    7. Calculate metrics
    
    Returns:
        results: [{expert, config, median_ms, p90_ms, correct, ...}]
    """
    results = []
    
    # Prepare inputs
    M, K, N = state["shape"]["M"], state["shape"]["K"], state["shape"]["N"]
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C_ref = (A.float() @ B.float()).half()  # Reference
    
    for item in state["valid_configs"]:
        config = item["config"]
        
        try:
            # Warmup
            for _ in range(10):
                _ = triton_matmul(
                    A, B,
                    block_m=config["block_m"],
                    block_n=config["block_n"],
                    block_k=config["block_k"],
                    num_warps=config["num_warps"],
                    num_stages=config["num_stages"]
                )
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                C = triton_matmul(A, B, **config)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            median_ms = float(np.median(times))
            p90_ms = float(np.percentile(times, 90))
            
            # Correctness
            max_abs_diff = (C - C_ref).abs().max().item()
            correct = max_abs_diff < 1e-2
            
            # Metrics
            metrics = evaluator.evaluate({
                "config": config,
                "median_ms": median_ms,
                "p90_ms": p90_ms,
                "max_abs_diff": max_abs_diff,
                "correct": correct
            })
            
            results.append({
                "expert": item["expert"],
                "config": config,
                "median_ms": median_ms,
                "p90_ms": p90_ms,
                "correct": correct,
                "max_abs_diff": max_abs_diff,
                "metrics": metrics
            })
            
        except Exception as e:
            # Mark as failed
            results.append({
                "expert": item["expert"],
                "config": config,
                "median_ms": float('inf'),
                "correct": False
            })
    
    return {"results": results, "valid_configs": []}
```

---

### 5. Analysis Node (Convergence Check)

```python
def analyze_results(state: TuningState) -> dict:
    """
    Analyzes benchmark results and decides next action.
    
    Logic:
    1. Filter for correct results only
    2. Find best (lowest median latency)
    3. Check if improvement over previous best (>2% threshold)
    4. Update history
    5. Check convergence:
       - Max iterations reached, OR
       - No improvement for 2+ iterations
    
    Returns:
        Updated state with best_config, best_latency, converged flag
    """
    valid_results = [r for r in state["results"] if r["correct"]]
    
    if not valid_results:
        return {
            "results": [],
            "iteration": state["iteration"] + 1,
            "converged": state["iteration"] + 1 >= state["max_iterations"]
        }
    
    best = min(valid_results, key=lambda r: r["median_ms"])
    
    # Update history
    history = state["history"] + [{
        "cfg": best["config"],
        "median_ms": best["median_ms"],
        "p90_ms": best["p90_ms"],
        "correct": best["correct"]
    }]
    
    # Check improvement
    improvement = False
    if state["best_config"] is None:
        # First iteration
        improvement = True
        best_config = best["config"]
        best_latency = best["median_ms"]
    elif best["median_ms"] < state["best_latency"] * 0.98:
        # 2% improvement threshold
        improvement = True
        best_config = best["config"]
        best_latency = best["median_ms"]
        speedup = ((state["best_latency"] / best_latency) - 1) * 100
        print(f"🎯 New best: {best_latency:.2f}ms (+{speedup:.1f}% speedup)")
    else:
        # No improvement
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
```

---

### 6. LangGraph Construction

```python
def build_graph():
    """
    Constructs the LangGraph workflow.
    
    Graph structure:
                    [START]
                       │
                       ▼
              ┌─────────────────┐
              │ throughput_expert│
              └─────────────────┘
                 │         │
                 ▼         ▼
    ┌──────────────┐  ┌──────────────┐
    │memory_expert │  │robustness_   │
    │              │  │expert        │
    └──────────────┘  └──────────────┘
           │                │
           └────────┬───────┘
                    ▼
            ┌──────────────┐
            │  validate    │
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │  execute     │
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │  analyze     │
            └──────────────┘
                    │
                    ▼
            [converged?]
              YES → END
              NO  → throughput_expert (loop)
    """
    workflow = StateGraph(TuningState)
    
    # Add nodes
    workflow.add_node("throughput_expert", throughput_expert)
    workflow.add_node("memory_expert", memory_expert)
    workflow.add_node("robustness_expert", robustness_expert)
    workflow.add_node("validate", validate_proposals)
    workflow.add_node("execute", execute_configs)
    workflow.add_node("analyze", analyze_results)
    
    # Entry point
    workflow.set_entry_point("throughput_expert")
    
    # All experts feed into validate
    workflow.add_edge("throughput_expert", "validate")
    workflow.add_edge("memory_expert", "validate")
    workflow.add_edge("robustness_expert", "validate")
    
    # Expert fan-out (parallel execution)
    workflow.add_edge("throughput_expert", "memory_expert")
    workflow.add_edge("throughput_expert", "robustness_expert")
    
    # Sequential after validation
    workflow.add_edge("validate", "execute")
    workflow.add_edge("execute", "analyze")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "propose": "throughput_expert",  # Loop back
            "end": END                        # Stop
        }
    )
    
    return workflow.compile()
```

---

## 🔌 INTEGRATION POINTS

### 1. OpenAI API

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Loads .env file with OPENAI_API_KEY

def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4") -> str:
    """
    Calls OpenAI API.
    
    Environment:
        OPENAI_API_KEY must be set in .env
    
    Args:
        system_prompt: Instructions for the LLM
        user_prompt: The specific query
        model: OpenAI model to use
    
    Returns:
        LLM response text (may contain JSON wrapped in prose)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env!")
    
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
```

### 2. Triton Matmul Integration

```python
# Import from triton-optimizer/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'triton-optimizer'))
from triton_matmul import triton_matmul

# Usage in execute_configs()
C = triton_matmul(
    A, B,
    block_m=config["block_m"],
    block_n=config["block_n"],
    block_k=config["block_k"],
    num_warps=config["num_warps"],
    num_stages=config["num_stages"]
)
```

**Triton Kernel Signature**:
```python
def triton_matmul(
    a: torch.Tensor,        # Shape: (M, K), dtype: float16
    b: torch.Tensor,        # Shape: (K, N), dtype: float16
    *,
    block_m: int = 128,     # Tile size M dimension
    block_n: int = 128,     # Tile size N dimension
    block_k: int = 32,      # Tile size K dimension (reduction)
    num_warps: int = 4,     # Number of warps per block
    num_stages: int = 2     # Pipeline stages
) -> torch.Tensor:          # Shape: (M, N), dtype: float16
    """
    Triton matmul kernel.
    
    Tunable parameters:
    - block_m, block_n: Output tile size
    - block_k: Reduction tile size
    - num_warps: Parallelism within block (2, 4, 8, 16)
    - num_stages: Pipeline depth (1-5)
    
    Returns:
        C = A @ B (matrix multiplication)
    """
```

### 3. Metrics Evaluator Integration

```python
from metrics_evaluator import MetricsEvaluator

evaluator = MetricsEvaluator()

# Usage in execute_configs()
metrics = evaluator.evaluate({
    "config": config,
    "median_ms": median_ms,
    "p90_ms": p90_ms,
    "max_abs_diff": max_abs_diff,
    "correct": correct
})

# metrics contains additional evaluation data
```

---

## 📊 DATA FLOW EXAMPLE

### Iteration 1:

**Input State**:
```python
{
    "kernel": "matmul",
    "shape": {"M": 2048, "K": 2048, "N": 2048},
    "knobs": [...],
    "history": [],
    "symptoms": ["initial run", "no baseline"],
    "proposals": [],
    "iteration": 0,
    "max_iterations": 5,
    "converged": False
}
```

**After Expert Calls** (parallel):
```python
{
    "proposals": [
        {"expert": "throughput", "config": {
            "block_m": 128, "block_n": 128, "block_k": 64,
            "num_warps": 8, "num_stages": 3
        }},
        {"expert": "memory", "config": {
            "block_m": 64, "block_n": 64, "block_k": 32,
            "num_warps": 4, "num_stages": 2
        }},
        {"expert": "robustness", "config": {
            "block_m": 128, "block_n": 64, "block_k": 32,
            "num_warps": 4, "num_stages": 2
        }}
    ]
}
```

**After Validation**:
```python
{
    "valid_configs": [
        {"expert": "throughput", "config": {...}, "smem_kb": 72.0},
        {"expert": "memory", "config": {...}, "smem_kb": 16.0},
        {"expert": "robustness", "config": {...}, "smem_kb": 24.0}
    ],
    "proposals": []  # Cleared
}
```

**After Execution**:
```python
{
    "results": [
        {
            "expert": "throughput",
            "config": {...},
            "median_ms": 45.23,
            "p90_ms": 49.75,
            "correct": True,
            "max_abs_diff": 2.1e-03
        },
        {
            "expert": "memory",
            "config": {...},
            "median_ms": 52.10,
            "p90_ms": 57.31,
            "correct": True,
            "max_abs_diff": 1.8e-03
        },
        {
            "expert": "robustness",
            "config": {...},
            "median_ms": 48.67,
            "p90_ms": 53.54,
            "correct": True,
            "max_abs_diff": 1.5e-03
        }
    ],
    "valid_configs": []  # Cleared
}
```

**After Analysis**:
```python
{
    "history": [
        {"cfg": {...}, "median_ms": 45.23, "correct": True}
    ],
    "results": [],  # Cleared
    "best_config": {
        "block_m": 128, "block_n": 128, "block_k": 64,
        "num_warps": 8, "num_stages": 3
    },
    "best_latency": 45.23,
    "iteration": 1,
    "converged": False  # Continue to iteration 2
}
```

---

## 🧪 CONFIGURATION SCHEMA

### MATMUL_SCHEMA

```python
{
    "kernel": "matmul",
    "shape": {
        "M": 2048,      # Matrix A rows
        "K": 2048,      # Shared dimension
        "N": 2048       # Matrix B columns
    },
    "knobs": [
        {
            "name": "block_m",
            "type": "int",
            "allowed": [64, 128, 256],
            "multiple_of": 16  # Tensor core alignment
        },
        {
            "name": "block_n",
            "type": "int",
            "allowed": [64, 128, 256],
            "multiple_of": 16
        },
        {
            "name": "block_k",
            "type": "int",
            "allowed": [16, 32, 64]
        },
        {
            "name": "num_warps",
            "type": "int",
            "allowed": [2, 4, 8]
        },
        {
            "name": "num_stages",
            "type": "int",
            "allowed": [1, 2, 3, 4]
        }
    ]
}
```

---

## 🔍 VALIDATION RULES

### 1. Schema Validation

```python
def validate_config(config, knobs):
    """
    Checks:
    - All required knobs present
    - No extra knobs
    - Correct types
    - Values in allowed set
    - Multiple-of constraints satisfied
    """
```

### 2. Resource Validation

```python
def estimate_shared_memory(config, kernel):
    """
    Estimates shared memory usage in bytes.
    
    For matmul:
        smem = (block_m * block_k + block_k * block_n) * 2 * num_stages
    
    Must be < 96KB for safety (actual limit varies by GPU)
    """
```

---

## ⚙️ TUNING PARAMETERS

### Convergence Criteria

```python
converged = (
    iteration >= max_iterations OR
    (no_improvement AND iteration > 2)
)
```

### Improvement Threshold

```python
# Consider it an improvement if at least 2% faster
improvement = new_latency < old_latency * 0.98
```

### Benchmarking

```python
WARMUP_RUNS = 10
BENCHMARK_TRIALS = 100
CORRECTNESS_THRESHOLD = 1e-2  # 1% relative error
```

---

## 🐛 COMMON ISSUES & DEBUGGING

### Issue 1: "OPENAI_API_KEY not found"

**Cause**: .env file missing or not in correct location

**Debug**:
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(f"API Key: {os.getenv('OPENAI_API_KEY')}")
```

**Fix**: Create .env file in project root:
```
OPENAI_API_KEY=sk-...
```

---

### Issue 2: "No module named 'triton_matmul'"

**Cause**: Python can't find triton-optimizer/ directory

**Debug**:
```python
import sys
print(sys.path)
# Check if triton-optimizer is in path
```

**Fix**: Ensure path setup:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'triton-optimizer'))
```

---

### Issue 3: LLM returns prose instead of JSON

**Symptom**: JSON parsing fails in expert nodes

**Debug**:
```python
print(f"LLM Response: {response}")
# Check if response has JSON wrapped in markdown or prose
```

**Fix**: Already handled with JSON extraction:
```python
start = response.find('{')
end = response.rfind('}') + 1
json_str = response[start:end]
config = json.loads(json_str)
```

---

### Issue 4: All configs fail validation

**Cause**: LLM proposing values outside allowed set

**Debug**:
```python
print(f"Proposed config: {config}")
print(f"Knob schema: {state['knobs']}")
# Check which knob value is invalid
```

**Fix**: 
- Widen allowed values in schema
- Improve expert prompts to emphasize constraints
- Add examples to prompts

---

### Issue 5: CUDA OOM during benchmarking

**Cause**: Shared memory exceeds GPU limits

**Debug**:
```python
smem = estimate_shared_memory(config, kernel)
print(f"Estimated SMEM: {smem / 1024}KB")
# Check GPU specs: nvidia-smi
```

**Fix**:
- Lower MAX_SMEM threshold in validation
- Add stricter SMEM estimation
- Reduce tile sizes in schema

---

### Issue 6: Incorrect results (correctness check fails)

**Cause**: Kernel implementation bug or numerical instability

**Debug**:
```python
print(f"Max abs diff: {max_abs_diff}")
print(f"Threshold: {CORRECTNESS_THRESHOLD}")
C_diff = (C - C_ref).abs()
print(f"Error distribution: mean={C_diff.mean()}, max={C_diff.max()}")
```

**Fix**:
- Check Triton kernel implementation
- Adjust correctness threshold
- Inspect specific failing configs

---

### Issue 7: No improvement after many iterations

**Symptom**: Same best config for multiple iterations

**Debug**:
```python
print(f"History: {state['history']}")
# Check if configs are diverse or all similar
```

**Fix**:
- Increase max_iterations
- Widen knob search space
- Check if already at optimal (might be working correctly!)
- Adjust expert biases to be more diverse

---

### Issue 8: LangGraph execution error

**Symptom**: Graph stream fails or hangs

**Debug**:
```python
for step in graph.stream(initial_state):
    print(f"Step: {step}")
    # Check which node is failing
```

**Fix**:
- Check all nodes return proper dict format
- Ensure state keys match TuningState schema
- Verify edge connections in build_graph()

---

## 📝 DEBUGGING CHECKLIST

When debugging, check these in order:

```
[ ] 1. Environment
    [ ] .env file exists
    [ ] OPENAI_API_KEY is set
    [ ] Python version >= 3.10

[ ] 2. Dependencies
    [ ] pip install openai python-dotenv langgraph torch triton numpy
    [ ] torch.cuda.is_available() returns True

[ ] 3. File Structure
    [ ] gpu_kernel_tuner_integrated.py in project root
    [ ] triton-optimizer/ directory exists
    [ ] triton_matmul.py in triton-optimizer/
    [ ] metrics_evaluator.py in triton-optimizer/

[ ] 4. LLM Integration
    [ ] call_llm() successfully calls OpenAI
    [ ] JSON extraction works (handles prose)
    [ ] All 3 expert nodes return proposals

[ ] 5. Validation
    [ ] Configs match schema
    [ ] SMEM estimates reasonable
    [ ] At least one config passes validation

[ ] 6. Execution
    [ ] Triton kernel compiles
    [ ] Tensors created correctly
    [ ] CUDA events work
    [ ] Correctness check passes

[ ] 7. Analysis
    [ ] Best config identified
    [ ] History updates
    [ ] Convergence logic works

[ ] 8. Graph Flow
    [ ] All edges connected
    [ ] State merging works (proposals)
    [ ] Conditional routing correct
```

---

## 🎯 EXPECTED BEHAVIOR

### Normal Run Output

```
======================================================================
🚀 Starting autotuning for MATMUL
   Shape: {'M': 2048, 'K': 2048, 'N': 2048}
   Using OpenAI API
======================================================================

📊 Throughput Expert: {'block_m': 128, 'block_n': 128, 'block_k': 64, 'num_warps': 8, 'num_stages': 3}
🧠 Memory Expert: {'block_m': 64, 'block_n': 64, 'block_k': 32, 'num_warps': 4, 'num_stages': 2}
🛡️  Robustness Expert: {'block_m': 128, 'block_n': 64, 'block_k': 32, 'num_warps': 4, 'num_stages': 2}

✅ throughput: {'block_m': 128, 'block_n': 128, 'block_k': 64, 'num_warps': 8, 'num_stages': 3} (SMEM: 72.0KB)
✅ memory: {'block_m': 64, 'block_n': 64, 'block_k': 32, 'num_warps': 4, 'num_stages': 2} (SMEM: 16.0KB)
✅ robustness: {'block_m': 128, 'block_n': 64, 'block_k': 32, 'num_warps': 4, 'num_stages': 2} (SMEM: 24.0KB)

⚡ Benchmarking throughput: block_m=128, block_n=128, block_k=64, warps=8, stages=3
✅ throughput  :  45.23ms (p90:  49.75ms, diff: 2.145e-03)

⚡ Benchmarking memory: block_m=64, block_n=64, block_k=32, warps=4, stages=2
✅ memory      :  52.10ms (p90:  57.31ms, diff: 1.876e-03)

⚡ Benchmarking robustness: block_m=128, block_n=64, block_k=32, warps=4, stages=2
✅ robustness  :  48.67ms (p90:  53.54ms, diff: 1.543e-03)

🎯 First best: 45.23ms

[Iteration 2...]

📊 Throughput Expert: {'block_m': 256, 'block_n': 128, 'block_k': 64, 'num_warps': 8, 'num_stages': 3}
...

🎯 New best: 42.15ms (was 45.23ms, +6.8% speedup)

[Iteration 3...]

📊 No improvement (best still 42.15ms)

✅ CONVERGED!

======================================================================
🏁 TUNING COMPLETE
======================================================================
Best Config: {
  "block_m": 256,
  "block_n": 128,
  "block_k": 64,
  "num_warps": 8,
  "num_stages": 3
}
Best Latency: 42.15ms
Iterations: 3
Speedup: 6.8%
======================================================================
```

---

## 💾 STATE PERSISTENCE

Currently the state is **not persisted** between runs. Each run starts fresh.

**To add persistence**:

```python
import pickle

# Save state after each iteration
def save_state(state, filename="tuning_state.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

# Load state at startup
def load_state(filename="tuning_state.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None
```

---

## 🎨 CUSTOMIZATION GUIDE

### Add a New Expert

1. **Create expert function**:
```python
def cache_expert(state: TuningState) -> dict:
    prompt = create_expert_prompt(state, "cache")
    response = call_llm(SYSTEM_PROMPT, prompt)
    config = extract_json(response)
    return {"proposals": [{"expert": "cache", "config": config}]}
```

2. **Add bias to prompt creator**:
```python
bias_instructions["cache"] = "Optimize for cache locality..."
```

3. **Wire into graph**:
```python
workflow.add_node("cache_expert", cache_expert)
workflow.add_edge("cache_expert", "validate")
workflow.add_edge("throughput_expert", "cache_expert")
```

### Add a New Kernel

1. **Create schema**:
```python
SOFTMAX_SCHEMA = {
    "kernel": "softmax",
    "shape": {"batch": 32, "seq_len": 2048, "hidden": 768},
    "knobs": [...]
}
```

2. **Update SMEM estimation**:
```python
elif kernel == "softmax":
    return config["block_rows"] * config["vector_width"] * 4
```

3. **Update execute_configs**:
```python
if state["kernel"] == "softmax":
    # Prepare inputs
    # Call softmax kernel
    # Benchmark
```

---

## 📚 KEY FILES REFERENCE

### gpu_kernel_tuner_integrated.py (Main File)

**Lines 1-50**: Imports, LLM setup, state definition  
**Lines 51-120**: Validation functions  
**Lines 121-220**: Expert nodes (3 experts)  
**Lines 221-250**: Validation node  
**Lines 251-380**: Execution node (benchmarking)  
**Lines 381-430**: Analysis node  
**Lines 431-470**: Graph construction  
**Lines 471-520**: Schema definition  
**Lines 521-560**: Main execution  

### triton_matmul.py

**Lines 1-40**: Triton kernel definition  
**Lines 41-60**: Wrapper function  
**Lines 61-80**: Testing/validation  

### metrics_evaluator.py

**Lines 1-50**: MetricsEvaluator class  
**Lines 51-100**: Evaluation methods  

---

This context document provides everything needed for debugging the GPU Kernel Autotuner project!