import time
import statistics
from dataclasses import dataclass
from typing import Callable, Any, Tuple
import torch

@dataclass
class TimingResult:
    median_ms: float
    p50_ms: float
    p90_ms: float
    std_ms: float
    iters: int

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_callable(callable_fn: Callable[[], Any], *, warmup: int = 5, iters: int = 50) -> TimingResult:
    # Warm-up
    for _ in range(warmup):
        callable_fn()
    _cuda_sync()

    times = []
    for _ in range(iters):
        _cuda_sync()
        t0 = time.perf_counter()
        callable_fn()
        _cuda_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms

    times.sort()
    median = statistics.median(times)
    p90 = times[int(0.9 * (len(times)-1))]
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return TimingResult(median, median, p90, std, iters)

def compare_against_reference(
    produce_out: Callable[[], torch.Tensor],
    produce_ref: Callable[[], torch.Tensor],
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2
) -> Tuple[bool, float]:
    """Returns (is_correct, max_abs_diff)"""
    with torch.no_grad():
        out = produce_out()
        ref = produce_ref()
        max_abs = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    return ok, max_abs

def speedup_ms(baseline_ms: float, new_ms: float) -> float:
    return baseline_ms / new_ms

if __name__ == "__main__":

    # Example wiring with the Triton function
    import triton_matmul as tmm

    torch.manual_seed(0)
    device = "cuda"
    M = N = K = 2048
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

    def run_triton():
        return tmm.triton_matmul(a, b)

    # Baseline using torch.mm in fp16→fp32
    def run_ref():
        return (a.float() @ b.float()).half()

    # Timing
    triton_time = time_callable(lambda: run_triton())
    ref_time = time_callable(lambda: run_ref())

    # Correctness
    ok, max_abs = compare_against_reference(run_triton, run_ref)

    print({
        "triton_ms_median": round(triton_time.median_ms, 3),
        "torch_ms_median": round(ref_time.median_ms, 3),
        "correct": ok,
        "max_abs_diff": max_abs,
        "speedup_vs_torch": round(speedup_ms(ref_time.median_ms, triton_time.median_ms), 3)
    })
