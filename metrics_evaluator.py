import time
import statistics
from dataclasses import dataclass
from typing import Callable, Any, Tuple, Dict
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
    # Warmup
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

class MetricsEvaluator:
    """
    Simple scoring wrapper expected by gpu_kernel_tuner.py.
    .evaluate(sample_dict) -> dict with a 'score' key (lower is better).
    """
    def __init__(self, w_med: float = 0.7, w_p90: float = 0.3, penalty: float = 1e6):
        self.w_med = w_med
        self.w_p90 = w_p90
        self.penalty = penalty

    def evaluate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # sample keys expected by your tuner: median_ms, p90_ms, correct, max_abs_diff, config
        correct = bool(sample.get("correct", False))
        median_ms = float(sample.get("median_ms", float("inf")))
        p90_ms = float(sample.get("p90_ms", float("inf")))
        max_abs = float(sample.get("max_abs_diff", float("inf")))

        if not correct or not torch.isfinite(torch.tensor([median_ms, p90_ms])).all():
            score = self.penalty
        else:
            # Weighted latency; add a tiny numerical-stability term with error
            score = self.w_med * median_ms + self.w_p90 * p90_ms + 1e3 * float(max_abs)

        return {
            "score": score,
            "correct": correct,
            "median_ms": median_ms,
            "p90_ms": p90_ms,
            "max_abs_diff": max_abs,
        }

if __name__ == "__main__":
    # Optional quick self-check
    print("MetricsEvaluator ready.")
