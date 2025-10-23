import torch
import csv
from triton_matmul import matmul_triton, time_pytorch, check_correctness, time_triton


def optimize_simple(M=2048, K=2048, N=2048, max_trials=20):
    print("=" * 70)
    print(f"🚀 SIMPLE TRITON OPTIMIZER")
    print("=" * 70)
    print(f"Matrix: {M}×{K} @ {K}×{N}")
    print(f"Max trials: {max_trials}")
    print("=" * 70)
    
    print("\nPreparing test matrices...")
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    print("Computing PyTorch baseline...")
    pytorch_result = torch.matmul(A, B)
    baseline_time = time_pytorch(A, B, warmup=10, measure=50)
    print(f"✅ Baseline: {baseline_time:.2f} ms")
    
    configs = []
    for block_m in [64, 128]:
        for block_n in [64, 128]:
            for block_k in [32, 64]:
                configs.append({
                    'block_m': block_m,
                    'block_n': block_n,
                    'block_k': block_k,
                    'num_warps': 4,
                    'num_stages': 2
                })
    
    best_config = None
    best_time = float('inf')
    
    with open('results_simple.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'trial', 'block_m', 'block_n', 'block_k', 
            'num_warps', 'num_stages',
            'time_ms', 'std_ms', 'speedup', 'correct'
        ])
        writer.writeheader()
        
        print(f"\n{'='*70}")
        print("Starting optimization...")
        print(f"{'='*70}\n")
        
        for trial, config in enumerate(configs[:max_trials]):
            print(f"Trial {trial+1}/{min(max_trials, len(configs))}")
            print(f"  Config: {config}")
            
            result, time_ms, std_ms = time_triton(
                A, B,
                block_m=config['block_m'],
                block_n=config['block_n'],
                block_k=config['block_k'],
                num_warps=config['num_warps'],
                num_stages=config['num_stages'],
                warmup=5,
                measure=50
            )
            
            correct, max_error = check_correctness(result, pytorch_result)
            
            if not correct:
                print(f"  ❌ INCORRECT (error: {max_error:.6f}) - skipping")
                writer.writerow({
                    'trial': trial + 1,
                    **config,
                    'time_ms': time_ms,
                    'std_ms': std_ms,
                    'speedup': 0,
                    'correct': False
                })
                continue
            
            speedup = baseline_time / time_ms
            
            writer.writerow({
                'trial': trial + 1,
                **config,
                'time_ms': time_ms,
                'std_ms': std_ms,
                'speedup': speedup,
                'correct': True
            })
            
            if time_ms < best_time * 0.98:
                improvement = ((best_time - time_ms) / best_time * 100) if best_time != float('inf') else 0
                best_time = time_ms
                best_config = config.copy()
                print(f"  ✅ NEW BEST! {time_ms:.2f}±{std_ms:.2f} ms "
                      f"({speedup:.2f}×{f', +{improvement:.1f}%' if improvement > 0 else ''})")
            else:
                print(f"     {time_ms:.2f}±{std_ms:.2f} ms ({speedup:.2f}×)")
    
    print(f"\n{'='*70}")
    print("🎉 OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    
    if best_config:
        speedup = baseline_time / best_time
        print(f"Best config: {best_config}")
        print(f"Best time: {best_time:.2f} ms")
        print(f"Baseline: {baseline_time:.2f} ms")
        print(f"Speedup: {speedup:.2f}×")
        print(f"\n📊 Results saved to results_simple.csv")
    else:
        print("❌ No valid configuration found!")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    optimize_simple()