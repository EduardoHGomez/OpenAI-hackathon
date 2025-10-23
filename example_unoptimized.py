"""
EXAMPLE PYTORCH TRAINING CODE - UNOPTIMIZED
============================================

This is an intentionally unoptimized training loop for demonstration.
The optimizer will improve: speed, memory, accuracy, stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train():
    """Unoptimized training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Suboptimal optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Fake data
    batch_size = 32  # Small batch
    num_batches = 100
    input_size = 784
    num_classes = 10
    
    # Training loop
    model.train()
    total_samples = 0
    losses = []
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        # Generate fake batch
        x = torch.randn(batch_size, input_size).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_samples += batch_size
        losses.append(loss.item())
    
    elapsed = time.time() - start_time
    
    return {
        "loss": sum(losses) / len(losses),
        "samples": total_samples,
        "elapsed": elapsed
    }


def validate():
    """Simple validation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(20):
            x = torch.randn(32, 784).to(device)
            y = torch.randint(0, 10, (32,)).to(device)
            
            output = model(x)
            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return correct / total


def get_metrics():
    """Measure all metrics"""
    import statistics
    
    # Speed measurement
    results = []
    for _ in range(3):
        result = train()
        results.append(result)
    
    avg_elapsed = sum(r['elapsed'] for r in results) / len(results)
    samples_per_second = results[0]['samples'] / avg_elapsed
    
    # Accuracy
    accuracy = validate()
    
    # Memory
    torch.cuda.reset_peak_memory_stats()
    _ = train()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Stability
    losses = [r['loss'] for r in results]
    variance = statistics.variance(losses) if len(losses) > 1 else 0.0
    
    # Cost efficiency
    throughput_per_gb = samples_per_second / (peak_memory_mb / 1024)
    
    return {
        "tokens_per_second": samples_per_second,  # Using samples as proxy
        "validation_accuracy": accuracy,
        "peak_gpu_memory_mb": peak_memory_mb,
        "run_variance": variance,
        "throughput_per_gb": throughput_per_gb
    }


if __name__ == "__main__":
    metrics = get_metrics()
    print("Baseline Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")