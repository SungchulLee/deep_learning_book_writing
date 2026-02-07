# Latency Requirements

## Overview

Financial applications impose stringent latency requirements that directly impact profitability and risk management. Model deployment in quantitative finance requires understanding the latency budget, measurement methodology, and optimization strategies specific to financial systems.

## Latency Budgets by Application

| Application | Target Latency | Measurement |
|------------|---------------|-------------|
| High-Frequency Trading | < 1 ms | p99 |
| Algorithmic Trading | < 10 ms | p95 |
| Real-Time Risk | < 100 ms | p99 |
| Portfolio Optimization | < 1 s | p95 |
| Batch Risk Reporting | < 1 min | mean |
| End-of-Day Processing | < 1 hour | total |

## Latency Decomposition

$$\text{Total Latency} = t_{\text{network}} + t_{\text{preprocess}} + t_{\text{inference}} + t_{\text{postprocess}} + t_{\text{network\_return}}$$

For a trading signal pipeline:
```
Market Data (0.1ms) → Feature Calc (0.5ms) → Model (2ms) → Signal (0.2ms) → OMS (0.5ms)
                                                                            Total: ~3.3ms
```

## Measurement

```python
import time
import numpy as np

class LatencyTracker:
    def __init__(self, name='inference'):
        self.name = name
        self.latencies = []
    
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self
    
    def __exit__(self, *args):
        elapsed_ns = time.perf_counter_ns() - self.start
        self.latencies.append(elapsed_ns / 1e6)  # Convert to ms
    
    def report(self):
        arr = np.array(self.latencies)
        return {
            'mean_ms': np.mean(arr),
            'p50_ms': np.percentile(arr, 50),
            'p95_ms': np.percentile(arr, 95),
            'p99_ms': np.percentile(arr, 99),
            'max_ms': np.max(arr),
            'std_ms': np.std(arr),
        }
```

## Optimization Priorities

1. **Quantization**: INT8 reduces inference time 2-4×
2. **TorchScript + freeze**: 1.5-2× speedup from graph optimization
3. **Batch size 1 optimization**: Critical for real-time (avoid batch overhead)
4. **ONNX Runtime**: Often faster than native PyTorch for inference
5. **TensorRT**: Maximum GPU performance for NVIDIA hardware

## References

1. Aldridge, I. "High-Frequency Trading: A Practical Guide." Wiley, 2013.
