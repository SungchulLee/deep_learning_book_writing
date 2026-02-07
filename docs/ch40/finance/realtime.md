# Real-Time Systems

## Overview

Real-time model inference for financial applications requires deterministic latency, fault tolerance, and the ability to process market data streams with microsecond-level timing requirements. This section covers the architecture patterns and implementation strategies for deploying deep learning models in real-time financial systems.

## Architecture Patterns

### Sidecar Pattern
```
Market Data Feed → Feature Service → Model Sidecar → Trading Engine
                        ↑                    ↑
                   Feature Store        Model Registry
```

### Pre-computed Signals
```
Model (batch, T-1) → Signal Cache → Trading Engine (real-time, T)
```

Pre-compute model outputs using end-of-day data, serve cached predictions during trading hours for sub-millisecond response.

## Implementation

```python
import torch
import time
from collections import OrderedDict

class RealTimeModelService:
    """Low-latency model serving for trading systems."""
    
    def __init__(self, model_path, warmup_iterations=1000):
        # Load optimized model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Pre-allocate tensors (avoid allocation during inference)
        self.input_buffer = torch.zeros(1, 50)
        self.output_buffer = torch.zeros(1, 1)
        
        # Warmup (JIT compilation, cache warming)
        self._warmup(warmup_iterations)
    
    def _warmup(self, n):
        for _ in range(n):
            with torch.no_grad():
                self.model(self.input_buffer)
    
    def predict(self, features):
        """Ultra-low-latency prediction."""
        # Copy into pre-allocated buffer (avoid tensor creation)
        self.input_buffer.copy_(features)
        
        with torch.no_grad():
            self.output_buffer = self.model(self.input_buffer)
        
        return self.output_buffer


class SignalCache:
    """LRU cache for pre-computed trading signals."""
    
    def __init__(self, capacity=10000):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

## Best Practices

- **Pre-allocate all tensors** to avoid garbage collection pauses
- **Use TorchScript** with `optimize_for_inference` for deterministic performance
- **Warm up models** before market open to avoid cold-start latency spikes
- **Implement circuit breakers** to fall back to simple models under stress
- **Co-locate model service** with trading engine to minimize network latency
- **Monitor tail latency** (p99, p99.9)—mean latency is insufficient for trading

## References

1. Patterson, S. "Dark Pools: The Rise of A.I. Trading Machines." Crown Business, 2012.
