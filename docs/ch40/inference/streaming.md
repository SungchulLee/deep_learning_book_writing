# Streaming Inference

## Overview

Streaming inference processes continuous data flows in real time, producing predictions as new data arrives. This pattern is essential for financial applications including live market data processing, real-time risk monitoring, and algorithmic trading signal generation.

## Architecture

```
Data Source → Preprocessing → Model Inference → Postprocessing → Action
   (feed)      (normalize)      (predict)        (threshold)     (trade)
```

## Stateful Streaming

For models that maintain state across time steps (RNNs, online learning):

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple
from collections import deque

class StreamingPredictor:
    """Stateful streaming inference with rolling window."""
    
    def __init__(self, model: nn.Module, window_size: int = 60,
                 device: str = 'cpu'):
        self.model = model.eval().to(device)
        self.device = device
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.hidden_state = None
    
    def process(self, new_data: torch.Tensor) -> Optional[torch.Tensor]:
        """Process a single new observation."""
        self.buffer.append(new_data)
        
        if len(self.buffer) < self.window_size:
            return None  # Not enough data yet
        
        # Create input tensor from buffer
        window = torch.stack(list(self.buffer)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(window)
        
        return prediction.cpu()
    
    def reset(self):
        """Reset state for new session."""
        self.buffer.clear()
        self.hidden_state = None


class RealTimeFeatureNormalizer:
    """Online normalization for streaming features."""
    
    def __init__(self, num_features: int, momentum: float = 0.01):
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.momentum = momentum
        self.count = 0
    
    def update_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.count += 1
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        
        self.running_mean = (1 - self.momentum) * self.running_mean + \
                           self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + \
                          self.momentum * batch_var
        
        return (x - self.running_mean) / (self.running_var.sqrt() + 1e-8)
```

## Latency-Optimized Pipeline

```python
import threading
import queue

class PipelinedInference:
    """Three-stage pipelined inference for minimum latency."""
    
    def __init__(self, preprocessor, model, postprocessor):
        self.preprocessor = preprocessor
        self.model = model.eval()
        self.postprocessor = postprocessor
        
        self.preprocess_queue = queue.Queue(maxsize=100)
        self.inference_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
    
    def start(self):
        threading.Thread(target=self._preprocess_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        threading.Thread(target=self._postprocess_loop, daemon=True).start()
    
    def _preprocess_loop(self):
        while True:
            raw = self.preprocess_queue.get()
            processed = self.preprocessor(raw)
            self.inference_queue.put(processed)
    
    def _inference_loop(self):
        while True:
            tensor = self.inference_queue.get()
            with torch.no_grad():
                output = self.model(tensor)
            self.output_queue.put(output)
    
    def _postprocess_loop(self):
        while True:
            prediction = self.output_queue.get()
            result = self.postprocessor(prediction)
            # Route result to downstream consumer
```

## Best Practices

- **Warm up the model** before processing live data to avoid cold-start latency
- **Use fixed-size buffers** with `deque(maxlen=N)` to bound memory usage
- **Pre-compile models** with TorchScript or ONNX for consistent latency
- **Monitor queue depths** to detect processing bottlenecks
- **Implement graceful degradation** — skip old data when falling behind

## References

1. PyTorch Real-Time Inference: https://pytorch.org/tutorials/intermediate/realtime_rpi.html
