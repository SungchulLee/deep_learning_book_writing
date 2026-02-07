# TorchServe

## Overview

TorchServe is PyTorch's official model serving framework, providing production-grade inference with features like multi-model serving, dynamic batching, model versioning, and monitoring. It is designed specifically for PyTorch models and integrates with the broader PyTorch ecosystem.

## Architecture

```
Client Request → Frontend (HTTP/gRPC) → Backend Workers → Model Inference
                     ↓                       ↓
              Request Batching          Model Management
                     ↓                       ↓
              Load Balancing            Health Monitoring
```

## Model Archiving

Package a model for TorchServe using `torch-model-archiver`:

```bash
# Install
pip install torchserve torch-model-archiver

# Archive model
torch-model-archiver \
    --model-name alpha_model \
    --version 1.0 \
    --serialized-file model.pt \
    --handler custom_handler.py \
    --export-path model_store/
```

## Custom Handler

```python
import torch
import json
import logging
from ts.torch_handler.base_handler import BaseHandler

class AlphaModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()
        self.initialized = True
    
    def preprocess(self, data):
        inputs = []
        for row in data:
            body = row.get("body") or row.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)
            features = torch.tensor(body["features"], dtype=torch.float32)
            inputs.append(features)
        return torch.stack(inputs)
    
    def inference(self, inputs):
        with torch.no_grad():
            return self.model(inputs)
    
    def postprocess(self, outputs):
        return [{"prediction": p.tolist()} for p in outputs]
```

## Starting TorchServe

```bash
torchserve --start \
    --model-store model_store \
    --models alpha_model=alpha_model.mar \
    --ts-config config.properties
```

Configuration (`config.properties`):
```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
job_queue_size=100
batch_size=32
max_batch_delay=100
```

## Best Practices

- **Use model archiver** for reproducible deployments
- **Configure batch size** based on latency requirements and hardware
- **Enable metrics** endpoint for Prometheus/Grafana monitoring
- **Use model versioning** for safe rollouts and A/B testing
- **Set resource limits** per model to prevent one model from starving others

## References

1. TorchServe Documentation: https://pytorch.org/serve/
2. TorchServe GitHub: https://github.com/pytorch/serve
