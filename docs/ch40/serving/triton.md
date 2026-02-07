# NVIDIA Triton Inference Server

## Overview

NVIDIA Triton Inference Server is an enterprise-grade serving platform that supports multiple frameworks (PyTorch, TensorFlow, ONNX, TensorRT) and provides advanced features like dynamic batching, model ensembles, and GPU scheduling. It is the standard choice for high-throughput GPU inference in production.

## Architecture

```
Client (HTTP/gRPC) → Triton Server → Model Repository
                          ↓
                    Scheduler (Dynamic Batching)
                          ↓
                    Backend (PyTorch/ONNX/TensorRT)
                          ↓
                    GPU Execution
```

## Model Repository Structure

```
model_repository/
├── alpha_model/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
└── risk_model/
    ├── config.pbtxt
    └── 1/
        └── model.pt
```

## Model Configuration

```protobuf
# config.pbtxt
name: "alpha_model"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [50]
  }
]

output [
  {
    name: "prediction"
    data_type: TYPE_FP32
    dims: [1]
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
```

## Python Client

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Create input
features = np.random.randn(1, 50).astype(np.float32)
inputs = [httpclient.InferInput("features", features.shape, "FP32")]
inputs[0].set_data_from_numpy(features)

# Request output
outputs = [httpclient.InferRequestedOutput("prediction")]

# Infer
result = client.infer("alpha_model", inputs, outputs=outputs)
prediction = result.as_numpy("prediction")
```

## Launching Triton

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

## Best Practices

- **Use TensorRT backend** for maximum GPU performance
- **Configure dynamic batching** to balance latency and throughput
- **Use model ensembles** for multi-stage pipelines (preprocessing → inference → postprocessing)
- **Monitor with Prometheus** metrics endpoint (port 8002)
- **Version models** for zero-downtime updates

## References

1. Triton Documentation: https://docs.nvidia.com/deeplearning/triton-inference-server/
2. Triton Model Configuration: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_configuration.html
