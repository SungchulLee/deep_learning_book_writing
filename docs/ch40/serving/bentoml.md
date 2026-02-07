# BentoML

## Overview

BentoML is an open-source framework for building production-ready AI services. It provides a Python-first approach to model serving with built-in support for containerization, batching, and deployment to cloud platforms. BentoML excels at simplifying the path from notebook to production.

## Core Concepts

BentoML organizes deployment around three concepts:

- **Service**: Python class defining the API
- **Bento**: Packaged artifact (model + code + dependencies)
- **Deployment**: Running service (local, Docker, cloud)

## Defining a Service

```python
import bentoml
import torch
import numpy as np
from bentoml.io import NumpyNdarray, JSON

# Save model to BentoML model store
model = torch.jit.load("model.pt")
bentoml.torchscript.save_model("alpha_model", model)

# Define service
runner = bentoml.torchscript.get("alpha_model:latest").to_runner()
svc = bentoml.Service("alpha_service", runners=[runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(features: np.ndarray):
    result = await runner.async_run(
        torch.tensor(features, dtype=torch.float32)
    )
    return {"prediction": result.numpy().tolist()}
```

## Building and Deploying

```bash
# Build Bento
bentoml build

# Containerize
bentoml containerize alpha_service:latest

# Run locally
bentoml serve alpha_service:latest

# Deploy to cloud
bentoml deploy alpha_service:latest --platform aws-lambda
```

## Adaptive Batching

BentoML automatically batches requests:

```python
@svc.api(
    input=NumpyNdarray(),
    output=JSON(),
    batch=True,  # Enable adaptive batching
)
async def predict_batch(features_batch: list):
    batch_tensor = torch.tensor(
        np.stack(features_batch), dtype=torch.float32
    )
    results = await runner.async_run(batch_tensor)
    return [{"prediction": r.tolist()} for r in results]
```

## Best Practices

- **Use runners** for parallel model execution
- **Enable adaptive batching** for throughput optimization
- **Version models** in the BentoML model store for reproducibility
- **Use Docker** for consistent deployment across environments
- **Implement health checks** and monitoring endpoints

## References

1. BentoML Documentation: https://docs.bentoml.com/
2. BentoML PyTorch Guide: https://docs.bentoml.com/en/latest/frameworks/pytorch.html
