# DeepSpeed

## Overview

DeepSpeed is Microsoft's deep learning optimization library that enables training of very large models through ZeRO (Zero Redundancy Optimizer), mixed precision, and efficient communication. It integrates with PyTorch and supports models with billions to trillions of parameters.

## ZeRO Stages

| Stage | What is Partitioned | Memory Reduction |
|-------|-------------------|-----------------|
| ZeRO-1 | Optimizer states | 4× |
| ZeRO-2 | + Gradients | 8× |
| ZeRO-3 | + Parameters | Linear with #GPUs |

## Basic Configuration

```json
{
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "contiguous_gradients": true,
        "overlap_comm": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4, "weight_decay": 0.01}
    }
}
```

## Integration

```python
import deepspeed

model, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

for data, target in dataloader:
    loss = model(data, target)
    model.backward(loss)
    model.step()
```

## References

1. DeepSpeed Documentation: https://www.deepspeed.ai/
2. ZeRO Paper: https://arxiv.org/abs/1910.02054
