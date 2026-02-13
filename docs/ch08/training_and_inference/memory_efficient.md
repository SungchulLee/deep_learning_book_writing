# Memory-Efficient Training

## Mixed Precision

Use BF16 for forward/backward passes with FP32 master weights. BF16 is preferred over FP16 for transformers due to its larger exponent range, which avoids overflow in attention logits.

```python
with autocast(dtype=torch.bfloat16):
    loss = model(batch)
scaler.scale(loss).backward()
```

## Gradient Checkpointing

Trade compute for memory by recomputing activations during backward: reduces memory from $O(L)$ to $O(\sqrt{L})$ layers, at ~33% compute overhead.

```python
from torch.utils.checkpoint import checkpoint
x = checkpoint(self._forward, x, use_reentrant=False)
```

## Memory Budget

For AdamW in mixed precision: parameters FP16 ($2P$), gradients FP16 ($2P$), optimizer states FP32 ($12P$: params + $m$ + $v$). Total: ~$16P$ bytes. A 1B parameter model needs ~16 GB before activations.

## FSDP

Fully Sharded Data Parallel shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds $1/N$ of the model state:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
```
