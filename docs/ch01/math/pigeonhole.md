# Pigeonhole Principle

The pigeonhole principle is a simple but powerful counting argument. In machine learning, it underpins hash collision analysis, proves existence of duplicate representations in finite-precision systems, and bounds the capacity of discrete models.

## Definition

If $n$ items are placed into $m$ containers and $n > m$, then at least one container holds more than one item. More precisely:

$$
n \text{ items into } m \text{ containers} \implies \exists \text{ container with } \geq \lceil n/m \rceil \text{ items}
$$

## Explanation

The principle is a direct consequence of counting. If every container held at most one item, we could accommodate at most $m$ items, contradicting $n > m$.

Applications in deep learning and computing:

- **Hash collisions**: When mapping $n$ items to $m$ hash buckets with $n > m$, collisions are unavoidable. This affects hash-based embedding tables used in recommendation systems.
- **Finite precision**: With 32-bit floats, there are $2^{32}$ representable values. Any mapping of more than $2^{32}$ real numbers to float32 must produce collisions (rounding). This is why numerical stability matters in deep learning.
- **Quantization**: When quantizing neural network weights from float32 to int8 (256 values), the pigeonhole principle guarantees that many distinct weights map to the same quantized value.

## Examples

```python
import torch

# Pigeonhole in quantization: many float values map to same int8 bucket
weights = torch.randn(1000)  # 1000 distinct float32 values
num_int8_buckets = 256

# Simulate uniform quantization to int8
w_min, w_max = weights.min(), weights.max()
scale = (w_max - w_min) / (num_int8_buckets - 1)
quantized = torch.round((weights - w_min) / scale).to(torch.int32)

unique_buckets = quantized.unique().numel()
print(f"Original values: {weights.numel()}")
print(f"Available int8 buckets: {num_int8_buckets}")
print(f"Occupied buckets: {unique_buckets}")
print(f"Pigeonhole: at least {weights.numel()} / {num_int8_buckets} "
      f"= {weights.numel() // num_int8_buckets} values share a bucket")

# Verify: find the most crowded bucket
counts = torch.zeros(num_int8_buckets, dtype=torch.int32)
for q in quantized:
    counts[q.item()] += 1
print(f"Max bucket occupancy: {counts.max().item()}")
print(f"Theoretical minimum max: {-(-(weights.numel()) // num_int8_buckets)}")
```
