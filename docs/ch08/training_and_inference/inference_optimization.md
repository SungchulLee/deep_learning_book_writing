# Inference Optimization

## KV Cache

Store key/value projections from previous tokens during autoregressive generation. Reduces per-step complexity from $O(n^2 d)$ to $O(nd)$, but cache memory grows as $2Lnd$ bytes.

```python
if kv_cache is not None:
    k = torch.cat([kv_cache[0], k], dim=1)
    v = torch.cat([kv_cache[1], v], dim=1)
new_cache = (k, v)
```

## torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

Fuses operations for significant speedup. Use fixed sequence lengths (pad to constant) to avoid recompilation.

## Quantization

Post-training quantization: INT8 gives 2-4x speedup with minimal quality loss; INT4 gives 4-8x compression with noticeable but often acceptable loss.

## Flash Attention

FlashAttention reorders computation to minimize HBM reads/writes, achieving 2-4x speedup without approximation. Available in PyTorch 2.0+ via `F.scaled_dot_product_attention`.

## Batched Inference

Static batching (pad to max length) or continuous batching (insert new requests as old ones finish, as in vLLM and TGI) for throughput-oriented serving.
