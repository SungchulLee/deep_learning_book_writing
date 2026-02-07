# Continuous Batching

## Learning Objectives

- Understand static vs. continuous batching
- Quantify the throughput improvement
- Implement iteration-level scheduling

## Static Batching Problem

In **static batching**, a batch of requests is processed together, and the entire batch waits for the longest sequence to finish:

```
Request 1: [====]____________  (done at step 4, waits until step 12)
Request 2: [============]     (done at step 12)
Request 3: [========]____     (done at step 8, waits until step 12)

GPU utilization: (4+12+8) / (12*3) = 67%
```

## Continuous Batching

**Continuous (iteration-level) batching** allows new requests to enter the batch as soon as any request completes:

```
Step 1-4:   [R1, R2, R3] → R1 completes, R4 enters
Step 5-8:   [R4, R2, R3] → R3 completes, R5 enters
Step 9-12:  [R4, R2, R5] → R2 completes, R6 enters
...

GPU utilization: ~95%+
```

## Throughput Impact

| Batching Strategy | Throughput (tokens/s) | GPU Utilization |
|------------------|----------------------|-----------------|
| No batching | 30-50 | <10% |
| Static (batch=32) | 500-800 | 40-60% |
| Continuous | 1500-3000 | 80-95% |

Continuous batching typically provides **2-4x throughput improvement** over static batching.

## Implementation Concept

```python
class ContinuousBatchScheduler:
    def __init__(self, model, max_batch_size=64):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests = []
        self.waiting_queue = []

    def step(self):
        # Remove completed requests
        completed = [r for r in self.active_requests if r.is_done()]
        for r in completed:
            self.active_requests.remove(r)
            r.callback(r.get_output())

        # Add new requests from queue
        while (len(self.active_requests) < self.max_batch_size
               and self.waiting_queue):
            new_req = self.waiting_queue.pop(0)
            self.active_requests.append(new_req)

        # Run one decode step for all active requests
        if self.active_requests:
            self.model.decode_step(self.active_requests)
```

## Implementations

| Framework | Continuous Batching | Additional Features |
|-----------|-------------------|-------------------|
| vLLM | Yes | PagedAttention |
| TGI (HuggingFace) | Yes | Flash Attention |
| TensorRT-LLM | Yes | FP8, speculative |
| SGLang | Yes | RadixAttention |

## References

1. Yu, G., et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI*.
