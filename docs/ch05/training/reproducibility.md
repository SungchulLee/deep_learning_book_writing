# Reproducibility

## Overview

Reproducibility ensures that training runs produce identical (or near-identical) results given the same code, data, and configuration. This is essential for debugging, experiment comparison, and scientific rigor.

## Sources of Non-Determinism

Non-determinism in PyTorch arises from multiple sources:

- **Random number generators**: Weight initialization, data shuffling, augmentation, dropout.
- **CUDA operations**: Certain GPU kernels use non-deterministic algorithms for speed.
- **Multi-threaded operations**: Thread scheduling affects reduction order in floating-point arithmetic.
- **cuDNN autotuning**: Selects the fastest algorithm variant, which may differ across runs.

## Setting Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
```

Setting seeds controls Python's `random`, NumPy, and PyTorch RNGs. However, seeds alone are insufficient for full determinism on GPU.

## Deterministic Operations

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

- **`cudnn.deterministic`**: Forces cuDNN to use deterministic convolution algorithms (slower).
- **`cudnn.benchmark`**: When `True`, cuDNN benchmarks multiple algorithms and selects the fastest. Disable for reproducibility.
- **`use_deterministic_algorithms`**: Raises an error if any operation uses a non-deterministic implementation.

## DataLoader Reproducibility

Multi-process data loading introduces additional randomness. Use a generator and worker init function:

```python
g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=4, generator=g,
                    worker_init_fn=seed_worker)
```

## Complete Reproducibility Setup

```python
def make_reproducible(seed=42):
    """Configure all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Optional: raise error on non-deterministic ops
    # torch.use_deterministic_algorithms(True)

make_reproducible(42)
```

## Performance vs. Reproducibility

Deterministic mode has a performance cost. In practice:

- **Development/debugging**: Use full deterministic mode.
- **Hyperparameter search**: Seeds are sufficient; exact reproducibility is less critical.
- **Final experiments**: Use deterministic mode for reported results.
- **Production training**: Prioritize speed; log seeds and configuration for approximate reproducibility.

## Logging for Reproducibility

Beyond seeds, log everything needed to reproduce a run:

```python
config = {
    'seed': 42,
    'model': 'ResNet18',
    'optimizer': 'AdamW',
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
}
```

## Key Takeaways

- Set seeds for Python, NumPy, and PyTorch RNGs.
- Enable `cudnn.deterministic` and disable `cudnn.benchmark` for exact GPU reproducibility.
- Use generator and worker init functions for DataLoader reproducibility.
- Full determinism has a performance cost—apply it during debugging and final experiments.
