# TorchScript

## Overview

TorchScript serializes PyTorch models into a format that can be loaded and executed in C++ (or other non-Python environments) without a Python runtime. This enables deployment in production systems, mobile devices, and embedded applications.

## Two Approaches

**Tracing** records operations executed during a forward pass with example inputs:

```python
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')
```

Tracing works for models with fixed control flow. It does not capture data-dependent branches.

**Scripting** analyzes the Python source code and compiles it:

```python
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

Scripting handles data-dependent control flow (if/else, loops based on input) but requires the code to be compatible with TorchScript's subset of Python.

## Loading in Python

```python
loaded = torch.jit.load('model_traced.pt')
output = loaded(torch.randn(1, 3, 224, 224))
```

## Loading in C++

```cpp
#include <torch/script.h>

torch::jit::script::Module module = torch::jit::load("model_traced.pt");
auto input = torch::randn({1, 3, 224, 224});
auto output = module.forward({input}).toTensor();
```

## Tracing vs. Scripting

| Aspect | Tracing | Scripting |
|---|---|---|
| Control flow | Fixed only | Dynamic |
| Ease of use | Simpler | Requires TorchScript-compatible code |
| Debugging | Harder (silent errors) | Better error messages |
| Recommended for | Standard CNNs, fixed architectures | Models with conditionals, loops |

## Optimization

TorchScript models can be optimized after export:

```python
traced = torch.jit.trace(model, example_input)
optimized = torch.jit.optimize_for_inference(traced)
optimized.save('model_optimized.pt')
```

## Key Takeaways

- TorchScript enables Python-free deployment via tracing or scripting.
- Use tracing for models with fixed control flow; scripting for dynamic control flow.
- Validate outputs match the original PyTorch model before deployment.
- `optimize_for_inference` can improve runtime performance of exported models.
