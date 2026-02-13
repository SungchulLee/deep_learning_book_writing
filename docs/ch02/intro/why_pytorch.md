# Why PyTorch

## Learning Objectives

By the end of this section, you will be able to:

- Articulate why PyTorch has become the dominant framework for deep learning research and increasingly for production
- Understand the key design principles that distinguish PyTorch from alternatives
- Appreciate the role of dynamic computation graphs in model development
- Evaluate when PyTorch is the right choice for a project versus alternatives

---

## The Deep Learning Framework Landscape

Deep learning frameworks translate mathematical operations — matrix multiplications, convolutions, nonlinear activations — into efficient code that runs on CPUs, GPUs, and specialised accelerators. Without a framework, practitioners would need to implement backpropagation by hand, manage GPU memory manually, and write CUDA kernels for every new layer type. Frameworks handle all of this, letting researchers focus on model design and experimentation.

Several frameworks have shaped the field over the past decade. Theano (2007–2017) pioneered symbolic computation graphs for automatic differentiation. Caffe (2014) dominated early computer vision research. TensorFlow (2015) introduced production-grade infrastructure at Google scale. PyTorch (2016), developed by Meta AI (formerly Facebook AI Research), took a fundamentally different approach — one that prioritised the researcher's experience above all else.

Today PyTorch is the most widely used framework in deep learning research and is rapidly gaining ground in production deployment. Understanding *why* is essential context for the rest of this curriculum.

---

## Design Philosophy: Python First

PyTorch's central design principle is that the framework should feel like native Python rather than a domain-specific language embedded in Python. This manifests in several concrete ways.

### Eager Execution

PyTorch operations execute immediately, just like standard Python code. When you write `y = torch.matmul(W, x) + b`, the computation happens right there — you can print `y`, inspect its shape, set a breakpoint, or branch on its value. This is called **eager execution** (or **define-by-run**), and it stands in contrast to the **define-then-run** approach used by early TensorFlow (1.x), where operations built a static graph that was compiled and executed separately.

The practical impact is enormous. Debugging a PyTorch model uses the same tools — `print()`, `pdb`, IDE breakpoints — that any Python developer already knows. There is no graph compilation step, no separate session to manage, and no opaque error messages pointing to nodes in an abstract graph.

```python
import torch

# This is real Python — every line executes immediately
x = torch.randn(32, 784)
W = torch.randn(256, 784, requires_grad=True)
b = torch.zeros(256, requires_grad=True)

y = torch.matmul(x, W.T) + b       # runs now, result available immediately
print(y.shape)                       # torch.Size([32, 256])
print(y.mean().item())               # a real number, right now
```

### Dynamic Computation Graphs

Because PyTorch builds the computation graph on-the-fly during the forward pass, the graph can change from one input to the next. This is called a **dynamic computation graph** (DCG). Control flow — `if` statements, `for` loops, recursion — works naturally:

```python
class TreeLSTM(nn.Module):
    """An LSTM that operates on tree-structured inputs.
    The computation graph depends on the input tree's shape."""
    
    def forward(self, node):
        if node.is_leaf:
            return self.leaf_transform(node.value)
        
        # Number of children varies per node
        child_states = [self.forward(child) for child in node.children]
        return self.merge(child_states)
```

This would be cumbersome or impossible in a static graph framework without special control-flow primitives. In PyTorch, it is just Python.

### Pythonic API

PyTorch follows Python conventions wherever possible. Models are classes that inherit from `nn.Module`. Layers are attributes. The forward pass is a method. Data pipelines use iterators. This means that Python developers can read PyTorch code without learning a new abstraction:

```python
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

---

## Automatic Differentiation

At the heart of any deep learning framework is **automatic differentiation** (autograd) — the ability to compute gradients of a loss function with respect to every parameter in a model, no matter how deep.

PyTorch's autograd system records operations on tensors that have `requires_grad=True`, building a directed acyclic graph of operations. Calling `.backward()` on a scalar loss traverses this graph in reverse to compute gradients via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} = \frac{\partial \mathcal{L}}{\partial z_n} \cdot \frac{\partial z_n}{\partial z_{n-1}} \cdots \frac{\partial z_{k+1}}{\partial \theta_i}$$

Because the graph is built dynamically, the autograd engine handles arbitrary Python control flow, variable-length sequences, and even stochastic computation paths — all without special syntax.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x ** 2 - 5 * x + 3    # f(x) = x³ + 2x² - 5x + 3

y.backward()
print(x.grad)    # df/dx = 3x² + 4x - 5 = 12 + 8 - 5 = 15.0
```

This is covered in depth in [Section 1.4: Automatic Differentiation](../autograd/computational_graphs.md).

---

## Research Dominance

PyTorch's design choices have led to overwhelming adoption in the research community. The evidence is clear across multiple metrics.

### Publication Adoption

At top machine learning venues (NeurIPS, ICML, ICLR), PyTorch's share of framework mentions in published papers has grown from roughly 10% in 2017 to over 75% by 2023–2024. Major research labs — Meta AI, Google DeepMind, OpenAI, Anthropic, Stanford, Berkeley, MIT — use PyTorch as their primary framework for most research.

### Landmark Models Built in PyTorch

Many of the most influential models in recent deep learning history were developed and released in PyTorch:

| Model / System | Domain | Year |
|---|---|---|
| AlphaFold 2 | Protein structure prediction | 2021 |
| Stable Diffusion | Image generation | 2022 |
| LLaMA | Large language models | 2023 |
| Segment Anything (SAM) | Computer vision | 2023 |
| Whisper | Speech recognition | 2022 |
| DINO / DINOv2 | Self-supervised vision | 2021–2023 |

When a new technique appears in a paper, the reference implementation is almost always in PyTorch. This means that staying current with the literature requires fluency in the framework.

### Quantitative Finance Relevance

For quantitative finance specifically, PyTorch offers several advantages. Its dynamic graphs handle the variable-length sequences common in financial time series (trading days vary, markets close on different holidays). Its probability distributions module (`torch.distributions`) supports Bayesian approaches to uncertainty quantification — critical for risk management. And the research community's adoption means that state-of-the-art techniques (transformers for time series, normalising flows for density estimation, graph neural networks for portfolio construction) are available first and best-supported in PyTorch.

---

## Production Readiness

A common criticism of early PyTorch was that it was "only for research." This is no longer true. Several developments have made PyTorch a credible production framework.

### TorchScript and torch.compile

**TorchScript** allows models to be serialised into a representation that can run without Python, enabling deployment in C++ services, mobile devices, and edge hardware. **`torch.compile`** (introduced in PyTorch 2.0) uses graph capture and compiler backends to optimise models for inference speed — often matching or exceeding TensorFlow's static-graph performance.

```python
# One-line compilation for production inference
optimized_model = torch.compile(model)
```

### TorchServe

**TorchServe** is PyTorch's model serving framework, providing REST/gRPC endpoints, batching, model versioning, and metrics out of the box.

### ONNX Export

Models can be exported to the **Open Neural Network Exchange** (ONNX) format, enabling deployment in runtimes optimised for specific hardware (TensorRT for NVIDIA GPUs, CoreML for Apple devices, ONNX Runtime for cross-platform inference).

### Mobile and Edge

**PyTorch Mobile** (and its successor, **ExecuTorch**) support deployment on iOS and Android, bringing models to the edge.

---

## PyTorch vs Alternatives

No framework is best for every situation. Here is a fair comparison:

| Aspect | PyTorch | TensorFlow / Keras | JAX |
|---|---|---|---|
| **Execution** | Eager (default) | Eager (TF 2.x default) | Functional transforms |
| **Graph construction** | Dynamic | Dynamic (2.x) or static (1.x) | Traced via `jit` |
| **Debugging** | Standard Python tools | Standard Python tools (2.x) | Requires understanding of tracing |
| **Research adoption** | Dominant | Declining in research | Growing (Google Brain/DeepMind) |
| **Production tooling** | TorchServe, torch.compile | TF Serving, TFLite, TF.js | Limited (early stage) |
| **Ecosystem** | Hugging Face, timm, Lightning | TFHub, TF Extended | Flax, Haiku, Optax |
| **Learning curve** | Moderate | Low (Keras), moderate (raw TF) | Steep (functional paradigm) |

**TensorFlow/Keras** remains strong in production pipelines, especially at Google-scale infrastructure, and Keras offers the lowest barrier to entry for beginners. **JAX** appeals to researchers who want fine-grained control over compilation and vectorisation (e.g., for large-scale distributed training or custom scientific computing). For the intersection of research flexibility, ecosystem breadth, and growing production capability, **PyTorch is the clear default** — which is why this curriculum is built on it.

---

## Summary

PyTorch's success stems from a simple insight: make the framework feel like Python, and researchers will adopt it. Eager execution, dynamic computation graphs, and a Pythonic API lower the barrier between mathematical ideas and working code. The resulting network effects — massive ecosystem, dominant research adoption, first-to-market implementations — create a virtuous cycle that reinforces PyTorch's position.

For quantitative finance practitioners, PyTorch provides the flexibility to implement novel architectures (transformers, normalising flows, graph networks), the research community's latest techniques, and an increasingly viable path to production deployment.

---

## References

1. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS.
2. PyTorch Documentation. https://pytorch.org/docs/stable/
3. He, H. & Deng, S. (2023). "Papers with Code: Framework Trends." https://paperswithcode.com/trends
4. PyTorch 2.0 Release Notes. https://pytorch.org/blog/pytorch-2.0-release/
