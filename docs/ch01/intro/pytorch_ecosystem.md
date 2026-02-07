# PyTorch Ecosystem

## Learning Objectives

By the end of this section, you will be able to:

- Navigate the major libraries and tools in the PyTorch ecosystem
- Understand how domain-specific libraries (vision, NLP, audio) relate to core PyTorch
- Identify the right ecosystem tool for a given task
- Appreciate the role of the broader community in PyTorch's development

---

## Overview

PyTorch is not a monolithic framework — it is a **core tensor library** surrounded by a rich ecosystem of domain-specific libraries, training utilities, deployment tools, and community-maintained packages. Understanding this ecosystem is essential for productive deep learning work, because most practical projects rely on several ecosystem components working together.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Community & Third-Party                         │
│    Hugging Face · timm · PyTorch Lightning · Weights & Biases      │
├─────────────────────────────────────────────────────────────────────┤
│                    Domain Libraries (Official)                      │
│         torchvision · torchtext · torchaudio · torchdata           │
├─────────────────────────────────────────────────────────────────────┤
│                    Specialised Extensions                           │
│     torch.distributions · torch.fx · torch.compile · TorchScript   │
├─────────────────────────────────────────────────────────────────────┤
│                         Core PyTorch                                │
│           Tensors · Autograd · nn.Module · Optimizers              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core PyTorch

The core library provides the building blocks that everything else is built on:

| Component | Purpose | Key Modules |
|---|---|---|
| **Tensors** | Multi-dimensional arrays with GPU support | `torch.Tensor`, `torch.cuda` |
| **Autograd** | Automatic differentiation engine | `torch.autograd` |
| **Neural Networks** | Layers, containers, loss functions | `torch.nn` |
| **Optimizers** | SGD, Adam, learning rate schedulers | `torch.optim` |
| **Data Loading** | Datasets, DataLoaders, samplers | `torch.utils.data` |
| **Serialization** | Save/load models and tensors | `torch.save`, `torch.load` |

These are covered in detail throughout Chapter 1 and form the foundation for all subsequent chapters.

---

## Official Domain Libraries

PyTorch maintains several domain-specific libraries that provide datasets, pretrained models, and domain-specific transforms.

### torchvision

**torchvision** is the computer vision companion library, providing:

- **Datasets**: MNIST, CIFAR-10/100, ImageNet, COCO, VOC, and many more via `torchvision.datasets`
- **Pretrained models**: ResNet, VGG, EfficientNet, Vision Transformer, Swin Transformer, and detection models (Faster R-CNN, YOLO wrappers) via `torchvision.models`
- **Transforms**: Data augmentation and preprocessing pipelines via `torchvision.transforms` (v1) and `torchvision.transforms.v2` (the newer, more flexible API)
- **Utilities**: Image I/O, drawing bounding boxes, grid visualisation

```python
import torchvision
from torchvision import models, transforms

# Pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### torchaudio

**torchaudio** provides audio processing capabilities:

- **I/O**: Load and save audio in various formats (WAV, MP3, FLAC)
- **Transforms**: Spectrograms, MFCCs, resampling, and augmentations
- **Datasets**: LibriSpeech, VCTK, CommonVoice
- **Pretrained models**: Wav2Vec 2.0, HuBERT for speech recognition

### torchtext

**torchtext** handles text data processing:

- **Datasets**: IMDB, AG News, SQuAD, and other NLP benchmarks
- **Tokenizers**: Wrappers for common tokenisation strategies
- **Vocabularies**: Building and managing token-to-index mappings

!!! note
    For most NLP work today, the **Hugging Face Transformers** library (discussed below) has largely superseded torchtext's model and tokeniser components, though torchtext's dataset utilities remain useful.

### TorchRec

**TorchRec** provides primitives for building large-scale recommendation systems, including efficient embedding tables and distributed training support.

### TorchGeo

**TorchGeo** provides datasets, transforms, and pretrained models for geospatial data — satellite imagery, land cover classification, and related remote sensing tasks.

---

## Training and Experiment Management

### PyTorch Lightning

[**PyTorch Lightning**](https://lightning.ai/docs/pytorch/stable/) is a high-level wrapper that removes boilerplate from training loops while preserving full flexibility. It separates *what* to train (your `LightningModule`) from *how* to train it (the `Trainer`):

```python
import lightning as L

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = L.Trainer(max_epochs=10, accelerator='auto')
trainer.fit(model, train_dataloader)
```

Lightning handles distributed training, mixed precision, checkpointing, logging, and early stopping automatically.

### Weights & Biases (W&B)

[**Weights & Biases**](https://wandb.ai/) is an experiment tracking platform that logs metrics, hyperparameters, model artifacts, and system resource usage. It integrates with PyTorch natively and with Lightning via a single callback:

```python
import wandb

wandb.init(project="my-project")
wandb.log({"loss": loss.item(), "accuracy": acc})
```

### TensorBoard

PyTorch includes native **TensorBoard** support via `torch.utils.tensorboard.SummaryWriter`. TensorBoard provides real-time visualisation of training metrics, model graphs, embeddings, and profiling data:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, global_step)
writer.add_histogram('layer1/weights', model.fc1.weight, global_step)
```

---

## Model Hubs and Pretrained Models

### Hugging Face

[**Hugging Face**](https://huggingface.co/) has become the central hub for pretrained models and datasets across modalities. Its key PyTorch-integrated libraries:

- **Transformers**: Thousands of pretrained models (BERT, GPT-2, LLaMA, T5, Whisper, Vision Transformer, …) with a unified API
- **Datasets**: Standardised access to thousands of NLP, vision, and audio datasets
- **Accelerate**: Simplified distributed training and mixed precision
- **PEFT**: Parameter-efficient fine-tuning (LoRA, prefix tuning, adapters)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

### timm (PyTorch Image Models)

[**timm**](https://github.com/huggingface/pytorch-image-models) provides a vast collection of pretrained image classification models (900+) with a consistent interface. It is the go-to source for state-of-the-art vision backbones:

```python
import timm

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=10)
```

### PyTorch Hub

**PyTorch Hub** (`torch.hub`) is PyTorch's built-in model-sharing mechanism. Any GitHub repository can publish models that users load with a single line:

```python
model = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
```

---

## Deployment and Optimisation

### torch.compile (PyTorch 2.0+)

`torch.compile` is a one-line optimisation that captures the computation graph and compiles it using backends like TorchInductor, achieving significant speedups without code changes:

```python
model = torch.compile(model)  # That's it
```

Under the hood, it traces the model's operations, applies graph-level optimisations (operator fusion, memory planning), and generates optimised kernels.

### TorchScript

**TorchScript** converts Python models into a serialised, optimisable format that can run without Python. This is critical for deployment in C++ services, mobile apps, and environments where a Python runtime is unavailable:

```python
scripted = torch.jit.script(model)
scripted.save("model.pt")

# Load in C++ or Python without the original model code
loaded = torch.jit.load("model.pt")
```

### ONNX Export

**ONNX** (Open Neural Network Exchange) provides interoperability between frameworks and deployment runtimes:

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

The exported model can then run on NVIDIA TensorRT, Intel OpenVINO, Microsoft ONNX Runtime, Apple CoreML, and other optimised runtimes.

### TorchServe

[**TorchServe**](https://pytorch.org/serve/) is PyTorch's official model serving solution, providing REST/gRPC APIs, request batching, model versioning, A/B testing, and monitoring.

### ExecuTorch

**ExecuTorch** is the next-generation on-device inference framework for mobile and edge deployment, succeeding PyTorch Mobile.

---

## Distributed Training

### torch.distributed

PyTorch's built-in distributed package supports multi-GPU and multi-node training:

| Strategy | Use Case |
|---|---|
| **DistributedDataParallel (DDP)** | Data parallelism across GPUs/nodes |
| **Fully Sharded Data Parallel (FSDP)** | Memory-efficient training for large models |
| **Pipeline Parallelism** | Split model layers across GPUs |
| **Tensor Parallelism** | Split individual layers across GPUs |

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model.cuda(), device_ids=[local_rank])
```

### Hugging Face Accelerate

**Accelerate** provides a simpler interface for distributed training that requires minimal code changes:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

---

## Probabilistic and Scientific Computing

### torch.distributions

The `torch.distributions` module provides a comprehensive library of probability distributions with support for sampling, log-probability computation, and entropy — all differentiable:

```python
from torch.distributions import Normal, MultivariateNormal

prior = Normal(loc=0.0, scale=1.0)
samples = prior.rsample((1000,))           # reparameterised (differentiable)
log_prob = prior.log_prob(samples)          # differentiable log-likelihood
kl = torch.distributions.kl_divergence(q, prior)  # KL divergence
```

This module is heavily used in Bayesian deep learning, variational autoencoders, normalising flows, and reinforcement learning — all topics with direct applications in quantitative finance (risk modelling, density estimation, portfolio optimisation).

### torch.linalg

`torch.linalg` provides GPU-accelerated linear algebra operations: matrix decompositions (SVD, QR, Cholesky, eigendecomposition), solvers, norms, and matrix functions. These are essential for financial applications involving covariance estimation, PCA, and optimisation.

### torch.fft

`torch.fft` provides Fast Fourier Transform operations, useful for signal processing, spectral analysis of financial time series, and certain convolutional architectures.

---

## Ecosystem for Quantitative Finance

While PyTorch does not have an official finance-specific library, several ecosystem components are particularly relevant to quantitative finance practitioners:

| Need | Ecosystem Tool |
|---|---|
| Time series forecasting | Hugging Face (PatchTST, Informer), custom architectures |
| Density estimation / risk modelling | `torch.distributions`, normalising flows (nflows, zuko) |
| Graph-based models (portfolio networks) | PyTorch Geometric |
| Bayesian inference | Pyro, NumPyro (JAX-based but conceptually aligned) |
| Tabular data | PyTorch Tabular, tab-transformer implementations |
| Reinforcement learning (trading agents) | Stable-Baselines3, CleanRL |
| Uncertainty quantification | Laplace Redux, MC Dropout patterns |
| Experiment tracking | W&B, MLflow, TensorBoard |

### Pyro

[**Pyro**](https://pyro.ai/) is a probabilistic programming library built on PyTorch, developed by Uber AI. It supports Bayesian neural networks, variational inference, MCMC, Gaussian processes, and causal inference — all with automatic differentiation through PyTorch's autograd:

```python
import pyro
import pyro.distributions as dist

def bayesian_regression(x, y=None):
    w = pyro.sample("w", dist.Normal(0, 1).expand([x.shape[1]]).to_event(1))
    b = pyro.sample("b", dist.Normal(0, 1))
    mean = x @ w + b
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mean, 0.1), obs=y)
```

### PyTorch Geometric

[**PyTorch Geometric**](https://pyg.org/) provides graph neural network layers, datasets, and utilities. Applications in finance include modelling relationships between assets, supply chain networks, and transaction graphs for fraud detection.

---

## Summary

The PyTorch ecosystem transforms a tensor computation library into a complete platform for deep learning research and deployment. Key takeaways:

1. **Core PyTorch** provides tensors, autograd, `nn.Module`, and optimisers — the foundation for everything else.

2. **Domain libraries** (torchvision, torchaudio, torchtext) offer datasets, transforms, and pretrained models for specific modalities.

3. **Community libraries** (Hugging Face, timm, Lightning) dramatically accelerate development by providing pretrained models, high-level training abstractions, and experiment tracking.

4. **Deployment tools** (torch.compile, TorchScript, ONNX, TorchServe) bridge the gap between research and production.

5. **Probabilistic tools** (torch.distributions, Pyro) support Bayesian methods and uncertainty quantification — critical for financial applications.

6. **Distributed training** (DDP, FSDP) scales to large models and datasets across multiple GPUs and nodes.

The rest of this curriculum builds on these ecosystem components progressively. Chapter 1 focuses on core PyTorch; subsequent chapters introduce domain libraries and community tools as needed.

---

## References

1. PyTorch Ecosystem. https://pytorch.org/ecosystem/
2. Hugging Face Documentation. https://huggingface.co/docs
3. PyTorch Lightning Documentation. https://lightning.ai/docs/pytorch/stable/
4. timm Documentation. https://huggingface.co/docs/timm/
5. Pyro Documentation. https://pyro.ai/
6. PyTorch Geometric Documentation. https://pyg.org/
7. TorchServe Documentation. https://pytorch.org/serve/
