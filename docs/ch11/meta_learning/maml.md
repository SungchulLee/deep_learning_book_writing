# Model-Agnostic Meta-Learning (MAML)

## Introduction

Model-Agnostic Meta-Learning (MAML), introduced by Finn et al. (2017), takes a fundamentally different approach to few-shot learning compared to metric-based methods. Instead of learning a fixed embedding space, MAML learns an **initialization** of model parameters that enables rapid adaptation to new tasks with just a few gradient steps.

The key insight is elegant: if we can find initial parameters that are close to the optimal parameters for many different tasks, then a few gradient steps will suffice to adapt to any new task from this neighborhood.

## The MAML Algorithm

### Problem Formulation

MAML assumes access to a distribution over tasks $p(\mathcal{T})$. Each task $\mathcal{T}_i$ consists of a loss function $\mathcal{L}_{\mathcal{T}_i}$ and samples from a distribution over inputs and outputs. The goal is to find parameters $\theta$ that can be quickly adapted to new tasks.

### Bi-Level Optimization

MAML involves two optimization loops:

**Inner Loop (Task Adaptation)**:
Given a task $\mathcal{T}_i$ with support set $\mathcal{D}_i^{\text{train}}$, adapt the parameters:
$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta; \mathcal{D}_i^{\text{train}})$$

**Outer Loop (Meta-Update)**:
Update the initialization using the adapted parameters' performance on query sets:
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i; \mathcal{D}_i^{\text{test}})$$

### The Meta-Gradient

The key mathematical insight is that we optimize the post-adaptation loss with respect to the **pre-adaptation** parameters. This requires computing gradients through the adaptation process:

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) = \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))$$

Using the chain rule:
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}(\theta'_i) \cdot \frac{\partial \theta'_i}{\partial \theta}$$

where:
$$\frac{\partial \theta'_i}{\partial \theta} = I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

This involves the **Hessian** of the task loss, making MAML a second-order optimization method.

## Mathematical Analysis

### Gradient Computation

For a single inner step, the meta-gradient is:

$$\nabla_\theta \mathcal{L}(\theta'_i) = \nabla_{\theta'_i} \mathcal{L}(\theta'_i) \cdot (I - \alpha H_{\mathcal{T}_i})$$

where $H_{\mathcal{T}_i} = \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$ is the Hessian.

For multiple inner steps, the computation becomes more complex but follows the same principle of backpropagating through the adaptation trajectory.

### First-Order Approximation (FOMAML)

Computing the full Hessian is expensive. FOMAML ignores the second-order terms:

$$\nabla_\theta \mathcal{L}(\theta'_i) \approx \nabla_{\theta'_i} \mathcal{L}(\theta'_i)$$

This approximation assumes $\frac{\partial \theta'_i}{\partial \theta} \approx I$, which is reasonable when:
- $\alpha$ is small
- The loss landscape is locally flat (small Hessian)

Empirically, FOMAML performs nearly as well as full MAML with significantly reduced computation.

### Reptile

Reptile (Nichol et al., 2018) is another first-order variant:

$$\theta \leftarrow \theta + \epsilon \cdot \frac{1}{n} \sum_{i=1}^n (\theta'_i - \theta)$$

This can be interpreted as moving toward the adapted parameters, averaging over tasks.

## PyTorch Implementation

### Complete MAML Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import copy


class MAMLModel(nn.Module):
    """
    Base model class for MAML with functional forward pass.
    
    Allows passing explicit parameters to enable gradient computation
    through the inner loop adaptation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional parameter override.
        
        Args:
            x: Input tensor
            params: Dict of parameters to use instead of self.parameters()
                   Keys should match state_dict keys
        """
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return parameters as a dictionary."""
        return OrderedDict(self.named_parameters())


class ConvMAMLModel(nn.Module):
    """
    Convolutional MAML model for image classification.
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        hidden_dim: int = 64, 
        output_dim: int = 5
    ):
        super().__init__()
        
        # Conv blocks
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Functional forward pass."""
        if params is None:
            params = dict(self.named_parameters())
        
        # Conv block 1
        x = F.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn1.weight'], params['bn1.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = F.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn2.weight'], params['bn2.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Conv block 3
        x = F.conv2d(x, params['conv3.weight'], params['conv3.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn3.weight'], params['bn3.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Conv block 4
        x = F.conv2d(x, params['conv4.weight'], params['conv4.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['bn4.weight'], params['bn4.bias'], training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Global average pooling and classifier
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['fc.weight'], params['fc.bias'])
        
        return x


class MAML:
    """
    Model-Agnostic Meta-Learning algorithm.
    
    Learns an initialization that enables fast adaptation to new tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """
        Args:
            model: The model to meta-train
            inner_lr: Learning rate for inner loop (task adaptation)
            meta_lr: Learning rate for outer loop (meta-update)
            inner_steps: Number of gradient steps in inner loop
            first_order: If True, use FOMAML (ignore second-order terms)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=meta_lr
        )
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop adaptation on support set.
        
        Args:
            support_x: Support inputs
            support_y: Support labels
            params: Initial parameters (will be adapted)
        
        Returns:
            adapted_params: Parameters after adaptation
        """
        # Clone parameters for adaptation
        adapted_params = OrderedDict(
            (name, param.clone()) for name, param in params.items()
        )
        
        for step in range(self.inner_steps):
            # Forward pass
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True
            )
            
            # Update parameters
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad if grad is not None else param)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        return adapted_params
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float]:
        """
        Perform one meta-training step on a batch of tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
        
        Returns:
            meta_loss: Average query loss across tasks
            meta_accuracy: Average query accuracy across tasks
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        # Get initial parameters
        init_params = OrderedDict(self.model.named_parameters())
        
        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt to task
            adapted_params = self.inner_loop(support_x, support_y, init_params)
            
            # Evaluate on query set
            query_logits = self.model(query_x, adapted_params)
            task_loss = F.cross_entropy(query_logits, query_y)
            
            meta_loss += task_loss
            
            # Compute accuracy
            with torch.no_grad():
                predictions = query_logits.argmax(dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy.item()
        
        # Average over tasks
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks)
        
        # Meta-update
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), meta_accuracy
    
    def adapt_and_evaluate(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        inner_steps: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Adapt to a new task and evaluate.
        
        Args:
            support_x, support_y: Support set for adaptation
            query_x, query_y: Query set for evaluation
            inner_steps: Override number of adaptation steps
        
        Returns:
            loss, accuracy on query set
        """
        if inner_steps is not None:
            original_steps = self.inner_steps
            self.inner_steps = inner_steps
        
        self.model.eval()
        
        with torch.no_grad():
            # Get initial parameters
            init_params = OrderedDict(
                (name, param.clone()) 
                for name, param in self.model.named_parameters()
            )
        
        # Adapt (with gradients for inner loop only)
        adapted_params = OrderedDict(
            (name, param.requires_grad_(True))
            for name, param in init_params.items()
        )
        
        for step in range(self.inner_steps):
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(loss, adapted_params.values())
            
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        # Evaluate
        with torch.no_grad():
            query_logits = self.model(query_x, adapted_params)
            query_loss = F.cross_entropy(query_logits, query_y)
            
            predictions = query_logits.argmax(dim=1)
            accuracy = (predictions == query_y).float().mean()
        
        if inner_steps is not None:
            self.inner_steps = original_steps
        
        return query_loss.item(), accuracy.item()
```

### FOMAML and Reptile Variants

```python
class FOMAML(MAML):
    """
    First-Order MAML - ignores second-order gradients.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, first_order=True, **kwargs)


class Reptile:
    """
    Reptile meta-learning algorithm.
    
    Simpler than MAML - just moves toward adapted parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.1,
        inner_steps: int = 5
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt model parameters on support set.
        Returns adapted parameters without computational graph.
        """
        # Clone model for task-specific training
        task_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.inner_steps):
            logits = task_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return OrderedDict(task_model.named_parameters())
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float]:
        """
        Reptile meta-update: move toward adapted parameters.
        """
        # Store initial parameters
        init_params = OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
        )
        
        # Collect adapted parameters from all tasks
        adapted_params_list = []
        meta_accuracy = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_params = self.inner_loop(support_x, support_y)
            adapted_params_list.append(adapted_params)
            
            # Evaluate (for logging)
            with torch.no_grad():
                # Create temporary model with adapted params
                for name, param in self.model.named_parameters():
                    param.data.copy_(adapted_params[name])
                
                logits = self.model(query_x)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy.item()
        
        meta_accuracy /= len(tasks)
        
        # Reptile update: move toward average of adapted parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Average adapted parameter
                avg_adapted = torch.stack([
                    adapted[name] for adapted in adapted_params_list
                ]).mean(dim=0)
                
                # Update: θ ← θ + ε(θ' - θ)
                param.data.add_(self.meta_lr * (avg_adapted - init_params[name]))
        
        return 0.0, meta_accuracy  # Reptile doesn't compute meta-loss
```

## Training Procedure

### Complete Training Loop

```python
def train_maml(
    maml: MAML,
    train_dataset,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    meta_batch_size: int = 4,
    n_iterations: int = 60000,
    eval_interval: int = 100,
    device: str = 'cuda'
):
    """
    Full MAML training loop.
    """
    maml.model.to(device)
    
    for iteration in range(n_iterations):
        # Sample batch of tasks
        tasks = []
        for _ in range(meta_batch_size):
            support_x, support_y, query_x, query_y = sample_task(
                train_dataset, n_way, k_shot, n_query
            )
            tasks.append((
                support_x.to(device),
                support_y.to(device),
                query_x.to(device),
                query_y.to(device)
            ))
        
        # Meta-training step
        loss, acc = maml.meta_train_step(tasks)
        
        if (iteration + 1) % eval_interval == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}, Acc = {acc:.4f}")
    
    return maml
```

### Hyperparameter Configuration

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Inner LR ($\alpha$) | 0.01 - 0.1 | Higher for simpler tasks |
| Meta LR ($\beta$) | 0.001 | Adam optimizer |
| Inner steps | 1-10 | 1-5 for training, more for test |
| Meta batch size | 2-8 | Number of tasks per update |
| Training iterations | 30K-60K | Until convergence |

## Variants and Extensions

### Meta-SGD

Learn per-parameter learning rates:

```python
class MetaSGD(MAML):
    """
    Meta-SGD: Learn per-parameter inner learning rates.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        # Learnable per-parameter learning rates
        self.inner_lrs = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(
                torch.ones_like(param) * self.inner_lr
            )
            for name, param in model.named_parameters()
        })
        
        # Add learning rates to optimizer
        self.meta_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.inner_lrs.values()),
            lr=self.meta_lr
        )
    
    def inner_loop(self, support_x, support_y, params):
        adapted_params = OrderedDict(
            (name, param.clone()) for name, param in params.items()
        )
        
        for step in range(self.inner_steps):
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=not self.first_order
            )
            
            # Use learned learning rates
            adapted_params = OrderedDict(
                (name, param - self.inner_lrs[name.replace('.', '_')] * grad)
                for (name, param), grad in zip(adapted_params.items(), grads)
            )
        
        return adapted_params
```

### ANIL (Almost No Inner Loop)

Only adapt the final layer:

```python
class ANIL(MAML):
    """
    Almost No Inner Loop - only adapt classification head.
    
    The feature extractor is frozen during inner loop,
    only the final classifier is adapted.
    """
    
    def __init__(self, model: nn.Module, head_names: List[str], **kwargs):
        """
        Args:
            head_names: Names of parameters to adapt (e.g., ['fc.weight', 'fc.bias'])
        """
        super().__init__(model, **kwargs)
        self.head_names = set(head_names)
    
    def inner_loop(self, support_x, support_y, params):
        # Only adapt head parameters
        adapted_params = OrderedDict(
            (name, param.clone() if name in self.head_names else param)
            for name, param in params.items()
        )
        
        # Get only head params for gradient computation
        head_params = {
            name: param for name, param in adapted_params.items()
            if name in self.head_names
        }
        
        for step in range(self.inner_steps):
            logits = self.model(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)
            
            grads = torch.autograd.grad(
                loss, head_params.values(),
                create_graph=not self.first_order
            )
            
            # Update only head
            for (name, grad) in zip(head_params.keys(), grads):
                adapted_params[name] = adapted_params[name] - self.inner_lr * grad
        
        return adapted_params
```

### Task-Adaptive MAML

Condition adaptation on task embedding:

```python
class TaskAdaptiveMAML(MAML):
    """
    Task-Adaptive MAML - condition learning rates on task.
    """
    
    def __init__(self, model: nn.Module, embed_dim: int = 64, **kwargs):
        super().__init__(model, **kwargs)
        
        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(list(model.parameters())))
        )
        
        # Add to optimizer
        self.meta_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.task_encoder.parameters()),
            lr=self.meta_lr
        )
    
    def compute_task_embedding(self, support_x, support_y):
        """Compute task representation from support set."""
        with torch.no_grad():
            features = self.model.encoder(support_x)
            task_embedding = features.mean(dim=0)
        return task_embedding
    
    def inner_loop(self, support_x, support_y, params):
        # Compute task-specific learning rates
        task_emb = self.compute_task_embedding(support_x, support_y)
        lr_scales = torch.sigmoid(self.task_encoder(task_emb))
        
        # ... rest similar to MAML but with scaled learning rates
```

## Theoretical Insights

### Implicit Gradient Descent

MAML's inner loop can be viewed as approximating the solution to:

$$\theta^*_{\mathcal{T}} = \argmin_\theta \mathcal{L}_{\mathcal{T}}(\theta)$$

with a few gradient steps from initialization $\theta_0$.

### Connection to Multi-Task Learning

MAML optimizes:
$$\min_\theta \sum_{\mathcal{T}} \mathcal{L}_{\mathcal{T}}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta))$$

This is related to multi-task learning but with the crucial difference that we evaluate after adaptation.

### Universal Function Approximation

Under certain conditions, MAML with sufficient capacity can learn to approximate any learning algorithm, making it a universal meta-learner.

## Practical Considerations

### BatchNorm in MAML

BatchNorm is tricky with MAML because running statistics shouldn't be updated during inner loop:

```python
# Set track_running_stats=False for all BatchNorm layers
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
```

### Gradient Clipping

Prevent exploding gradients in second-order computation:

```python
# After meta_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### Memory Efficiency

For large models, use gradient checkpointing:

```python
from torch.utils.checkpoint import checkpoint

def inner_loop_checkpointed(self, support_x, support_y, params):
    def adaptation_step(params_flat, support_x, support_y):
        # Unflatten and perform one step
        ...
    
    for step in range(self.inner_steps):
        params_flat = checkpoint(adaptation_step, params_flat, support_x, support_y)
    
    return unflatten(params_flat)
```

## Summary

MAML provides a powerful optimization-based approach to few-shot learning:

1. **Learn to Learn**: Finds an initialization that enables rapid adaptation
2. **Model-Agnostic**: Works with any differentiable model
3. **Theoretically Grounded**: Bi-level optimization with second-order information

Key trade-offs:
- More flexible than metric learning but computationally expensive
- Second-order computation can be approximated (FOMAML, Reptile)
- Requires careful handling of BatchNorm and other stateful operations

## References

1. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
2. Nichol, A., et al. "On First-Order Meta-Learning Algorithms." arXiv 2018.
3. Li, Z., et al. "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning." arXiv 2017.
4. Raghu, A., et al. "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML." ICLR 2020.
5. Finn, C., et al. "Online Meta-Learning." ICML 2019.
