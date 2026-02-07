# Few-Shot Learning Fundamentals

## Introduction

Few-shot learning addresses one of the most significant limitations of traditional deep learning: the requirement for massive labeled datasets. While humans can recognize new objects from just one or two examples, standard neural networks typically require thousands of labeled samples per class. Few-shot learning develops algorithms that can learn from minimal data, mimicking the human ability to generalize from limited experience.

## The Problem Setting

### Formal Definition

In few-shot learning, we consider a task $\mathcal{T}$ defined over a set of classes $\mathcal{C}$. The key distinction from standard supervised learning lies in the data availability:

**Training Phase**: We have access to a large labeled dataset $\mathcal{D}_{\text{train}}$ covering classes $\mathcal{C}_{\text{train}}$.

**Testing Phase**: We must classify examples from novel classes $\mathcal{C}_{\text{test}}$ where $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$, using only a few labeled examples per class.

### N-way K-shot Classification

The standard evaluation protocol involves **N-way K-shot** tasks:

- **N-way**: The number of classes in each task
- **K-shot**: The number of labeled examples per class

Common configurations include:
- **5-way 1-shot**: 5 classes with 1 example each (most challenging)
- **5-way 5-shot**: 5 classes with 5 examples each
- **20-way 1-shot**: More classes, testing scalability

### Support and Query Sets

Each few-shot task is structured with two sets:

**Support Set** $\mathcal{S}$: The small set of labeled examples used for learning
$$\mathcal{S} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_{N \times K}, y_{N \times K})\}$$

**Query Set** $\mathcal{Q}$: The examples we need to classify
$$\mathcal{Q} = \{(x_1^q, y_1^q), (x_2^q, y_2^q), \ldots, (x_{N \times Q}, y_{N \times Q})\}$$

where $Q$ is the number of query examples per class.

## Mathematical Framework

### Episodic Training

Few-shot models are trained using **episodic training**, where each training iteration simulates the few-shot test scenario:

1. Sample $N$ classes from $\mathcal{C}_{\text{train}}$
2. For each class, sample $K$ support examples and $Q$ query examples
3. Train the model to classify query examples using only support examples
4. Update model parameters based on query set performance

This training paradigm ensures the model learns **how to learn** from limited data rather than memorizing specific class features.

### Meta-Learning Objective

The meta-learning objective can be formalized as:

$$\theta^* = \argmin_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}(\mathcal{T}; \theta) \right]$$

where:
- $\theta$ represents model parameters
- $\mathcal{T}$ is a task sampled from task distribution $p(\mathcal{T})$
- $\mathcal{L}(\mathcal{T}; \theta)$ is the loss on task $\mathcal{T}$

### Generalization Bound

The generalization capability of few-shot learning can be analyzed through the lens of learning theory. For a model with hypothesis class $\mathcal{H}$, the expected error on novel tasks is bounded by:

$$\mathbb{E}_{\mathcal{T}}[\text{err}(\mathcal{T})] \leq \hat{\text{err}}_{\text{train}} + \sqrt{\frac{d_{\mathcal{H}} \log(m) + \log(1/\delta)}{2m}}$$

where $d_{\mathcal{H}}$ is the VC dimension of the hypothesis class and $m$ is the number of training tasks.

## Taxonomy of Approaches

Few-shot learning methods can be categorized into three main families:

### 1. Metric-Based Methods

Learn an embedding space where classification can be performed by comparing distances:

$$p(y = c | x, \mathcal{S}) = \frac{\exp(-d(f_\theta(x), \mu_c))}{\sum_{c'} \exp(-d(f_\theta(x), \mu_{c'}))}$$

where $f_\theta$ is the embedding function, $d$ is a distance metric, and $\mu_c$ represents class $c$ in the embedding space.

**Key Methods**: Siamese Networks, Matching Networks, Prototypical Networks, Relation Networks

### 2. Optimization-Based Methods (Meta-Learning)

Learn to quickly adapt to new tasks through gradient-based optimization:

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{S}}(\theta)$$

The model learns an initialization $\theta$ that enables rapid adaptation with few gradient steps.

**Key Methods**: MAML, Reptile, Meta-SGD, FOMAML

### 3. Model-Based Methods

Use model architectures with external memory or specialized components:

$$h_t = f(x_t, h_{t-1}, \mathcal{M})$$

where $\mathcal{M}$ is an external memory module.

**Key Methods**: Memory-Augmented Neural Networks, Neural Turing Machines for few-shot

## Key Concepts

### Inductive Bias

Effective few-shot learning relies heavily on appropriate inductive biases:

1. **Compositionality**: Objects are composed of reusable parts
2. **Similarity Structure**: Similar classes should have similar representations
3. **Task Structure**: Tasks share common underlying structure

### Transfer vs. Few-Shot Learning

| Aspect | Transfer Learning | Few-Shot Learning |
|--------|------------------|-------------------|
| Target Data | Some labeled data available | Minimal labeled data (1-5 examples) |
| Class Overlap | May share classes | Disjoint classes |
| Adaptation | Fine-tuning on target | Rapid adaptation mechanism |
| Training | Standard supervised | Episodic training |

### The Role of Pre-training

Modern few-shot learning often leverages pre-trained representations:

$$f_\theta = g_\psi \circ h_\phi$$

where $h_\phi$ is a pre-trained encoder (frozen or fine-tuned) and $g_\psi$ is a task-specific adaptation layer.

## Benchmark Datasets

### Omniglot

The "transpose" of MNIST with 1,623 characters from 50 alphabets:
- 20 examples per character
- 28×28 grayscale images
- Standard split: 964 train, 659 test characters

### mini-ImageNet

A subset of ImageNet for few-shot evaluation:
- 100 classes, 600 images per class
- 84×84 color images
- Split: 64 train, 16 validation, 20 test classes

### tiered-ImageNet

Larger scale with hierarchical structure:
- 608 classes grouped into 34 categories
- Categories split to ensure semantic separation
- More challenging transfer setting

### Meta-Dataset

A benchmark spanning multiple domains:
- ImageNet, Omniglot, Aircraft, Birds, Textures, Quick Draw, Fungi, VGG Flower, Traffic Signs, MSCOCO
- Tests cross-domain generalization

## Evaluation Protocol

### Standard Procedure

1. Sample 600-1000 test episodes
2. For each episode:
   - Sample N classes from test set
   - Sample K support + Q query examples per class
   - Predict query labels using support set
   - Compute accuracy
3. Report mean accuracy with 95% confidence interval

### Computing Confidence Intervals

Given episode accuracies $\{a_1, a_2, \ldots, a_n\}$:

$$\bar{a} = \frac{1}{n}\sum_{i=1}^n a_i, \quad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (a_i - \bar{a})^2}$$

The 95% confidence interval is:
$$\text{CI} = \bar{a} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}}$$

## PyTorch Implementation

### Episode Sampler

```python
import torch
import numpy as np
from collections import defaultdict

class EpisodeSampler:
    """
    Samples episodes for few-shot learning training and evaluation.
    
    Each episode contains a support set and query set sampled from
    N randomly selected classes.
    """
    
    def __init__(self, labels, n_way, k_shot, n_query, n_episodes):
        """
        Args:
            labels: Tensor of class labels for all samples
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Total number of episodes to generate
        """
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Build index mapping: class -> list of sample indices
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label.item()].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
        # Validate we have enough samples per class
        min_samples = k_shot + n_query
        for c, indices in self.class_to_indices.items():
            if len(indices) < min_samples:
                raise ValueError(
                    f"Class {c} has {len(indices)} samples, "
                    f"need at least {min_samples}"
                )
    
    def sample_episode(self):
        """
        Sample a single episode.
        
        Returns:
            support_indices: Indices for support set
            query_indices: Indices for query set
            class_mapping: Dict mapping original labels to 0...N-1
        """
        # Sample N classes for this episode
        episode_classes = np.random.choice(
            self.classes, 
            self.n_way, 
            replace=False
        )
        
        support_indices = []
        query_indices = []
        class_mapping = {}
        
        for new_label, original_class in enumerate(episode_classes):
            class_mapping[original_class] = new_label
            
            # Get all indices for this class
            available_indices = self.class_to_indices[original_class]
            
            # Sample without replacement
            selected = np.random.choice(
                available_indices,
                self.k_shot + self.n_query,
                replace=False
            )
            
            support_indices.extend(selected[:self.k_shot])
            query_indices.extend(selected[self.k_shot:])
        
        return support_indices, query_indices, class_mapping
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.sample_episode()
    
    def __len__(self):
        return self.n_episodes


class FewShotDataset(torch.utils.data.Dataset):
    """
    Wrapper that creates few-shot episodes from a standard dataset.
    """
    
    def __init__(self, data, labels, n_way, k_shot, n_query, n_episodes):
        """
        Args:
            data: Tensor of shape (N, *input_shape)
            labels: Tensor of shape (N,)
            n_way: Number of classes per episode
            k_shot: Support examples per class
            n_query: Query examples per class
            n_episodes: Number of episodes
        """
        self.data = data
        self.labels = labels
        self.sampler = EpisodeSampler(
            labels, n_way, k_shot, n_query, n_episodes
        )
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
    
    def __getitem__(self, idx):
        # Sample an episode
        support_idx, query_idx, class_map = self.sampler.sample_episode()
        
        # Extract data
        support_data = self.data[support_idx]
        query_data = self.data[query_idx]
        
        # Remap labels to 0...N-1
        support_labels = torch.tensor([
            class_map[self.labels[i].item()] 
            for i in support_idx
        ])
        query_labels = torch.tensor([
            class_map[self.labels[i].item()] 
            for i in query_idx
        ])
        
        return support_data, support_labels, query_data, query_labels
    
    def __len__(self):
        return self.sampler.n_episodes
```

### Base Encoder Architecture

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard convolutional block used in few-shot learning.
    Conv -> BatchNorm -> ReLU -> MaxPool
    """
    
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = pool
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.pool:
            x = F.max_pool2d(x, 2)
        return x


class Conv4Encoder(nn.Module):
    """
    4-layer convolutional encoder commonly used in few-shot learning.
    
    This architecture is the standard backbone in papers like
    Prototypical Networks, Matching Networks, and MAML.
    
    Output: 64-dimensional embedding per image (for 28x28 input)
    """
    
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),      # 28 -> 14
            ConvBlock(hidden_dim, hidden_dim),       # 14 -> 7
            ConvBlock(hidden_dim, hidden_dim),       # 7 -> 3
            ConvBlock(hidden_dim, hidden_dim),       # 3 -> 1
        )
        
        # Calculate output dimension based on input size
        # For 28x28: 64 * 1 * 1 = 64
        # For 84x84: 64 * 5 * 5 = 1600
        self.output_dim = hidden_dim
    
    def forward(self, x):
        """
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            embeddings: (batch, output_dim)
        """
        features = self.encoder(x)
        # Global average pooling for variable input sizes
        features = F.adaptive_avg_pool2d(features, 1)
        return features.view(features.size(0), -1)


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for few-shot learning.
    
    Uses a pre-trained or randomly initialized ResNet backbone
    with the final classification layer removed.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        import torchvision.models as models
        
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512
    
    def forward(self, x):
        features = self.encoder(x)
        return features.view(features.size(0), -1)
```

### Evaluation Utilities

```python
import numpy as np
from scipy import stats

def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: (n,) predicted class indices
        labels: (n,) ground truth class indices
    
    Returns:
        accuracy: float in [0, 1]
    """
    return (predictions == labels).float().mean().item()


def compute_confidence_interval(accuracies, confidence=0.95):
    """
    Compute confidence interval for accuracy measurements.
    
    Uses Student's t-distribution for proper small-sample inference.
    
    Args:
        accuracies: List of accuracy values from multiple episodes
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        mean: Mean accuracy
        ci: Half-width of confidence interval
    """
    accuracies = np.array(accuracies)
    n = len(accuracies)
    
    mean = np.mean(accuracies)
    std_error = stats.sem(accuracies)  # Standard error of the mean
    
    # t-value for given confidence level
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_value * std_error
    
    return mean, ci


def evaluate_few_shot(model, test_dataset, n_episodes=600, device='cuda'):
    """
    Standard few-shot evaluation protocol.
    
    Args:
        model: Few-shot model with forward(support, support_labels, query)
        test_dataset: FewShotDataset for evaluation
        n_episodes: Number of episodes to evaluate
        device: Device to run evaluation on
    
    Returns:
        mean_accuracy: Mean accuracy across episodes
        ci: 95% confidence interval
    """
    model.eval()
    accuracies = []
    
    dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=True
    )
    
    with torch.no_grad():
        for i, (support, support_labels, query, query_labels) in enumerate(dataloader):
            if i >= n_episodes:
                break
            
            # Remove batch dimension
            support = support.squeeze(0).to(device)
            support_labels = support_labels.squeeze(0).to(device)
            query = query.squeeze(0).to(device)
            query_labels = query_labels.squeeze(0).to(device)
            
            # Get predictions
            logits = model(support, support_labels, query)
            predictions = logits.argmax(dim=1)
            
            # Record accuracy
            acc = compute_accuracy(predictions, query_labels)
            accuracies.append(acc)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    
    return mean_acc, ci
```

## Practical Considerations

### Data Augmentation

Data augmentation is crucial in few-shot learning due to limited examples:

```python
import torchvision.transforms as T

few_shot_transforms = T.Compose([
    T.RandomResizedCrop(84, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])
```

### Hyperparameter Sensitivity

Key hyperparameters and their typical ranges:

| Hyperparameter | Typical Range | Notes |
|---------------|---------------|-------|
| Learning rate | 1e-4 to 1e-3 | Often requires warmup |
| Embedding dimension | 64 to 1600 | Depends on task complexity |
| Training episodes | 20K to 100K | More for complex tasks |
| Episodes per update | 1 to 8 | Batch of episodes |

### Common Pitfalls

1. **Label Leakage**: Ensure test classes are never seen during training
2. **Imbalanced Episodes**: Always sample equal examples per class
3. **Overfitting to Training Classes**: Use early stopping based on validation tasks
4. **Ignoring Pre-training**: Modern methods benefit significantly from pre-trained features

## Summary

Few-shot learning tackles the fundamental challenge of learning from limited data by:

1. Training models on many small tasks (episodic training)
2. Learning representations that enable rapid adaptation
3. Leveraging inductive biases about similarity and compositionality

The field has evolved from simple metric-based methods to sophisticated meta-learning algorithms, with recent advances incorporating large-scale pre-training and multi-modal learning.

## References

1. Vinyals, O., et al. "Matching Networks for One Shot Learning." NeurIPS 2016.
2. Snell, J., et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
3. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
4. Koch, G., et al. "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop 2015.
5. Triantafillou, E., et al. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR 2020.
