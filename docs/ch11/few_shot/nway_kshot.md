# N-way K-shot Setup

The N-way K-shot framework is the standard evaluation protocol for few-shot learning. It structures both training and testing around small classification tasks (episodes) that simulate the data-scarce conditions encountered at test time.

## Protocol Definition

Each episode consists of:

- **N classes** sampled from the available class pool
- **K labeled examples per class** forming the **support set** $\mathcal{S}$
- **Q query examples per class** forming the **query set** $\mathcal{Q}$

$$\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{N \times K}, \quad \mathcal{Q} = \{(x_j^q, y_j^q)\}_{j=1}^{N \times Q}$$

The model must classify query examples using only the support set.

## Common Configurations

| Configuration | N | K | Difficulty | Primary use |
|--------------|---|---|-----------|-------------|
| 5-way 1-shot | 5 | 1 | High | Standard benchmark |
| 5-way 5-shot | 5 | 5 | Moderate | Standard benchmark |
| 20-way 1-shot | 20 | 1 | Very high | Scalability test |
| 2-way 1-shot | 2 | 1 | Lower | Verification tasks |

## Episodic Training

```python
import torch
import random
from torch.utils.data import Sampler


class EpisodicSampler(Sampler):
    """
    Sample episodes for N-way K-shot training.
    
    Each episode contains:
    - N classes (randomly sampled)
    - K support + Q query examples per class
    """
    
    def __init__(self, labels, n_way, k_shot, q_query, num_episodes):
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        
        # Organise indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            # Sample N classes
            episode_classes = random.sample(self.classes, self.n_way)
            
            episode_indices = []
            for cls in episode_classes:
                cls_indices = self.class_indices[cls]
                selected = random.sample(cls_indices, self.k_shot + self.q_query)
                episode_indices.extend(selected)
            
            yield episode_indices
    
    def __len__(self):
        return self.num_episodes


def split_support_query(batch, n_way, k_shot, q_query):
    """
    Split a batch into support and query sets.
    
    Args:
        batch: (images, labels) where images is (N*(K+Q), C, H, W)
        n_way: Number of classes
        k_shot: Support examples per class
        q_query: Query examples per class
    
    Returns:
        support_images: (N*K, C, H, W)
        support_labels: (N*K,)
        query_images: (N*Q, C, H, W)
        query_labels: (N*Q,)
    """
    images, labels = batch
    samples_per_class = k_shot + q_query
    
    support_images, support_labels = [], []
    query_images, query_labels = [], []
    
    for i in range(n_way):
        start = i * samples_per_class
        support_images.append(images[start:start + k_shot])
        support_labels.append(labels[start:start + k_shot])
        query_images.append(images[start + k_shot:start + samples_per_class])
        query_labels.append(labels[start + k_shot:start + samples_per_class])
    
    support_images = torch.cat(support_images)
    support_labels = torch.cat(support_labels)
    query_images = torch.cat(query_images)
    query_labels = torch.cat(query_labels)
    
    # Remap labels to 0..N-1
    unique_labels = support_labels.unique()
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    support_labels = torch.tensor([label_map[l.item()] for l in support_labels])
    query_labels = torch.tensor([label_map[l.item()] for l in query_labels])
    
    return support_images, support_labels, query_images, query_labels
```

## Evaluation Protocol

Standard evaluation averages accuracy over many episodes:

```python
def evaluate_few_shot(model, test_loader, n_way, k_shot, device,
                      num_episodes=600):
    """
    Standard few-shot evaluation.
    
    Convention: report mean accuracy ± 95% confidence interval
    over 600+ episodes.
    """
    model.eval()
    accuracies = []
    
    for episode_idx, batch in enumerate(test_loader):
        if episode_idx >= num_episodes:
            break
        
        support_x, support_y, query_x, query_y = split_support_query(
            batch, n_way, k_shot, q_query=15
        )
        
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        with torch.no_grad():
            predictions = model.predict(support_x, support_y, query_x)
        
        acc = (predictions == query_y).float().mean().item()
        accuracies.append(acc)
    
    mean_acc = torch.tensor(accuracies).mean().item() * 100
    ci_95 = 1.96 * torch.tensor(accuracies).std().item() / (len(accuracies) ** 0.5) * 100
    
    print(f"{n_way}-way {k_shot}-shot: {mean_acc:.2f}% ± {ci_95:.2f}%")
    return mean_acc, ci_95
```

## Class Splits

Standard benchmarks define disjoint train/val/test class splits:

| Dataset | Total classes | Train | Val | Test |
|---------|-------------|-------|-----|------|
| Omniglot | 1623 | 1200 | — | 423 |
| mini-ImageNet | 100 | 64 | 16 | 20 |
| tiered-ImageNet | 608 | 351 | 97 | 160 |
| CUB-200 | 200 | 100 | 50 | 50 |

The critical requirement is $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$: test classes are never seen during training.

## References

1. Vinyals, O., et al. (2016). "Matching Networks for One Shot Learning." *NeurIPS*.
2. Triantafillou, E., et al. (2020). "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." *ICLR*.
