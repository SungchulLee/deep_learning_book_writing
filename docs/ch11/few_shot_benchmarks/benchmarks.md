# Few-Shot Learning Benchmarks

Standard benchmarks enable fair comparison of few-shot learning methods. This section surveys the most widely used datasets and evaluation protocols.

## Omniglot

The "MNIST of few-shot learning." Contains 1,623 handwritten characters from 50 alphabets, with 20 examples per character.

| Property | Value |
|----------|-------|
| Classes | 1,623 characters |
| Samples/class | 20 |
| Image size | 105 × 105 grayscale |
| Train/test split | 1,200 / 423 classes |
| Standard protocol | 5-way 1-shot, 20-way 1-shot |

## mini-ImageNet

The most widely used few-shot benchmark. A subset of ImageNet with 100 classes and 600 images per class.

| Property | Value |
|----------|-------|
| Classes | 100 (64 train / 16 val / 20 test) |
| Samples/class | 600 |
| Image size | 84 × 84 RGB |
| Standard protocol | 5-way 1-shot, 5-way 5-shot |

### State-of-the-Art Results (mini-ImageNet)

| Method | Type | 5-way 1-shot | 5-way 5-shot |
|--------|------|-------------|-------------|
| Matching Networks | Metric | 43.56% | 55.31% |
| Prototypical Networks | Metric | 49.42% | 68.20% |
| Relation Network | Metric | 50.44% | 65.32% |
| MAML | Meta | 48.70% | 63.11% |
| MetaOptNet | Optimisation | 62.64% | 78.63% |
| FEAT | Transformer | 66.78% | 82.05% |
| P>M>F | Foundation model | 85.53% | 93.27% |

## tiered-ImageNet

A larger and more challenging variant with hierarchical class splits ensuring train and test classes come from different high-level categories.

| Property | Value |
|----------|-------|
| Classes | 608 (351 train / 97 val / 160 test) |
| Samples/class | ~1,281 |
| Split criterion | Semantic hierarchy (WordNet) |

## CUB-200-2011

Fine-grained bird classification. Tests whether methods can distinguish visually similar classes.

| Property | Value |
|----------|-------|
| Classes | 200 bird species (100 train / 50 val / 50 test) |
| Samples/class | ~60 |
| Challenge | Fine-grained distinctions |

## Meta-Dataset

A benchmark of benchmarks that evaluates generalisation across diverse data sources.

| Dataset | Domain | Classes |
|---------|--------|---------|
| ILSVRC 2012 | Natural images | 1,000 |
| Omniglot | Handwritten characters | 1,623 |
| Aircraft | Fine-grained | 100 |
| CUB | Birds | 200 |
| Textures | DTD | 47 |
| Quick Draw | Sketches | 345 |
| Fungi | Nature | 1,394 |
| VGG Flower | Flowers | 102 |
| Traffic Signs | Road signs | 43 |
| MSCOCO | Objects | 80 |

## Evaluation Protocol

Standard evaluation reports:

1. **Mean accuracy** over 600+ test episodes
2. **95% confidence interval**: $\mu \pm 1.96 \sigma / \sqrt{n}$
3. **Both 1-shot and 5-shot** results on the same split

```python
import torch
import numpy as np


def standard_evaluation(model, test_loader, n_way, k_shot, device,
                       num_episodes=600):
    """Standard few-shot evaluation with confidence intervals."""
    accuracies = []
    
    for episode in test_loader:
        support_x, support_y, query_x, query_y = episode
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        with torch.no_grad():
            preds = model.predict(support_x, support_y, query_x)
        
        acc = (preds == query_y).float().mean().item()
        accuracies.append(acc)
        
        if len(accuracies) >= num_episodes:
            break
    
    accs = np.array(accuracies) * 100
    mean = accs.mean()
    ci95 = 1.96 * accs.std() / np.sqrt(len(accs))
    
    print(f"{n_way}-way {k_shot}-shot: {mean:.2f}% ± {ci95:.2f}%")
    return mean, ci95
```

## References

1. Lake, B.M., et al. (2015). "Human-Level Concept Learning Through Probabilistic Program Induction." *Science*.
2. Vinyals, O., et al. (2016). "Matching Networks for One Shot Learning." *NeurIPS*.
3. Ren, M., et al. (2018). "Meta-Learning for Semi-Supervised Few-Shot Classification." *ICLR*.
4. Triantafillou, E., et al. (2020). "Meta-Dataset: A Dataset of Datasets for Learning to Learn." *ICLR*.
