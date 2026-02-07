# Ensemble Methods for Image Classification

## Overview

Ensemble methods combine predictions from multiple models to achieve better accuracy and robustness than any single model. This is a standard technique for competitions and production systems requiring maximum reliability.

## Learning Objectives

1. Understand ensemble strategies (averaging, voting, stacking)
2. Implement model ensembles for classification
3. Select diverse models for effective ensembles
4. Balance accuracy gains against computational cost

## Ensemble Strategies

### 1. Simple Averaging

Average prediction probabilities across models:

```python
def ensemble_average(models, x):
    """Average softmax probabilities."""
    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            probs.append(prob)
    
    # Average probabilities
    avg_prob = torch.stack(probs).mean(dim=0)
    return avg_prob.argmax(dim=1)
```

### 2. Weighted Averaging

Weight models by validation accuracy:

```python
def ensemble_weighted(models, weights, x):
    """Weighted average based on model performance."""
    weights = torch.tensor(weights) / sum(weights)  # Normalize
    
    probs = []
    for model, w in zip(models, weights):
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)
            probs.append(w * prob)
    
    return torch.stack(probs).sum(dim=0).argmax(dim=1)
```

### 3. Majority Voting

Each model votes for a class:

```python
def ensemble_voting(models, x):
    """Hard voting - majority wins."""
    votes = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
            votes.append(pred)
    
    # Stack and find mode
    votes = torch.stack(votes, dim=0)  # (num_models, batch)
    final_pred, _ = torch.mode(votes, dim=0)
    return final_pred
```

### 4. Stacking (Meta-Learning)

Train a meta-model on base model predictions:

```python
class StackingEnsemble(nn.Module):
    def __init__(self, base_models, num_classes, hidden_dim=128):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        
        # Freeze base models
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Meta-classifier
        input_dim = len(base_models) * num_classes
        self.meta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Get base predictions
        base_preds = []
        for model in self.base_models:
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                base_preds.append(pred)
        
        # Concatenate and pass through meta-classifier
        stacked = torch.cat(base_preds, dim=1)
        return self.meta(stacked)
```

## Model Selection for Ensembles

### Diversity Matters

Ensembles work best with **diverse** models:

```python
# Good ensemble: Different architectures
ensemble_diverse = [
    resnet50(num_classes=10),
    efficientnet_b0(num_classes=10),
    densenet121(num_classes=10),
    mobilenet_v2(num_classes=10),
]

# Less effective: Same architecture, different seeds
ensemble_similar = [
    resnet50(num_classes=10),  # seed 1
    resnet50(num_classes=10),  # seed 2
    resnet50(num_classes=10),  # seed 3
]
```

### Correlation Analysis

Measure prediction correlation to ensure diversity:

```python
def compute_ensemble_diversity(models, dataloader, device):
    """Compute pairwise prediction correlation."""
    predictions = []
    
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in dataloader:
                pred = model(x.to(device)).argmax(1).cpu()
                preds.append(pred)
        predictions.append(torch.cat(preds))
    
    # Compute correlation matrix
    n_models = len(models)
    corr_matrix = torch.zeros(n_models, n_models)
    
    for i in range(n_models):
        for j in range(n_models):
            agreement = (predictions[i] == predictions[j]).float().mean()
            corr_matrix[i, j] = agreement
    
    return corr_matrix
```

## Test-Time Augmentation (TTA)

Ensemble predictions from augmented versions of same image:

```python
def test_time_augmentation(model, image, num_augments=5):
    """Apply augmentations at test time and average predictions."""
    
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    ]
    
    model.eval()
    probs = []
    
    with torch.no_grad():
        # Original
        probs.append(F.softmax(model(image), dim=1))
        
        # Augmented versions
        for aug in augmentations:
            aug_image = aug(image)
            prob = F.softmax(model(aug_image), dim=1)
            probs.append(prob)
    
    return torch.stack(probs).mean(dim=0)
```

## Complete Ensemble Pipeline

```python
class EnsembleClassifier:
    def __init__(self, model_configs, num_classes, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        self.models = []
        self.weights = []
        
        for config in model_configs:
            model = config['factory'](num_classes=num_classes)
            model.load_state_dict(torch.load(config['checkpoint']))
            model = model.to(device).eval()
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
    
    def predict(self, x, method='weighted_avg', use_tta=False):
        x = x.to(self.device)
        
        if use_tta:
            probs = [self._predict_with_tta(m, x) for m in self.models]
        else:
            probs = []
            with torch.no_grad():
                for model in self.models:
                    prob = F.softmax(model(x), dim=1)
                    probs.append(prob)
        
        if method == 'average':
            ensemble_prob = torch.stack(probs).mean(dim=0)
        elif method == 'weighted_avg':
            weights = torch.tensor(self.weights, device=self.device)
            weights = weights / weights.sum()
            ensemble_prob = sum(w * p for w, p in zip(weights, probs))
        elif method == 'voting':
            votes = torch.stack([p.argmax(1) for p in probs])
            return torch.mode(votes, dim=0).values
        
        return ensemble_prob.argmax(dim=1)
    
    def _predict_with_tta(self, model, x):
        # Implement TTA
        pass


# Usage
configs = [
    {'factory': resnet50, 'checkpoint': 'resnet50.pth', 'weight': 0.95},
    {'factory': efficientnet_b0, 'checkpoint': 'effnet.pth', 'weight': 0.93},
    {'factory': densenet121, 'checkpoint': 'densenet.pth', 'weight': 0.91},
]

ensemble = EnsembleClassifier(configs, num_classes=10)
predictions = ensemble.predict(test_images, method='weighted_avg', use_tta=True)
```

## Performance Gains

### Typical Improvements

| Setup | Accuracy |
|-------|----------|
| Single ResNet-50 | 76.1% |
| 3× ResNet-50 (diff seeds) | 77.0% |
| ResNet + EfficientNet | 78.5% |
| 5 diverse models | 79.5% |
| 5 models + TTA | 80.2% |

### Diminishing Returns

```
Number of Models vs Accuracy Gain:

Models  Gain
1       baseline
2       +1.0%
3       +1.5%
4       +1.7%
5       +1.9%
10      +2.1%
20      +2.2%  ← Diminishing returns
```

## Exercises

1. Ensemble ResNet-18, VGG-16, and MobileNet on CIFAR-10
2. Compare averaging vs voting vs stacking
3. Analyze diversity with correlation matrix
4. Implement knowledge distillation from ensemble to single model

## Key Takeaways

1. **Diversity** is more important than individual model accuracy
2. **Weighted averaging** typically outperforms simple averaging
3. **TTA** provides free accuracy boost at inference cost
4. **Diminishing returns** after 3-5 models
5. Consider **knowledge distillation** for deployment

## References

1. Dietterich (2000). Ensemble Methods in Machine Learning.
2. Hinton et al. (2015). Distilling the Knowledge in a Neural Network.

---

**Previous**: [Transfer Learning](transfer_learning.md) | **Next Chapter**: [Video Understanding](../video_understanding/temporal_modeling.md)
