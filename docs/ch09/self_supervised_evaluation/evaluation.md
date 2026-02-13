# SSL Evaluation Protocols

Evaluating self-supervised learning methods requires standardised protocols that measure the quality of learned representations independent of the pretext task.

## Linear Probing

The standard evaluation: freeze the encoder and train a linear classifier on top.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def linear_evaluation(encoder, train_loader, test_loader, device,
                      num_classes=10, epochs=100, lr=0.1):
    """
    Standard linear evaluation protocol.
    Freeze encoder, train linear classifier. Reports top-1 accuracy.
    """
    encoder.eval()

    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                feat = encoder(images)
                if len(feat.shape) > 2:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                features.append(feat.cpu())
                labels.append(targets)
        return torch.cat(features).numpy(), torch.cat(labels).numpy()

    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    feature_dim = train_features.shape[1]
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        classifier.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            loss = F.cross_entropy(classifier(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    test_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = classifier(test_tensor).argmax(1)
        accuracy = (preds == test_labels_tensor).float().mean().item() * 100

    return accuracy


def knn_evaluation(encoder, train_loader, test_loader, device, k=200):
    """k-NN evaluation: no training required. Measures feature clustering quality."""
    encoder.eval()

    def extract(loader):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                feat = encoder(images.to(device))
                if len(feat.shape) > 2:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                features.append(F.normalize(feat, dim=1).cpu())
                labels.append(targets)
        return torch.cat(features), torch.cat(labels)

    train_feat, train_labels = extract(train_loader)
    test_feat, test_labels = extract(test_loader)

    sim = test_feat @ train_feat.T
    _, indices = sim.topk(k, dim=1)

    neighbor_labels = train_labels[indices]
    predictions = torch.mode(neighbor_labels, dim=1).values
    accuracy = (predictions == test_labels).float().mean().item() * 100
    return accuracy
```

## Representation Quality Metrics

Beyond classification accuracy, representation quality can be measured through alignment and uniformity (Wang & Isola, 2020):

```python
def alignment_uniformity(features, labels):
    """
    Alignment: positive pairs should be close (lower is better).
    Uniformity: features should be spread on the hypersphere (lower is better).
    """
    features = F.normalize(torch.tensor(features, dtype=torch.float32), dim=1)

    sq_pdist = torch.cdist(features, features, p=2).pow(2)
    n = features.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)

    uniformity = torch.log(torch.exp(-2 * sq_pdist[mask]).mean())

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(features.numpy(), labels)

    return {'uniformity': uniformity.item(), 'silhouette': sil}
```

## Standard Benchmarks

| Benchmark | Task | Metric | State-of-the-art |
|-----------|------|--------|-----------------|
| ImageNet linear | Classification | Top-1 acc | ~86% (DINOv2 ViT-g) |
| CIFAR-10 linear | Classification | Top-1 acc | ~97% |
| VOC07 | Detection | mAP | ~85% |
| ADE20K | Segmentation | mIoU | ~49% |

## Method Comparison

| Method | ImageNet Linear | k-NN | Batch size | Key property |
|--------|----------------|------|------------|-------------|
| SimCLR | 69.3% | 60.6% | 4096 | Needs large batches |
| MoCo v2 | 71.1% | 61.9% | 256 | Queue reduces batch need |
| BYOL | 74.3% | 66.5% | 4096 | No negatives |
| SimSiam | 71.3% | — | 256 | Simplest method |
| Barlow Twins | 73.2% | — | 2048 | Redundancy reduction |
| DINO (ViT-B) | 78.2% | 76.1% | 1024 | Self-distillation |
| MAE (ViT-L) | 75.8%* | — | 4096 | Masked modeling |
| DINOv2 (ViT-g) | 86.3% | — | — | Scaled self-distillation |

*MAE benefits more from fine-tuning than linear probing.

## Choosing an Evaluation Protocol

| Protocol | When to use | Measures |
|----------|-------------|---------|
| Linear probing | Standard comparison | Feature quality for classification |
| k-NN | Quick evaluation, no training | Feature clustering quality |
| Fine-tuning | Practical transfer | Representation + adaptation |
| Few-shot | Low-data transfer | Representation generalisation |
| Dense probing | Segmentation/detection | Spatial feature quality |

## References

1. Wang, T., & Isola, P. (2020). "Understanding Contrastive Representation Learning through Alignment and Uniformity." *ICML*.
2. Goyal, P., et al. (2019). "Scaling and Benchmarking Self-Supervised Visual Representation Learning." *ICCV*.
