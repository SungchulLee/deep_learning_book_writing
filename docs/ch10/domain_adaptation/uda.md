# Unsupervised Domain Adaptation

Unsupervised Domain Adaptation (UDA) addresses the setting where labeled source data and *unlabeled* target data are available. The goal is to learn a model that performs well on the target domain without target labels.

## Problem Setting

$$\text{Given: } \{(\mathbf{x}_i^s, y_i^s)\}_{i=1}^{n_s} \sim \mathcal{D}_S, \quad \{\mathbf{x}_j^t\}_{j=1}^{n_t} \sim \mathcal{D}_T$$

$$\text{Goal: } \min_{h} \epsilon_T(h) \text{ using only source labels}$$

This is the most practically relevant setting—labeled data is expensive, but unlabeled target data is readily available.

## Taxonomy of UDA Methods

| Category | Approach | Key Methods |
|----------|----------|-------------|
| Discrepancy-based | Minimise distribution distance | MMD, CORAL, CDD |
| Adversarial | Domain-invariant features via adversarial training | DANN, ADDA, CDAN |
| Self-training | Pseudo-label target data | CBST, CRST |
| Reconstruction-based | Shared encoder with domain-specific decoders | DSN, DRCN |

## Discrepancy-Based Methods

### Maximum Mean Discrepancy (MMD)

MMD measures the distance between feature distributions using kernel embeddings:

$$\text{MMD}^2(\mathcal{D}_S, \mathcal{D}_T) = \left\| \frac{1}{n_s}\sum_{i=1}^{n_s}\phi(\mathbf{x}_i^s) - \frac{1}{n_t}\sum_{j=1}^{n_t}\phi(\mathbf{x}_j^t) \right\|_{\mathcal{H}}^2$$

```python
import torch
import torch.nn as nn


def compute_mmd(source_features, target_features, gamma=1.0):
    """Compute MMD between source and target features using RBF kernel."""
    n_s, n_t = source_features.size(0), target_features.size(0)

    def rbf_kernel(x, y):
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        distances = xx + yy.t() - 2 * torch.mm(x, y.t())
        return torch.exp(-gamma * distances)

    ss = rbf_kernel(source_features, source_features)
    tt = rbf_kernel(target_features, target_features)
    st = rbf_kernel(source_features, target_features)

    mmd = (ss.sum() / (n_s * n_s) +
           tt.sum() / (n_t * n_t) -
           2 * st.sum() / (n_s * n_t))

    return torch.sqrt(torch.clamp(mmd, min=0))


class MMDAdaptationModel(nn.Module):
    """Model with MMD regularisation for domain adaptation."""
    
    def __init__(self, backbone, num_classes, feature_dim=512):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(feature_dim, num_classes)

        for param in self.feature_extractor[:5].parameters():
            param.requires_grad = False

    def forward(self, x, return_features=False):
        features = self.flatten(self.feature_extractor(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
```

### Training with MMD Loss

```python
def train_with_mmd(model, source_loader, target_loader,
                   num_epochs=20, mmd_weight=0.1, device='cuda'):
    """Train with classification + MMD domain adaptation losses."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        target_iter = iter(target_loader)

        for source_inputs, source_labels in source_loader:
            try:
                target_inputs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_inputs, _ = next(target_iter)

            source_inputs = source_inputs.to(device)
            source_labels = source_labels.to(device)
            target_inputs = target_inputs.to(device)

            optimizer.zero_grad()

            source_logits, source_features = model(source_inputs, return_features=True)
            _, target_features = model(target_inputs, return_features=True)

            cls_loss = criterion(source_logits, source_labels)
            mmd_loss = compute_mmd(source_features, target_features)
            loss = cls_loss + mmd_weight * mmd_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(source_loader):.4f}")

    return model
```

### CORAL (Correlation Alignment)

CORAL aligns second-order statistics (covariance matrices) between domains:

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \| C_S - C_T \|_F^2$$

```python
def coral_loss(source_features, target_features):
    """Compute CORAL loss between source and target feature covariances."""
    d = source_features.size(1)

    # Centre features
    source_centered = source_features - source_features.mean(0)
    target_centered = target_features - target_features.mean(0)

    # Covariance matrices
    cs = (source_centered.t() @ source_centered) / (source_features.size(0) - 1)
    ct = (target_centered.t() @ target_centered) / (target_features.size(0) - 1)

    loss = (cs - ct).pow(2).sum() / (4 * d * d)
    return loss
```

## Self-Training for UDA

Self-training generates pseudo-labels on target data using the current model, then trains on them:

```python
def self_training_adaptation(model, source_loader, target_loader,
                              num_rounds=5, confidence_threshold=0.9, device='cuda'):
    """Self-training with pseudo-labels for UDA."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for round_idx in range(num_rounds):
        # Step 1: Generate pseudo-labels for target data
        model.eval()
        pseudo_inputs, pseudo_labels = [], []

        with torch.no_grad():
            for inputs, _ in target_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                max_probs, predictions = probs.max(dim=1)

                # Keep only high-confidence predictions
                mask = max_probs > confidence_threshold
                if mask.any():
                    pseudo_inputs.append(inputs[mask].cpu())
                    pseudo_labels.append(predictions[mask].cpu())

        if pseudo_inputs:
            pseudo_inputs = torch.cat(pseudo_inputs)
            pseudo_labels = torch.cat(pseudo_labels)
            print(f"Round {round_idx+1}: {len(pseudo_labels)} pseudo-labeled samples")
        else:
            print(f"Round {round_idx+1}: No confident predictions, skipping")
            continue

        # Step 2: Train on source + pseudo-labeled target
        model.train()
        pseudo_dataset = torch.utils.data.TensorDataset(pseudo_inputs, pseudo_labels)
        pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for source_batch, pseudo_batch in zip(source_loader, pseudo_loader):
            s_inputs, s_labels = source_batch[0].to(device), source_batch[1].to(device)
            p_inputs, p_labels = pseudo_batch[0].to(device), pseudo_batch[1].to(device)

            optimizer.zero_grad()
            s_loss = criterion(model(s_inputs), s_labels)
            p_loss = criterion(model(p_inputs), p_labels)
            loss = s_loss + 0.5 * p_loss
            loss.backward()
            optimizer.step()

    return model
```

## Batch Normalization Adaptation

The simplest UDA technique: update BatchNorm statistics using target data:

```python
def adapt_batchnorm(model, target_loader, device='cuda'):
    """
    Update BatchNorm statistics using target domain data.
    Simple but surprisingly effective domain adaptation technique.
    """
    model.train()  # Enable BN stat updates

    with torch.no_grad():
        for inputs, _ in target_loader:
            inputs = inputs.to(device)
            _ = model(inputs)  # Updates running mean/var

    model.eval()
    return model
```

## Evaluation

```python
def evaluate_uda(model, source_loader, target_loader, device='cuda'):
    """Evaluate domain adaptation performance."""
    model.eval()
    model = model.to(device)

    def accuracy(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total

    src_acc = accuracy(source_loader)
    tgt_acc = accuracy(target_loader)

    print(f"Source accuracy: {src_acc:.2f}%")
    print(f"Target accuracy: {tgt_acc:.2f}%")
    print(f"Adaptation gap:  {src_acc - tgt_acc:.2f}%")

    return src_acc, tgt_acc
```

## Summary

| Method | Requires | Complexity | When to use |
|--------|----------|-----------|-------------|
| BN Adaptation | Unlabeled target | Very simple | Always try first |
| MMD | Unlabeled target | Moderate | Moderate gap |
| CORAL | Unlabeled target | Moderate | Feature alignment |
| Self-training | Unlabeled target | Moderate | Confident predictions |
| DANN | Unlabeled target | Complex | Large gap |

## References

1. Long, M., et al. (2015). "Learning Transferable Features with Deep Adaptation Networks." *ICML*.
2. Sun, B., & Saenko, K. (2016). "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." *ECCV Workshops*.
3. Zou, Y., et al. (2018). "Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training." *ECCV*.
4. Li, Y., et al. (2016). "Revisiting Batch Normalization for Practical Domain Adaptation." *ICLR Workshop*.
