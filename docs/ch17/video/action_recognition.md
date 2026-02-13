# Action Recognition

## Learning Objectives

By the end of this section, you will be able to:

- Design complete action recognition pipelines
- Choose appropriate architectures for different scenarios
- Implement training and evaluation procedures
- Handle common challenges (class imbalance, temporal variation)
- Deploy models for real-world applications

## Problem Formulation

**Video Classification**: Given video $V = \{I_1, \ldots, I_T\}$, predict action class $y \in \{1, \ldots, K\}$.

### Challenges

1. **Temporal variation**: Same action can have different durations
2. **Viewpoint changes**: Actions look different from different angles
3. **Intra-class variation**: Same action, many visual appearances
4. **Fine-grained distinctions**: Similar actions (e.g., "open door" vs "close door")

## Complete Pipeline

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionRecognitionPipeline:
    """Complete action recognition system."""
    
    def __init__(self, model, num_frames=16, frame_size=(224, 224)):
        self.model = model
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.model.eval()
    
    def preprocess(self, video_path):
        """Load and preprocess video."""
        frames = self._load_frames(video_path)
        frames = self._sample_frames(frames, self.num_frames)
        frames = self._apply_transforms(frames)
        return torch.stack(frames).unsqueeze(0)
    
    def predict(self, video_path, top_k=5):
        """Predict action from video."""
        video = self.preprocess(video_path)
        
        with torch.no_grad():
            logits = self.model(video)
            probs = F.softmax(logits, dim=1)
        
        top_probs, top_indices = torch.topk(probs[0], top_k)
        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]
```

## Training

```python
def train_action_model(model, train_loader, val_loader, epochs=50):
    """Training loop for action recognition."""
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for videos, labels in train_loader:
            videos, labels = videos.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.cuda(), labels.cuda()
                outputs = model(videos)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return model
```

## Evaluation

```python
def evaluate_action_model(model, test_loader):
    """Comprehensive evaluation."""
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.cuda()
            outputs = model(videos)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Top-1 Accuracy
    top1 = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Top-5 Accuracy
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    top5_correct = sum(l in np.argsort(p)[-5:] for l, p in zip(labels, probs))
    top5 = top5_correct / len(labels)
    
    return {'top1': top1, 'top5': top5}
```

## Handling Class Imbalance

```python
def focal_loss(logits, targets, gamma=2.0):
    """Focal loss for class imbalance."""
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    p = torch.exp(-ce_loss)
    loss = (1 - p) ** gamma * ce_loss
    return loss.mean()
```

## Benchmarks

| Model | UCF-101 | Kinetics-400 |
|-------|---------|--------------|
| I3D | 95.1% | 71.1% |
| SlowFast | 96.8% | 79.8% |
| TimeSformer | 96.0% | 80.7% |
| VideoMAE | 96.1% | 87.4% |

## Summary

Action recognition requires:
- Appropriate architecture for speed/accuracy trade-off
- Proper data augmentation
- Multi-clip evaluation for robustness
- Transfer learning from large datasets
