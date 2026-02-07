# Generalized Zero-Shot Learning

## Overview

Generalized Zero-Shot Learning (GZSL) extends conventional ZSL to a more realistic setting where test instances may come from either seen or unseen classes. This section covers the GZSL problem formulation, evaluation metrics, and techniques to address the bias toward seen classes.

## Problem Formulation

### Conventional ZSL vs GZSL

**Conventional ZSL:**
- Training: Learn on seen classes $\mathcal{Y}^s$
- Testing: Predict only on unseen classes $\mathcal{Y}^u$
- Assumption: Test instances are known to be from unseen classes

**Generalized ZSL:**
- Training: Learn on seen classes $\mathcal{Y}^s$
- Testing: Predict on all classes $\mathcal{Y}^s \cup \mathcal{Y}^u$
- No assumption about which domain test instances belong to

### Formal Definition

Given:
- Training set: $\mathcal{D}^{tr} = \{(\mathbf{x}_i, y_i) : y_i \in \mathcal{Y}^s\}$
- Test set: $\mathcal{D}^{te} = \{(\mathbf{x}_j, y_j) : y_j \in \mathcal{Y}^s \cup \mathcal{Y}^u\}$
- Semantic embeddings for all classes

Goal: Learn a classifier $f: \mathcal{X} \rightarrow \mathcal{Y}^s \cup \mathcal{Y}^u$

### The Bias Problem

Models trained on seen classes exhibit strong **bias toward seen classes**:

$$P(\hat{y} \in \mathcal{Y}^s | y \in \mathcal{Y}^u) >> P(\hat{y} \in \mathcal{Y}^u | y \in \mathcal{Y}^u)$$

**Causes:**
1. Seen class features are in-distribution for the trained model
2. Compatibility scores are naturally higher for seen classes
3. Visual features may be more discriminative for seen classes

**Example:**
- Seen accuracy: 85%
- Unseen accuracy: 5%
- Model almost always predicts seen classes!

## Evaluation Metrics

### Per-Domain Accuracy

**Accuracy on seen classes:**
$$\text{acc}^s = \frac{1}{|\mathcal{D}^{te}_s|} \sum_{(\mathbf{x}, y) \in \mathcal{D}^{te}_s} \mathbb{1}[f(\mathbf{x}) = y]$$

**Accuracy on unseen classes:**
$$\text{acc}^u = \frac{1}{|\mathcal{D}^{te}_u|} \sum_{(\mathbf{x}, y) \in \mathcal{D}^{te}_u} \mathbb{1}[f(\mathbf{x}) = y]$$

### Harmonic Mean (Primary GZSL Metric)

$$H = \frac{2 \times \text{acc}^s \times \text{acc}^u}{\text{acc}^s + \text{acc}^u}$$

**Properties:**
- Penalizes imbalance between seen and unseen performance
- $H = 0$ if either $\text{acc}^s = 0$ or $\text{acc}^u = 0$
- $H = 1$ only if both accuracies are perfect
- Symmetric: treats seen and unseen equally

**Why Harmonic Mean?**
- Arithmetic mean would allow a model predicting all seen to score well
- Harmonic mean forces balance between domains
- Reflects real-world utility where both domains matter

### Area Under Seen-Unseen Curve (AUSUC)

Plot $\text{acc}^s$ vs $\text{acc}^u$ for different calibration parameters:

$$\text{AUSUC} = \int_0^1 \text{acc}^s(\text{acc}^u) \, d(\text{acc}^u)$$

This captures the full tradeoff space, not just a single operating point.

### Evaluation Code

```python
import numpy as np
from collections import defaultdict

def evaluate_gzsl(predictions, labels, seen_classes, unseen_classes):
    """
    Comprehensive GZSL evaluation.
    
    Args:
        predictions: Predicted class names
        labels: True class names
        seen_classes: List of seen class names
        unseen_classes: List of unseen class names
    
    Returns:
        Dictionary with all metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Masks for seen and unseen test samples
    seen_mask = np.isin(labels, seen_classes)
    unseen_mask = np.isin(labels, unseen_classes)
    
    # Per-domain accuracy
    acc_seen = np.mean(predictions[seen_mask] == labels[seen_mask]) if seen_mask.sum() > 0 else 0
    acc_unseen = np.mean(predictions[unseen_mask] == labels[unseen_mask]) if unseen_mask.sum() > 0 else 0
    
    # Harmonic mean
    if acc_seen + acc_unseen > 0:
        harmonic = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    else:
        harmonic = 0
    
    # Overall accuracy
    overall = np.mean(predictions == labels)
    
    # Per-class accuracy
    per_class_acc = {}
    for cls in np.unique(labels):
        cls_mask = labels == cls
        per_class_acc[cls] = np.mean(predictions[cls_mask] == labels[cls_mask])
    
    return {
        'acc_seen': acc_seen,
        'acc_unseen': acc_unseen,
        'harmonic_mean': harmonic,
        'overall': overall,
        'per_class': per_class_acc
    }
```

## Calibration Techniques

### Threshold Calibration

Add a constant offset to unseen class scores:

$$\text{score}^{calib}_c = \begin{cases}
\text{score}_c & \text{if } c \in \mathcal{Y}^s \\
\text{score}_c + \gamma & \text{if } c \in \mathcal{Y}^u
\end{cases}$$

where $\gamma$ is a calibration parameter optimized on validation data.

```python
def calibrated_prediction(scores_seen, scores_unseen, calibration):
    """
    Predict with calibration.
    
    Args:
        scores_seen: (batch, n_seen) scores for seen classes
        scores_unseen: (batch, n_unseen) scores for unseen classes
        calibration: Offset added to unseen scores
    
    Returns:
        predictions, domain (seen/unseen)
    """
    # Apply calibration
    scores_unseen_calib = scores_unseen + calibration
    
    # Combine scores
    max_seen = scores_seen.max(dim=1)
    max_unseen = scores_unseen_calib.max(dim=1)
    
    # Decide domain
    predict_unseen = max_unseen.values > max_seen.values
    
    predictions = torch.where(
        predict_unseen,
        max_unseen.indices,
        max_seen.indices
    )
    
    return predictions, predict_unseen
```

### Temperature Scaling

Divide seen class scores by temperature $T > 1$:

$$\text{score}^{scaled}_c = \begin{cases}
\text{score}_c / T & \text{if } c \in \mathcal{Y}^s \\
\text{score}_c & \text{if } c \in \mathcal{Y}^u
\end{cases}$$

This makes seen class probabilities more uniform, reducing confidence.

### Finding Optimal Calibration

```python
def find_optimal_calibration(model, val_loader, class_embeddings,
                             seen_classes, unseen_classes,
                             calibration_range=np.arange(-2, 2, 0.1)):
    """
    Find calibration that maximizes harmonic mean on validation set.
    """
    model.eval()
    
    # Get all predictions and scores
    all_scores_seen = []
    all_scores_unseen = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # Compute scores for seen classes
            seen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in seen_classes
            ])
            scores_seen = model(batch_x, seen_embs)
            
            # Compute scores for unseen classes
            unseen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in unseen_classes
            ])
            scores_unseen = model(batch_x, unseen_embs)
            
            all_scores_seen.append(scores_seen)
            all_scores_unseen.append(scores_unseen)
            all_labels.extend(batch_y)
    
    scores_seen = torch.cat(all_scores_seen)
    scores_unseen = torch.cat(all_scores_unseen)
    labels = np.array(all_labels)
    
    # Search for optimal calibration
    best_harmonic = 0
    best_calibration = 0
    
    results = []
    
    for calib in calibration_range:
        # Apply calibration
        scores_unseen_calib = scores_unseen + calib
        
        # Predict
        all_scores = torch.cat([scores_seen, scores_unseen_calib], dim=1)
        all_classes = seen_classes + unseen_classes
        
        pred_indices = all_scores.argmax(dim=1).numpy()
        predictions = [all_classes[i] for i in pred_indices]
        
        # Evaluate
        metrics = evaluate_gzsl(predictions, labels, seen_classes, unseen_classes)
        
        results.append({
            'calibration': calib,
            **metrics
        })
        
        if metrics['harmonic_mean'] > best_harmonic:
            best_harmonic = metrics['harmonic_mean']
            best_calibration = calib
    
    return best_calibration, results
```

## Gating Mechanisms

### Concept

Train a gating network to predict whether an instance belongs to seen or unseen domain:

$$g(\mathbf{x}) = P(\text{domain} = \text{unseen} | \mathbf{x})$$

Then route prediction:
- If $g(\mathbf{x}) > 0.5$: Predict from unseen classes
- Otherwise: Predict from seen classes

### Implementation

```python
class GatedGZSL(nn.Module):
    """
    GZSL model with learned gating mechanism.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int,
                 embedding_dim: int = 256):
        super().__init__()
        
        # Visual-semantic compatibility model
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
        self.semantic_encoder = nn.Linear(semantic_dim, embedding_dim)
        
        # Gating network: predicts seen vs unseen
        self.gate = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def encode_visual(self, x):
        return F.normalize(self.visual_encoder(x), dim=1)
    
    def encode_semantic(self, s):
        return F.normalize(self.semantic_encoder(s), dim=1)
    
    def compatibility(self, visual, semantic):
        v_enc = self.encode_visual(visual)
        s_enc = self.encode_semantic(semantic)
        
        if len(s_enc.shape) == 2 and s_enc.shape[0] != visual.shape[0]:
            return v_enc @ s_enc.T
        return torch.sum(v_enc * s_enc, dim=1)
    
    def predict_gate(self, visual):
        """Predict probability of unseen domain."""
        return self.gate(visual)
    
    def gated_predict(self, visual, seen_embeddings, unseen_embeddings,
                      seen_classes, unseen_classes):
        """
        Predict with gating mechanism.
        """
        # Get gate predictions
        gate_prob = self.predict_gate(visual).squeeze()
        
        # Get compatibility scores
        scores_seen = self.compatibility(visual, seen_embeddings)
        scores_unseen = self.compatibility(visual, unseen_embeddings)
        
        # Gated prediction
        predictions = []
        for i in range(len(visual)):
            if gate_prob[i] > 0.5:
                # Predict from unseen
                pred_idx = scores_unseen[i].argmax().item()
                predictions.append(unseen_classes[pred_idx])
            else:
                # Predict from seen
                pred_idx = scores_seen[i].argmax().item()
                predictions.append(seen_classes[pred_idx])
        
        return predictions


def train_gated_gzsl(model, train_loader, class_embeddings, 
                     seen_classes, unseen_classes, epochs=50):
    """
    Train GZSL model with gating.
    
    Note: Gate training requires some unseen class samples,
    typically from a validation split or generated features.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare embeddings
    seen_embs = torch.stack([
        torch.tensor(class_embeddings[c], dtype=torch.float32)
        for c in seen_classes
    ])
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # Compatibility loss (ranking)
            pos_emb = torch.stack([
                torch.tensor(class_embeddings[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            # Random negative
            neg_idx = torch.randint(0, len(seen_classes), (len(batch_y),))
            neg_emb = seen_embs[neg_idx]
            
            pos_score = model.compatibility(batch_x, pos_emb)
            neg_score = model.compatibility(batch_x, neg_emb)
            
            ranking_loss = torch.clamp(0.2 - pos_score + neg_score, min=0).mean()
            
            # Gate loss (all training samples are from seen domain)
            gate_pred = model.predict_gate(batch_x)
            gate_loss = F.binary_cross_entropy(gate_pred, torch.zeros_like(gate_pred))
            
            # Combined loss
            loss = ranking_loss + 0.5 * gate_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
```

## Generative Approaches

### Concept

Generate synthetic visual features for unseen classes, then train a standard classifier:

$$G: \mathcal{S} \times \mathcal{Z} \rightarrow \mathcal{V}$$

where $\mathcal{Z}$ is a noise distribution.

### Feature Generation with VAE

```python
class FeatureVAE(nn.Module):
    """
    VAE for generating visual features from semantic embeddings.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, latent_dim: int = 64):
        super().__init__()
        
        # Encoder: visual + semantic -> latent
        self.encoder = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder: latent + semantic -> visual
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + semantic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, visual_dim)
        )
    
    def encode(self, visual, semantic):
        x = torch.cat([visual, semantic], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, semantic):
        x = torch.cat([z, semantic], dim=1)
        return self.decoder(x)
    
    def forward(self, visual, semantic):
        mu, logvar = self.encode(visual, semantic)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, semantic)
        return recon, mu, logvar
    
    def generate(self, semantic, n_samples=1):
        """Generate visual features for given semantic embedding."""
        z = torch.randn(n_samples, self.fc_mu.out_features)
        semantic_expanded = semantic.unsqueeze(0).expand(n_samples, -1)
        return self.decode(z, semantic_expanded)


def train_feature_vae(vae, train_loader, class_embeddings, epochs=100):
    """Train feature VAE."""
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        
        for batch_v, batch_y in train_loader:
            batch_s = torch.stack([
                torch.tensor(class_embeddings[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            recon, mu, logvar = vae(batch_v, batch_s)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, batch_v)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_v.shape[0]
            
            loss = recon_loss + 0.001 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def generate_unseen_features(vae, class_embeddings, unseen_classes, 
                              n_per_class=100):
    """Generate synthetic features for unseen classes."""
    vae.eval()
    
    X_gen = []
    y_gen = []
    
    with torch.no_grad():
        for cls in unseen_classes:
            semantic = torch.tensor(class_embeddings[cls], dtype=torch.float32)
            features = vae.generate(semantic, n_samples=n_per_class)
            
            X_gen.append(features)
            y_gen.extend([cls] * n_per_class)
    
    return torch.cat(X_gen), y_gen
```

## Transductive GZSL

### Concept

Leverage unlabeled test data during training to adapt to unseen class distributions.

### Self-Training Approach

```python
def transductive_gzsl(model, train_loader, test_loader, 
                      class_embeddings, seen_classes, unseen_classes,
                      iterations=5, confidence_threshold=0.9):
    """
    Transductive GZSL with self-training.
    """
    # Initial training on seen classes
    train_model(model, train_loader, class_embeddings, seen_classes)
    
    for iteration in range(iterations):
        print(f"\nTransductive iteration {iteration + 1}/{iterations}")
        
        # Predict on test data
        model.eval()
        pseudo_X = []
        pseudo_y = []
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                # Get predictions with confidence
                all_classes = seen_classes + unseen_classes
                all_embs = torch.stack([
                    torch.tensor(class_embeddings[c], dtype=torch.float32)
                    for c in all_classes
                ])
                
                scores = model(batch_x, all_embs)
                probs = F.softmax(scores, dim=1)
                
                max_probs, pred_indices = probs.max(dim=1)
                
                # Keep high-confidence predictions on unseen classes
                for i in range(len(batch_x)):
                    if max_probs[i] > confidence_threshold:
                        pred_class = all_classes[pred_indices[i]]
                        if pred_class in unseen_classes:
                            pseudo_X.append(batch_x[i])
                            pseudo_y.append(pred_class)
        
        if len(pseudo_X) > 0:
            print(f"  Added {len(pseudo_X)} pseudo-labeled unseen samples")
            
            # Combine with training data
            # ... (create combined dataset and retrain)
        else:
            print("  No confident predictions on unseen classes")
            break
```

## Summary

Generalized Zero-Shot Learning presents significant challenges beyond conventional ZSL:

1. **Bias toward seen classes** is the primary obstacle in GZSL
2. **Harmonic mean** is the standard evaluation metric, penalizing imbalanced performance
3. **Calibration techniques** (threshold, temperature) can adjust the seen-unseen balance
4. **Gating mechanisms** learn to route predictions to the appropriate domain
5. **Generative approaches** synthesize features for unseen classes, converting GZSL to supervised learning
6. **Transductive methods** leverage unlabeled test data for domain adaptation

Best practices:
- Always evaluate using harmonic mean, not just overall accuracy
- Cross-validate calibration parameters on a held-out validation set
- Consider generative approaches when bias is severe
- Report the full seen-unseen tradeoff curve (AUSUC)
