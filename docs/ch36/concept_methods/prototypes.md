# Prototype Networks for Interpretability
## Introduction

**Prototype-based explanation** methods make predictions by comparing inputs to learned prototypical examples. Instead of abstract feature attributions, these methods explain: "This input is classified as X because it looks like **this** prototypical case." This provides intuitive, example-based explanations that domain experts find natural to understand.

## Mathematical Foundation

### Prototypical Network for Interpretability

Given a set of $P$ prototypes $\{\mathbf{p}_1, \ldots, \mathbf{p}_P\}$ in a learned embedding space, the model computes:

$$
f(\mathbf{x}) = h\left(d(g(\mathbf{x}), \mathbf{p}_1), \ldots, d(g(\mathbf{x}), \mathbf{p}_P)\right)
$$

where:

- $g(\mathbf{x})$ is the embedding of input $\mathbf{x}$
- $d(\cdot, \cdot)$ is a distance or similarity function
- $h$ combines prototype similarities into a prediction

### ProtoPNet Architecture (Chen et al., 2019)

ProtoPNet learns class-specific prototypical parts:

1. **Convolutional backbone** extracts feature maps
2. **Prototype layer** computes similarity to learned prototypes
3. **Fully connected layer** weights prototype activations for classification

The prediction is:

$$
\hat{y}_c = \sum_{p \in P_c} w_{cp} \max_{(i,j)} \log\left(\frac{\|z_{(i,j)} - \mathbf{p}_p\|^2 + 1}{\|z_{(i,j)} - \mathbf{p}_p\|^2 + \epsilon}\right)
$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeNetwork(nn.Module):
    """
    Interpretable prototype-based classifier.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        n_prototypes: int,
        n_classes: int,
        prototype_dim: int = 128
    ):
        super().__init__()
        self.backbone = backbone
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        
        # Projection to prototype space
        self.projection = nn.Linear(feature_dim, prototype_dim)
        
        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, prototype_dim)
        )
        
        # Classification from prototype similarities
        self.classifier = nn.Linear(n_prototypes, n_classes, bias=False)
    
    def compute_similarities(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        
        # L2 distance to each prototype
        distances = torch.cdist(projected.unsqueeze(1), 
                               self.prototypes.unsqueeze(0))
        distances = distances.squeeze(1)
        
        # Convert to similarity
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        return similarities, distances
    
    def forward(self, x):
        similarities, _ = self.compute_similarities(x)
        logits = self.classifier(similarities)
        return logits
    
    def explain(self, x, prototype_images=None):
        """
        Explain prediction by showing nearest prototypes.
        """
        similarities, distances = self.compute_similarities(x)
        prediction = self.classifier(similarities).argmax(dim=1).item()
        
        # Class weights for prototypes
        class_weights = self.classifier.weight[prediction].detach().cpu().numpy()
        
        sim_values = similarities[0].detach().cpu().numpy()
        contributions = sim_values * class_weights
        
        sorted_idx = np.argsort(contributions)[::-1]
        
        explanation = []
        for idx in sorted_idx[:5]:
            explanation.append({
                'prototype_idx': idx,
                'similarity': sim_values[idx],
                'contribution': contributions[idx],
                'distance': distances[0, idx].item()
            })
        
        return prediction, explanation
```

## Applications in Quantitative Finance

Prototype networks are particularly useful in finance for:

- **Credit risk**: "This applicant's profile is similar to these historically defaulting/non-defaulting cases"
- **Fraud detection**: "This transaction pattern matches known fraud prototype #3"
- **Regime classification**: "Current market conditions most resemble the 2018 volatility spike"

## Summary

Prototype networks provide case-based reasoning that aligns with how domain experts naturally think. They sacrifice some model capacity for interpretability, producing explanations that reference concrete examples rather than abstract feature attributions.

## References

1. Chen, C., et al. (2019). "This Looks Like That: Deep Learning for Interpretable Image Recognition." *NeurIPS*.

2. Snell, J., et al. (2017). "Prototypical Networks for Few-shot Learning." *NeurIPS*.

3. Li, O., et al. (2018). "Deep Learning for Case-Based Reasoning through Prototypes." *AAAI*.

## Exercises

**Exercise 1.**
Apply the interpretability method described in this section to a 2-layer neural network with ReLU activations classifying XOR inputs. Compute the explanation for the input $x = [1, 1]$.

??? success "Solution to Exercise 1"
    For a trained XOR network with weights $W_1, b_1, W_2, b_2$, the output is $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$. The explanation method produces attributions for each input feature. For $x = [1, 1]$ (class 0), both features contribute to the negative classification. The specific attribution values depend on the method: gradient-based methods compute $\partial f / \partial x_i$; perturbation-based methods measure output change when features are masked. The XOR problem demonstrates that linear explanation methods can mislead because the decision boundary is non-linear. $\square$

---

**Exercise 2.**
Prove or disprove that the explanation method in this section satisfies the completeness axiom: the sum of all feature attributions equals $f(x) - f(x_0)$ for some baseline $x_0$.

??? success "Solution to Exercise 2"
    The completeness axiom (also called efficiency in Shapley value theory) states that attributions sum to the difference between the model output at the input and at the baseline. Whether this method satisfies completeness depends on its formulation. Gradient methods do not satisfy completeness (gradients are local, not path-integrated). Integrated Gradients satisfies completeness by construction (fundamental theorem of calculus along the path). SHAP values satisfy efficiency by the Shapley axiom. Methods that violate completeness may over- or under-attribute, making the total attribution unreliable as a global explanation. $\square$

---

**Exercise 3.**
Design an experiment to evaluate the faithfulness of the explanations produced by this method. Use insertion and deletion curves to measure whether highlighted features are truly important to the model.

??? success "Solution to Exercise 3"
    Protocol: (1) Compute feature attributions for each test image. (2) Deletion: progressively mask features in order of decreasing attribution, recording the model confidence drop. Faithful explanations cause rapid confidence decrease. (3) Insertion: progressively reveal features in order of decreasing attribution from a blank baseline, recording confidence increase. Faithful explanations cause rapid confidence increase. (4) Compute AUC for both curves. (5) Compare against random ordering (baseline) and other methods. A faithful method should have low deletion AUC and high insertion AUC. Repeat over 1000+ test samples for statistical reliability. $\square$

---

**Exercise 4.**
Discuss how this interpretability method could be applied to a financial model predicting credit default. What regulatory requirements must the explanations satisfy?

??? success "Solution to Exercise 4"
    For credit models, regulations (ECOA, GDPR Article 22) require individualized explanations for adverse decisions. The method must produce: (1) the top factors contributing to the denial (adverse action reasons); (2) explanations that are consistent (similar applicants get similar explanations); (3) explanations that are actionable (the applicant understands what to change). The interpretability method from this section can identify feature importances, but must be validated for stability (small input changes should not drastically alter the explanation) and correctness (removing important features should change the prediction). Protected attributes must be handled carefully to avoid revealing proxy discrimination. $\square$
