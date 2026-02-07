# Concept Bottleneck Models

## Introduction

**Concept Bottleneck Models (CBMs)** achieve interpretability by design: the network is forced to first predict a set of human-interpretable concepts, then use only those concepts to make the final prediction. This creates a "bottleneck" of interpretable intermediate representations.

Unlike post-hoc methods (SHAP, Grad-CAM) that explain after the fact, CBMs are **inherently interpretable**—every prediction can be traced through explicit concept activations.

## Architecture

### Standard CBM

```
Input x → Concept Predictor → [c₁, c₂, ..., cₖ] → Task Predictor → y
```

Given input $\mathbf{x}$, the model first predicts concept values:

$$
\hat{c}_i = g_i(\mathbf{x}), \quad i = 1, \ldots, k
$$

Then uses concepts for the final prediction:

$$
\hat{y} = h(\hat{c}_1, \hat{c}_2, \ldots, \hat{c}_k)
$$

### Training Objective

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{task}}(\hat{y}, y)}_{\text{task loss}} + \lambda \underbrace{\sum_{i=1}^{k} \mathcal{L}_{\text{concept}}(\hat{c}_i, c_i)}_{\text{concept loss}}
$$

where $c_i$ are ground-truth concept annotations.

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model with separate concept and task heads.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        n_concepts: int,
        n_classes: int,
        concept_names: list = None
    ):
        super().__init__()
        self.backbone = backbone
        self.concept_names = concept_names or [f'c_{i}' for i in range(n_concepts)]
        
        # Concept predictor
        self.concept_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_concepts),
            nn.Sigmoid()
        )
        
        # Task predictor (from concepts only)
        self.task_head = nn.Sequential(
            nn.Linear(n_concepts, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x, return_concepts=False):
        features = self.backbone(x)
        concepts = self.concept_head(features)
        output = self.task_head(concepts)
        
        if return_concepts:
            return output, concepts
        return output
    
    def explain(self, x):
        """Generate human-readable explanation."""
        output, concepts = self.forward(x, return_concepts=True)
        
        concept_values = concepts[0].detach().cpu().numpy()
        prediction = output.argmax(dim=1).item()
        
        # Task head weights show concept → prediction relationship
        task_weights = self.task_head[0].weight.data[prediction].cpu().numpy()
        contributions = concept_values * task_weights
        
        explanation = []
        sorted_idx = np.argsort(np.abs(contributions))[::-1]
        for idx in sorted_idx:
            explanation.append({
                'concept': self.concept_names[idx],
                'value': concept_values[idx],
                'contribution': contributions[idx]
            })
        
        return prediction, explanation
    
    def intervene(self, x, concept_idx, new_value):
        """
        Test counterfactual: what if concept had a different value?
        This is a unique advantage of CBMs.
        """
        _, concepts = self.forward(x, return_concepts=True)
        concepts_modified = concepts.clone()
        concepts_modified[0, concept_idx] = new_value
        return self.task_head(concepts_modified)
```

## Applications in Quantitative Finance

### Credit Scoring CBM

```python
# Concepts: debt_ratio_high, income_stable, long_credit_history, 
#           low_utilization, no_recent_delinquency
concept_names = [
    'High Debt Ratio', 'Stable Income', 'Long Credit History',
    'Low Utilization', 'No Recent Delinquency', 'Diverse Credit Mix'
]

model = ConceptBottleneckModel(
    backbone=feature_extractor,
    backbone_dim=512,
    n_concepts=6,
    n_classes=2,
    concept_names=concept_names
)

# Explain a decision
pred, explanation = model.explain(applicant_features)
print(f"Decision: {'Approved' if pred == 0 else 'Declined'}")
for item in explanation[:4]:
    print(f"  {item['concept']}: {item['value']:.2f} "
          f"(contribution: {item['contribution']:+.3f})")
```

## Summary

Concept Bottleneck Models provide interpretability by construction, enabling both explanations and counterfactual interventions. The trade-off is requiring concept annotations during training and potentially reduced accuracy if the concept set is incomplete.

## References

1. Koh, P. W., et al. (2020). "Concept Bottleneck Models." *ICML*.

2. Yuksekgonul, M., et al. (2022). "Post-hoc Concept Bottleneck Models." *ICLR*.
