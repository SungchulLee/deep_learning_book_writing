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
