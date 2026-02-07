# TCAV: Testing with Concept Activation Vectors

## Introduction

**Testing with CAVs (TCAV)** extends Concept Activation Vectors to provide a quantitative, statistical test of how important a concept is for a model's predictions. While a CAV defines the direction of a concept in activation space, TCAV measures **what fraction of inputs for a given class are positively influenced by that concept**.

## Mathematical Foundation

### TCAV Score

For a class $c$, concept $k$, and layer $l$, the TCAV score is:

$$
\text{TCAV}_{c,k,l} = \frac{|\{x \in X_c : S_{c,k,l}(x) > 0\}|}{|X_c|}
$$

where $S_{c,k,l}(x) = \nabla_{h_l(x)} f_c(x) \cdot v_l^k$ is the conceptual sensitivity and $X_c$ is the set of inputs belonging to class $c$.

A TCAV score of 0.8 means 80% of class $c$ inputs are positively influenced by concept $k$.

### Statistical Testing

To assess significance, TCAV uses random CAVs as a null hypothesis:

1. Train multiple CAVs using random concepts (random sets of images)
2. Compute TCAV scores for each random CAV
3. Test whether the real concept's TCAV score is significantly different from random

A concept is considered meaningful if its TCAV score is statistically significantly different from 0.5 (random chance).

## PyTorch Implementation

```python
import torch
import numpy as np
from scipy.stats import ttest_1samp

class TCAV:
    """Testing with Concept Activation Vectors."""
    
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.cav_module = ConceptActivationVector(model, target_layer)
    
    def compute_tcav_score(
        self,
        class_inputs: torch.Tensor,
        cav: np.ndarray,
        target_class: int
    ) -> float:
        """Compute TCAV score for a concept and class."""
        positive_count = 0
        total = len(class_inputs)
        
        for i in range(total):
            sensitivity = self.cav_module.conceptual_sensitivity(
                class_inputs[i:i+1].to(self.device),
                cav, target_class
            )
            if sensitivity > 0:
                positive_count += 1
        
        return positive_count / total
    
    def tcav_with_significance(
        self,
        class_inputs: torch.Tensor,
        concept_examples: torch.Tensor,
        random_example_sets: list,
        target_class: int,
        alpha: float = 0.05
    ) -> dict:
        """
        Compute TCAV score with statistical significance test.
        
        Args:
            class_inputs: Inputs of the target class
            concept_examples: Examples of the concept
            random_example_sets: List of random example sets for null hypothesis
            target_class: Class to test
            alpha: Significance level
        """
        # Train concept CAV
        random_neg = random_example_sets[0]
        concept_cav = self.cav_module.train_cav(concept_examples, random_neg)
        concept_score = self.compute_tcav_score(
            class_inputs, concept_cav, target_class
        )
        
        # Train random CAVs for null hypothesis
        random_scores = []
        for i in range(0, len(random_example_sets) - 1, 2):
            random_cav = self.cav_module.train_cav(
                random_example_sets[i], random_example_sets[i + 1]
            )
            score = self.compute_tcav_score(
                class_inputs, random_cav, target_class
            )
            random_scores.append(score)
        
        # Two-sided t-test against 0.5
        t_stat, p_value = ttest_1samp(
            [concept_score] + random_scores, 0.5
        )
        
        return {
            'tcav_score': concept_score,
            'random_scores': random_scores,
            'p_value': p_value,
            'significant': p_value < alpha,
            'concept_meaningful': concept_score > 0.5 and p_value < alpha
        }
```

## Applications in Quantitative Finance

TCAV enables testing whether financial models have learned economically meaningful concepts:

```python
def test_financial_concepts(model, layer, device):
    """Test whether a return prediction model uses financial concepts."""
    
    tcav = TCAV(model, layer, device)
    
    concepts_to_test = {
        'momentum': momentum_examples,
        'mean_reversion': reversion_examples,
        'volatility_regime': vol_regime_examples,
        'credit_stress': credit_stress_examples
    }
    
    for concept_name, examples in concepts_to_test.items():
        result = tcav.tcav_with_significance(
            class_inputs=positive_return_samples,
            concept_examples=examples,
            random_example_sets=random_sets,
            target_class=1  # Positive return class
        )
        
        sig = "***" if result['significant'] else "n.s."
        print(f"{concept_name:20s}: TCAV={result['tcav_score']:.3f} "
              f"p={result['p_value']:.4f} {sig}")
```

## Summary

TCAV provides a rigorous, quantitative framework for testing concept importance in neural networks. Its statistical testing ensures only genuinely meaningful concepts are identified, making it suitable for regulated environments where explanation quality must be validated.

## References

1. Kim, B., et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors." *ICML*.

2. Ghorbani, A., Wexler, J., Zou, J., & Kim, B. (2019). "Towards Automatic Concept-based Explanations." *NeurIPS*.
