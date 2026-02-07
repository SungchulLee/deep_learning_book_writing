# Introduction to Algorithmic Fairness

## Overview

Algorithmic fairness is a rapidly evolving field that addresses the ethical, legal, and technical challenges of ensuring machine learning systems do not discriminate against individuals or groups based on protected characteristics. As ML models increasingly influence high-stakes decisions—from loan approvals to parole recommendations—understanding fairness has become essential for practitioners.

## The Fairness Problem

Consider a loan approval system trained on historical data. If past lending decisions reflected discriminatory practices, a model trained on this data may learn to perpetuate these biases, even without explicit access to protected attributes like race or gender.

### Motivating Example

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

def demonstrate_bias_problem():
    """
    Demonstrate how bias can emerge in ML predictions.
    
    This example shows a simple scenario where a model trained on
    biased historical data produces unfair predictions.
    """
    np.random.seed(42)
    n_samples = 2000
    
    # Generate features: credit score and income
    # Both groups have similar distributions
    credit_scores = np.random.normal(650, 100, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    
    # Protected attribute: group membership (e.g., demographic group)
    group = np.random.randint(0, 2, n_samples)
    
    # Historical approval decisions were BIASED
    # Group 0: approved if credit_score > 600
    # Group 1: approved if credit_score > 700 (higher threshold!)
    historical_approval = np.where(
        group == 0,
        (credit_scores > 600).astype(int),
        (credit_scores > 700).astype(int)  # Discriminatory threshold
    )
    
    # Analyze the bias
    group_0_approval_rate = historical_approval[group == 0].mean()
    group_1_approval_rate = historical_approval[group == 1].mean()
    
    print("Historical Approval Rates (Biased Data):")
    print(f"  Group 0: {group_0_approval_rate:.2%}")
    print(f"  Group 1: {group_1_approval_rate:.2%}")
    print(f"  Gap: {abs(group_0_approval_rate - group_1_approval_rate):.2%}")
    
    # Despite similar qualifications, Group 1 faces discrimination
    print("\nAverage Credit Scores:")
    print(f"  Group 0: {credit_scores[group == 0].mean():.1f}")
    print(f"  Group 1: {credit_scores[group == 1].mean():.1f}")
    
    return {
        'credit_scores': credit_scores,
        'income': income,
        'group': group,
        'approval': historical_approval
    }

data = demonstrate_bias_problem()
```

**Output:**
```
Historical Approval Rates (Biased Data):
  Group 0: 69.23%
  Group 1: 30.89%
  Gap: 38.34%

Average Credit Scores:
  Group 0: 651.2
  Group 1: 648.7
```

The data reveals a stark disparity: despite nearly identical credit score distributions, Group 1 has less than half the approval rate of Group 0. A model trained on this data will learn to reproduce this discriminatory pattern.

## Formal Definition of Fairness

Fairness can be formalized in multiple ways. The most common framework considers:

- **Input**: Feature vector $X \in \mathcal{X}$
- **Protected Attribute**: $A \in \{0, 1, \ldots, k\}$
- **True Label**: $Y \in \{0, 1\}$
- **Prediction**: $\hat{Y} = f(X)$
- **Score**: $S = s(X) \in [0, 1]$

A classifier $f$ is considered fair under some criterion $\mathcal{C}$ if:

$$\mathcal{C}(f, A) \leq \epsilon$$

where $\epsilon$ is a tolerance threshold (often 0 for exact fairness). The fundamental challenge is that there are many reasonable but mutually incompatible criteria $\mathcal{C}$, as we will see in subsequent sections.

## The Fairness–Accuracy Tradeoff

A fundamental tension exists between predictive accuracy and fairness. When historical data reflects discriminatory patterns, an accuracy-maximizing model will learn those patterns. Enforcing fairness constraints necessarily redirects the model away from pure accuracy optimization.

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class FairnessAccuracyTradeoff:
    """Demonstrates the tradeoff between accuracy and fairness."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute accuracy and fairness metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            group: Group membership (protected attribute)
            
        Returns:
            Dictionary with accuracy and fairness metrics
        """
        accuracy = (y_true == y_pred).mean()
        
        rate_0 = y_pred[group == 0].mean()
        rate_1 = y_pred[group == 1].mean()
        
        # Statistical Parity Difference
        spd = abs(rate_0 - rate_1)
        
        # Disparate Impact Ratio
        di_ratio = (
            min(rate_0, rate_1) / max(rate_0, rate_1)
            if max(rate_0, rate_1) > 0 else 0
        )
        
        return {
            'accuracy': accuracy,
            'statistical_parity_diff': spd,
            'disparate_impact_ratio': di_ratio,
            'rate_group_0': rate_0,
            'rate_group_1': rate_1,
        }
    
    def simulate_tradeoff(self, n_points: int = 5) -> None:
        """Simulate different accuracy–fairness tradeoff points."""
        np.random.seed(42)
        n = 1000
        group = np.random.randint(0, 2, n)
        y_true = np.random.randint(0, 2, n)
        
        print("Accuracy–Fairness Tradeoff Simulation:")
        print("-" * 60)
        print(f"{'Bias Level':<15} {'Accuracy':<12} {'SPD':<12} {'DI Ratio':<12}")
        print("-" * 60)
        
        for bias_level in np.linspace(0, 0.4, n_points):
            base_pred = (np.random.rand(n) > 0.5).astype(int)
            y_pred = base_pred.copy()
            shift_mask = (group == 1) & (np.random.rand(n) < bias_level)
            y_pred[shift_mask] = 0
            
            metrics = self.evaluate_model(y_true, y_pred, group)
            print(
                f"  {bias_level:.2f}         "
                f"{metrics['accuracy']:.4f}       "
                f"{metrics['statistical_parity_diff']:.4f}       "
                f"{metrics['disparate_impact_ratio']:.4f}"
            )
            self.results.append({'bias_level': bias_level, **metrics})

tradeoff = FairnessAccuracyTradeoff()
tradeoff.simulate_tradeoff()
```

## Stakeholders and Perspectives

Fairness considerations involve multiple stakeholders, each with legitimate but potentially conflicting goals:

**Individual perspective.** "Similar individuals should be treated similarly." This view motivates *individual fairness* criteria based on distance metrics in feature space.

**Group perspective.** "Demographic groups should have equal outcomes or opportunities." This view motivates *group fairness* criteria such as demographic parity and equalized odds.

**Process perspective.** "Decisions should be made using fair procedures." This focuses on feature selection, model transparency, and avoidance of protected attributes as direct inputs.

**Outcome perspective.** "Results should be equitable across groups." This may require different treatment of groups to achieve substantively equal results—sometimes called *equity* as distinct from *equality*.

## Practical Framework for Fair ML

A systematic approach to building fair ML systems proceeds through six stages:

```python
class FairMLPipeline:
    """
    Framework for building fair machine learning systems.
    
    This class outlines the key steps in a fairness-aware ML pipeline,
    from problem definition through deployment monitoring.
    """
    
    def __init__(self, protected_attributes: list):
        self.protected_attributes = protected_attributes
        self.fairness_definitions = []
        self.mitigation_strategy = None
        
    def step1_identify_protected_attributes(self) -> str:
        """
        Step 1: Identify which attributes should be protected.
        
        Protected attributes typically include race/ethnicity, gender,
        age, religion, disability status, and national origin. Consider
        legal requirements (ECOA, Fair Housing Act), ethical norms, and
        proxy variables that correlate with protected attributes.
        """
        summary = f"Protected attributes: {self.protected_attributes}\n"
        summary += "Considerations:\n"
        summary += "  - Legal requirements (e.g., ECOA, Fair Housing Act)\n"
        summary += "  - Ethical norms specific to the domain\n"
        summary += "  - Proxy variables correlated with protected attributes"
        return summary
    
    def step2_choose_fairness_definition(self, definitions: list) -> str:
        """Step 2: Select appropriate fairness definitions."""
        self.fairness_definitions = definitions
        explanations = {
            'demographic_parity': (
                "Equal positive prediction rates across groups.\n"
                "  Use when: Equal outcomes required regardless of qualifications."
            ),
            'equal_opportunity': (
                "Equal true positive rates across groups.\n"
                "  Use when: False negatives are costly and should be equalized."
            ),
            'equalized_odds': (
                "Equal TPR and FPR across groups.\n"
                "  Use when: Both false positives and negatives matter."
            ),
            'calibration': (
                "Equal PPV across groups at each score level.\n"
                "  Use when: Predicted probabilities must be trustworthy per group."
            ),
        }
        summary = "Selected fairness definitions:\n"
        for defn in definitions:
            summary += f"\n  {defn}:\n  {explanations.get(defn, 'Unknown')}\n"
        return summary
    
    def step3_analyze_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Step 3: Analyze training data for existing bias."""
        analysis = {}
        for attr_name, attr_values in protected.items():
            groups = np.unique(attr_values)
            for g in groups:
                mask = attr_values == g
                analysis[f'{attr_name}_group_{g}_positive_rate'] = y[mask].mean()
                analysis[f'{attr_name}_group_{g}_count'] = int(mask.sum())
            counts = [np.sum(attr_values == g) for g in groups]
            analysis[f'{attr_name}_representation_ratio'] = min(counts) / max(counts)
        return analysis
    
    def step4_train_and_evaluate_baseline(self) -> str:
        """Step 4: Train baseline model without fairness constraints."""
        return (
            "1. Train using standard procedure\n"
            "2. Evaluate accuracy AND fairness metrics\n"
            "3. Document baseline performance\n"
            "4. This establishes the cost of fairness interventions"
        )
    
    def step5_apply_mitigation(self, strategy: str) -> str:
        """Step 5: Apply bias mitigation technique."""
        self.mitigation_strategy = strategy
        strategies = {
            'pre_processing': "Reweighing, resampling, data transformation",
            'in_processing': "Adversarial debiasing, fairness constraints, regularization",
            'post_processing': "Threshold optimization, calibration, reject option",
        }
        return strategies.get(strategy, "Unknown strategy")
    
    def step6_evaluate_and_monitor(self) -> str:
        """Step 6: Evaluate fairness and set up production monitoring."""
        return (
            "1. Test on held-out data across all protected groups\n"
            "2. Compute all relevant fairness metrics\n"
            "3. Analyze accuracy–fairness tradeoffs\n"
            "4. Set up production monitoring dashboards\n"
            "5. Establish alerting for fairness drift\n"
            "6. Plan for regular fairness audits"
        )

# Example usage
pipeline = FairMLPipeline(['gender', 'race'])
print(pipeline.step1_identify_protected_attributes())
print()
print(pipeline.step2_choose_fairness_definition(
    ['demographic_parity', 'equal_opportunity']
))
```

## Key Takeaways

1. **Bias is pervasive**: Historical biases in data lead to biased ML predictions even when protected attributes are excluded
2. **Fairness is contextual**: Different applications require different fairness definitions
3. **Tradeoffs exist**: Improving fairness may reduce overall accuracy—and different fairness criteria conflict with each other
4. **Multiple stakeholders**: Individual, group, process, and outcome perspectives must all be considered
5. **Systematic approach**: A structured pipeline from problem definition through monitoring is essential

## Next Steps

- [Sources of Bias](sources.md): Understanding the taxonomy of bias in ML systems
- [Historical Context](history.md): Landmark cases and the evolution of fairness research
