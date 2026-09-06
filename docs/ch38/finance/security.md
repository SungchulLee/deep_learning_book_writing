# Model Security in Financial Systems
## Introduction

Deploying machine learning models in production financial systems introduces security considerations that extend beyond adversarial robustness of individual predictions. This section covers the end-to-end security of ML-based financial systems, from model theft and data poisoning to secure deployment practices.

## Threat Landscape

### Model Extraction

An adversary with API access can **steal** the model by training a surrogate on query-response pairs:

$$
f_{\text{surrogate}} \approx f_{\text{target}} \quad \text{via} \quad \{(\mathbf{x}_i, f_{\text{target}}(\mathbf{x}_i))\}_{i=1}^N
$$

**Financial impact**: Proprietary trading signals, credit scoring models, and risk models represent significant intellectual property. Extraction enables both model theft and subsequent white-box adversarial attacks.

**Defenses**:

- Query rate limiting and anomaly detection
- Output perturbation (add calibrated noise to predictions)
- Watermarking (embed detectable patterns in model behavior)

### Data Poisoning

Adversaries who can influence training data can inject **poisoned examples** that degrade model performance or create targeted backdoors:

$$
\mathcal{D}_{\text{poisoned}} = \mathcal{D}_{\text{clean}} \cup \{(\mathbf{x}_{\text{poison}}, y_{\text{target}})\}
$$

**Financial examples**:

- Manipulating historical price data used for model training
- Injecting fraudulent transactions labeled as legitimate into training sets
- Corrupting alternative data sources (satellite imagery, web scraping)

#### Targeted Poisoning via Feature Collision

A particularly insidious form of data poisoning is **clean-label poisoning**, where the attacker does not need to change any labels. Instead, the attacker crafts poisoned training instances that collide with the target in feature space:

$$
\mathbf{x}_{\text{poison}} = \arg\min_{\mathbf{x}} \| f(\mathbf{x}) - f(\mathbf{x}_{\text{target}}) \|_2^2 + \beta \| \mathbf{x} - \mathbf{x}_{\text{base}} \|_2^2
$$

where $f(\cdot)$ extracts the learned feature representation, $\mathbf{x}_{\text{target}}$ is the instance the attacker wants misclassified at test time, and $\mathbf{x}_{\text{base}}$ is a legitimate example from the attacker's chosen class. The first term ensures feature-space collision with the target, while the second keeps the poison visually similar to a legitimate base instance (so its label appears correct).

When the model trains on this poisoned example, it learns features that associate the target's representation with the poison's class — causing misclassification of the target at test time without any label corruption.

**Why this matters for quant**: In financial ML, training data often comes from third-party vendors, alternative data providers, or scraped sources. An adversary could craft synthetic market data points that are statistically consistent with legitimate data (pass all quality checks) yet subtly shift the model's decision boundary to misclassify specific securities or market conditions.

#### Defenses Against Data Poisoning

| Defense | Mechanism | Trade-off |
|---------|-----------|-----------|
| Data sanitization | Remove outliers in feature space before training | May discard legitimate rare events |
| Influence functions | Identify training points with high influence on specific predictions | Computationally expensive for large datasets |
| Spectral signatures | Detect poisoned samples via spectral analysis of feature covariance | Requires sufficient poison concentration |
| Differential privacy | Add noise during training to limit any single point's influence | Reduces model accuracy |
| Ensemble agreement | Compare predictions across models trained on different data subsets | Increases compute cost |


### Backdoor Attacks

A **backdoor** is a hidden trigger pattern that causes targeted misclassification when present:

$$
f(\mathbf{x} + \text{trigger}) = y_{\text{target}} \quad \forall \mathbf{x}
$$

while $f(\mathbf{x}) = y_{\text{correct}}$ on clean inputs.

**Financial risk**: A backdoored credit model could approve specific fraudulent applications when they contain a particular feature pattern.

## Secure Deployment Practices

### Defense-in-Depth Architecture

```
Input Validation → Feature Monitoring → Model Ensemble → Output Validation → Decision
       ↓                  ↓                   ↓                ↓
   Anomaly             Distribution       Disagreement      Range &
   Detection            Shift Alert        Detection       Consistency
```

### Input Validation

```python
import torch
from typing import Dict, Optional

class InputValidator:
    """
    Validate model inputs before prediction.
    
    Checks for out-of-distribution inputs, adversarial
    indicators, and data quality issues.
    """
    
    def __init__(
        self,
        feature_means: torch.Tensor,
        feature_stds: torch.Tensor,
        feature_mins: torch.Tensor,
        feature_maxs: torch.Tensor,
        z_threshold: float = 5.0
    ):
        self.means = feature_means
        self.stds = feature_stds
        self.mins = feature_mins
        self.maxs = feature_maxs
        self.z_threshold = z_threshold
    
    def validate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Validate inputs and return quality flags.
        """
        # Z-score check: flag extreme values
        z_scores = (x - self.means) / (self.stds + 1e-8)
        extreme = (z_scores.abs() > self.z_threshold).any(dim=1)
        
        # Range check: flag out-of-training-range values
        below = (x < self.mins).any(dim=1)
        above = (x > self.maxs).any(dim=1)
        out_of_range = below | above
        
        # Missing value check
        has_nan = torch.isnan(x).any(dim=1)
        
        return {
            'valid': ~(extreme | out_of_range | has_nan),
            'extreme_values': extreme,
            'out_of_range': out_of_range,
            'has_missing': has_nan,
            'max_z_score': z_scores.abs().max(dim=1)[0]
        }
```

### Model Monitoring

Continuous monitoring for signs of adversarial activity or model degradation:

| Signal | Indicates | Action |
|--------|-----------|--------|
| Prediction distribution shift | Data drift or manipulation | Alert + investigate |
| Unusual query patterns | Model extraction attempt | Rate limit + block |
| Sudden accuracy drop | Poisoning or distribution shift | Fallback to backup model |
| Feature importance change | Concept drift or attack | Retrain with validation |
| Ensemble disagreement spike | Out-of-distribution inputs | Flag for human review |

### Regulatory Considerations

Financial ML models face specific regulatory requirements:

- **Model Risk Management (SR 11-7)**: Models must be validated, documented, and monitored
- **Explainability requirements**: Adversarial robustness must be documented and tested
- **Fair lending laws**: Robustness evaluation must consider protected attributes
- **Data governance**: Training data integrity must be maintained and auditable

## Comprehensive Security Checklist

For deploying ML models in financial production systems:

- [ ] Adversarial robustness evaluation (AutoAttack or domain-specific)
- [ ] Input validation and anomaly detection
- [ ] Model extraction defenses (rate limiting, output perturbation)
- [ ] Data poisoning resistance (training data validation)
- [ ] Backdoor detection (neural cleanse or similar)
- [ ] Ensemble disagreement monitoring
- [ ] Distribution shift detection
- [ ] Fallback mechanisms (rule-based backup)
- [ ] Audit trail for all model decisions
- [ ] Regular red-team exercises

## Summary

Model security in financial systems requires a holistic approach that goes beyond adversarial robustness of individual predictions. Defense-in-depth—combining input validation, robust models, output monitoring, and organizational processes—provides the most reliable protection against the diverse threat landscape of production financial ML.

## References

1. Kumar, R. S. S., et al. (2020). "Adversarial Machine Learning—Industry Perspectives." IEEE S&P Workshop.
2. Goldblum, M., et al. (2022). "Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses." IEEE TPAMI.
3. Board of Governors of the Federal Reserve System (2011). "Supervisory Guidance on Model Risk Management (SR 11-7)."

## Exercises

**Exercise 1.**
For a linear classifier $f(x) = w^T x + b$, compute the minimal $\ell_\infty$ perturbation needed to change the predicted class. Relate this to the robustness of neural networks.

??? success "Solution to Exercise 1"
    For a linear classifier, the distance to the decision boundary under $\ell_\infty$ norm is $\frac{|w^T x + b|}{\|w\|_1}$. The minimal perturbation is $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$. For neural networks, the local linear approximation $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$ explains why FGSM (which uses the sign of the gradient) is effective. The vulnerability of high-dimensional models comes from the fact that $\|w\|_1$ grows with dimension while $|w^T x + b|$ does not necessarily, making the robustness margin shrink. $\square$

---

**Exercise 2.**
Implement the attack or defense described in this section for a ResNet-18 model on CIFAR-10. Report clean accuracy and robust accuracy under PGD-20 attack with $\epsilon = 8/255$.

??? success "Solution to Exercise 2"
    A standard ResNet-18 achieves $\sim$93% clean accuracy but $\sim$0% robust accuracy under PGD-20 ($\epsilon = 8/255$, step size $2/255$). After applying the method from this section, typical results depend on the specific technique: adversarial training achieves $\sim$83% clean / $\sim$50% robust; certified defenses achieve lower but provable bounds. The accuracy-robustness trade-off is fundamental: improving robustness typically costs 5--15% clean accuracy. Report results averaged over 3 random seeds with standard errors. $\square$

---

**Exercise 3.**
Prove that no defense can simultaneously achieve high accuracy on clean data and high robustness against $\ell_\infty$ perturbations without increasing model capacity, under the assumption that the data distribution has overlapping class-conditional supports within the perturbation ball.

??? success "Solution to Exercise 3"
    If two classes have support overlap within distance $\epsilon$ (i.e., $\exists x_1 \in \text{class 1}, x_2 \in \text{class 2}$ with $\|x_1 - x_2\|_\infty \leq 2\epsilon$), then any classifier robust at both $x_1$ and $x_2$ must misclassify at least one of them (the perturbation balls overlap). This is the fundamental accuracy-robustness trade-off: the fraction of overlapping support determines the unavoidable accuracy loss. For natural image distributions, significant overlap exists at $\epsilon = 8/255$, explaining the observed 10--15% accuracy drop. Increased model capacity (wider networks) can better approximate the complex robust decision boundary, partially mitigating the trade-off. $\square$

---

**Exercise 4.**
Discuss how adversarial robustness concerns manifest in a financial machine learning system (e.g., fraud detection or trading signal generation). How does the threat model differ from computer vision?

??? success "Solution to Exercise 4"
    In finance, adversaries are strategic agents (fraudsters, market manipulators) who actively adapt to detection systems. Key differences from vision: (1) the perturbation space is constrained by what is economically feasible (a fraudster cannot change their entire transaction history); (2) attacks are sequential and adaptive (the adversary observes the system's response and adjusts); (3) the cost of false positives and false negatives is asymmetric (blocking a legitimate transaction vs. missing fraud); (4) $\ell_p$ norms are not meaningful -- domain-specific perturbation models are needed. Defenses must be robust to adaptive adversaries, which rules out many detection-based approaches that can be evaded once the detection criterion is known. $\square$
