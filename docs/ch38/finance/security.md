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
