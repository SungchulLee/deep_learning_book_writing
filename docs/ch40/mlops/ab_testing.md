# A/B Testing for Models

## Overview

A/B testing for ML models compares the performance of different model versions in production by routing traffic to each variant and measuring business metrics. Unlike offline evaluation, A/B testing captures real-world effects including user behavior changes, data distribution shifts, and system interactions.

## Architecture

```
Incoming Request → Traffic Router → Model A (control, 90%)
                                  → Model B (treatment, 10%)
                        ↓
               Metrics Collection
                        ↓
               Statistical Analysis
                        ↓
               Decision (promote/rollback)
```

## Implementation

### Traffic Routing

```python
import hashlib
import random

class ABRouter:
    """Route requests to model variants."""
    
    def __init__(self, variants: dict, default: str = 'control'):
        self.variants = variants  # {'control': 0.9, 'treatment': 0.1}
        self.default = default
    
    def route(self, request_id: str) -> str:
        """Deterministic routing based on request ID."""
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        normalized = (hash_val % 10000) / 10000.0
        
        cumulative = 0
        for variant, weight in self.variants.items():
            cumulative += weight
            if normalized < cumulative:
                return variant
        
        return self.default

router = ABRouter({'control': 0.9, 'treatment_v2': 0.1})
```

### Metrics Collection

```python
from dataclasses import dataclass, field
from typing import List
import numpy as np
from scipy import stats

@dataclass
class ABMetrics:
    variant: str
    predictions: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    business_metric: List[float] = field(default_factory=list)

def statistical_significance(control: ABMetrics, treatment: ABMetrics,
                            alpha: float = 0.05) -> dict:
    """Test for statistically significant difference."""
    t_stat, p_value = stats.ttest_ind(
        control.business_metric,
        treatment.business_metric
    )
    
    control_mean = np.mean(control.business_metric)
    treatment_mean = np.mean(treatment.business_metric)
    lift = (treatment_mean - control_mean) / control_mean
    
    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'lift': lift,
        'p_value': p_value,
        'significant': p_value < alpha,
        'recommendation': 'promote' if (p_value < alpha and lift > 0) else 'keep_control'
    }
```

## Best Practices

- **Use deterministic routing** (hash-based) for consistent user experience
- **Run tests long enough** for statistical significance (minimum 1-2 weeks typical)
- **Monitor guardrail metrics** (latency, error rate) alongside business metrics
- **Use sequential testing** for early stopping when results are clear
- **Document and review** all A/B test results for organizational learning

## References

1. Kohavi, R., et al. "Trustworthy Online Controlled Experiments." Cambridge University Press, 2020.
2. Google A/B Testing: https://developers.google.com/machine-learning/crash-course/static/experiment-design.pdf
