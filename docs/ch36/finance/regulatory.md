# Regulatory Compliance

## Introduction

Financial model interpretability is not just a technical concern—it is a regulatory requirement. This section covers the key regulatory frameworks requiring model explainability and provides implementation patterns for compliance.

## Regulatory Frameworks

### SR 11-7: Model Risk Management

The Federal Reserve's SR 11-7 guidance requires:

- **Model validation**: Understanding model assumptions and limitations
- **Documentation**: Explaining model logic and behavior
- **Ongoing monitoring**: Detecting model degradation

### GDPR Article 22

The EU's GDPR establishes rights for individuals to receive explanations of automated decisions, including "meaningful information about the logic involved."

### MiFID II

The Markets in Financial Instruments Directive requires transparency in algorithmic trading decisions and best execution analysis.

### Basel III/IV

Risk model frameworks demand transparency in model assumptions, validation of risk calculations, and documentation of model limitations.

## Model Documentation

```python
import numpy as np
import shap

def generate_model_documentation(model, test_data, feature_names):
    """Generate documentation for regulatory compliance."""
    doc = {
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }
    
    # Global feature importance
    explainer = shap.Explainer(model, test_data[:100])
    shap_values = explainer(test_data[:1000])
    
    global_importance = np.abs(shap_values.values).mean(axis=0)
    doc['global_feature_importance'] = dict(
        zip(feature_names, global_importance)
    )
    
    sorted_idx = np.argsort(global_importance)[::-1]
    doc['top_10_features'] = [feature_names[i] for i in sorted_idx[:10]]
    doc['base_prediction'] = explainer.expected_value
    
    return doc
```

## VaR Attribution

```python
class RiskModelExplainer:
    """Explain risk model predictions (VaR, CVaR)."""
    
    def __init__(self, risk_model, factor_names):
        self.risk_model = risk_model
        self.factor_names = factor_names
    
    def explain_var(self, portfolio_exposures):
        """Explain VaR prediction by attributing to factors."""
        var_95 = self.risk_model.predict_var(
            portfolio_exposures, confidence=0.95
        )
        
        # Component VaR using marginal contributions
        epsilon = 1e-6
        marginal_var = np.zeros(len(portfolio_exposures))
        
        for i in range(len(portfolio_exposures)):
            perturbed = portfolio_exposures.copy()
            perturbed[i] += epsilon
            var_perturbed = self.risk_model.predict_var(perturbed, 0.95)
            marginal_var[i] = (var_perturbed - var_95) / epsilon
        
        # Euler allocation
        component_var = marginal_var * portfolio_exposures
        component_var = component_var * var_95 / component_var.sum()
        
        return {
            'total_var_95': var_95,
            'component_var': dict(zip(self.factor_names, component_var)),
            'marginal_var': dict(zip(self.factor_names, marginal_var)),
            'diversification_benefit': component_var.sum() - var_95
        }
```

## Explanation Drift Monitoring

```python
from scipy.stats import ks_2samp

class ExplanationMonitor:
    """Monitor for drift in model explanations over time."""
    
    def __init__(self, model, feature_names, baseline_explanations):
        self.model = model
        self.feature_names = feature_names
        self.baseline = baseline_explanations
    
    def compute_explanation_drift(self, current_data):
        """Detect drift in feature attributions."""
        explainer = shap.Explainer(self.model)
        current_shap = explainer(current_data)
        
        drift_metrics = {}
        for i, feature in enumerate(self.feature_names):
            baseline_dist = self.baseline[:, i]
            current_dist = current_shap.values[:, i]
            
            ks_stat, p_value = ks_2samp(baseline_dist, current_dist)
            mean_shift = current_dist.mean() - baseline_dist.mean()
            
            drift_metrics[feature] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'mean_shift': mean_shift,
                'significant_drift': p_value < 0.05
            }
        
        return drift_metrics
    
    def generate_drift_alert(self, drift_metrics):
        """Generate alert if significant drift detected."""
        drifted_features = [
            f for f, m in drift_metrics.items() if m['significant_drift']
        ]
        
        if drifted_features:
            return {
                'status': 'WARNING',
                'message': f'Explanation drift in {len(drifted_features)} features',
                'features': drifted_features,
                'details': {f: drift_metrics[f] for f in drifted_features}
            }
        return {'status': 'OK', 'message': 'No significant drift'}
```

## Interpretability Checklist

```python
def interpretability_checklist():
    """Checklist for deploying interpretable financial models."""
    return {
        'documentation': [
            'Model assumptions documented',
            'Feature engineering explained',
            'Training data characteristics recorded',
            'Performance metrics on different segments'
        ],
        'explanation_methods': [
            'Global feature importance computed',
            'Local explanations available for individual predictions',
            'Interaction effects analyzed',
            'Explanations validated against domain knowledge'
        ],
        'regulatory_compliance': [
            'Explanations meet GDPR requirements',
            'SR 11-7 model documentation complete',
            'Adverse action reasons identifiable',
            'Audit trail for explanations maintained'
        ],
        'monitoring': [
            'Explanation drift monitoring in place',
            'Feature importance stability tracked',
            'Alerts for significant changes configured',
            'Regular explanation quality reviews scheduled'
        ]
    }
```

## Summary

Regulatory compliance requires documented, stable, and auditable model explanations. A comprehensive interpretability framework combines global and local explanation methods with ongoing monitoring and drift detection.

## References

1. Federal Reserve. (2011). "Supervisory Guidance on Model Risk Management (SR 11-7)."
2. European Union. (2016). "General Data Protection Regulation (GDPR)."
3. Rudin, C. (2019). "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead." *Nature Machine Intelligence*.
4. Chen, J., et al. (2018). "The Model Explanation System." *FAT** Conference.
5. Bhatt, U., et al. (2020). "Explainable Machine Learning in Deployment." *FAT** Conference.
