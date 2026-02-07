# Credit Risk Explanation

## Introduction

Credit risk models face unique interpretability demands: regulatory requirements mandate that lenders provide specific reasons when adverse actions are taken (loan denial, unfavorable terms). The Equal Credit Opportunity Act (ECOA), Fair Credit Reporting Act (FCRA), and GDPR all require explanations for automated credit decisions.

## Regulatory Requirements

### Adverse Action Reasons

Under US regulations, lenders must provide the **top 4 specific reasons** for adverse credit decisions. Interpretability methods must identify concrete, actionable factors—not vague statistical attributions.

### GDPR Right to Explanation

The EU's GDPR requires that individuals receive "meaningful information about the logic involved" in automated decisions affecting them.

## Implementation

### Comprehensive Credit Decision Explainer

```python
import numpy as np
import shap

class CreditDecisionExplainer:
    """
    Comprehensive explainer for credit risk models.
    """
    
    def __init__(self, model, feature_names, training_data):
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.explainer = shap.Explainer(model.predict_proba, training_data)
    
    def explain_decision(self, applicant):
        """Generate full explanation for credit decision."""
        prob_default = self.model.predict_proba(
            applicant.reshape(1, -1)
        )[0, 1]
        decision = "Approved" if prob_default < 0.3 else "Declined"
        
        shap_values = self.explainer(applicant.reshape(1, -1))
        
        explanation = {
            'decision': decision,
            'default_probability': prob_default,
            'base_rate': self.explainer.expected_value[1],
            'factors': []
        }
        
        for i, (name, value, shap_val) in enumerate(zip(
            self.feature_names, applicant,
            shap_values.values[0, :, 1]
        )):
            explanation['factors'].append({
                'name': name,
                'value': value,
                'contribution': shap_val,
                'percentile': self._compute_percentile(i, value)
            })
        
        explanation['factors'].sort(
            key=lambda x: abs(x['contribution']), reverse=True
        )
        
        return explanation
    
    def _compute_percentile(self, feature_idx, value):
        feature_values = self.training_data[:, feature_idx]
        return (feature_values < value).mean() * 100
    
    def generate_report(self, applicant):
        """Generate regulatory-compliant credit report."""
        explanation = self.explain_decision(applicant)
        
        report = []
        report.append("=" * 60)
        report.append("CREDIT DECISION REPORT")
        report.append("=" * 60)
        report.append(f"\nDecision: {explanation['decision']}")
        report.append(f"Default Probability: {explanation['default_probability']:.1%}")
        report.append(f"Threshold: 30%")
        report.append(f"\nBase Rate: {explanation['base_rate']:.1%}")
        
        report.append("\n" + "-" * 60)
        report.append("KEY FACTORS")
        report.append("-" * 60)
        
        for factor in explanation['factors'][:10]:
            direction = "increases risk" if factor['contribution'] > 0 else "decreases risk"
            report.append(
                f"\n{factor['name']:30s}: {factor['value']:10.2f}"
                f" (percentile: {factor['percentile']:.0f}%)"
                f"\n  Impact: {factor['contribution']:+.4f} {direction}"
            )
        
        return "\n".join(report)
    
    def generate_adverse_action_reasons(self, applicant, n_reasons=4):
        """Generate top adverse action reasons for regulatory compliance."""
        explanation = self.explain_decision(applicant)
        
        negative_factors = [
            f for f in explanation['factors'] if f['contribution'] > 0
        ]
        
        reasons = []
        for factor in negative_factors[:n_reasons]:
            reasons.append({
                'reason': factor['name'],
                'your_value': factor['value'],
                'impact': factor['contribution'],
                'percentile': factor['percentile']
            })
        
        return reasons
```

### Consumer-Friendly Explanation

```python
def generate_consumer_explanation(model, applicant, feature_names, feature_values):
    """Generate consumer-friendly explanation for GDPR compliance."""
    shap_values = explain_prediction(model, applicant)
    
    explanation = {
        'decision': 'Approved' if model.predict(applicant) == 1 else 'Declined',
        'primary_factors': [],
        'suggestions_for_improvement': []
    }
    
    sorted_idx = np.argsort(shap_values)
    
    for idx in sorted_idx[:3]:
        if shap_values[idx] < 0:
            explanation['primary_factors'].append({
                'factor': feature_names[idx],
                'impact': 'negative',
                'your_value': feature_values[idx]
            })
    
    return explanation


def format_consumer_explanation(explanation):
    """Format for non-technical audience."""
    text = f"CREDIT DECISION: {explanation['decision']}\n\n"
    text += "Main factors in your application:\n"
    
    for factor in explanation['primary_factors']:
        impact = "worked against" if factor['impact'] == 'negative' else "worked for"
        text += f"\n- Your {factor['factor']} {impact} your application"
    
    if explanation['suggestions_for_improvement']:
        text += "\n\nSuggestions for improving future applications:"
        for suggestion in explanation['suggestions_for_improvement']:
            text += f"\n- {suggestion}"
    
    return text
```

## Best Practices

1. **Use SHAP for regulatory compliance**: Well-documented theoretical properties make audit defense easier
2. **Validate against domain knowledge**: Explanations should align with credit risk expertise
3. **Test stability**: Same applicant profile should get consistent explanations across runs
4. **Document methodology**: Record baseline choices, background data, and any approximations
5. **Provide actionable feedback**: Explanations should suggest how applicants can improve

## Summary

Credit risk explanation requires methods that produce stable, documented, and actionable explanations meeting regulatory requirements. SHAP values with proper documentation provide the strongest foundation for compliance.

## References

1. Federal Reserve. (2011). "Supervisory Guidance on Model Risk Management (SR 11-7)."
2. European Union. (2016). "General Data Protection Regulation (GDPR)."
3. Bhatt, U., et al. (2020). "Explainable Machine Learning in Deployment." *FAT** Conference.
