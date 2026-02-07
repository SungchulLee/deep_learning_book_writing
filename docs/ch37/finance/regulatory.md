# Regulatory Framework for AI Fairness in Finance

## Overview

Financial institutions deploying ML models face a complex web of regulations that either explicitly or implicitly require fairness. This section maps the regulatory landscape and provides guidance for compliance.

## Key Regulations

### United States

| Regulation | Agency | Scope | Fairness Requirement |
|-----------|--------|-------|---------------------|
| ECOA | CFPB | Lending | Prohibits discrimination on protected characteristics |
| Fair Housing Act | HUD | Housing/mortgage | Prohibits discriminatory housing practices |
| SR 11-7 | OCC/Fed | Model risk | Requires model validation including fairness |
| Fair Credit Reporting Act | FTC/CFPB | Credit reporting | Accuracy and fairness of credit information |
| Dodd-Frank §1071 | CFPB | Small business lending | Requires demographic data collection and fair lending |

### European Union

| Regulation | Scope | Fairness Requirement |
|-----------|-------|---------------------|
| GDPR Art. 22 | Automated decisions | Right to explanation, human review |
| EU AI Act | High-risk AI | Mandatory fairness assessment, conformity testing |
| Gender Directive | Insurance | Prohibits gender-based pricing |
| PSD2/MiFID II | Trading/payments | Best execution, fair access |

## Compliance Framework

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"

@dataclass
class ComplianceChecklist:
    """Regulatory compliance checklist for fair ML in finance."""
    model_name: str
    use_case: str
    risk_level: RiskLevel
    checks: Dict[str, bool] = field(default_factory=dict)
    documentation: Dict[str, str] = field(default_factory=dict)
    
    def assess(self) -> Dict[str, bool]:
        """Run compliance assessment."""
        self.checks = {
            # Data governance
            'protected_attrs_identified': False,
            'proxy_analysis_completed': False,
            'data_quality_assessed': False,
            
            # Model fairness
            'fairness_metrics_computed': False,
            'four_fifths_rule_passed': False,
            'statistical_significance_tested': False,
            
            # Documentation
            'model_card_created': False,
            'adverse_action_reasons_defined': False,
            'fairness_tradeoffs_documented': False,
            
            # Governance
            'model_risk_review_completed': False,
            'ongoing_monitoring_established': False,
            'escalation_procedures_defined': False,
        }
        return self.checks
    
    def generate_model_card(self) -> str:
        """Generate a model card template for regulatory submission."""
        return f"""
MODEL CARD — {self.model_name}
{'=' * 50}
Use Case: {self.use_case}
Risk Level: {self.risk_level.value}

1. MODEL DETAILS
   - Architecture: [describe]
   - Training data: [describe, including demographic breakdown]
   - Features used: [list, noting any proxy risk]
   - Protected attributes: [list]

2. INTENDED USE
   - Primary: {self.use_case}
   - Out-of-scope uses: [list]

3. FAIRNESS EVALUATION
   - Criteria used: [DP / EO / Calibration / etc.]
   - Metrics:
     - SPD: [value]
     - DIR: [value]
     - TPR difference: [value]
   - Threshold: 4/5 rule (DIR >= 0.8)
   - Result: [PASS/FAIL]

4. TRADEOFF ANALYSIS
   - Accuracy impact of fairness constraints: [quantify]
   - Impossibility theorem implications: [discuss if base rates differ]

5. MONITORING PLAN
   - Frequency: [e.g., monthly]
   - Metrics tracked: [list]
   - Alert thresholds: [list]
   - Escalation: [procedure]

6. LIMITATIONS
   - [Known limitations and boundary conditions]
"""


def demo():
    checklist = ComplianceChecklist(
        model_name="CreditScore_v3",
        use_case="Consumer credit approval",
        risk_level=RiskLevel.HIGH,
    )
    
    checks = checklist.assess()
    print("Regulatory Compliance Checklist")
    print("=" * 50)
    for check, status in checks.items():
        print(f"  {'✓' if status else '☐'} {check}")
    
    print("\n" + checklist.generate_model_card())

if __name__ == "__main__":
    demo()
```

## Best Practices for Financial Institutions

1. **Treat fairness as a first-class model risk**: Include fairness metrics in standard model validation
2. **Document everything**: Regulatory examiners expect detailed records of fairness decisions and tradeoffs
3. **Use model cards**: Standardized documentation for each deployed model
4. **Monitor continuously**: Fairness can degrade over time due to data drift and population shifts
5. **Engage legal counsel**: The interaction between ML fairness and anti-discrimination law is complex
6. **Prepare for adverse action**: Models must be able to explain why an applicant was denied

## Summary

- Financial ML operates under **extensive regulatory oversight** requiring fairness
- **Model cards** and **compliance checklists** standardize the documentation process
- The **four-fifths rule** is the most common legal test but not the only one
- **Continuous monitoring** is required by model risk management frameworks
- Practitioners must navigate tensions between **actuarial accuracy** and **non-discrimination**
