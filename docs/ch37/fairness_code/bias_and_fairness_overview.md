# Bias and Fairness in Deep Learning

This repository contains Python implementations of bias detection, fairness metrics, and bias mitigation techniques for deep learning models.

## Overview

Machine learning models can inadvertently perpetuate or amplify biases present in training data, leading to unfair outcomes for certain groups. This toolkit provides comprehensive methods for:

1. **Detecting Bias**: Identify biases in datasets and model predictions
2. **Measuring Fairness**: Quantify fairness using various metrics
3. **Mitigating Bias**: Apply techniques to reduce bias in models

## Contents

### 1. `bias_detection.py`
**Bias Detection and Measurement**

Provides the `BiasDetector` class with methods to detect and measure bias:

- **Statistical Parity Difference**: Measures difference in positive prediction rates across groups
- **Disparate Impact Ratio**: Ratio of positive rates (80% rule)
- **Equal Opportunity Difference**: Difference in true positive rates
- **Equalized Odds**: Difference in both TPR and FPR across groups
- **Comprehensive Reporting**: Generate detailed bias reports

**Usage Example:**
```python
from bias_detection import BiasDetector

detector = BiasDetector(['gender', 'race'])
report = detector.generate_bias_report(y_true, y_pred, sensitive_attrs)
print(report)
```

### 2. `fairness_metrics.py`
**Fairness Metrics and Evaluation**

Implements comprehensive fairness metrics:

- **Demographic Parity**: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
- **Equal Opportunity**: TPR should be equal across groups
- **Equalized Odds**: Both TPR and FPR equal across groups
- **Predictive Parity**: Precision should be equal across groups
- **Calibration Metrics**: Assess probability calibration by group
- **Composite Fairness Score**: Combined metric for overall fairness

**Usage Example:**
```python
from fairness_metrics import FairnessMetrics, comprehensive_fairness_evaluation

report = comprehensive_fairness_evaluation(
    y_true, y_pred, y_pred_proba,
    sensitive_attrs={'gender': gender, 'race': race}
)
print(report)
```

### 3. `bias_mitigation.py`
**Bias Mitigation Techniques**

Implements various bias mitigation approaches:

#### Pre-processing:
- **Reweighing**: Assigns weights to training samples to achieve fairness

#### In-processing:
- **Adversarial Debiasing**: Uses adversarial training to remove bias
- **Fair Representation Learning**: Learns representations invariant to sensitive attributes

#### Post-processing:
- **Threshold Optimization**: Optimizes decision thresholds per group

**Usage Example:**
```python
from bias_mitigation import ReweighingMitigation, AdversarialDebiasing

# Reweighing
reweigh = ReweighingMitigation()
weights = reweigh.compute_weights(y, sensitive_attr)

# Adversarial Debiasing
model = AdversarialDebiasing(input_dim=20)
model = train_adversarial_debiasing(model, X_train, y_train, sensitive_train)
```

### 4. `practical_example.py`
**Complete Practical Example**

Demonstrates the full pipeline on a synthetic loan approval dataset:

1. Generates biased dataset
2. Detects data-level bias
3. Trains baseline model
4. Analyzes model fairness
5. Applies bias mitigation
6. Compares results

**Run the example:**
```bash
python practical_example.py
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- PyTorch
- scikit-learn
- pandas

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Concepts

### Fairness Definitions

**Statistical Parity (Demographic Parity)**
- Positive predictions should be equal across groups
- P(Ŷ=1|A=0) = P(Ŷ=1|A=1)

**Equal Opportunity**
- True positive rates should be equal across groups
- P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)

**Equalized Odds**
- Both TPR and FPR should be equal across groups
- Combines equal opportunity with equal false positive rates

**Predictive Parity**
- Precision should be equal across groups
- P(Y=1|Ŷ=1, A=0) = P(Y=1|Ŷ=1, A=1)

### Bias Mitigation Approaches

**Pre-processing**
- Modify training data before model training
- Example: Reweighing samples

**In-processing**
- Modify the learning algorithm during training
- Example: Adversarial debiasing, fair constraints

**Post-processing**
- Adjust model outputs after training
- Example: Threshold optimization

## Best Practices

1. **Understand Your Context**: Different fairness definitions may be appropriate for different applications

2. **Multiple Metrics**: No single metric captures all aspects of fairness. Use multiple metrics.

3. **Trade-offs**: There are often trade-offs between accuracy and fairness, and between different fairness definitions

4. **Domain Expertise**: Involve domain experts and affected communities in defining fairness

5. **Regular Monitoring**: Continuously monitor models for bias in production

6. **Documentation**: Document fairness considerations and decisions

## Limitations

- Binary sensitive attributes: Current implementations primarily handle binary sensitive attributes
- Intersectionality: Limited handling of intersectional identities
- Causality: Metrics are observational, not causal
- Impossibility theorems: Perfect fairness across all definitions is often impossible

## References

- Mehrabi et al. (2021): "A Survey on Bias and Fairness in Machine Learning"
- Barocas et al. (2019): "Fairness and Machine Learning"
- Chouldechova (2017): "Fair Prediction with Disparate Impact"
- Hardt et al. (2016): "Equality of Opportunity in Supervised Learning"
- Feldman et al. (2015): "Certifying and Removing Disparate Impact"

## Contributing

Contributions are welcome! Areas for improvement:
- Support for multi-class sensitive attributes
- Additional mitigation techniques
- More comprehensive examples
- Better handling of intersectionality

## License

MIT License - Feel free to use and modify for your projects.

## Disclaimer

These tools are for educational and research purposes. Bias mitigation is complex and context-dependent. Always:
- Consult with domain experts
- Involve affected communities
- Consider legal and ethical implications
- Validate thoroughly before deployment
- Monitor continuously in production

## Contact

For questions, issues, or contributions, please open an issue on the repository.
