# Human-Centered Evaluation

## Introduction

Ultimately, interpretability exists for human benefit. While automated metrics measure explanation quality algorithmically, **human studies** evaluate whether explanations actually help people understand, trust, and work effectively with AI systems.

## Evaluation Paradigms

### Forward Simulation

**Can users predict the model's output given the explanation?**

Show participants an input and its explanation, ask them to predict the model's output, and measure accuracy. Higher accuracy means the explanation effectively communicates the model's reasoning.

### Trust Calibration

**Do explanations help users appropriately calibrate their trust?**

Show model predictions with and without explanations. Users decide whether to follow or override the model. Good explanations should increase trust in correct predictions and decrease trust in incorrect ones.

### Debugging Task

**Can users identify model errors from explanations?**

Train models with known biases, show participants predictions and explanations, and ask them to identify whether the model uses valid features.

### Comparative Evaluation

Show the same prediction with explanations from multiple methods and ask users to rate usefulness, clarity, and trustworthiness.

## Metrics

| Metric | Measures | Scale |
|--------|----------|-------|
| Prediction accuracy | Forward simulation quality | 0-100% |
| Trust calibration | Appropriate AI reliance | AUROC |
| Task completion time | Cognitive load | Seconds |
| User satisfaction | Subjective quality | Likert 1-7 |
| Inter-rater agreement | Explanation clarity | Cohen's kappa |

## Study Design Considerations

### Participant Expertise

Different audiences evaluate explanations differently:

| Audience | Focus | Method Preference |
|----------|-------|-------------------|
| ML engineers | Technical accuracy | Detailed attribution maps |
| Domain experts | Domain relevance | Concept-based, factor-level |
| Regulators | Compliance, documentation | Stable, reproducible methods |
| End users | Actionability | Simple, text-based explanations |

### Sample Size and Power

Human studies in XAI typically require 30-100 participants for between-subjects designs. Within-subjects designs are more powerful but risk order effects.

## Applications in Finance

Human evaluation is particularly important in finance:

- **Regulators** must understand explanations to approve model deployment
- **Portfolio managers** must trust explanations to act on model signals
- **Compliance officers** must verify explanations meet audit requirements
- **Risk managers** must assess whether explanations reveal true risk drivers

## Summary

Human studies provide the ultimate validation of interpretability methods, capturing clarity, actionability, and trust calibration that automated metrics miss.

## References

1. Doshi-Velez, F., & Kim, B. (2017). "Towards A Rigorous Science of Interpretable Machine Learning." *arXiv:1702.08608*.

2. Hase, P., & Bansal, M. (2020). "Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?" *ACL*.

3. Chandrasekaran, A., et al. (2018). "Do Explanations make VQA Models more Predictable to a Human?" *EMNLP*.
