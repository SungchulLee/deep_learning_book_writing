# Evaluation of Interpretability Methods
## Overview

How do we know if an explanation is "good"? Evaluating interpretability methods is challenging because we typically lack ground truth for what the correct explanation should be. This section introduces the key evaluation dimensions—faithfulness, stability, comprehensiveness, and human-grounded metrics—providing the conceptual foundation for the detailed evaluation methods in Section 35.7.

## Why Evaluation Matters

Beautiful visualizations can be misleading. Adebayo et al. (2018) demonstrated that some popular saliency methods produce visually compelling heatmaps that are **independent of both the model's learned parameters and the training data**. This means the "explanations" could be generated for a random, untrained network and would look nearly identical—a devastating finding for methods that are supposed to reveal what the model has learned.

This motivates rigorous, quantitative evaluation of every interpretability method before trusting its outputs.

## Evaluation Dimensions

### Faithfulness

**Does the explanation accurately reflect the model's decision process?**

An explanation is faithful if removing (or inserting) features identified as important actually changes the model's prediction accordingly. Faithfulness is the most critical property—an unfaithful explanation is worse than no explanation at all.

Key metrics:

$$
\text{Faithfulness} \propto \text{Correlation}\left(\phi_i, \Delta f_{\text{when removing } i}\right)
$$

Methods: Insertion/deletion curves, ROAR (RemOve And Retrain), pixel flipping. See Section 35.7.1 for detailed implementations.

### Stability (Robustness)

**Do similar inputs produce similar explanations?**

An explanation method is stable if small, semantically meaningless perturbations to the input do not dramatically change the explanation. Instability undermines trust: if adding imperceptible noise changes the explanation entirely, practitioners cannot rely on it.

$$
\text{Stability} = 1 - \frac{\|E(\mathbf{x}) - E(\mathbf{x} + \boldsymbol{\epsilon})\|}{\|\boldsymbol{\epsilon}\|}
$$

Methods: Lipschitz estimation, sensitivity to noise, relative stability metrics. See [Section 35.7.2](../evaluation/stability.md).

### Comprehensiveness

**Does the explanation capture all important aspects of the decision?**

A comprehensive explanation identifies all the features that matter, not just a few. The complement of comprehensiveness is **sufficiency**—whether the identified features alone are sufficient to reproduce the prediction.

$$
\text{Comprehensiveness}(E) = f(\mathbf{x}) - f(\mathbf{x}_{\setminus E})
$$

$$
\text{Sufficiency}(E) = f(\mathbf{x}) - f(\mathbf{x}_{E})
$$

where $\mathbf{x}_{\setminus E}$ removes features in explanation $E$ and $\mathbf{x}_E$ keeps only those features. See [Section 35.7.3](../evaluation/comprehensiveness.md).

### Human-Grounded Evaluation

**Do humans find the explanations useful and understandable?**

Ultimately, interpretability exists for human benefit. Human evaluation measures whether explanations help people understand, predict, and appropriately trust model decisions.

Key paradigms: forward simulation (can users predict model output from the explanation?), trust calibration (do explanations improve human-AI team performance?), debugging (can users identify model flaws from explanations?). See Section 35.7.4.

## Sanity Checks

Before detailed quantitative evaluation, every explanation method should pass basic sanity checks:

### Model Randomization Test

Explanations should change meaningfully when model parameters are randomized. If $E(f, \mathbf{x}) \approx E(f_{\text{random}}, \mathbf{x})$, the method is not actually reflecting model behavior.

### Data Randomization Test

Explanations should differ between a model trained on real data and one trained on randomized labels. If not, the method captures input structure rather than model-specific patterns.

### Known-Pattern Test

For synthetic data with known ground-truth attribution (e.g., only features 1 and 3 are used), the method should correctly identify those features.

## Practical Evaluation Protocol

A recommended evaluation protocol for any new interpretability application:

1. **Sanity checks**: Model and data randomization tests
2. **Faithfulness**: Insertion/deletion curves on held-out data
3. **Stability**: Sensitivity to noise and random seeds
4. **Comprehensiveness**: Sufficiency and comprehensiveness scores
5. **Cross-method comparison**: Compare multiple methods on the same inputs
6. **Domain validation**: Have domain experts review explanations for plausibility
7. **Human study** (if resources allow): Forward simulation or debugging tasks

## Summary

Rigorous evaluation is essential for trustworthy interpretability. No single metric captures all desirable properties, so a multi-faceted evaluation approach is recommended. The subsequent sections in 35.7 provide detailed implementations and guidance for each evaluation dimension.

## References

1. Adebayo, J., et al. (2018). "Sanity Checks for Saliency Maps." *NeurIPS*.

2. Hooker, S., et al. (2019). "A Benchmark for Interpretability Methods in Deep Neural Networks." *NeurIPS*.

3. DeYoung, J., et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL*.

4. Nauta, M., et al. (2023). "From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI." *ACM Computing Surveys*.

5. Zhou, J., et al. (2021). "Evaluating the Quality of Machine Learning Explanations: A Survey on Methods and Metrics." *Electronics*.

## Exercises

**Exercise 1.**
Apply the interpretability method described in this section to a 2-layer neural network with ReLU activations classifying XOR inputs. Compute the explanation for the input $x = [1, 1]$.

??? success "Solution to Exercise 1"
    For a trained XOR network with weights $W_1, b_1, W_2, b_2$, the output is $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$. The explanation method produces attributions for each input feature. For $x = [1, 1]$ (class 0), both features contribute to the negative classification. The specific attribution values depend on the method: gradient-based methods compute $\partial f / \partial x_i$; perturbation-based methods measure output change when features are masked. The XOR problem demonstrates that linear explanation methods can mislead because the decision boundary is non-linear. $\square$

---

**Exercise 2.**
Prove or disprove that the explanation method in this section satisfies the completeness axiom: the sum of all feature attributions equals $f(x) - f(x_0)$ for some baseline $x_0$.

??? success "Solution to Exercise 2"
    The completeness axiom (also called efficiency in Shapley value theory) states that attributions sum to the difference between the model output at the input and at the baseline. Whether this method satisfies completeness depends on its formulation. Gradient methods do not satisfy completeness (gradients are local, not path-integrated). Integrated Gradients satisfies completeness by construction (fundamental theorem of calculus along the path). SHAP values satisfy efficiency by the Shapley axiom. Methods that violate completeness may over- or under-attribute, making the total attribution unreliable as a global explanation. $\square$

---

**Exercise 3.**
Design an experiment to evaluate the faithfulness of the explanations produced by this method. Use insertion and deletion curves to measure whether highlighted features are truly important to the model.

??? success "Solution to Exercise 3"
    Protocol: (1) Compute feature attributions for each test image. (2) Deletion: progressively mask features in order of decreasing attribution, recording the model confidence drop. Faithful explanations cause rapid confidence decrease. (3) Insertion: progressively reveal features in order of decreasing attribution from a blank baseline, recording confidence increase. Faithful explanations cause rapid confidence increase. (4) Compute AUC for both curves. (5) Compare against random ordering (baseline) and other methods. A faithful method should have low deletion AUC and high insertion AUC. Repeat over 1000+ test samples for statistical reliability. $\square$

---

**Exercise 4.**
Discuss how this interpretability method could be applied to a financial model predicting credit default. What regulatory requirements must the explanations satisfy?

??? success "Solution to Exercise 4"
    For credit models, regulations (ECOA, GDPR Article 22) require individualized explanations for adverse decisions. The method must produce: (1) the top factors contributing to the denial (adverse action reasons); (2) explanations that are consistent (similar applicants get similar explanations); (3) explanations that are actionable (the applicant understands what to change). The interpretability method from this section can identify feature importances, but must be validated for stability (small input changes should not drastically alter the explanation) and correctness (removing important features should change the prediction). Protected attributes must be handled carefully to avoid revealing proxy discrimination. $\square$
