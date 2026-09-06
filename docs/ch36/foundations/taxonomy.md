# Taxonomy of Interpretability Methods
## Overview

The landscape of interpretability methods is vast and rapidly growing. A clear taxonomy helps practitioners select the right tool for their specific needs. This section classifies methods along three orthogonal dimensions: scope (local vs global), model access (agnostic vs specific), and explanation type (feature, example, or concept-based).

## Classification by Scope

### Local Interpretability

Local methods explain individual predictions. For a specific input $\mathbf{x}$, local methods answer: "Why did the model predict $f(\mathbf{x}) = \hat{y}$?"

*Example*: "This loan application was rejected because the debt-to-income ratio exceeded 0.45 and the applicant has fewer than 2 years of credit history."

Local methods produce explanations of the form:

$$
f(\mathbf{x}) \approx g(\mathbf{x}) = \phi_0 + \sum_{i=1}^{d} \phi_i(\mathbf{x})
$$

where $\phi_i(\mathbf{x})$ is the attribution of feature $i$ for this specific input.

**Key local methods**: LIME, SHAP values, Grad-CAM, Integrated Gradients, saliency maps, counterfactual explanations.

### Global Interpretability

Global methods explain overall model behavior. They answer: "What general patterns has the model learned across the input space?"

*Example*: "The credit model primarily relies on payment history (35% importance), credit utilization (25%), and length of credit history (20%)."

Global methods aggregate local explanations or directly analyze model structure:

$$
I_j = \mathbb{E}_{\mathbf{x}}[|\phi_j(\mathbf{x})|]
$$

**Key global methods**: Permutation importance, aggregated SHAP, TCAV, concept bottleneck models, partial dependence plots.

### Semi-Local Methods

Some methods operate between local and global scope, explaining model behavior in a region of input space. Subgroup explanations and rule extraction methods fall in this category.

## Classification by Model Access

### Model-Agnostic Methods

Model-agnostic methods treat the model as a black box, using only input-output relationships. These methods work with any model architecture.

| Method | Mechanism | Complexity |
|--------|-----------|------------|
| LIME | Local surrogate fitting | $O(N \cdot d)$ per sample |
| Kernel SHAP | Weighted regression on coalitions | $O(2^d)$ exact, sampled in practice |
| Permutation Importance | Feature shuffling | $O(N \cdot d)$ |
| Partial Dependence | Marginal effect estimation | $O(N \cdot G)$ grid points |
| Counterfactual Explanations | Optimization for minimal change | Varies |

**Advantages**: Universal applicability, no architecture assumptions, can compare across model types.

**Limitations**: Computationally expensive (many model evaluations), may miss architecture-specific insights, sampling can introduce variance.

### Model-Specific Methods

Model-specific methods exploit internal model structure for more precise and efficient explanations.

**For differentiable models (neural networks)**:

| Method | Exploits | Architecture |
|--------|----------|-------------|
| Vanilla Gradients | Backpropagation | Any differentiable model |
| Grad-CAM | Feature map gradients | CNNs with spatial feature maps |
| Integrated Gradients | Path integral of gradients | Any differentiable model |
| SmoothGrad | Averaged gradients | Any differentiable model |
| LRP | Layer-wise decomposition | Neural networks |
| DeepLIFT | Reference-based activations | Neural networks |

**For attention-based models (transformers)**:

| Method | Exploits | Architecture |
|--------|----------|-------------|
| Attention Visualization | Raw attention weights | Transformers |
| Attention Rollout | Cumulative attention | Multi-layer transformers |
| Attention Flow | Gradient-weighted attention | Transformers |
| Probing Classifiers | Hidden representations | Any encoder |

**For tree-based models**:

| Method | Exploits | Architecture |
|--------|----------|-------------|
| Tree SHAP | Tree structure | Decision trees, ensembles |
| Feature Importance (Gini/gain) | Split statistics | Tree ensembles |
| Decision Path | Path through tree | Single trees |

## Classification by Explanation Type

### Feature Attribution

Feature attribution assigns importance scores to input features. These methods answer: "How much did each feature contribute to this prediction?"

$$
f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{d} \phi_i(\mathbf{x})
$$

This additive decomposition is the unifying framework for SHAP, Integrated Gradients, LRP, and DeepLIFT. The key distinction among methods is *how* they compute the attributions $\phi_i$.

| Method | Attribution Mechanism | Completeness |
|--------|----------------------|-------------|
| Vanilla Gradients | Local sensitivity $\partial f / \partial x_i$ | No |
| Gradient × Input | Sensitivity × value $x_i \cdot \partial f / \partial x_i$ | No |
| Integrated Gradients | Path integral from baseline | Yes |
| SHAP | Marginal contributions over coalitions | Yes |
| LRP | Backward relevance propagation | Yes |
| DeepLIFT | Difference from reference | Yes |

### Example-Based Explanations

Example-based methods explain predictions by referencing similar or influential training instances.

**Prototype-based explanations**: "This input is classified as class $c$ because it is most similar to prototype $p_c$."

**Influential instances**: "This prediction was most affected by training examples $\{x_1, x_3, x_{42}\}$."

**Counterfactual explanations**: "The prediction would change from 'reject' to 'approve' if the debt-to-income ratio were reduced from 0.45 to 0.35."

### Concept-Based Explanations

Concept-based methods explain in terms of human-understandable concepts rather than raw features.

**Concept Activation Vectors (CAVs)**: Learn a direction in activation space that corresponds to a human concept (e.g., "striped", "furry", "high volatility").

**Testing with CAVs (TCAV)**: Quantifies how sensitive a model's predictions are to a specific concept:

$$
\text{TCAV}_{c,k,l} = \frac{|\{x \in X_c : \nabla h_l(x) \cdot v_l^k > 0\}|}{|X_c|}
$$

**Concept Bottleneck Models**: Force the network to first predict human-interpretable concepts, then use those concepts for the final prediction.

## Unified View

Many seemingly different methods are connected through a unified framework. Lundberg and Lee (2017) showed that several methods are special cases of additive feature attribution:

$$
g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i
$$

where $z' \in \{0, 1\}^M$ is a simplified binary representation of the input and $\phi_i$ are feature attributions. Different choices of the loss function, weighting kernel, and regularization recover LIME, SHAP, and other methods.

| Method | Kernel $\pi_{x'}(z')$ | Loss $\mathcal{L}$ | Regularization $\Omega$ |
|--------|----------------------|---------------------|------------------------|
| LIME | $\exp(-D(x, z)^2 / \sigma^2)$ | Squared error | $L_1$ or feature count |
| SHAP | $\frac{M-1}{\binom{M}{|z'|}|z'|(M-|z'|)}$ | Squared error | None (unique solution) |
| Integrated Gradients | — | — | Path integral |

## Method Selection Decision Tree

```
Start
├── Need to explain a specific prediction?
│   ├── Yes → Local method
│   │   ├── Model is differentiable?
│   │   │   ├── Yes → Gradient-based (fast) or SHAP (rigorous)
│   │   │   └── No → LIME or Kernel SHAP
│   │   ├── Need theoretical guarantees?
│   │   │   ├── Yes → SHAP or Integrated Gradients
│   │   │   └── No → Grad-CAM (CNN) or vanilla gradients (quick)
│   │   └── Regulatory requirement?
│   │       ├── Yes → SHAP (documented properties)
│   │       └── No → Any appropriate method
│   └── No → Global method
│       ├── Feature importance ranking?
│       │   ├── Yes → Permutation importance or aggregated SHAP
│       │   └── No → Partial dependence or concept methods
│       └── Concept-level understanding?
│           ├── Yes → TCAV, Concept Bottleneck
│           └── No → Global surrogate or rule extraction
└── Model architecture?
    ├── CNN → Grad-CAM, feature visualization
    ├── Transformer → Attention analysis, probing
    ├── Tree ensemble → Tree SHAP
    └── Any → LIME, Kernel SHAP
```

## Summary

Understanding the taxonomy of interpretability methods is essential for selecting the right tool. The three key dimensions—scope, model access, and explanation type—provide a structured framework for navigating the growing landscape of methods. In practice, combining methods from different categories provides the most comprehensive understanding of model behavior.

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." *KDD*.

3. Molnar, C. (2020). *Interpretable Machine Learning*. Chapter 5: Model-Agnostic Methods.

4. Guidotti, R., et al. (2018). "A Survey of Methods for Explaining Black Box Models." *ACM Computing Surveys*.

5. Murdoch, W. J., et al. (2019). "Definitions, methods, and applications in interpretable machine learning." *PNAS*.

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
