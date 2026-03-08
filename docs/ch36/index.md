# Chapter 36: Model Interpretability

This chapter provides a comprehensive treatment of interpretability methods for deep learning models, from foundational gradient-based techniques to advanced concept-level explanations. Understanding why models make specific predictions is essential for regulatory compliance, debugging, and building trust in high-stakes domains such as quantitative finance, healthcare, and autonomous systems. The chapter covers both post-hoc explanation methods and inherently interpretable architectures, with rigorous evaluation frameworks.

---

## Foundations

Core concepts and taxonomies for understanding the interpretability landscape.

- [Introduction to Model Interpretability](foundations/introduction.md) -- The black box problem, regulatory drivers, and why interpretability matters
- [Taxonomy of Interpretability Methods](foundations/taxonomy.md) -- Classification by scope (local/global), model access, and explanation type
- [Evaluation of Interpretability Methods](foundations/evaluation.md) -- Faithfulness, stability, comprehensiveness, and human-grounded evaluation dimensions

## Gradient Methods

Gradient-based visualization techniques that reveal which input features influence model predictions.

- Model Interpretability Overview -- Comprehensive toolkit overview for Grad-CAM and attention visualization
- Saliency Maps Overview -- Module overview of gradient-based saliency map techniques
- Quick Start Guide -- Getting started with saliency map implementations in 5 minutes
- [Saliency Maps and Vanilla Gradients](gradient_methods/saliency_maps.md) -- Mathematical foundations of gradient-based saliency for neural network predictions
- [Integrated Gradients](gradient_methods/integrated_gradients.md) -- Path-integrated attributions satisfying sensitivity and implementation invariance axioms
- [Grad-CAM](gradient_methods/gradcam.md) -- Gradient-weighted class activation mapping for CNN visual explanations
- [Grad-CAM++](gradient_methods/gradcam_plusplus.md) -- Improved localization for multiple object instances and small regions
- [Guided Backpropagation](gradient_methods/guided_backprop.md) -- High-resolution saliency maps via modified ReLU gradient propagation
- [SmoothGrad](gradient_methods/smoothgrad.md) -- Noise-based gradient averaging for sharper, cleaner saliency visualizations

## Attention Visualization

Techniques for understanding transformer attention patterns and information flow.

- Attention Visualization Overview -- Module overview of attention visualization tools and techniques
- Quick Start Guide -- Running attention visualization examples with pre-trained transformers
- Attention Fundamentals -- Mathematical foundations of scaled dot-product and multi-head attention for visualization
- Attention Rollout -- Cumulative attention computation across all layers for total input influence
- Attention Flow -- Combining attention weights with gradients for accurate attribution
- [Attention Pattern Analysis](attention_viz/pattern_analysis.md) -- Multi-head analysis, layer-wise progression, and cross-attention interpretation

## Feature Attribution

Model-agnostic methods for attributing predictions to input features using game theory.

- SHAP: SHapley Additive exPlanations -- Unified feature attribution based on Shapley values from cooperative game theory
- LIME: Local Interpretable Model-agnostic Explanations -- Local surrogate models for explaining individual predictions of any black-box model
- [Kernel SHAP](feature_attribution/kernel_shap.md) -- Approximating SHAP values via weighted linear regression
- [Deep SHAP](feature_attribution/deep_shap.md) -- Combining DeepLIFT backpropagation with SHAP for efficient neural network attribution
- [Tree SHAP](feature_attribution/tree_shap.md) -- Exact polynomial-time Shapley values for tree-based models
- [Feature Interaction Effects](feature_attribution/interactions.md) -- SHAP interaction values for detecting non-linear feature dependencies

## Concept Methods

Human-interpretable concept-level explanations beyond raw feature attributions.

- [Concept Activation Vectors (CAV)](concept_methods/cav.md) -- Testing whether neural networks have learned specific human-understandable concepts
- [TCAV: Testing with Concept Activation Vectors](concept_methods/tcav.md) -- Quantitative statistical tests for concept importance in model predictions
- [Concept Bottleneck Models](concept_methods/concept_bottleneck.md) -- Inherently interpretable architectures with explicit concept prediction layers
- [Prototype Networks](concept_methods/prototypes.md) -- Example-based explanations via learned prototypical cases

## Model-Specific Methods

Interpretability techniques tailored to specific neural network architectures.

- [CNN Visualization and Decomposition](model_specific/cnn_visualization.md) -- Layer-wise Relevance Propagation and DeepLIFT for convolutional networks
- [GNN Explanation Methods](model_specific/gnn_explanation.md) -- GNNExplainer, PGExplainer, and SubgraphX for graph neural networks
- [Transformer Probing](model_specific/transformer_probing.md) -- Probing classifiers and BertViz for analyzing transformer hidden representations
- [Feature Inversion](model_specific/feature_inversion.md) -- Reconstructing inputs from intermediate representations to reveal what the model sees

## Evaluation

Rigorous methods for assessing explanation quality and reliability.

- Faithfulness Evaluation -- Insertion/deletion curves measuring whether explanations reflect true model behavior
- [Stability Evaluation](evaluation/stability.md) -- Relative input stability and max-sensitivity metrics for explanation robustness
- [Comprehensiveness Evaluation](evaluation/comprehensiveness.md) -- Testing whether explanations capture all important features via sufficiency scores
- Human-Centered Evaluation -- Forward simulation, trust calibration, and user study paradigms

## Finance Applications

Applying interpretability methods to quantitative finance use cases.

- Regulatory Compliance -- SR 11-7, GDPR Article 22, and implementation patterns for financial model explainability
- Trading Signal Analysis -- Decomposing buy/sell signals by feature category for validation and surveillance
- [Factor Attribution](finance/factor_attribution.md) -- SHAP-based decomposition of portfolio returns into systematic factor contributions
- Credit Risk Explanation -- Adverse action reasons and regulatory-compliant explanations for credit decisions
