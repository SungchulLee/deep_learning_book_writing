# Interpretation Pitfalls

## Attention Is Not Explanation

Jain and Wallace (2019) showed that alternative attention distributions can produce identical predictions. Gradient-based importance and attention weights frequently disagree on which tokens matter most.

## The Residual Connection Problem

In transformers, $x_{l+1} = x_l + \text{Attn}(\text{LN}(x_l))$. The residual connection means information bypasses attention entirely. Raw attention captures only what attention *adds*, not total information flow. Attention rollout (Abnar & Zuidema, 2020) addresses this by recursively combining attention with identity matrices.

## Softmax Saturation

Near-one-hot attention distributions look decisive but may simply reflect confident key-query matches rather than semantic importance.

## Recommendations

1. Never rely on attention alone as explanation — combine with gradient-based methods (Integrated Gradients, SHAP)
2. Use attention rollout instead of raw single-layer attention
3. Compare heads systematically rather than cherry-picking appealing patterns
4. Validate with perturbation: if a high-attention token is masked, does the output change?
5. Report entropy alongside attention maps

## When Attention Is Informative

Cross-attention in translation (aligns well with linguistic alignment), retrieval-augmented models (shows which documents are used), and relative comparisons between correct and incorrect predictions.
