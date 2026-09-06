# GNN Explanation Methods
## Introduction

**Graph Neural Networks (GNNs)** process graph-structured data where both node features and graph topology affect predictions. Explaining GNN predictions requires attributing importance not just to input features but also to **edges** and **subgraph structures**. This section covers GNNExplainer, PGExplainer, and SubgraphX.

## The GNN Explanation Problem

For a GNN prediction $\hat{y} = f(G, X)$ where $G = (V, E)$ is the graph and $X$ are node features, we seek:

1. **Node importance**: Which nodes matter for this prediction?
2. **Edge importance**: Which connections are critical?
3. **Feature importance**: Which node features drive the prediction?
4. **Subgraph explanation**: What minimal subgraph explains the prediction?

## GNNExplainer

GNNExplainer (Ying et al., 2019) learns soft masks on edges and features:

$$
\max_{M_E, M_F} MI(Y, (G_s, X_s)) = H(Y) - H(Y | G = G_s, X = X_s)
$$

where $G_s = G \odot M_E$ is the masked graph and $X_s = X \odot M_F$ are masked features.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNExplainer:
    """GNNExplainer for node and graph classification."""
    
    def __init__(self, model, num_hops=3, lr=0.01, epochs=100):
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.epochs = epochs
    
    def explain_node(self, node_idx, x, edge_index, target=None):
        self.model.eval()
        
        if target is None:
            with torch.no_grad():
                out = self.model(x, edge_index)
                target = out[node_idx].argmax().item()
        
        num_edges = edge_index.shape[1]
        edge_mask = nn.Parameter(torch.randn(num_edges) * 0.1)
        
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            mask = torch.sigmoid(edge_mask)
            masked_edge_weight = mask
            
            out = self.model(x, edge_index, edge_weight=masked_edge_weight)
            log_prob = F.log_softmax(out[node_idx], dim=0)
            
            # Maximize prediction probability
            pred_loss = -log_prob[target]
            
            # Sparsity regularization
            size_loss = mask.sum() * 0.01
            
            # Entropy for discrete masks
            entropy = -mask * torch.log(mask + 1e-10) - (1-mask) * torch.log(1-mask + 1e-10)
            entropy_loss = entropy.mean() * 0.1
            
            loss = pred_loss + size_loss + entropy_loss
            loss.backward()
            optimizer.step()
        
        return torch.sigmoid(edge_mask).detach()
```

## Applications in Quantitative Finance

GNN explanation is relevant for financial networks:

- **Counterparty risk**: Which connections in a banking network contribute to systemic risk?
- **Supply chain**: Which supplier relationships most affect a company's risk profile?
- **Social networks**: Which information flows in trader networks predict market movements?

```python
def explain_financial_network(model, node_idx, x, edge_index):
    """Explain predictions in a financial network."""
    explainer = GNNExplainer(model)
    edge_importance = explainer.explain_node(node_idx, x, edge_index)
    
    # Top connections
    top_edges = edge_importance.argsort(descending=True)[:10]
    print(f"Most important connections for node {node_idx}:")
    for idx in top_edges:
        src, dst = edge_index[:, idx]
        print(f"  {src.item()} -> {dst.item()}: {edge_importance[idx]:.3f}")
    
    return edge_importance
```

## Summary

GNN explanation methods extend interpretability to graph-structured data by attributing importance to nodes, edges, and substructures. This is particularly relevant for financial network analysis where relational structure drives risk and return.

## References

1. Ying, R., et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *NeurIPS*.

2. Luo, D., et al. (2020). "Parameterized Explainer for Graph Neural Network." *NeurIPS*.

3. Yuan, H., et al. (2021). "On Explainability of Graph Neural Networks via Subgraph Explorations." *ICML*.\n

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
