# Transformer Probing
## Introduction

**Probing classifiers** investigate what information is encoded in transformer hidden representations by training simple classifiers (typically linear) on frozen activations. Combined with visualization tools like **BertViz**, probing provides a comprehensive picture of what transformers learn at each layer.

## Probing Methodology

### Linear Probes

Given frozen representations $h_l(x)$ at layer $l$, train a linear classifier for a linguistic or task-specific property $y$:

$$
\hat{y} = \sigma(W h_l(x) + b)
$$

If accuracy is high, layer $l$ encodes information about property $y$. The simplicity of the probe ensures that the information exists in the representation rather than being constructed by the probe itself.

### What Probes Reveal

| Layer | Typical Encodings |
|-------|-------------------|
| 0 (embeddings) | Token identity, position |
| 1-3 | POS tags, morphology |
| 4-8 | Syntax, dependency relations |
| 9-12 | Semantics, coreference, sentiment |

### Implementation

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class TransformerProbe:
    """Probing classifier for transformer representations."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_representations(self, texts, layer):
        """Extract hidden states at a specific layer."""
        self.model.eval()
        representations = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors='pt',
                padding=True, truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hidden = outputs.hidden_states[layer]
            cls_rep = hidden[0, 0].cpu().numpy()  # [CLS] token
            representations.append(cls_rep)
        
        return np.array(representations)
    
    def probe_layer(self, texts, labels, layer, test_size=0.2):
        """Train and evaluate a probe at a specific layer."""
        from sklearn.model_selection import train_test_split
        
        reps = self.extract_representations(texts, layer)
        
        X_train, X_test, y_train, y_test = train_test_split(
            reps, labels, test_size=test_size, random_state=42
        )
        
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)
        
        accuracy = probe.score(X_test, y_test)
        return accuracy
    
    def probe_all_layers(self, texts, labels):
        """Probe all layers to find where information is encoded."""
        n_layers = self.model.config.num_hidden_layers + 1
        
        results = {}
        for layer in range(n_layers):
            acc = self.probe_layer(texts, labels, layer)
            results[layer] = acc
            print(f"Layer {layer:2d}: accuracy = {acc:.3f}")
        
        return results
```

## BertViz Integration

BertViz provides interactive visualization of attention patterns:

```python
from bertviz import head_view, model_view

def visualize_with_bertviz(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Interactive visualizations
    head_view(outputs.attentions, tokens)
    model_view(outputs.attentions, tokens)
```

## Probing for Financial Concepts

```python
def probe_financial_model(model, tokenizer, device):
    """Probe a financial text model for domain-specific knowledge."""
    
    probe = TransformerProbe(model, tokenizer, device)
    
    # Test: does the model encode sentiment?
    texts = financial_texts  # List of financial news
    sentiment_labels = [0, 1, 1, 0, ...]  # Bearish/bullish
    
    print("Sentiment encoding by layer:")
    results = probe.probe_all_layers(texts, sentiment_labels)
    
    best_layer = max(results, key=results.get)
    print(f"Best layer for sentiment: {best_layer} (acc={results[best_layer]:.3f})")
```

## Summary

Probing classifiers reveal what information transformers encode at each layer, providing a mechanistic understanding complementary to attention visualization. Combined with BertViz for interactive exploration, they form a powerful toolkit for transformer interpretability.

## References

1. Belinkov, Y. (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics*.

2. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.

3. Tenney, I., et al. (2019). "BERT Rediscovers the Classical NLP Pipeline." *ACL*.

4. Hewitt, J., & Manning, C. D. (2019). "A Structural Probe for Finding Syntax in Word Representations." *NAACL*.\n

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
