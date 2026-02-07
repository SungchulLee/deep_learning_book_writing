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