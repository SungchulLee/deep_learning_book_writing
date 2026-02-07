# Trading Signal Analysis

## Introduction

Understanding why a trading model generates specific signals is critical for risk management, model validation, and regulatory compliance. Interpretability methods reveal which features drive buy/sell decisions, enabling traders to validate economic intuition, detect overfitting to spurious patterns, and satisfy market surveillance requirements.

## Signal Explanation Framework

### Feature Category Attribution

Trading signals typically combine features from multiple categories. Decomposing attribution by category reveals the signal's economic drivers:

```python
import numpy as np
import shap

class TradingSignalExplainer:
    """Explain trading model signals."""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def explain_signal(self, market_state):
        """Explain why model generated a particular signal."""
        signal = self.model.predict(market_state.reshape(1, -1))[0]
        signal_prob = self.model.predict_proba(
            market_state.reshape(1, -1)
        )[0]
        
        explainer = shap.Explainer(self.model)
        shap_values = explainer(market_state.reshape(1, -1))
        
        # Categorize features
        technical_features = [f for f in self.feature_names 
                            if any(x in f.lower() for x in 
                                  ['rsi', 'macd', 'sma', 'bb', 'volume'])]
        fundamental_features = [f for f in self.feature_names 
                               if any(x in f.lower() for x in 
                                     ['pe', 'pb', 'roe', 'eps'])]
        sentiment_features = [f for f in self.feature_names 
                             if any(x in f.lower() for x in 
                                   ['sentiment', 'news', 'social'])]
        
        def category_contribution(features):
            indices = [self.feature_names.index(f) for f in features 
                      if f in self.feature_names]
            if indices:
                return shap_values.values[0, indices].sum()
            return 0
        
        explanation = {
            'signal': ['SELL', 'HOLD', 'BUY'][signal + 1],
            'confidence': signal_prob.max(),
            'technical_contribution': category_contribution(technical_features),
            'fundamental_contribution': category_contribution(fundamental_features),
            'sentiment_contribution': category_contribution(sentiment_features),
            'top_features': []
        }
        
        sorted_idx = np.argsort(np.abs(shap_values.values[0]))[::-1]
        for idx in sorted_idx[:5]:
            explanation['top_features'].append({
                'name': self.feature_names[idx],
                'value': market_state[idx],
                'contribution': shap_values.values[0, idx]
            })
        
        return explanation
```

### Temporal Attribution

For sequential models (LSTM, Transformer), saliency reveals which historical time steps drive the current signal:

```python
import torch

def temporal_signal_attribution(
    model, sequence, target_class=None
):
    """
    Identify which historical time steps drive the trading signal.
    """
    sequence = sequence.clone().requires_grad_(True)
    
    model.eval()
    output = model(sequence)
    
    if target_class is not None:
        output = output[0, target_class]
    else:
        output = output.squeeze()
    
    model.zero_grad()
    output.backward()
    
    saliency = sequence.grad.abs().squeeze().cpu().numpy()
    
    if saliency.ndim == 2:
        saliency = saliency.sum(axis=1)  # Sum across features
    
    return saliency
```

## Signal Validation

Interpretability helps validate that trading signals reflect genuine market patterns rather than spurious correlations:

### Sanity Check: Calendar Effects

```python
def check_calendar_spuriousness(model, explanations, feature_names):
    """
    Check if model relies on calendar features (potential spurious signals).
    """
    calendar_features = [f for f in feature_names 
                        if any(x in f.lower() for x in 
                              ['day_of_week', 'month', 'quarter', 'holiday'])]
    
    calendar_attribution = 0
    total_attribution = 0
    
    for exp in explanations:
        for feat in exp['top_features']:
            total_attribution += abs(feat['contribution'])
            if feat['name'] in calendar_features:
                calendar_attribution += abs(feat['contribution'])
    
    ratio = calendar_attribution / (total_attribution + 1e-10)
    
    if ratio > 0.2:
        print(f"WARNING: {ratio:.1%} of attribution goes to calendar features")
        print("Model may be learning spurious temporal patterns")
    
    return ratio
```

## Summary

Trading signal explanation reveals the economic drivers behind model decisions, enables detection of spurious patterns, and supports market surveillance compliance. Decomposing attribution by feature category provides actionable insight for portfolio managers.

## References

1. Chen, J., et al. (2018). "The Model Explanation System." *FAT** Conference.
2. Bhatt, U., et al. (2020). "Explainable Machine Learning in Deployment." *FAT** Conference.
