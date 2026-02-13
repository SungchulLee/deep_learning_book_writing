# Domain Adaptation for NER

## Overview

NER models trained on news data (CoNLL) degrade significantly on other domains. Domain adaptation techniques bridge this gap.

## Challenge: Domain Shift

| Source Domain | Target Domain | Typical F1 Drop |
|--------------|---------------|-----------------|
| News → Biomedical | Gene/protein names | -15 to -25 |
| News → Financial | Company/instrument names | -10 to -20 |
| News → Social media | Informal/noisy text | -15 to -30 |

## Adaptation Strategies

### Continued Pre-training

Pre-train the language model further on target domain text (no labels needed):

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Continue MLM pre-training on financial text
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# Train on domain corpus with masked language modeling...
# Then fine-tune for NER on available labeled data
```

### Few-Shot Fine-Tuning

Fine-tune a pre-trained NER model on small amounts of target domain labels. Even 50-100 labeled sentences can recover 5-10 F1 points.

### Active Learning

Select the most informative target domain examples for annotation:

$$x^* = \arg\max_x \; H(Y | x) = \arg\max_x \left(-\sum_y P(y|x) \log P(y|x)\right)$$

### Data Augmentation

- **Entity replacement**: Swap entities with domain-specific alternatives
- **Contextual augmentation**: Use LMs to generate new training sentences
- **Back-translation**: Translate to another language and back

## Summary

1. Domain shift is a primary bottleneck for NER deployment
2. Continued pre-training on target domain text is the most reliable first step
3. Even small amounts of target domain annotations provide large gains
4. Active learning minimizes annotation cost for domain adaptation
