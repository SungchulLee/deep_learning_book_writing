# Low-Resource Machine Translation

## Learning Objectives

- Understand data augmentation techniques for MT
- Apply transfer learning and multilingual pre-training
- Recognize the challenges of truly low-resource languages

## The Low-Resource Challenge

High-quality MT typically requires millions of parallel sentence pairs. Most of the world's 7,000+ languages have little to no parallel data available. Strategies for low-resource MT aim to maximize translation quality with minimal supervised data.

## Back-Translation

The most effective data augmentation technique (Sennrich et al., 2016). Train a target-to-source model on available parallel data, then use it to translate abundant monolingual target text into synthetic source text:

1. Train reverse model: $P(\mathbf{x} \mid \mathbf{y})$ on available parallel data
2. Translate target monolingual data: $\hat{\mathbf{x}} = \text{Translate}(\mathbf{y}_{\text{mono}})$
3. Train forward model on real + synthetic parallel data: $\{(\mathbf{x}, \mathbf{y})\} \cup \{(\hat{\mathbf{x}}, \mathbf{y}_{\text{mono}})\}$

Back-translation provides diverse training signal and is complementary to the original parallel data. Iterative back-translation alternates between forward and reverse models for progressive improvement.

## Transfer Learning

### Parent-Child Transfer

1. Train a "parent" model on a high-resource language pair (e.g., FR-EN)
2. Fine-tune on the low-resource "child" pair (e.g., Wolof-EN)

Related languages transfer best (e.g., Portuguese model for Galician). Even unrelated parent languages provide useful decoder and attention initialization.

### Multilingual Pre-training

Models like mBART and mT5 pre-train on large multilingual corpora, then fine-tune on specific translation pairs. This provides strong initialization even for languages with minimal parallel data.

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

tokenizer.src_lang = "en_XX"
inputs = tokenizer("Machine translation enables cross-lingual communication.", return_tensors="pt")
generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

## Unsupervised MT

For language pairs with zero parallel data, unsupervised MT (Lample et al., 2018) relies on:

1. **Shared BPE**: Joint subword vocabulary captures cross-lingual overlap
2. **Denoising autoencoding**: Reconstruct corrupted sentences in each language
3. **Back-translation**: Iteratively improve using synthetic parallel data
4. **Cross-lingual word embeddings**: Align monolingual embedding spaces

## Data Mining

Automatically mine parallel sentences from the web:

- **Bitext mining**: Use multilingual sentence embeddings (LASER, LaBSE) to find translation pairs in web crawls
- **CCMatrix**: Mined 4.5B parallel sentences across 576 language pairs from CommonCrawl
- **OPUS**: Collection of parallel corpora from web sources

## References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). Improving NMT Models with Monolingual Data. *ACL*.
2. Zoph, B., et al. (2016). Transfer Learning for Low-Resource Neural MT. *EMNLP*.
3. Lample, G., et al. (2018). Unsupervised Machine Translation Using Monolingual Corpora Only. *ICLR*.
