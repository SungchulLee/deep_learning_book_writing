# Multilingual Machine Translation

## Learning Objectives

- Understand single-model many-to-many translation
- Explore zero-shot translation capabilities
- Review key multilingual MT models

## Motivation

Training separate models for every language pair is impractical — $N$ languages require $N(N-1)$ directed pairs. Multilingual MT trains a single model for all directions simultaneously.

## Target Language Tag

Johnson et al. (2017) showed that a simple target language tag enables multilingual translation:

```
Input:  <2fr> The cat sat on the mat .
Output: Le chat était assis sur le tapis .

Input:  <2de> The cat sat on the mat .
Output: Die Katze saß auf der Matte .
```

The model learns shared representations across languages, with the tag controlling the output language.

## Zero-Shot Translation

A remarkable emergent capability: the model can translate between language pairs it never saw during training. If trained on EN-FR and EN-DE, it can translate FR-DE directly.

### How It Works

Shared encoder representations create a language-agnostic "interlingua" space. The decoder, conditioned on the target language tag, generates in the requested language regardless of the source.

### Limitations

Zero-shot quality is typically lower than supervised pairs. It often suffers from "off-target" translation (generating in the wrong language). Pivoting through English (FR->EN->DE) remains a competitive baseline.

## Key Models

### mBART (Liu et al., 2020)

Multilingual denoising autoencoder pre-trained on 25 languages. Fine-tuned for MT, excels in low-resource settings.

### M2M-100 (Fan et al., 2021)

First truly many-to-many model trained on 100 languages without using English as a pivot. 2,200 language directions with direct parallel data.

### NLLB-200 (Costa-jussa et al., 2022)

No Language Left Behind covers 200 languages, including many low-resource languages. Uses curriculum training and data augmentation. Available in sizes from 600M to 54B parameters.

## Model Comparison

| Model | Languages | Directions | Parameters |
|-------|-----------|------------|------------|
| mBART-50 | 50 | 2,450 | 680M |
| M2M-100 | 100 | 9,900 | 12B |
| NLLB-200 | 200 | 39,800 | 600M-54B |

## Practical Considerations

- **Capacity bottleneck**: A single model must handle all languages — larger models generally improve quality
- **Data imbalance**: High-resource languages (EN, FR, DE) dominate training; temperature sampling helps balance
- **Evaluation**: Must evaluate across many directions, not just EN-XX

## References

1. Johnson, M., et al. (2017). Google's Multilingual NMT System. *TACL*.
2. Fan, A., et al. (2021). Beyond English-Centric Multilingual MT. *JMLR*.
3. Costa-jussa, M. R., et al. (2022). No Language Left Behind. *arXiv:2207.04672*.
