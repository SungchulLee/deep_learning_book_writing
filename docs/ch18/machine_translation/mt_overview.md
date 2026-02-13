# Machine Translation Overview

## Learning Objectives

- Trace the evolution of MT from rule-based to neural approaches
- Understand the key paradigm shifts and their motivations
- Identify remaining challenges in modern MT

## Historical Evolution

### Rule-Based MT (1950s-1990s)

The earliest MT systems used hand-crafted linguistic rules: morphological analysis, syntactic parsing, transfer rules, and target-language generation. These required extensive linguistic expertise for each language pair and scaled poorly.

### Statistical MT (1990s-2014)

IBM researchers revolutionized MT by treating translation as a statistical problem. The noisy channel model decomposes translation into:

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x}) = \arg\max_{\mathbf{y}} P(\mathbf{x} \mid \mathbf{y}) \cdot P(\mathbf{y})$$

where $P(\mathbf{x} \mid \mathbf{y})$ is the translation model and $P(\mathbf{y})$ is the language model.

Phrase-based SMT (Koehn et al., 2003) translated phrase-by-phrase rather than word-by-word, capturing local reordering and idioms.

### Neural MT (2014-present)

Sutskever et al. (2014) introduced the encoder-decoder architecture with LSTMs. The attention mechanism (Bahdanau et al., 2015) eliminated the information bottleneck. The Transformer (Vaswani et al., 2017) enabled parallel training and achieved state-of-the-art results.

## Key Paradigm Comparison

| Aspect | Rule-Based | Statistical | Neural |
|--------|-----------|-------------|--------|
| Knowledge | Linguistic rules | Parallel corpora | Parallel corpora |
| Features | Hand-crafted | Phrase tables | Learned representations |
| Fluency | Rigid | Moderate | High |
| Training | Manual rules | EM + MERT | SGD + backprop |
| Scalability | Poor | Moderate | Good |

## Remaining Challenges

- **Low-resource languages**: Many languages lack sufficient parallel data
- **Domain adaptation**: Medical, legal, and financial translation require specialized models
- **Document-level consistency**: Maintaining coherence across paragraphs
- **Rare phenomena**: Idioms, cultural references, humor
- **Evaluation**: Automated metrics imperfectly correlate with human judgment

## References

1. Koehn, P. (2020). *Neural Machine Translation*. Cambridge University Press.
2. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
3. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS*.
