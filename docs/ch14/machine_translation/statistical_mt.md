# Statistical Machine Translation

## Learning Objectives

- Understand the noisy channel formulation of MT
- Learn IBM alignment models and phrase-based translation
- Appreciate the transition from statistical to neural approaches

## The Noisy Channel Model

Statistical MT frames translation as a Bayesian inference problem. To translate source sentence $\mathbf{x}$ (e.g., French) into target $\mathbf{y}$ (e.g., English):

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x}) = \arg\max_{\mathbf{y}} \underbrace{P(\mathbf{x} \mid \mathbf{y})}_{\text{translation model}} \cdot \underbrace{P(\mathbf{y})}_{\text{language model}}$$

The language model ensures fluency; the translation model ensures adequacy.

## Word Alignment

Before translating phrases, we must learn which source words correspond to which target words. The **alignment** $\mathbf{a}$ maps each source position to a target position:

$$P(\mathbf{x}, \mathbf{a} \mid \mathbf{y}) = \prod_{j=1}^{J} P(a_j \mid j, J, I) \cdot P(x_j \mid y_{a_j})$$

### IBM Models

The IBM alignment models (Brown et al., 1993) form a progression of increasing complexity:

| Model | Alignment | Fertility | Distortion |
|-------|-----------|-----------|------------|
| IBM-1 | Uniform | No | No |
| IBM-2 | Position-dependent | No | No |
| IBM-3 | Position-dependent | Yes | Yes |
| IBM-4 | Relative distortion | Yes | Yes |
| IBM-5 | Deficiency fix | Yes | Yes |

Training uses the EM algorithm: E-step computes expected alignments, M-step updates translation probabilities.

## Phrase-Based SMT

Word-level alignment is too rigid. Phrase-based SMT (Koehn et al., 2003) translates contiguous phrase pairs:

### Phrase Table Construction

1. Run word alignment (e.g., GIZA++) in both directions
2. Take intersection and grow using heuristics
3. Extract all phrase pairs consistent with the alignment
4. Score phrases: $\phi(\bar{f} \mid \bar{e})$ and $\phi(\bar{e} \mid \bar{f})$

### Log-Linear Model

The decoder combines multiple features:

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \sum_{m=1}^{M} \lambda_m h_m(\mathbf{x}, \mathbf{y})$$

Features include: phrase translation probabilities, language model score, distortion penalty, word penalty, and phrase penalty. Feature weights $\lambda_m$ are tuned using Minimum Error Rate Training (MERT).

### Decoding

Beam search over partial hypotheses, maintaining coverage vectors to track which source words have been translated.

## Limitations Leading to Neural MT

- **Sparsity**: Many valid phrase pairs never seen in training data
- **No generalization**: Similar phrases treated independently
- **Feature engineering**: Requires manual design of feature functions
- **Limited context**: Each phrase translated relatively independently

## References

1. Brown, P. F., et al. (1993). The Mathematics of Statistical Machine Translation. *Computational Linguistics*.
2. Koehn, P., Och, F. J., & Marcu, D. (2003). Statistical Phrase-Based Translation. *NAACL*.
3. Och, F. J. (2003). Minimum Error Rate Training in Statistical MT. *ACL*.
