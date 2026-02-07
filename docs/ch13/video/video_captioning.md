# Video Captioning

## Learning Objectives

By the end of this section, you will be able to:

- Understand the sequence-to-sequence formulation for video captioning
- Explain how temporal features are aggregated for caption generation
- Describe the evolution from CNN-LSTM to transformer-based video captioning

## Problem Formulation

Video captioning generates natural language descriptions from video input. Given a video $V = \{f_1, \ldots, f_T\}$ with $T$ frames, the goal is to produce a caption $W = (w_1, \ldots, w_L)$:

$$P(W | V) = \prod_{t=1}^{L} P(w_t | w_{1:t-1}, V)$$

## Architecture Approaches

### Encoder-Decoder (CNN-LSTM)

1. **Encode**: Extract per-frame features with a CNN, then aggregate temporally
2. **Decode**: Generate words autoregressively with an LSTM/GRU conditioned on video features

### Transformer-Based

Modern approaches use transformer decoders with cross-attention to video features, often from pre-trained video encoders (CLIP, TimeSformer).

## Evaluation Metrics

Video captioning uses the same metrics as image captioning: BLEU, METEOR, CIDEr, with CIDEr being the most informative as it rewards informativeness through TF-IDF weighting.

## References

1. Venugopalan, S., et al. (2015). Sequence to Sequence - Video to Text. ICCV.
2. Zhou, L., et al. (2018). End-to-End Dense Video Captioning with Masked Transformer. CVPR.
