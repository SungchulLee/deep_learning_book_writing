# Conditioning Mechanisms

## Overview

WaveNet can be conditioned on external information to control the generated audio, enabling applications like text-to-speech, music generation conditioned on MIDI, and speaker-conditioned speech synthesis.

## Global Conditioning

A single conditioning vector $h$ (e.g., speaker identity) is applied uniformly across all time steps:

$$y = \tanh(W_f * x + V_f h) \odot \sigma(W_g * x + V_g h)$$

where $V_f$ and $V_g$ are linear projections of the conditioning vector, broadcast across time.

## Local Conditioning

A time-varying conditioning signal $h_t$ (e.g., linguistic features from a TTS frontend) provides different information at each time step:

$$y_t = \tanh(W_f * x_t + V_f * h_t) \odot \sigma(W_g * x_t + V_g * h_t)$$

The conditioning signal typically operates at a lower temporal resolution than the audio and is upsampled (via transposed convolution or repetition) to match.

## Text-to-Speech Pipeline

```
Text → Text Analysis → Linguistic Features → WaveNet → Audio
         (phonemes,      (local conditioning
          duration,       signal, ~200 Hz)
          prosody)
```

The linguistic features encode phoneme identity, duration, pitch contour, and other prosodic information at ~5 ms resolution. WaveNet upsamples this to 16 kHz audio.

## Multi-Speaker Conditioning

Combine global (speaker embedding) and local (linguistic features) conditioning:

$$h = h_{\text{local}}(t) + h_{\text{speaker}}$$

The speaker embedding is learned end-to-end and captures voice characteristics (timbre, pitch range, speaking rate).
