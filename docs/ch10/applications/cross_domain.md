# Cross-Domain Transfer

## Overview

Cross-domain transfer applies knowledge from a well-resourced source domain to a different, data-scarce target domain, presenting additional challenges due to distribution mismatch.

## Domain Distance

Success depends on source-target similarity. Ben-David et al. (2010) formalized this: $\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(S,T) + \lambda^*$, where the middle term measures domain divergence.

## Strategies by Gap Size

Small gap (same modality, similar task): standard fine-tuning. Medium gap (same modality, different task): discriminative learning rates (lower for early layers). Large gap (different modality): feature-based transfer or intermediate domain adaptation.

## Multi-Source Transfer

Weighted combination of multiple sources: $\hat{h}_T = \sum_i w_i \hat{h}_{S_i}$, with weights learned from domain similarity or validation performance.

## Negative Transfer

Transfer hurts when domains are too dissimilar. Detection: compare against randomly initialized baseline. Mitigation: freeze fewer layers, use smaller LR for transferred layers, or use only early layers (which learn more general features).
