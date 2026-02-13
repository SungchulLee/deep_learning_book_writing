# Attention Map Extraction

## Overview

Attention maps reveal which input tokens a transformer layer attends to when producing each output. The attention weights are computed as $\alpha_{ij} = \text{softmax}(q_i \cdot k_j / \sqrt{d_k})$, forming a matrix $A \in \mathbb{R}^{n \times n}$ per head.

## Hook-Based Extraction (PyTorch)

```python
class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self._hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                self.attention_maps[name] = output[1].detach().cpu()
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self._hooks.append(
                    module.register_forward_hook(self._hook_fn(name)))

    def extract(self, input_ids, **kwargs):
        self.attention_maps.clear()
        self.register_hooks()
        with torch.no_grad():
            self.model(input_ids, **kwargs)
        for h in self._hooks: h.remove()
        self._hooks.clear()
        return self.attention_maps
```

## HuggingFace Extraction

```python
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
outputs = model(**inputs)
attention = outputs.attentions  # tuple: 12 layers, each (B, H, T, T)
```

## Raw vs Aggregated Attention

For $L$ layers and $H$ heads, raw attention gives $L \times H$ matrices. Aggregation methods include head averaging $\bar{A}_l = \frac{1}{H}\sum_h A_l^{(h)}$, max-pooling, and attention rollout which accounts for residual connections across layers.

## Cross-Attention vs Self-Attention

Encoder self-attention has shape $(n_{src}, n_{src})$, decoder self-attention $(n_{tgt}, n_{tgt})$, and cross-attention $(n_{tgt}, n_{src})$. Cross-attention maps are particularly useful for understanding alignment in translation tasks.

## Practical Notes

Memory for a 12-layer, 12-head model on 512-token input: approximately 150 MB per sample. Consider gradient-weighted attention (multiply weights by gradients) to focus on task-relevant patterns.
