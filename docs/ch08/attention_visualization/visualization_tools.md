# Visualization Tools

## BertViz

The most widely used tool for interactive attention visualization:

```python
from bertviz import model_view, head_view
model_view(outputs.attentions, tokens)  # all layers and heads
head_view(outputs.attentions, tokens)   # detailed view of one layer
```

## Matplotlib Heatmaps

For publication-quality static figures:

```python
def plot_attention_heatmap(attention, tokens, title="Attention"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention.numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
```

## Attention Flow Diagrams

Bipartite graph visualizations show attention flow between source and target tokens, weighted by attention magnitude. Particularly useful for cross-attention in translation and summarization.

## Summary Statistics

For large-scale analysis, visualize aggregate statistics: entropy distributions across layers, head importance rankings, attention distance histograms (average distance of attended tokens), and proportion of attention to special tokens.
