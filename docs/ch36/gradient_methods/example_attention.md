# Example Usage

Example Usage: Attention Visualization for Transformers This script demonstrates how to visualize attention patterns in transformer models.

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
Example Usage: Attention Visualization for Transformers

This script demonstrates how to visualize attention patterns in transformer models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from attention_visualization import (

# ========================================================================
# Main
# ========================================================================
    AttentionVisualizer, 
    BERTAttentionVisualizer,
    visualize_token_attention
)

# Note: These examples require transformers library
# Install with: pip install transformers


def example_simple_attention():
    """
    Example 1: Visualize a synthetic attention pattern
    """
    print("=" * 60)
    print("Example 1: Simple Attention Visualization")
    print("=" * 60)
    
    # Create synthetic attention weights
    # Shape: (batch=1, heads=8, seq_len=10, seq_len=10)
    seq_len = 10
    num_heads = 8
    
    attention = torch.zeros(1, num_heads, seq_len, seq_len)
    
    # Create different patterns for different heads
    for h in range(num_heads):
        if h == 0:  # Diagonal pattern (self-attention)
            attention[0, h] = torch.eye(seq_len)
        elif h == 1:  # Attend to previous token
            attention[0, h] = torch.diag(torch.ones(seq_len - 1), -1)
        elif h == 2:  # Attend to first token
            attention[0, h, :, 0] = 1.0
        else:  # Random pattern
            attention[0, h] = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
    
    # Create tokens
    tokens = [f"Token_{i}" for i in range(seq_len)]
    
    # Visualize
    print("Creating visualization...")
    fig = visualize_token_attention(attention, tokens, layer_idx=0)
    plt.savefig('attention_simple.png', dpi=150, bbox_inches='tight')
    print("Saved to 'attention_simple.png'")
    plt.close()
    
    # Visualize all heads
    from attention_visualization import AttentionVisualizer
    visualizer = AttentionVisualizer(None)  # No model needed for this example
    fig = visualizer.visualize_all_heads(attention, tokens, max_heads=8)
    plt.savefig('attention_all_heads.png', dpi=150, bbox_inches='tight')
    print("Saved to 'attention_all_heads.png'")
    plt.close()


def example_bert_attention():
    """
    Example 2: Visualize BERT attention
    
    Requires: pip install transformers
    """
    print("\n" + "=" * 60)
    print("Example 2: BERT Attention Visualization")
    print("=" * 60)
    
    try:
        from transformers import BertTokenizer, BertModel
    except ImportError:
        print("Transformers library not found. Install with: pip install transformers")
        return
    
    # Load model and tokenizer
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    
    # Input text
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Input text: {text}")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Tokens: {tokens}")
    
    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract attention weights
    # outputs.attentions is a tuple of (num_layers) tensors
    # Each tensor has shape (batch, heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    print(f"\nNumber of layers: {len(attentions)}")
    print(f"Attention shape per layer: {attentions[0].shape}")
    
    # Visualize last layer
    print("\nVisualizing last layer attention...")
    fig = visualize_token_attention(attentions, tokens, layer_idx=-1)
    plt.savefig('bert_attention_last_layer.png', dpi=150, bbox_inches='tight')
    print("Saved to 'bert_attention_last_layer.png'")
    plt.close()
    
    # Visualize all heads of last layer
    visualizer = BERTAttentionVisualizer(model)
    fig = visualizer.visualize_all_heads(attentions[-1], tokens, max_heads=12)
    plt.savefig('bert_attention_all_heads.png', dpi=150, bbox_inches='tight')
    print("Saved to 'bert_attention_all_heads.png'")
    plt.close()
    
    # Visualize attention flow from a specific token
    query_token = "fox"
    query_idx = tokens.index(query_token)
    fig = visualizer.plot_attention_flow(attentions[-1], tokens, query_idx)
    plt.savefig('bert_attention_flow.png', dpi=150, bbox_inches='tight')
    print("Saved to 'bert_attention_flow.png'")
    plt.close()


def example_gpt2_attention():
    """
    Example 3: Visualize GPT-2 attention
    
    Requires: pip install transformers
    """
    print("\n" + "=" * 60)
    print("Example 3: GPT-2 Attention Visualization")
    print("=" * 60)
    
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
    except ImportError:
        print("Transformers library not found. Install with: pip install transformers")
        return
    
    # Load model and tokenizer
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
    model.eval()
    
    # Input text
    text = "Artificial intelligence is transforming"
    print(f"Input text: {text}")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Tokens: {tokens}")
    
    # Get outputs with attention
    with torch.no_grad():
        outputs = model(**inputs)
    
    attentions = outputs.attentions
    
    print(f"\nNumber of layers: {len(attentions)}")
    print(f"Attention shape per layer: {attentions[0].shape}")
    
    # Visualize causal attention pattern (GPT-2 has causal masking)
    print("\nVisualizing causal attention pattern...")
    fig = visualize_token_attention(attentions, tokens, layer_idx=-1)
    plt.savefig('gpt2_attention_causal.png', dpi=150, bbox_inches='tight')
    print("Saved to 'gpt2_attention_causal.png'")
    plt.close()
    
    # Visualize multiple layers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    layer_indices = [0, 2, 4, 6, 8, 11]  # Sample layers
    
    for idx, layer_idx in enumerate(layer_indices):
        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()
        
        im = axes[idx].imshow(attn, cmap='viridis', aspect='auto')
        axes[idx].set_title(f'Layer {layer_idx}')
        axes[idx].set_xlabel('Key Position')
        axes[idx].set_ylabel('Query Position')
        
        if len(tokens) <= 10:
            axes[idx].set_xticks(range(len(tokens)))
            axes[idx].set_yticks(range(len(tokens)))
            axes[idx].set_xticklabels(tokens, rotation=90, fontsize=8)
            axes[idx].set_yticklabels(tokens, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('gpt2_attention_layers.png', dpi=150, bbox_inches='tight')
    print("Saved to 'gpt2_attention_layers.png'")
    plt.close()


def example_attention_patterns():
    """
    Example 4: Analyze different attention patterns
    """
    print("\n" + "=" * 60)
    print("Example 4: Common Attention Patterns")
    print("=" * 60)
    
    seq_len = 12
    tokens = [f"T{i}" for i in range(seq_len)]
    
    # Create different attention patterns
    patterns = {
        'Local': torch.zeros(seq_len, seq_len),
        'Global': torch.zeros(seq_len, seq_len),
        'Causal': torch.zeros(seq_len, seq_len),
        'Dilated': torch.zeros(seq_len, seq_len),
    }
    
    # Local attention (window of 3)
    for i in range(seq_len):
        start = max(0, i - 1)
        end = min(seq_len, i + 2)
        patterns['Local'][i, start:end] = 1.0
    patterns['Local'] = torch.softmax(patterns['Local'], dim=-1)
    
    # Global attention (attend to all)
    patterns['Global'] = torch.ones(seq_len, seq_len) / seq_len
    
    # Causal attention (attend to past)
    for i in range(seq_len):
        patterns['Causal'][i, :i+1] = 1.0
    patterns['Causal'] = torch.softmax(patterns['Causal'], dim=-1)
    
    # Dilated attention (attend to positions at fixed intervals)
    for i in range(seq_len):
        for j in range(0, seq_len, 3):
            patterns['Dilated'][i, j] = 1.0
    patterns['Dilated'] = torch.softmax(patterns['Dilated'], dim=-1)
    
    # Visualize all patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        im = axes[idx].imshow(pattern.numpy(), cmap='viridis', aspect='auto')
        axes[idx].set_title(f'{name} Attention Pattern')
        axes[idx].set_xlabel('Key Position')
        axes[idx].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    print("Saved to 'attention_patterns.png'")
    plt.close()


def example_attention_statistics():
    """
    Example 5: Compute attention statistics
    """
    print("\n" + "=" * 60)
    print("Example 5: Attention Statistics")
    print("=" * 60)
    
    # Create synthetic multi-layer attention
    num_layers = 12
    num_heads = 8
    seq_len = 16
    
    attentions = []
    for _ in range(num_layers):
        attn = torch.softmax(torch.randn(1, num_heads, seq_len, seq_len), dim=-1)
        attentions.append(attn)
    
    # Compute statistics
    print("\nComputing attention statistics...")
    
    # Entropy (how focused is the attention?)
    entropies = []
    for layer_attn in attentions:
        # Average over batch and heads
        attn = layer_attn[0].mean(dim=0)
        # Compute entropy for each query position
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
        entropies.append(entropy.item())
    
    # Attention distance (how far does attention reach?)
    distances = []
    for layer_attn in attentions:
        attn = layer_attn[0].mean(dim=0)
        positions = torch.arange(seq_len).float()
        avg_dist = 0
        for i in range(seq_len):
            weighted_pos = (attn[i] * positions).sum()
            avg_dist += abs(weighted_pos - i)
        distances.append(avg_dist.item() / seq_len)
    
    # Plot statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(range(num_layers), entropies, marker='o')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Entropy')
    axes[0].set_title('Attention Entropy by Layer')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(num_layers), distances, marker='o', color='coral')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average Distance')
    axes[1].set_title('Attention Distance by Layer')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_statistics.png', dpi=150, bbox_inches='tight')
    print("Saved to 'attention_statistics.png'")
    plt.close()
    
    print(f"\nLayer-wise statistics:")
    for i in range(num_layers):
        print(f"  Layer {i}: Entropy={entropies[i]:.3f}, Distance={distances[i]:.3f}")


if __name__ == "__main__":
    print("Attention Visualization Examples\n")
    
    # Run examples
    example_simple_attention()
    example_attention_patterns()
    example_attention_statistics()
    
    # These require transformers library
    try:
        example_bert_attention()
        example_gpt2_attention()
    except Exception as e:
        print(f"\nSkipping transformer examples: {e}")
        print("Install transformers with: pip install transformers")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)```

## Discussion

Visualization plays an important role in understanding model behavior and diagnosing training issues. The plotting code provides insight into the learned representations, convergence dynamics, or evaluation metrics, making abstract computations tangible.

The patterns demonstrated here extend naturally to more complex scenarios. Experimenting with hyperparameters, architectural variations, and different datasets deepens understanding and builds practical intuition for model interpretability tasks.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for gradient-based explanation.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

---

**Exercise 2.**
Add a dropout layer after the attention weights (before multiplying with values). Use a dropout rate of 0.1 during training. Explain why attention dropout helps with regularization.

??? success "Solution to Exercise 2"
    Add `self.attn_dropout = nn.Dropout(0.1)` in `__init__` and apply it after the softmax: `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. Attention dropout randomly zeroes some attention weights during training, preventing the model from relying too heavily on specific token-to-token relationships. This encourages the model to distribute attention more broadly and learn more robust representations, similar to how standard dropout prevents co-adaptation of neurons.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Write a comprehensive test function that validates the Example Usage implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_example usage():
        model = Example Usage(...)
        # Normal input
        assert model(normal_input).shape == expected_shape
        # Single element batch
        assert model(single_input).shape == (1, ...)
        # Large values (check for overflow)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # Gradient flow
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    Testing gradient flow is especially important to ensure the architecture supports end-to-end training.
