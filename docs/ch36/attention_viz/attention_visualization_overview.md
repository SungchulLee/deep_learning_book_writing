# Module 59: Attention Visualization

## Overview
This module provides comprehensive Python implementations for visualizing attention mechanisms in neural networks, particularly transformers. Attention visualization is crucial for understanding how models process and weigh different parts of input sequences.

## Learning Objectives
By completing this module, students will:
- Understand various attention visualization techniques
- Visualize self-attention and cross-attention weights
- Implement attention heatmaps and rollout methods
- Analyze multi-head attention patterns
- Apply attention flow visualization techniques
- Use attention visualization for model interpretability

## Prerequisites
- Module 25: Attention Mechanisms
- Module 26: Transformers (NLP)
- Module 27: Vision Transformers
- Basic knowledge of PyTorch
- Understanding of attention theory

## Module Structure

### 01_beginner/
**Focus**: Basic attention weight visualization
- `attention_basics.py` - Simple attention weight extraction and visualization
- `self_attention_heatmap.py` - Creating heatmaps for self-attention
- `attention_patterns.py` - Common attention patterns visualization

**Topics Covered**:
- Extracting attention weights from transformer layers
- Creating basic heatmaps with matplotlib/seaborn
- Visualizing token-to-token attention
- Understanding attention score distributions

**Time Estimate**: 3-4 hours

### 02_intermediate/
**Focus**: Multi-head attention and attention rollout
- `multihead_visualization.py` - Visualizing multiple attention heads
- `attention_rollout.py` - Attention rollout algorithm implementation
- `attention_comparison.py` - Comparing attention across layers
- `cross_attention_viz.py` - Cross-attention visualization for seq2seq

**Topics Covered**:
- Multi-head attention analysis
- Attention rollout for capturing long-range dependencies
- Layer-wise attention evolution
- Cross-attention in encoder-decoder architectures
- Statistical analysis of attention distributions

**Time Estimate**: 4-5 hours

### 03_advanced/
**Focus**: Advanced techniques and attention flow
- `attention_flow.py` - Attention flow analysis through layers
- `gradient_attention.py` - Gradient-based attention attribution
- `bertviz_style.py` - Interactive BertViz-style visualizations
- `vit_attention.py` - Vision Transformer attention visualization
- `attention_entropy.py` - Entropy and information-theoretic analysis
- `integrated_attention.py` - Integrated gradients for attention

**Topics Covered**:
- Attention flow computation
- Gradient-weighted attention visualization
- Interactive visualization techniques
- Spatial attention in Vision Transformers
- Entropy and uncertainty in attention
- Attribution methods for attention mechanisms
- Attention head importance analysis

**Time Estimate**: 6-8 hours

## Mathematical Background

### Attention Mechanism Recap
For self-attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q: Query matrix (n × d_k)
- K: Key matrix (n × d_k)
- V: Value matrix (n × d_v)
- n: sequence length
- d_k: key/query dimension

### Attention Weights
The attention weight matrix A is:
```
A = softmax(QK^T / √d_k)
```
where A_ij represents how much token i attends to token j.

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Attention Rollout
For L layers with attention matrices A^(l):
```
Ã^(l) = A^(l) × Ã^(l-1)
Ã^(0) = I (identity)
```

### Attention Flow
Combines attention weights with value gradients:
```
Flow^(l) = A^(l) ⊙ |∇V^(l)|
```

## Installation & Requirements

### Required Libraries
```bash
pip install torch torchvision --break-system-packages
pip install transformers --break-system-packages
pip install matplotlib seaborn --break-system-packages
pip install numpy pandas --break-system-packages
pip install plotly --break-system-packages  # For interactive visualizations
pip install pillow --break-system-packages  # For image processing
pip install scikit-learn --break-system-packages
```

### Optional Libraries
```bash
pip install bertviz --break-system-packages  # For comparison
pip install captum --break-system-packages  # For attribution methods
```

## Usage Examples

### Basic Self-Attention Visualization
```python
from beginner.attention_basics import AttentionVisualizer

# Initialize visualizer
viz = AttentionVisualizer()

# Load or create attention weights
attention_weights = model.get_attention_weights(input_text)

# Visualize
viz.plot_attention_heatmap(
    attention_weights,
    tokens=["The", "cat", "sat", "on", "mat"],
    save_path="attention_heatmap.png"
)
```

### Multi-Head Attention Analysis
```python
from intermediate.multihead_visualization import MultiHeadVisualizer

# Initialize visualizer
viz = MultiHeadVisualizer(num_heads=8)

# Visualize all heads
viz.plot_all_heads(
    attention_weights,  # Shape: (num_heads, seq_len, seq_len)
    tokens=tokens,
    layer_name="Layer 6"
)
```

### Attention Rollout
```python
from intermediate.attention_rollout import AttentionRollout

# Initialize rollout
rollout = AttentionRollout()

# Compute rollout across layers
rollout_attention = rollout.compute_rollout(
    attention_layers,  # List of attention matrices
    discard_ratio=0.1
)

# Visualize
rollout.visualize(rollout_attention, tokens)
```

### Vision Transformer Attention
```python
from advanced.vit_attention import ViTAttentionVisualizer

# Initialize ViT visualizer
viz = ViTAttentionVisualizer()

# Visualize spatial attention
viz.visualize_patch_attention(
    model,
    image,
    patch_size=16,
    head_idx=0,
    layer_idx=-1  # Last layer
)
```

## Key Concepts

### 1. Attention Heatmaps
Visual representations of attention weight matrices where:
- Rows represent query tokens
- Columns represent key tokens
- Color intensity shows attention strength
- Helps identify which tokens the model focuses on

### 2. Multi-Head Patterns
Different attention heads learn different patterns:
- **Local patterns**: Adjacent token attention
- **Syntactic patterns**: Grammar-related attention
- **Semantic patterns**: Meaning-related attention
- **Positional patterns**: Position-based attention

### 3. Attention Rollout
Aggregates attention across layers:
- Captures long-range dependencies
- Shows cumulative attention effects
- More interpretable than single-layer attention
- Useful for understanding deep transformers

### 4. Attention Flow
Combines attention weights with gradients:
- Shows which attention paths are important for predictions
- Gradient-weighted attention
- Helps identify critical information flow
- More accurate for attribution than raw attention

### 5. Attention Entropy
Measures attention distribution:
- Low entropy: Focused attention (peaky distribution)
- High entropy: Diffuse attention (uniform distribution)
- Helps identify attention head specialization
- Useful for model analysis and pruning

## Common Attention Patterns

### Pattern 1: Diagonal Pattern (Local Attention)
```
[1.0 0.1 0.0 0.0]
[0.1 1.0 0.1 0.0]
[0.0 0.1 1.0 0.1]
[0.0 0.0 0.1 1.0]
```
Indicates tokens attend to themselves and immediate neighbors.

### Pattern 2: Vertical Stripes (Broadcasting)
```
[0.1 0.8 0.05 0.05]
[0.1 0.8 0.05 0.05]
[0.1 0.8 0.05 0.05]
[0.1 0.8 0.05 0.05]
```
One token (column) receives attention from all others.

### Pattern 3: Block Pattern (Segment Attention)
```
[1.0 1.0 0.0 0.0]
[1.0 1.0 0.0 0.0]
[0.0 0.0 1.0 1.0]
[0.0 0.0 1.0 1.0]
```
Tokens within segments attend to each other.

### Pattern 4: Beginning-of-Sequence (BOS Attention)
```
[1.0 0.0 0.0 0.0]
[0.8 0.2 0.0 0.0]
[0.6 0.2 0.2 0.0]
[0.5 0.2 0.2 0.1]
```
All tokens attend strongly to the first token.

## Practical Applications

### 1. Model Debugging
- Identify attention collapse (all heads attending similarly)
- Detect degenerate attention patterns
- Find layers that don't learn useful patterns
- Validate model training

### 2. Model Interpretability
- Understand model decisions
- Explain predictions to end users
- Identify important input features
- Build trust in model outputs

### 3. Model Analysis
- Compare different model architectures
- Analyze the effect of design choices
- Study the impact of hyperparameters
- Research attention mechanisms

### 4. Fine-tuning Guidance
- Identify which layers to freeze/unfreeze
- Understand task-specific attention patterns
- Guide architecture modifications
- Optimize model compression

## Best Practices

### 1. Choosing Visualization Type
- **Single sequence**: Use heatmaps
- **Multiple sequences**: Use attention flow
- **Long sequences**: Use attention rollout
- **Vision tasks**: Use spatial visualizations
- **Comparative analysis**: Use head comparison plots

### 2. Interpretation Guidelines
- High attention ≠ high importance (attention is not explanation)
- Consider multiple layers, not just one
- Compare across different heads
- Use gradient-based methods for attribution
- Validate findings with ablation studies

### 3. Computational Considerations
- Attention matrices can be memory-intensive
- Use sampling for very long sequences
- Cache computations when possible
- Vectorize operations for efficiency

### 4. Visualization Design
- Use perceptually uniform colormaps (viridis, plasma)
- Clearly label axes with tokens
- Include color bars with scales
- Save high-resolution images for papers
- Consider accessibility (colorblind-friendly palettes)

## Common Pitfalls

### 1. Over-interpreting Attention
**Issue**: Assuming attention weights directly indicate importance
**Solution**: Use gradient-based attribution methods alongside attention

### 2. Ignoring Layer Effects
**Issue**: Only looking at one layer's attention
**Solution**: Analyze attention across all layers using rollout or flow

### 3. Memory Issues
**Issue**: Loading full attention matrices for long sequences
**Solution**: Use chunking or sample representative portions

### 4. Visualization Overload
**Issue**: Too many plots making interpretation difficult
**Solution**: Focus on key heads/layers, use interactive visualizations

## Exercises

### Beginner Level
1. Extract and visualize attention weights from a pre-trained BERT model
2. Create attention heatmaps for different sentences
3. Identify common attention patterns in your visualizations
4. Compare attention in early vs. late layers

### Intermediate Level
1. Implement attention rollout from scratch
2. Analyze multi-head attention patterns in a 12-layer transformer
3. Visualize cross-attention in a translation model
4. Compute and visualize attention entropy across layers

### Advanced Level
1. Implement attention flow with gradient integration
2. Create an interactive attention visualization dashboard
3. Analyze Vision Transformer attention on image classification
4. Compare attention patterns across different model architectures
5. Implement your own attention attribution method

## Further Reading

### Foundational Papers
1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "Analyzing Multi-Head Self-Attention" (Voita et al., 2019)
3. "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
4. "Attention is not Explanation" (Jain & Wallace, 2019)
5. "Attention is not not Explanation" (Wiegreffe & Pinter, 2019)

### Visualization Tools
1. BertViz: Visualization tool for BERT attention
2. Captum: Model interpretability library
3. Transformer Explainability: Attribution methods for transformers

### Vision Transformer Attention
1. "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
2. "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)

## Assessment

Students should be able to:
- [ ] Extract attention weights from transformer models
- [ ] Create clear attention heatmaps
- [ ] Implement attention rollout
- [ ] Analyze multi-head attention patterns
- [ ] Visualize cross-attention in seq2seq models
- [ ] Compute attention flow with gradients
- [ ] Create interactive attention visualizations
- [ ] Interpret attention for model debugging
- [ ] Apply visualization techniques to Vision Transformers
- [ ] Understand limitations of attention-based interpretability

## License
MIT License - Free for educational use

## Citation
If you use this module in your course or research, please cite:
```
Deep Learning Curriculum - Module 59: Attention Visualization
Comprehensive Python implementations for attention mechanism visualization
```

## Contact & Contributions
For questions, suggestions, or contributions, please reach out to the course instructor.

---

**Last Updated**: November 2025
**Module Difficulty**: Intermediate to Advanced
**Estimated Total Time**: 13-17 hours
