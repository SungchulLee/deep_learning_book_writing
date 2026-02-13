# Quick Start Guide - Module 59: Attention Visualization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt --break-system-packages
```

## Running Examples

### Beginner Level

Start with these to understand attention basics:

```bash
# Basic attention visualization
python 01_beginner_attention_basics.py

# Real transformer models
python 02_beginner_self_attention_heatmap.py

# Pattern recognition
python 03_beginner_attention_patterns.py
```

### Intermediate Level

Progress to multi-head and rollout techniques:

```bash
# Multi-head attention analysis
python 04_intermediate_multihead_visualization.py

# Attention rollout
python 05_intermediate_attention_rollout.py

# Cross-attention for seq2seq
python 06_intermediate_cross_attention.py
```

### Advanced Level

Explore sophisticated visualization techniques:

```bash
# Vision Transformer attention
python 07_advanced_vit_attention.py

# Attention flow with gradients
python 08_advanced_attention_flow.py

# Gradient-based attribution
python 09_advanced_gradient_attention.py
```

## Customization

All scripts can be easily customized:
- Modify the `if __name__ == "__main__"` section
- Change hyperparameters in class initialization
- Add your own examples

## Using with Real Models

To use with Hugging Face models:

```python
from transformers import AutoModel, AutoTokenizer
from beginner.attention_basics import AttentionVisualizer

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Your analysis code here...
```

## Directory Structure

```
module_59_attention_visualization/
├── README.md                                  # Full documentation
├── QUICKSTART.md                              # This file
├── requirements.txt                           # Dependencies
├── 01_beginner_attention_basics.py            # Beginner level
├── 02_beginner_self_attention_heatmap.py
├── 03_beginner_attention_patterns.py
├── 04_intermediate_multihead_visualization.py # Intermediate level
├── 05_intermediate_attention_rollout.py
├── 06_intermediate_cross_attention.py
├── 07_advanced_vit_attention.py               # Advanced level
├── 08_advanced_attention_flow.py
└── 09_advanced_gradient_attention.py
```

## Tips

1. **Start with synthetic data**: Run beginner examples first
2. **GPU recommended**: For real model examples (but CPU works)
3. **First run downloads models**: ~400MB for BERT
4. **Comment out examples**: Each script has multiple examples you can enable/disable
5. **Read the comments**: Heavily documented for learning

## Troubleshooting

**Issue**: ImportError for transformers
**Solution**: `pip install transformers --break-system-packages`

**Issue**: CUDA out of memory
**Solution**: Use smaller models or CPU: `device='cpu'`

**Issue**: Slow visualization
**Solution**: Reduce sequence length or use fewer examples

## Next Steps

After completing these modules:
1. Try with your own models and data
2. Combine multiple visualization techniques
3. Explore the Research Papers section in README.md
4. Build custom attention analysis tools

Happy Learning!
