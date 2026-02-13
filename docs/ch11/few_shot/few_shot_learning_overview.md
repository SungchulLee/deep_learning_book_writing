# Few-Shot Learning Examples

This collection contains Python implementations of various few-shot learning techniques for advanced students.

## What is Few-Shot Learning?

Few-shot learning is a machine learning paradigm where models learn to recognize new classes from only a few labeled examples (typically 1-5 examples per class). This is in contrast to traditional deep learning which requires thousands of examples.

## Files Included

1. **prototypical_networks.py** - Implementation of Prototypical Networks, a popular metric-learning approach
2. **matching_networks.py** - Matching Networks with attention mechanisms
3. **maml.py** - Model-Agnostic Meta-Learning (MAML) implementation
4. **siamese_network.py** - Siamese Networks for one-shot learning
5. **few_shot_transformer.py** - Modern approach using transformers
6. **data_loader.py** - Episodic data loading utilities for few-shot tasks
7. **evaluation.py** - Evaluation metrics and testing utilities

## Key Concepts

- **N-way K-shot**: N classes with K examples each
- **Support Set**: Small set of labeled examples for each class
- **Query Set**: Examples to classify using the support set
- **Episodic Training**: Training on multiple small tasks to learn how to learn
- **Meta-Learning**: Learning to learn from limited data

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
torchvision>=0.15.0
scikit-learn>=1.0.0
```

## Usage

Each file can be run independently. Most include example usage at the bottom.

```python
python prototypical_networks.py
```

## References

- Snell et al. "Prototypical Networks for Few-shot Learning" (2017)
- Vinyals et al. "Matching Networks for One Shot Learning" (2016)
- Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation" (2017)
