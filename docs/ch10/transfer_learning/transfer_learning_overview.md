# PyTorch Transfer Learning Tutorial for Undergraduates

Welcome! This tutorial package contains four progressively challenging examples of transfer learning using PyTorch. Each example is fully commented and designed to help you understand transfer learning concepts step by step.

## 📚 What is Transfer Learning?

Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task. Instead of training a neural network from scratch, we leverage knowledge learned from large datasets (like ImageNet) and adapt it to our specific problem.

**Why use transfer learning?**
- ⚡ Faster training times
- 🎯 Better performance with limited data
- 💾 Requires less computational resources
- 🏆 Often achieves state-of-the-art results

## 📂 Project Structure

```
pytorch_transfer_learning_tutorial/
│
├── README.md (this file)
├── requirements.txt
│
├── example_1_basic/
│   ├── README.md
│   └── basic_transfer_learning.py
│
├── example_2_fine_tuning/
│   ├── README.md
│   └── fine_tuning.py
│
├── example_3_custom_dataset/
│   ├── README.md
│   └── custom_dataset_transfer.py
│
└── example_4_advanced/
    ├── README.md
    └── advanced_techniques.py
```

## 🎓 Examples Overview

### Example 1: Basic Transfer Learning (Feature Extraction)
**Difficulty: ⭐ Beginner**

Learn the fundamentals of transfer learning by using a pre-trained ResNet18 model as a fixed feature extractor. You'll only train a new classifier head on the CIFAR-10 dataset.

**Key Concepts:**
- Loading pre-trained models
- Freezing layers
- Feature extraction
- Training only the final classifier

### Example 2: Fine-Tuning
**Difficulty: ⭐⭐ Intermediate**

Build on Example 1 by learning how to fine-tune the entire network or specific layers. This allows the model to adapt more closely to your specific task.

**Key Concepts:**
- Selective layer unfreezing
- Different learning rates for different layers
- Fine-tuning strategies
- Validation and early stopping

### Example 3: Custom Dataset
**Difficulty: ⭐⭐⭐ Intermediate-Advanced**

Apply transfer learning to your own custom dataset using PyTorch's Dataset and DataLoader classes. Learn proper data handling and augmentation techniques.

**Key Concepts:**
- Custom Dataset classes
- Data augmentation
- Train/validation/test splits
- Handling imbalanced datasets

### Example 4: Advanced Techniques
**Difficulty: ⭐⭐⭐⭐ Advanced**

Explore advanced transfer learning techniques including learning rate scheduling, different architectures, and training optimizations.

**Key Concepts:**
- Learning rate schedulers
- Multiple architectures (ResNet, VGG, EfficientNet)
- Mixed precision training
- Model ensemble
- Gradient accumulation

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python and neural networks
- Familiarity with PyTorch (helpful but not required)

### Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Start with Example 1 and progress through the examples in order.

### Running the Examples

Each example is self-contained and can be run independently:

```bash
# Example 1
cd example_1_basic
python basic_transfer_learning.py

# Example 2
cd example_2_fine_tuning
python fine_tuning.py

# Example 3
cd example_3_custom_dataset
python custom_dataset_transfer.py

# Example 4
cd example_4_advanced
python advanced_techniques.py
```

## 📖 Learning Path

We recommend following this learning path:

1. **Start with Example 1**: Understand the basics of transfer learning
2. **Move to Example 2**: Learn fine-tuning strategies
3. **Practice with Example 3**: Apply to custom datasets
4. **Explore Example 4**: Master advanced techniques

## 💡 Tips for Success

- Read the code comments carefully - they explain every step
- Experiment with different hyperparameters
- Try using different pre-trained models
- Visualize your results (accuracy, loss curves)
- Start with small datasets to iterate quickly

## 🔧 Common Issues & Solutions

### Out of Memory Errors
- Reduce batch size
- Use smaller input image sizes
- Enable gradient checkpointing (shown in Example 4)

### Model Not Learning
- Check learning rate (try 1e-3, 1e-4, 1e-5)
- Verify data normalization
- Ensure proper train/eval mode switching

### Slow Training
- Use GPU if available
- Enable mixed precision training (Example 4)
- Reduce image resolution during prototyping

## 📚 Additional Resources

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

## 🤝 Contributing

Feel free to modify and extend these examples for your own learning! Try:
- Using different datasets
- Experimenting with various architectures
- Adding new augmentation techniques
- Implementing additional metrics

## 📝 License

This educational material is provided as-is for learning purposes.

## ⚠️ Note

These examples use publicly available datasets. The first run will download datasets automatically, which may take some time depending on your internet connection.

---

**Happy Learning! 🎉**

If you find these examples helpful, consider sharing them with fellow students!
