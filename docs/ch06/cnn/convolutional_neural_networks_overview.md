# PyTorch CNN Tutorial Package for Undergraduates

A comprehensive, progressively challenging collection of Convolutional Neural Network (CNN) tutorials using PyTorch. Perfect for undergraduate students learning computer vision and deep learning!

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Tutorial Structure](#tutorial-structure)
- [Quick Start](#quick-start)
- [Dataset Information](#dataset-information)
- [Architecture Overview](#architecture-overview)
- [Usage Examples](#usage-examples)
- [Common Issues](#common-issues)
- [Additional Resources](#additional-resources)

## 🎯 Prerequisites

- Python 3.7 or higher
- Basic understanding of neural networks
- Familiarity with PyTorch basics (tensors, autograd)
- Basic linear algebra and calculus knowledge

## 🚀 Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision numpy matplotlib
```

## 📚 Tutorial Structure

### Level 1: Easy - Dataset Visualization (30 min each)

#### **01_mnist_dataset.py**
- **Topics:** Loading and visualizing MNIST handwritten digits
- **Dataset:** 60,000 training images of digits 0-9
- **Skills:** Data loading, visualization, DataLoader basics
- **Run:** `python 01_mnist_dataset.py`

#### **02_fashion_mnist_dataset.py**
- **Topics:** Fashion-MNIST dataset exploration
- **Dataset:** 60,000 images of clothing items (10 categories)
- **Skills:** Working with different datasets, class labels
- **Run:** `python 02_fashion_mnist_dataset.py`

#### **03_cifar10_dataset.py**
- **Topics:** Color image dataset (CIFAR-10)
- **Dataset:** 60,000 32×32 RGB images (10 classes)
- **Skills:** Handling color images, normalization
- **Run:** `python 03_cifar10_dataset.py`

### Level 2: Intermediate - Basic CNN Training (1-2 hours each)

#### **04_mnist_classifier.py**
- **Topics:** Complete CNN training pipeline on MNIST
- **Architecture:** 2 Conv layers + 2 FC layers
- **Expected Accuracy:** ~99%
- **Skills:** Training loop, evaluation, model saving
- **Run:** `python 04_mnist_classifier.py --epochs 10`

#### **05_fashion_mnist_classifier.py**
- **Topics:** CNN for fashion items classification
- **Architecture:** Same CNN, different domain
- **Expected Accuracy:** ~90-92%
- **Skills:** Transfer of architecture knowledge, domain differences
- **Run:** `python 05_fashion_mnist_classifier.py --epochs 10`

#### **06_cifar10_basic.py**
- **Topics:** Basic CNN for color images (simpler version)
- **Architecture:** Simple CNN for understanding
- **Expected Accuracy:** ~60-70%
- **Skills:** Working with RGB images, larger networks
- **Run:** `python 06_cifar10_basic.py --epochs 5`

#### **07_cifar10_advanced.py**
- **Topics:** Advanced CNN architecture for CIFAR-10
- **Architecture:** Deeper network with more filters
- **Expected Accuracy:** ~75-80%
- **Skills:** Network depth, batch normalization, advanced techniques
- **Run:** `python 07_cifar10_advanced.py --epochs 15`

### Level 3: Advanced - Specialized Topics (2-3 hours each)

#### **08_binary_classification.py**
- **Topics:** All pairwise binary classification on MNIST
- **Task:** Train 45 binary classifiers (digits 0vs1, 0vs2, ..., 8vs9)
- **Skills:** Binary classification, multiple models, analysis
- **Run:** `python 08_binary_classification.py`
- **Output:** 10×10 grid showing learning curves

#### **09_hogwild_training.py**
- **Topics:** Multi-process parallel training (Hogwild!)
- **Method:** Lock-free parallel SGD
- **Skills:** Distributed training, multiprocessing, speedup analysis
- **Run:** `python 09_hogwild_training.py --num-processes 4`
- **Note:** Requires multi-core CPU for benefits

## 📊 Dataset Information

### MNIST (Modified National Institute of Standards and Technology)
- **Size:** 60,000 training + 10,000 test images
- **Image Size:** 28×28 grayscale
- **Classes:** 10 (digits 0-9)
- **Difficulty:** Easy (baseline ~97% with simple models)
- **Use Case:** Handwriting recognition, quick prototyping

### Fashion-MNIST
- **Size:** 60,000 training + 10,000 test images
- **Image Size:** 28×28 grayscale
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Difficulty:** Medium (more challenging than MNIST)
- **Use Case:** Fashion classification, MNIST replacement

### CIFAR-10 (Canadian Institute For Advanced Research)
- **Size:** 50,000 training + 10,000 test images
- **Image Size:** 32×32 RGB (color)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Difficulty:** Hard (natural images with variation)
- **Use Case:** Object recognition, real-world vision

## 🏗️ Architecture Overview

### Basic CNN Architecture (MNIST/Fashion-MNIST)
```
Input (1×28×28)
    ↓
Conv2D(1→32, 3×3) + ReLU
    ↓
MaxPool2D(2×2)
    ↓
Conv2D(32→64, 3×3) + ReLU
    ↓
MaxPool2D(2×2)
    ↓
Flatten
    ↓
Linear(64×7×7 → 128) + ReLU + Dropout(0.5)
    ↓
Linear(128 → 10)
    ↓
Output (10 classes)
```

### Advanced CNN Architecture (CIFAR-10)
```
Input (3×32×32)
    ↓
Conv2D(3→32, 3×3) + ReLU
    ↓
Conv2D(32→32, 3×3) + ReLU
    ↓
MaxPool2D(2×2)
    ↓
Conv2D(32→64, 3×3) + ReLU
    ↓
Conv2D(64→64, 3×3) + ReLU
    ↓
MaxPool2D(2×2)
    ↓
Flatten
    ↓
Linear(64×8×8 → 512) + ReLU + Dropout(0.5)
    ↓
Linear(512 → 10)
    ↓
Output (10 classes)
```

## 💡 Usage Examples

### Basic Training
```bash
# Train on MNIST with default parameters
python 04_mnist_classifier.py

# Train on Fashion-MNIST with custom parameters
python 05_fashion_mnist_classifier.py --epochs 15 --lr 0.01 --batch-size 128

# Train on CIFAR-10 with GPU
python 07_cifar10_advanced.py --epochs 20 --device cuda
```

### Advanced Options
```bash
# Hogwild training with 4 processes
python 09_hogwild_training.py --num-processes 4 --epochs 10

# Binary classification analysis
python 08_binary_classification.py --lr 0.001

# Save trained model
python 04_mnist_classifier.py --save-model --path ./models/mnist_cnn.pth
```

### Command Line Arguments

All training scripts support these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 64 | Training batch size |
| `--test-batch-size` | 1000 | Test batch size |
| `--epochs` | 14 | Number of training epochs |
| `--lr` | 0.01 | Learning rate |
| `--momentum` | 0.5 | SGD momentum |
| `--gamma` | 0.7 | Learning rate decay |
| `--seed` | 1 | Random seed |
| `--log-interval` | 10 | Logging frequency |
| `--save-model` | False | Save trained model |
| `--path` | "./model.pth" | Model save path |
| `--device` | auto | Device (cuda/cpu) |

## 🔧 Common Issues and Solutions

### Issue: CUDA out of memory
**Solution:** Reduce batch size
```bash
python 04_mnist_classifier.py --batch-size 32
```

### Issue: Training too slow on CPU
**Solution:** 
1. Reduce number of epochs
2. Use smaller batch size
3. Enable GPU if available

### Issue: Low accuracy on CIFAR-10
**Solution:**
1. Train for more epochs (20-30)
2. Use data augmentation
3. Try learning rate scheduling
4. Use the advanced architecture (07_cifar10_advanced.py)

### Issue: Import errors
**Solution:** Install missing packages
```bash
pip install torch torchvision matplotlib numpy
```

### Issue: Hogwild training not faster
**Solution:**
- Ensure you have multiple CPU cores
- Try different number of processes (2-8)
- Note: Benefits vary by system

## 📈 Expected Results

| Dataset | Model | Training Time* | Expected Accuracy |
|---------|-------|----------------|-------------------|
| MNIST | Basic CNN | 2-3 min | 98-99% |
| Fashion-MNIST | Basic CNN | 2-3 min | 90-92% |
| CIFAR-10 (Basic) | Simple CNN | 5-10 min | 60-70% |
| CIFAR-10 (Advanced) | Deep CNN | 15-20 min | 75-80% |

*On modern CPU; much faster on GPU

## 🎓 Learning Path

### For Beginners (Week 1-2)
1. Start with dataset visualization (01-03)
2. Understand the data and preprocessing
3. Move to MNIST classifier (04)
4. Study the training loop carefully

### Intermediate (Week 3-4)
1. Complete Fashion-MNIST (05)
2. Understand why it's harder than MNIST
3. Move to CIFAR-10 basic (06)
4. Compare grayscale vs RGB challenges

### Advanced (Week 5-6)
1. Study CIFAR-10 advanced architecture (07)
2. Experiment with binary classification (08)
3. Learn distributed training concepts (09)
4. Start your own project!

## 🔬 Experimentation Ideas

### Easy Experiments
1. Change learning rate and observe effects
2. Modify batch size and measure speed/accuracy
3. Try different optimizers (SGD → Adam)
4. Add/remove dropout layers

### Intermediate Experiments
1. Add data augmentation to CIFAR-10
2. Implement early stopping
3. Try different activation functions
4. Visualize learned filters

### Advanced Experiments
1. Implement residual connections (ResNet)
2. Add batch normalization
3. Transfer learning from CIFAR-10 to Fashion-MNIST
4. Implement custom augmentation strategies

## 📖 Helper Modules

### `cnn_utils.py`
Core utilities for training and evaluation:
- `parse_args()`: Command line argument parsing
- `set_seed()`: Reproducibility
- `load_data()`: Dataset loading with transforms
- `train()`: Training loop
- `compute_accuracy()`: Evaluation
- `save_model()` / `load_model()`: Model persistence
- `show_predictions()`: Visualization

### `models.py`
CNN architectures:
- `CNN()`: Basic CNN for MNIST/Fashion-MNIST
- `CNN_CIFAR10()`: Advanced CNN for CIFAR-10
- `BinaryClassifier()`: Binary classification model

### `binary_classification_utils.py`
Binary classification helpers:
- Data preparation for pairwise classification
- Training for binary tasks
- Visualization utilities

### `hogwild_utils.py`
Distributed training:
- Multi-process data loading
- Shared memory model setup
- Lock-free parameter updates

## 🌟 Key Concepts Covered

1. **Convolutional Layers**
   - Feature extraction
   - Parameter sharing
   - Translation invariance

2. **Pooling Layers**
   - Dimension reduction
   - Translation robustness
   - Computational efficiency

3. **Training Techniques**
   - Data normalization
   - Learning rate scheduling
   - Momentum optimization
   - Dropout regularization

4. **Evaluation Metrics**
   - Accuracy
   - Per-class accuracy
   - Confusion matrices
   - Loss curves

5. **Advanced Topics**
   - Binary classification
   - Multi-process training
   - Model saving/loading
   - Visualization

## 🔗 Additional Resources

### Official Documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html)
- [PyTorch Forum](https://discuss.pytorch.org/)

### Research Papers
- [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - Original CNN architecture
- [ImageNet Classification](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) - AlexNet
- [Hogwild!](https://arxiv.org/abs/1106.5730) - Lock-free parallel SGD

### Online Courses
- Stanford CS231n: Convolutional Neural Networks
- Fast.ai: Practical Deep Learning
- Coursera: Deep Learning Specialization

## 🤝 Tips for Success

1. **Start Simple**: Don't skip the easy tutorials
2. **Experiment**: Modify hyperparameters and observe
3. **Visualize**: Always plot training curves
4. **Debug**: Print tensor shapes frequently
5. **Read Code**: Understand every line before running
6. **Ask Questions**: Use PyTorch forums and communities
7. **Build Projects**: Apply to your own datasets
8. **Be Patient**: Deep learning takes time to master

## 📝 Notes

- All scripts include extensive comments explaining each step
- Models are intentionally simple for learning purposes
- Production models would be more complex
- GPU training is much faster but not required
- Results may vary slightly due to randomness

## 🎯 Next Steps After This Tutorial

1. **Transfer Learning**: Use pre-trained models (ResNet, VGG)
2. **Object Detection**: YOLO, Faster R-CNN
3. **Image Segmentation**: U-Net, DeepLab
4. **Generative Models**: GANs, VAEs
5. **Transformers**: Vision Transformers (ViT)
6. **Real Projects**: Kaggle competitions, research projects

## 📄 License

Educational use only. Feel free to modify and share!

## 🎉 Conclusion

By completing these tutorials, you will:
- ✅ Understand CNN architectures deeply
- ✅ Train models on real datasets
- ✅ Evaluate and visualize results
- ✅ Handle different image types
- ✅ Implement advanced techniques
- ✅ Be ready for real-world projects

**Happy Learning! 🚀**

---

*Last Updated: November 2025*
*For questions or feedback, please open an issue or discussion.*
