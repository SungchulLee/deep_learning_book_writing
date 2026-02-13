# Level 2: Building Neural Networks

## 🎯 Purpose

Learn different approaches to building neural networks, work with real datasets (MNIST), and understand the trade-offs between different architectural choices, activation functions, and loss functions.

## 📚 What You'll Master

- **Real Datasets**: MNIST digit classification
- **Multiple Architectures**: Sequential vs custom modules
- **Activation Functions**: When to use ReLU, Sigmoid, Tanh, etc.
- **Loss Functions**: Cross-entropy vs MSE and when to use each
- **Data Loading**: PyTorch's Dataset and DataLoader

## 📖 Files in This Level

### 08_mnist_basic.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Your first real computer vision model - classify handwritten digits!

**What You'll Learn**:
- Loading MNIST dataset with torchvision
- Flattening images (28x28 → 784)
- Training on real data with train/test split
- Evaluation and accuracy metrics

**Architecture**:
```
Input (784) → Linear(784→100) → ReLU → Linear(100→10) → Output
```

**Dataset**: 60,000 training images, 10,000 test images

**Expected Accuracy**: ~97-98% on test set

**Why It Matters**: MNIST is the "Hello World" of deep learning. It's simple enough to train quickly but complex enough to be interesting.

---

### 09_mnist_classification_detailed.py ⭐⭐⭐
**Difficulty**: Intermediate | **Time**: 60-90 min

A more thorough treatment of the same problem with extensive explanations.

**What You'll Learn**:
- Detailed data preprocessing
- Training vs evaluation mode
- Logging and monitoring
- Saving/loading model checkpoints
- Confusion matrix and per-class accuracy

**Additional Features**:
- Visualization of predictions
- Learning curves (loss and accuracy over time)
- Model checkpointing
- Detailed performance metrics

**Compare with File 08**: 
- File 08 = minimal working example
- File 09 = production-ready code with best practices

**Why It Matters**: See the difference between a quick prototype and a well-engineered solution.

---

### 10_using_sequential.py ⭐
**Difficulty**: Beginner | **Time**: 30-45 min

Quick model building with `nn.Sequential` - for when you need to prototype fast.

**What You'll Learn**:
- `nn.Sequential` container
- When Sequential is appropriate
- Limitations of Sequential
- Code simplicity vs flexibility trade-off

**Comparison**:
```python
# Custom Module (more flexible)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

# Sequential (simpler)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

**When to Use Sequential**:
✅ Simple feedforward architectures  
✅ Prototyping  
✅ No branching or skip connections  

**When NOT to Use Sequential**:
❌ Complex architectures (ResNet, U-Net)  
❌ Multiple inputs/outputs  
❌ Custom forward logic  

**Why It Matters**: Choose the right tool for the job. Sequential is great for simple models.

---

### 11_custom_module.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Deep dive into creating flexible, reusable custom modules.

**What You'll Learn**:
- Proper `nn.Module` subclassing
- Organizing complex architectures
- Reusable components
- Model composition

**Techniques Covered**:
- Dynamic layer creation
- Configurable architectures
- Nested modules
- Module registration

**Best Practices**:
```python
class FlexibleNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        # Create layers dynamically based on config
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

**Why It Matters**: For real projects, you need flexible, configurable architectures. This shows you how.

---

### 12_activation_functions.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Compare different activation functions and understand when to use each.

**What You'll Learn**:
- ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish
- Advantages and disadvantages of each
- The vanishing gradient problem
- How to choose the right activation

**Activation Function Comparison**:

| Activation | Range | Use Case | Pros | Cons |
|------------|-------|----------|------|------|
| **ReLU** | [0, ∞) | Hidden layers (default) | Fast, no vanishing gradient | Dead neurons |
| **LeakyReLU** | (-∞, ∞) | When ReLU has dead neurons | Fixes dead ReLU problem | Extra hyperparameter |
| **Sigmoid** | (0, 1) | Binary output, gates | Smooth, probabilistic | Vanishing gradient |
| **Tanh** | (-1, 1) | RNNs, centered data | Zero-centered | Vanishing gradient |
| **ELU** | (-α, ∞) | Deep networks | Smooth, no dead neurons | Slower than ReLU |
| **Swish** | (-∞, ∞) | Very deep networks | State-of-art performance | Computationally expensive |

**Rule of Thumb**:
- **Hidden layers**: ReLU (start here!)
- **Output layer (classification)**: None (use with CrossEntropyLoss) or Softmax
- **Output layer (regression)**: None (linear output)
- **Output layer (binary)**: Sigmoid

**Why It Matters**: Wrong activation can prevent your network from learning!

---

### 13_loss_functions.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Master different loss functions and when to use each.

**What You'll Learn**:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Cross-Entropy Loss
- Binary Cross-Entropy
- When to use each

**Loss Function Guide**:

| Loss | Problem Type | Output Activation | When to Use |
|------|--------------|-------------------|-------------|
| **MSE** | Regression | None | Predicting continuous values (price, temperature) |
| **MAE** | Regression | None | Robust to outliers |
| **CrossEntropy** | Multi-class | None* | Image classification (MNIST, CIFAR-10) |
| **BCE** | Binary | Sigmoid | Binary classification (spam/not spam) |
| **BCEWithLogits** | Binary | None* | More numerically stable than BCE |

*PyTorch's CrossEntropyLoss includes softmax. Don't add softmax in your model!

**Common Mistakes**:
❌ Using MSE for classification  
❌ Adding Softmax before CrossEntropyLoss  
❌ Using CrossEntropyLoss for regression  
❌ Wrong number of output neurons  

**Why It Matters**: Wrong loss function = model won't learn the right thing!

## 🎓 Learning Path

```
08_mnist_basic.py (Required) → Learn the basics
    ↓
09_mnist_classification_detailed.py (Recommended) → See best practices
    ↓
10_using_sequential.py (Required) → Quick prototyping
    ↓
11_custom_module.py (Required) → Flexible architectures
    ↓
12_activation_functions.py (Required) → Choose activations wisely
    ↓
13_loss_functions.py (Required) → Choose loss correctly
    ↓
Ready for Level 3! 🎉
```

## 💡 Study Tips

- **Compare 08 vs 09**: Notice the differences in code quality and features
- **Experiment with activations (12)**: Train same model with different activations, compare accuracy
- **Mix loss functions (13)**: Try wrong loss function, see what happens
- **Modify architectures**: Change number of layers, neurons, see effect on accuracy

## 🧪 Experiments to Try

1. **Network Size**: Try hidden sizes of [32], [64], [128], [256], [512]
2. **Network Depth**: Try 1, 2, 3, 4, 5 hidden layers
3. **Activation Sweep**: Compare all activations on same architecture
4. **Wrong Loss**: Use MSE instead of CrossEntropy for MNIST, see what happens
5. **Overfitting**: Train tiny network (2 layers, 10 neurons) vs huge network (5 layers, 1000 neurons)

## ✅ Level 2 Completion Checklist

Before moving to Level 3, make sure you can:

- [ ] Load and preprocess real datasets
- [ ] Build models with both Sequential and custom modules
- [ ] Choose appropriate activation functions for different layers
- [ ] Select the correct loss function for your problem
- [ ] Evaluate model performance with accuracy metrics
- [ ] Visualize training progress (loss curves)
- [ ] Save and load trained models

## 🎯 Next Level Preview

**Level 3: Advanced Techniques** will teach you:
- Preventing overfitting with dropout and L2 regularization
- Batch normalization for stable training
- Learning rate scheduling strategies
- Proper weight initialization (Xavier, He)
- Techniques used in production systems

---

**Excellent work!** You can now build and train real neural networks. Time to learn advanced optimization! 🚀

*Pro tip: MNIST is just the beginning. The techniques you learned here apply to any dataset!*
