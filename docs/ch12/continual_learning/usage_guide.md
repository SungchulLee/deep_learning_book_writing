# Module 57: Continual Learning - Usage Guide

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# If you're using a specific PyTorch version for your GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Running the Scripts

The module is organized into three levels of difficulty:

#### **Beginner Level (01_*.py)**
Start here to understand the fundamentals:

```bash
# 1. See catastrophic forgetting in action
python 01_catastrophic_forgetting_demo.py

# 2. Establish naive baseline with proper metrics
python 01_naive_sequential_learning.py

# 3. Try your first continual learning method
python 01_simple_experience_replay.py
```

#### **Intermediate Level (02_*.py)**
Learn sophisticated continual learning techniques:

```bash
# 1. Elastic Weight Consolidation (regularization-based)
python 02_elastic_weight_consolidation.py

# 2. Learning Without Forgetting (knowledge distillation)
python 02_learning_without_forgetting.py
```

#### **Advanced Level (03_*.py)**
Compare methods and analyze trade-offs:

```bash
# Comprehensive comparison of all methods
python 03_comprehensive_comparison.py
```

---

## File Descriptions

### Documentation
- **README.md**: Comprehensive theory, mathematical foundations, and references
- **USAGE_GUIDE.md**: This file - practical usage instructions
- **requirements.txt**: Python package dependencies

### Beginner Level
1. **01_catastrophic_forgetting_demo.py**
   - Demonstrates the catastrophic forgetting problem
   - Visualizes accuracy degradation over tasks
   - Establishes motivation for continual learning
   - Runtime: ~5-10 minutes on CPU

2. **01_naive_sequential_learning.py**
   - Implements proper evaluation protocol
   - Calculates standard continual learning metrics
   - Establishes baseline for comparison
   - Runtime: ~3-5 minutes on CPU

3. **01_simple_experience_replay.py**
   - First continual learning method: memory buffer
   - Shows significant improvement over baseline
   - Introduces memory-efficiency trade-offs
   - Runtime: ~4-6 minutes on CPU

### Intermediate Level
1. **02_elastic_weight_consolidation.py**
   - Regularization-based continual learning
   - Fisher information matrix computation
   - No storage of previous examples
   - Runtime: ~8-12 minutes on CPU (Fisher computation adds overhead)

2. **02_learning_without_forgetting.py**
   - Knowledge distillation for continual learning
   - Temperature scaling and soft targets
   - Privacy-preserving (no example storage)
   - Runtime: ~5-8 minutes on CPU

### Advanced Level
1. **03_comprehensive_comparison.py**
   - Compares all methods on same benchmark
   - Analyzes performance vs computational trade-offs
   - Generates comprehensive visualization
   - Runtime: ~15-20 minutes on CPU

---

## Customization Guide

### Changing the Dataset

The scripts use Split MNIST by default. To use a different dataset:

```python
# In any script, modify the dataset loading section:

# Example: Split CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Adjust task configuration
num_tasks = 5
classes_per_task = 10 // num_tasks  # For CIFAR-10
```

### Tuning Hyperparameters

#### Experience Replay
```python
# Key hyperparameters in 01_simple_experience_replay.py

memory_size = 1000           # Total memory buffer size
                            # Try: [500, 1000, 2000]
                            
examples_per_task = 200      # Examples to store per task
                            # Try: [100, 200, 500]
                            
# Trade-off: Larger memory → less forgetting but more storage
```

#### Elastic Weight Consolidation (EWC)
```python
# Key hyperparameters in 02_elastic_weight_consolidation.py

ewc_lambda = 5000           # Regularization strength
                           # Try: [100, 1000, 5000, 10000]
                           # Higher → more protection, less plasticity
                           
num_fisher_samples = 1000   # Samples for Fisher computation
                           # Try: [500, 1000, 2000]
                           # Higher → more accurate but slower
```

#### Learning Without Forgetting (LWF)
```python
# Key hyperparameters in 02_learning_without_forgetting.py

distill_lambda = 1.0        # Distillation loss weight
                           # Try: [0.5, 1.0, 2.0, 5.0]
                           # Higher → more preservation
                           
temperature = 2.0           # Temperature for soft targets
                           # Try: [1.0, 2.0, 3.0, 4.0]
                           # Higher → softer distributions
```

### Modifying the Network Architecture

```python
# In any script, find create_model() or SimpleNetwork class:

def create_custom_model():
    """Create a custom architecture."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),      # Larger hidden layer
        nn.ReLU(),
        nn.Dropout(0.2),          # Add dropout
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)         # Output layer
    )

# Replace model creation:
model = create_custom_model().to(device)
```

---

## Understanding the Output

### Accuracy Matrix
```
Task |  After T0 |  After T1 |  After T2 |  After T3 |  After T4 |
------------------------------------------------------------------
  0  |   96.5%   |   62.3%   |   58.1%   |   55.2%   |   52.8%   |
  1  |     -     |   97.1%   |   64.5%   |   61.2%   |   58.9%   |
  2  |     -     |     -     |   96.8%   |   65.3%   |   62.1%   |
  3  |     -     |     -     |     -     |   97.2%   |   66.7%   |
  4  |     -     |     -     |     -     |     -     |   96.5%   |
```

**Interpretation:**
- **Diagonal:** Learning accuracy (performance right after learning)
- **Last column:** Final accuracy (performance at the end)
- **Horizontal trend:** Shows forgetting over time

### Key Metrics

1. **Average Accuracy (AA)**
   - Mean of last column
   - Overall performance across all tasks
   - Higher is better

2. **Backward Transfer (BWT)**
   - Measure of forgetting
   - Negative values indicate forgetting
   - Closer to 0 or positive is better

3. **Learning Accuracy (LA)**
   - Mean of diagonal
   - Ability to learn each task initially
   - Should be high (>90%) for good methods

### Visualization Outputs

Each script generates PNG files:
- `catastrophic_forgetting_demonstration.png`
- `naive_sequential_learning_results.png`
- `experience_replay_comparison.png`
- `ewc_results.png`
- `lwf_results.png`
- `continual_learning_comparison.png`

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```python
# Reduce batch size
batch_size = 64  # Instead of 128

# Or reduce model size
hidden_size = 128  # Instead of 256
```

#### 2. Slow Training
```python
# Reduce epochs per task
epochs_per_task = 3  # Instead of 5

# Or reduce number of tasks
num_tasks = 3  # Instead of 5
```

#### 3. Poor Performance on All Methods
```python
# Increase learning rate
learning_rate = 0.01  # Instead of 0.001

# Or increase epochs
epochs_per_task = 10  # Instead of 5
```

#### 4. EWC Taking Too Long
```python
# Reduce Fisher samples
num_fisher_samples = 500  # Instead of 1000

# Or use fewer tasks
num_tasks = 3  # Instead of 5
```

---

## Experimental Best Practices

### 1. Start Simple
- Begin with 01_catastrophic_forgetting_demo.py
- Understand the problem before learning solutions
- Use default hyperparameters first

### 2. Fair Comparison
- Use same random seed across experiments
- Same model architecture for all methods
- Same dataset splits and preprocessing
- Report mean and standard deviation over multiple runs

### 3. Proper Evaluation
- Always evaluate on all previous tasks after each task
- Use separate test set (never train on test data)
- Calculate standard continual learning metrics
- Consider computational cost and memory usage

### 4. Hyperparameter Tuning
- Grid search over key hyperparameters
- Use validation set for tuning (not test set)
- Report hyperparameters used in final results
- Consider task-specific vs task-agnostic tuning

---

## Integration with Course

### Suggested Teaching Sequence

**Week 1: Problem Introduction**
- Lecture: What is continual learning?
- Lab: Run 01_catastrophic_forgetting_demo.py
- Assignment: Analyze forgetting patterns

**Week 2: Evaluation & Baselines**
- Lecture: Continual learning metrics
- Lab: Run 01_naive_sequential_learning.py
- Assignment: Implement custom evaluation metrics

**Week 3: Memory-Based Methods**
- Lecture: Experience replay strategies
- Lab: Run 01_simple_experience_replay.py
- Assignment: Compare different memory sizes

**Week 4: Regularization Methods**
- Lecture: EWC and Fisher information
- Lab: Run 02_elastic_weight_consolidation.py
- Assignment: Tune λ hyperparameter

**Week 5: Knowledge Distillation**
- Lecture: LWF and soft targets
- Lab: Run 02_learning_without_forgetting.py
- Assignment: Experiment with temperature scaling

**Week 6: Comprehensive Analysis**
- Lecture: Method comparison and trade-offs
- Lab: Run 03_comprehensive_comparison.py
- Project: Apply to custom dataset

---

## Citation

If you use these materials in your research or teaching, please cite:

```bibtex
@misc{continual_learning_module57,
  title={Module 57: Continual Learning - Educational Package},
  author={Your Name},
  year={2024},
  howpublished={\url{https://your-url.com}}
}
```

---

## Additional Resources

### Papers to Read
1. Kirkpatrick et al. (2017) - EWC
2. Li & Hoiem (2017) - Learning Without Forgetting
3. Lopez-Paz & Ranzato (2017) - Gradient Episodic Memory
4. Parisi et al. (2019) - Continual Learning Survey

### Online Resources
- Continual Learning Benchmarks: https://github.com/ContinualAI
- Tutorial: https://avalanche.continualai.org/
- Community: https://www.continualai.org/

---

## Support

For questions or issues:
1. Check this USAGE_GUIDE.md
2. Review comments in the code
3. Consult README.md for theory
4. Contact course instructor

---

**Happy Learning! 🚀**
