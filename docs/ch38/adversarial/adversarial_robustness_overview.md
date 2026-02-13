# Module 62: Adversarial Robustness

## Overview
This module provides a comprehensive introduction to adversarial robustness in deep learning, covering attack methods, defense mechanisms, and evaluation techniques. The materials progress from basic concepts to advanced certified defenses.

## Learning Objectives
By completing this module, students will:
1. Understand the vulnerability of neural networks to adversarial examples
2. Implement classic adversarial attack methods (FGSM, PGD, C&W)
3. Develop and evaluate defense mechanisms
4. Apply adversarial training techniques
5. Understand certified robustness approaches
6. Evaluate model robustness using standardized metrics

## Mathematical Background

### What are Adversarial Examples?
Adversarial examples are inputs to machine learning models that are intentionally designed to cause the model to make mistakes. They are created by adding small, often imperceptible perturbations to legitimate inputs.

**Formal Definition:**
Given a classifier f(x) and input x with true label y, an adversarial example x' satisfies:
- f(x') ≠ y (misclassification)
- ||x' - x|| ≤ ε (bounded perturbation)

### Threat Models
1. **White-box attacks**: Attacker has full access to model parameters and gradients
2. **Black-box attacks**: Attacker only has query access to the model
3. **Gray-box attacks**: Attacker has partial information about the model

### Attack Formulation
Most attacks solve the optimization problem:
```
maximize L(f(x + δ), y)  subject to ||δ|| ≤ ε
```
where:
- δ is the perturbation
- L is a loss function
- ε is the perturbation budget
- ||·|| is a norm (typically L∞, L2, or L1)

## Module Structure

### Beginner Level
**File: `01_fgsm_basic.py`**
- Fast Gradient Sign Method (FGSM)
- Single-step gradient-based attack
- Mathematical derivation and implementation
- Visualization of adversarial examples
- Attack success rate evaluation

**Mathematical Foundation:**
FGSM uses the sign of the gradient to create adversarial perturbations:
```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```

### Intermediate Level
**File: `02_pgd_attack.py`**
- Projected Gradient Descent (PGD) attack
- Multi-step iterative attack
- Projection onto ε-ball
- Comparison with FGSM

**Mathematical Foundation:**
PGD iteratively applies gradient steps with projection:
```
x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))
```
where Π projects back onto the allowed perturbation set S.

**File: `03_cw_attack.py`**
- Carlini & Wagner (C&W) attack
- Optimization-based attack
- Different distance metrics (L2, L∞)
- Stronger than gradient-based methods

**Mathematical Foundation:**
C&W formulates attack as optimization problem:
```
minimize ||δ||_p + c · f(x + δ)
```
where f is a carefully designed loss function that encourages misclassification.

### Advanced Level
**File: `04_adversarial_training.py`**
- Adversarial training as defense
- Min-max optimization formulation
- Training with adversarial examples
- Robustness evaluation

**Mathematical Foundation:**
Adversarial training solves:
```
min_θ E_{(x,y)}[max_{||δ||≤ε} L(θ, x + δ, y)]
```
This is a min-max game: inner maximization finds worst-case perturbation, outer minimization trains robust model.

**File: `05_certified_defenses.py`**
- Randomized smoothing
- Certified robustness guarantees
- Provable defenses
- Certification radius computation

**Mathematical Foundation:**
Randomized smoothing creates certifiably robust classifier g(x):
```
g(x) = argmax_c P(f(x + ε) = c)  where ε ~ N(0, σ²I)
```
Guarantees: if g(x) = c with probability ≥ p, then prediction is certified robust within radius R.

## Prerequisites
- PyTorch fundamentals (Module 02-05)
- Neural network training (Module 20)
- Convolutional networks (Module 23)
- Loss functions and optimizers (Module 14-15)

## Installation

```bash
pip install torch torchvision numpy matplotlib scipy tqdm
```

Or use the provided requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic FGSM Attack
```python
from utils import load_model, load_data
from fgsm_basic import FGSM

# Load pretrained model and data
model = load_model('resnet18', pretrained=True)
images, labels = load_data('cifar10', batch_size=32)

# Create FGSM attack
attack = FGSM(model, epsilon=0.03)

# Generate adversarial examples
adv_images = attack.generate(images, labels)

# Evaluate
accuracy = attack.evaluate(images, labels, adv_images)
print(f"Adversarial accuracy: {accuracy:.2%}")
```

### PGD Attack
```python
from pgd_attack import PGD

# Create PGD attack (stronger than FGSM)
attack = PGD(model, epsilon=0.03, alpha=0.01, num_iter=40)
adv_images = attack.generate(images, labels)
```

### Adversarial Training
```python
from adversarial_training import AdversarialTrainer

# Train robust model
trainer = AdversarialTrainer(
    model=model,
    epsilon=0.03,
    attack_type='pgd',
    num_iter=10
)

robust_model = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
```

## File Descriptions

### Core Implementation Files

1. **`01_fgsm_basic.py`** (Beginner - ~300 lines)
   - FGSM attack implementation
   - Step-by-step mathematical derivation
   - Visualization functions
   - Success rate computation

2. **`02_pgd_attack.py`** (Intermediate - ~350 lines)
   - PGD attack implementation
   - Random initialization strategies
   - Early stopping mechanisms
   - Comparison utilities

3. **`03_cw_attack.py`** (Intermediate - ~400 lines)
   - C&W L2 and L∞ attacks
   - Binary search for optimal c parameter
   - Adam optimizer for attack generation
   - Confidence-based objectives

4. **`04_adversarial_training.py`** (Advanced - ~450 lines)
   - Adversarial training loop
   - TRADES defense implementation
   - MART defense implementation
   - Robust evaluation protocols

5. **`05_certified_defenses.py`** (Advanced - ~400 lines)
   - Randomized smoothing
   - Certification algorithm
   - Monte Carlo sampling
   - Confidence interval computation

6. **`utils.py`** (~250 lines)
   - Data loading utilities
   - Model loading and architecture definitions
   - Visualization functions
   - Metric computation helpers

## Key Concepts Covered

### 1. Attack Methods
- **Gradient-based attacks**: FGSM, PGD, I-FGSM
- **Optimization-based attacks**: C&W, EAD
- **Decision-based attacks**: Boundary attack
- **Transfer attacks**: Black-box scenarios

### 2. Defense Mechanisms
- **Adversarial training**: PGD-AT, TRADES, MART
- **Input transformations**: Denoising, JPEG compression
- **Defensive distillation**: Temperature scaling
- **Certified defenses**: Randomized smoothing, interval bound propagation

### 3. Evaluation Metrics
- **Clean accuracy**: Performance on unperturbed data
- **Robust accuracy**: Performance under attack
- **Attack success rate**: Percentage of successful attacks
- **Perturbation magnitude**: Average L∞, L2 distance
- **Certified accuracy**: Provably robust predictions

### 4. Robustness Benchmarks
- **AutoAttack**: Ensemble of parameter-free attacks
- **RobustBench**: Standardized evaluation platform
- **Adversarial robustness toolbox**: IBM's comprehensive library

## Theoretical Insights

### Why Do Adversarial Examples Exist?

1. **High-dimensional geometry**: Small perturbations in high dimensions can accumulate
2. **Linear behavior**: Neural networks are locally approximately linear
3. **Overconfidence**: Models assign high confidence to incorrect predictions
4. **Decision boundaries**: Complex, non-smooth boundaries in input space

### Robustness vs. Accuracy Tradeoff

There's an inherent tradeoff between clean accuracy and robust accuracy:
- Robust models may sacrifice some clean accuracy
- Standard training optimizes for clean accuracy only
- Adversarial training balances both objectives

**Mathematical Formulation:**
```
Standard training: min_θ E[(x,y)][L(θ, x, y)]
Robust training:   min_θ E[(x,y)][max_{||δ||≤ε} L(θ, x+δ, y)]
```

### Certified vs. Empirical Robustness

- **Empirical robustness**: Tested against specific attacks (no guarantees)
- **Certified robustness**: Provable guarantees for all perturbations in ε-ball
- Certified methods provide lower bounds on adversarial accuracy

## Computational Complexity

| Method | Time Complexity | Memory | Strength |
|--------|----------------|---------|----------|
| FGSM | O(1) | Low | Weak |
| PGD | O(K) | Low | Strong |
| C&W | O(K·B) | Medium | Very Strong |
| Adv Training | O(K·E) | High | Defense |
| Randomized Smoothing | O(N) | Medium | Certified |

Where: K = iterations, E = epochs, B = binary search steps, N = samples for certification

## Common Pitfalls and Best Practices

### Pitfalls
1. **Gradient masking**: Defenses that hide gradients without true robustness
2. **Weak attacks**: Evaluating only against FGSM
3. **Improper normalization**: Not accounting for data preprocessing
4. **Small ε values**: Testing with unrealistic perturbation budgets

### Best Practices
1. **Use strong attacks**: PGD with random restarts, AutoAttack
2. **Multiple ε values**: Test across range of perturbation budgets
3. **White-box evaluation**: Assume worst-case attacker knowledge
4. **Standardized protocols**: Follow RobustBench guidelines
5. **Checkpoint selection**: Early stopping based on robust validation accuracy

## Research Directions

1. **Scalability**: Adversarial training on large models and datasets
2. **Real-world robustness**: Beyond Lp perturbations
3. **Compositional robustness**: Multiple simultaneous perturbations
4. **Certified defenses**: Tighter bounds, faster certification
5. **Theoretical understanding**: Why adversarial examples transfer

## References

### Foundational Papers
1. Szegedy et al. (2013) - "Intriguing properties of neural networks"
2. Goodfellow et al. (2015) - "Explaining and Harnessing Adversarial Examples" (FGSM)
3. Madry et al. (2018) - "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
4. Carlini & Wagner (2017) - "Towards Evaluating the Robustness of Neural Networks"

### Defense Methods
5. Zhang et al. (2019) - "Theoretically Principled Trade-off between Robustness and Accuracy" (TRADES)
6. Wang et al. (2020) - "Improving Adversarial Robustness Requires Revisiting Misclassified Examples" (MART)
7. Cohen et al. (2019) - "Certified Adversarial Robustness via Randomized Smoothing"

### Evaluation Standards
8. Croce & Hein (2020) - "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks" (AutoAttack)
9. RobustBench: https://robustbench.github.io/

## Exercise Suggestions

### Beginner Exercises
1. Implement FGSM and visualize adversarial examples on MNIST
2. Compare FGSM with different ε values
3. Measure attack success rate vs. perturbation magnitude

### Intermediate Exercises
1. Implement PGD with random initialization
2. Compare FGSM, PGD, and C&W attacks
3. Analyze transferability of adversarial examples
4. Implement targeted vs. untargeted attacks

### Advanced Exercises
1. Implement adversarial training from scratch
2. Compare TRADES, MART, and standard adversarial training
3. Implement randomized smoothing certification
4. Evaluate models using AutoAttack
5. Study the accuracy-robustness tradeoff empirically

## Additional Resources

- **PyTorch tutorials**: Official adversarial example tutorial
- **Adversarial Robustness Toolbox (ART)**: IBM's comprehensive library
- **Foolbox**: Python library for adversarial attacks
- **CleverHans**: Google's adversarial examples library
- **RobustBench**: Standardized leaderboard and evaluation

## Assessment Rubric

### Knowledge (30%)
- Understanding of threat models
- Mathematical foundations of attacks and defenses
- Awareness of evaluation protocols

### Implementation (40%)
- Correct implementation of attacks
- Proper gradient computation
- Efficient code structure

### Analysis (30%)
- Critical evaluation of results
- Understanding of tradeoffs
- Insights from experiments

## Time Estimates

- **Beginner level**: 2-3 hours
- **Intermediate level**: 4-6 hours
- **Advanced level**: 6-8 hours
- **Complete module**: 12-17 hours (including exercises and reading)

## Contributing

Students are encouraged to:
- Add new attack methods
- Implement additional defenses
- Extend to other datasets and architectures
- Improve visualization utilities
- Add more comprehensive experiments

## License

This educational material is provided for academic use. Please cite appropriately if used in courses or research.

---

**Last Updated**: November 2025
**Module Version**: 1.0
**Difficulty**: Intermediate to Advanced
**Prerequisites**: Modules 02-05, 14-15, 20, 23
