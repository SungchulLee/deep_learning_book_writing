# Complete Learning Rate Scheduler Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Why Learning Rate Scheduling?](#why-learning-rate-scheduling)
3. [Scheduler Deep Dive](#scheduler-deep-dive)
4. [Comparison Table](#comparison-table)
5. [Decision Tree](#decision-tree)
6. [Mathematical Formulas](#mathematical-formulas)
7. [Practical Examples](#practical-examples)

## Introduction

Learning rate is arguably the most important hyperparameter in neural network training. Learning rate schedulers dynamically adjust this parameter during training to achieve better and faster convergence.

## Why Learning Rate Scheduling?

### The Problem with Fixed Learning Rate

**Too High:**
- Training becomes unstable
- Loss oscillates or diverges
- Model overshoots minima

**Too Low:**
- Training is slow
- May get stuck in suboptimal local minima
- Requires many epochs to converge

### The Solution: Scheduling

Start with a higher learning rate for:
- Faster initial progress
- Exploration of parameter space
- Escaping poor initializations

Reduce learning rate later for:
- Fine-tuning parameters
- Settling into better minima
- Improved final performance

## Scheduler Deep Dive

### 1. StepLR

**Concept:** Reduce LR by a factor at regular intervals

**Formula:**
```
LR = initial_lr × gamma^(epoch // step_size)
```

**Parameters:**
- `step_size`: Epochs between reductions
- `gamma`: Multiplicative factor (0 < gamma < 1)

**Example Timeline (lr=0.1, step_size=30, gamma=0.1):**
```
Epoch   0-29:  LR = 0.1
Epoch  30-59:  LR = 0.01
Epoch  60-89:  LR = 0.001
Epoch 90-119:  LR = 0.0001
```

**Pros:**
✅ Simple and intuitive
✅ Predictable behavior
✅ Well-established in literature

**Cons:**
❌ Sharp drops can cause temporary instability
❌ Requires tuning step_size
❌ One-size-fits-all approach

**Best For:**
- Image classification
- When you have experience with the dataset
- Reproducing classical papers

**Common Settings:**
```bash
# Conservative (gentle decay)
--step_size 40 --gamma 0.5

# Moderate (balanced)
--step_size 30 --gamma 0.1

# Aggressive (strong decay)
--step_size 20 --gamma 0.1
```

---

### 2. MultiStepLR

**Concept:** StepLR with custom milestones

**Formula:**
```
LR = initial_lr × gamma^k
where k = number of milestones reached
```

**Parameters:**
- `milestones`: List of epochs [e1, e2, e3, ...]
- `gamma`: Multiplicative factor

**Example Timeline (lr=0.1, milestones=[30,80], gamma=0.1):**
```
Epoch   0-29:  LR = 0.1
Epoch  30-79:  LR = 0.01
Epoch  80+:    LR = 0.001
```

**Pros:**
✅ Precise control over decay timing
✅ Can adapt to training phases
✅ More flexible than StepLR

**Cons:**
❌ Requires domain knowledge
❌ Must plan milestones in advance
❌ Not adaptive to actual training progress

**Best For:**
- Multi-stage training
- When you know optimal decay points
- Fine-tuning pre-trained models

**Common Settings:**
```bash
# Early training dataset
--milestones 10,20,30 --gamma 0.3

# Standard training
--milestones 30,60,90 --gamma 0.1

# Long training
--milestones 50,100,150 --gamma 0.1
```

---

### 3. ExponentialLR

**Concept:** Exponential decay every epoch

**Formula:**
```
LR = initial_lr × gamma^epoch
```

**Parameters:**
- `gamma`: Decay rate (0 < gamma < 1)

**Example Timeline (lr=0.1, gamma=0.95):**
```
Epoch  0:  LR = 0.1
Epoch 10:  LR = 0.0599
Epoch 20:  LR = 0.0358
Epoch 50:  LR = 0.0077
```

**Pros:**
✅ Smooth, continuous decay
✅ No sudden jumps
✅ Simple single parameter

**Cons:**
❌ Can decay too slowly or too fast
❌ Requires careful gamma tuning
❌ Less commonly used

**Best For:**
- Smooth optimization landscapes
- When gradual decay is preferred
- Smaller datasets

**Common Settings:**
```bash
# Slow decay (conservative)
--gamma 0.98

# Moderate decay (balanced)
--gamma 0.95

# Fast decay (aggressive)
--gamma 0.90
```

---

### 4. CosineAnnealingLR

**Concept:** LR follows cosine curve from initial to minimum

**Formula:**
```
LR = eta_min + (initial_lr - eta_min) × (1 + cos(π × epoch / T_max)) / 2
```

**Parameters:**
- `T_max`: Number of epochs for one cycle
- `eta_min`: Minimum learning rate

**Visual Pattern:**
```
LR
 |     
 |⌒⌒⌒⌒⌒╲
 |        ╲___
 |             ╲___
 +----------------→ epoch
```

**Pros:**
✅ Smooth decrease
✅ Fast at extremes, slow in middle
✅ Can restart for multiple cycles
✅ Modern, popular approach

**Cons:**
❌ T_max must match total epochs
❌ May not reach minimum if epochs < T_max
❌ Single cycle may not be optimal

**Best For:**
- Image classification (especially ResNets)
- Modern architectures
- When you want smooth decay
- CIFAR-10, ImageNet training

**Common Settings:**
```bash
# Standard (full cycle)
--t_max 50 --eta_min 0

# With minimum LR
--t_max 100 --eta_min 1e-6

# Short cycle (for testing)
--t_max 20 --eta_min 0
```

**Advanced: Warm Restarts (SGDR)**
Not implemented in basic version, but possible with `CosineAnnealingWarmRestarts`:
- Restart cosine schedule periodically
- Can escape local minima
- Popular in competition settings

---

### 5. OneCycleLR

**Concept:** One cycle: warmup → annealing (based on 1cycle policy)

**Phases:**
1. **Warmup (pct_start):** LR increases from initial to max_lr
2. **Annealing (1-pct_start):** LR decreases from max_lr to final_lr

**Formula:** Complex (see PyTorch docs for details)

**Parameters:**
- `max_lr`: Peak learning rate
- `pct_start`: Fraction of cycle for warmup (typically 0.3)
- `anneal_strategy`: 'cos' or 'linear'

**Visual Pattern:**
```
LR
 |      /⌒⌒⌒╲
 |     /      ╲___
 |    /            ╲___
 |___/                  ╲___
 +------------------------→ epoch
   warmup  |  annealing
```

**Pros:**
✅ Very fast convergence
✅ Often best final performance
✅ Regularization effect from high LR
✅ Minimal hyperparameter tuning
✅ Works well out-of-the-box

**Cons:**
❌ Requires knowing total steps in advance
❌ Not suitable for indefinite training
❌ Can be unstable if max_lr too high

**Best For:**
- Fast training (5-20 epochs)
- Competitions
- Modern architectures
- When you want quick results
- SOTA performance

**Common Settings:**
```bash
# Standard (5-10x initial LR)
--max_lr 0.5 --pct_start 0.3

# Conservative
--max_lr 0.2 --pct_start 0.3

# Aggressive
--max_lr 1.0 --pct_start 0.2

# Long warmup
--max_lr 0.5 --pct_start 0.5
```

**Research Background:**
- Based on Leslie Smith's papers
- Super-convergence phenomenon
- Used by winners of many ML competitions

---

### 6. CyclicLR

**Concept:** Cycle LR between base and max repeatedly

**Modes:**
1. **triangular:** Constant amplitude
2. **triangular2:** Amplitude halves each cycle
3. **exp_range:** Amplitude decays exponentially

**Parameters:**
- `base_lr`: Minimum LR
- `max_lr`: Maximum LR
- `step_size_up`: Iterations to reach max_lr
- `mode`: Cycling pattern

**Visual Patterns:**
```
triangular:
LR  /\  /\  /\
   /  \/  \/  \

triangular2:
LR  /\  /\  /\
   /  \/  \/  \  (decreasing amplitude)

exp_range:
LR  /\  /╲  /⌒╲
   /  \/  \/    (exponentially decreasing)
```

**Pros:**
✅ Helps escape local minima
✅ Explores LR range
✅ Can find optimal LR automatically
✅ Multiple cycles per training

**Cons:**
❌ More complex to tune
❌ Updates per batch (not epoch)
❌ Can be unstable if range too wide
❌ May hurt final convergence

**Best For:**
- Finding optimal LR range
- Avoiding local minima
- Difficult optimization landscapes
- When training seems stuck

**Common Settings:**
```bash
# Standard triangular
--base_lr 1e-3 --max_lr 1e-1 --step_size_up 2000 --mode triangular

# Decreasing amplitude
--base_lr 1e-3 --max_lr 1e-1 --step_size_up 2000 --mode triangular2

# Exponential decay
--base_lr 1e-4 --max_lr 1e-2 --step_size_up 2000 --mode exp_range
```

**Important:** Use `--scheduler_step_on batch` for CyclicLR!

---

### 7. ReduceLROnPlateau

**Concept:** Adaptive - reduce LR when metric plateaus

**Logic:**
```
If metric hasn't improved for 'patience' epochs:
    new_lr = current_lr × factor
```

**Parameters:**
- `mode`: 'min' (for loss) or 'max' (for accuracy)
- `factor`: Reduction factor (typically 0.1-0.5)
- `patience`: Epochs to wait before reducing
- `threshold`: Minimum improvement to count

**Example Timeline:**
```
Epoch    Val Loss    Action
0-10:    Decreasing  No change
11-15:   Plateaus    No change (patience=5)
16:      Plateaus    Reduce LR × 0.1
17-25:   Decreasing  No change
26-30:   Plateaus    No change
31:      Plateaus    Reduce LR × 0.1
```

**Pros:**
✅ Adaptive to training dynamics
✅ No need to plan schedule
✅ Works without domain knowledge
✅ Safe default choice
✅ Monitors actual performance

**Cons:**
❌ Reactive, not proactive
❌ May reduce LR too late
❌ Can be overly conservative
❌ Requires validation set

**Best For:**
- Unknown datasets
- Exploratory training
- When unsure of schedule
- Production systems

**Common Settings:**
```bash
# Standard
--patience 10 --factor 0.1 --mode min

# Conservative (waits longer)
--patience 20 --factor 0.5 --mode min

# Aggressive (reacts quickly)
--patience 5 --factor 0.1 --mode min

# For accuracy monitoring
--patience 10 --factor 0.5 --mode max
```

**Important:** Use `--scheduler_step_on val_loss` for ReduceLROnPlateau!

---

## Comparison Table

| Scheduler | Complexity | Tuning Effort | Adaptivity | Best Use Case |
|-----------|------------|---------------|------------|---------------|
| StepLR | Low | Medium | None | Classical training |
| MultiStepLR | Low | High | None | Multi-stage training |
| ExponentialLR | Low | Medium | None | Smooth decay |
| CosineAnnealingLR | Medium | Low | None | Modern architectures |
| OneCycleLR | Medium | Low | None | Fast convergence |
| CyclicLR | High | High | None | Exploration |
| ReduceLROnPlateau | Low | Low | High | Unknown datasets |

**Legend:**
- **Complexity:** Implementation complexity
- **Tuning Effort:** How much hyperparameter tuning needed
- **Adaptivity:** Adapts to training dynamics
- **Best Use Case:** Primary application

---

## Decision Tree

```
START: Need to choose a scheduler?
│
├─ Are you unsure what will work?
│  └─ YES → Use ReduceLROnPlateau (safe default)
│
├─ Want fastest training (5-20 epochs)?
│  └─ YES → Use OneCycleLR
│
├─ Training modern architecture (ResNet, etc.)?
│  └─ YES → Use CosineAnnealingLR
│
├─ Know optimal decay epochs?
│  ├─ Single decay point → Use StepLR
│  └─ Multiple decay points → Use MultiStepLR
│
├─ Want to explore LR range?
│  └─ YES → Use CyclicLR
│
├─ Want smooth, gentle decay?
│  └─ YES → Use ExponentialLR
│
└─ Default choice → Use ReduceLROnPlateau or OneCycleLR
```

---

## Mathematical Formulas

### StepLR
```
η_t = η_0 × γ^⌊t/k⌋

where:
  η_t = learning rate at epoch t
  η_0 = initial learning rate
  γ = gamma (decay factor)
  k = step_size
  ⌊·⌋ = floor function
```

### ExponentialLR
```
η_t = η_0 × γ^t

where:
  t = epoch number
```

### CosineAnnealingLR
```
η_t = η_min + (η_max - η_min) × [1 + cos(πt/T)] / 2

where:
  T = T_max (period)
  η_min = eta_min
  η_max = initial learning rate
```

### OneCycleLR (Simplified)
```
Phase 1 (Warmup, t < pct_start × T):
  η_t = η_0 + (η_max - η_0) × t / (pct_start × T)

Phase 2 (Annealing, t ≥ pct_start × T):
  η_t = η_max × (1 - (t - pct_start×T) / ((1-pct_start)×T))
  
where:
  T = total steps
  pct_start = warmup fraction
```

---

## Practical Examples

### Example 1: Image Classification on CIFAR-10

**Scenario:** Training ResNet on CIFAR-10 for 200 epochs

**Recommended:** CosineAnnealingLR
```bash
python scheduler.py \
    --scheduler cosine \
    --epochs 200 \
    --t_max 200 \
    --lr 0.1 \
    --eta_min 0
```

**Why:** CosineAnnealing is popular for image models, smooth decay works well with ResNets.

### Example 2: Quick Prototype/Competition

**Scenario:** Fast training, 10 epochs, need good results quickly

**Recommended:** OneCycleLR
```bash
python scheduler.py \
    --scheduler onecycle \
    --epochs 10 \
    --max_lr 1.0 \
    --lr 0.1 \
    --pct_start 0.3
```

**Why:** OneCycle converges very fast and often gives best results in few epochs.

### Example 3: Unknown Dataset

**Scenario:** New dataset, unsure of optimal schedule

**Recommended:** ReduceLROnPlateau
```bash
python scheduler.py \
    --scheduler plateau \
    --epochs 100 \
    --patience 10 \
    --factor 0.1 \
    --scheduler_step_on val_loss
```

**Why:** Adaptive, doesn't require domain knowledge, safe default.

### Example 4: Finding Learning Rate Range

**Scenario:** Want to discover optimal LR range

**Recommended:** CyclicLR
```bash
python scheduler.py \
    --scheduler cyclical \
    --base_lr 1e-5 \
    --max_lr 1.0 \
    --mode triangular \
    --scheduler_step_on batch \
    --epochs 10
```

**Why:** Cycles through range, helps identify sweet spot. Use resulting plot to choose fixed LR or schedule.

### Example 5: Classical Approach

**Scenario:** Reproducing a paper that uses StepLR

**Recommended:** StepLR
```bash
python scheduler.py \
    --scheduler step \
    --epochs 90 \
    --step_size 30 \
    --gamma 0.1 \
    --lr 0.1
```

**Why:** Standard in many papers, decay at epochs 30 and 60.

---

## Summary

**Quick Recommendations:**

| Situation | Scheduler | Priority |
|-----------|-----------|----------|
| Fastest results | OneCycleLR | ⭐⭐⭐⭐⭐ |
| Unknown dataset | ReduceLROnPlateau | ⭐⭐⭐⭐⭐ |
| Modern architectures | CosineAnnealingLR | ⭐⭐⭐⭐ |
| Known decay points | StepLR/MultiStepLR | ⭐⭐⭐⭐ |
| LR exploration | CyclicLR | ⭐⭐⭐ |
| Smooth decay | ExponentialLR | ⭐⭐⭐ |

**Final Advice:**
1. Start with OneCycleLR or ReduceLROnPlateau
2. If unsatisfied, try CosineAnnealingLR
3. Use this guide's examples as starting points
4. Always visualize LR schedule and training curves
5. Experiment and compare!

---

**Remember:** The best scheduler depends on your specific problem. Experiment and compare results!
