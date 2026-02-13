# Bayesian Methods & Neural Networks: Complete Educational Curriculum

**A Comprehensive Python-Based Course from Bayesian Foundations to Modern Deep Learning**

## 📚 Course Overview

This comprehensive educational package provides a complete journey from classical Bayesian inference to modern Bayesian neural networks and score-based generative models. Designed for undergraduate computer science and mathematics students, this curriculum bridges theoretical foundations with practical implementations, emphasizing both mathematical rigor and computational intuition.

### Author Information
**Professor Sungchul**  
Yonsei University  
Email: sungchulyonsei@gmail.com

---

## 🎯 Learning Objectives

By completing this curriculum, students will be able to:

1. **Understand Bayesian Foundations**: Master Bayes' theorem, conjugate priors, and posterior inference
2. **Implement Computational Methods**: Grid approximation, importance sampling, and MCMC algorithms
3. **Apply Variational Inference**: Understand ELBO, mean-field approximation, and modern variational methods
4. **Connect to Deep Learning**: See how Bayesian principles underlie modern neural network techniques
5. **Comprehend Generative Models**: Understand score-based methods and their connection to diffusion models
6. **Develop Practical Skills**: Implement all methods from scratch with heavily commented Python code

---

## 📖 Curriculum Structure

### **Module 01: Bayesian Inference Foundations** 
**Duration**: 2-3 weeks | **Level**: Beginner to Advanced

#### Topics Covered:
- **01. Bayes' Theorem Basics**: Medical testing, coin flipping, sequential updating
- **02. Continuous Distributions**: Beta-Binomial, Normal-Normal conjugate models
- **03. Conjugate Priors**: Complete conjugate families with analytical solutions
- **04. MAP Estimation**: Point estimates, regularization connections
- **05. Credible Intervals**: HPD intervals, comparison with confidence intervals
- **06. Hypothesis Testing**: Bayes factors, model comparison, Savage-Dickey ratio
- **07. Hierarchical Models**: Partial pooling, shrinkage estimation
- **08. Empirical Bayes**: Hyperparameter estimation, James-Stein estimator
- **09. Bayesian Linear Regression**: Full posterior inference, predictive distributions
- **10. Advanced Applications**: A/B testing, Bayesian optimization

#### Learning Path:
```
Basic Probability → Bayes' Theorem → Conjugate Models → 
Point Estimates → Interval Estimates → Model Comparison →
Hierarchical Structure → Practical Applications
```

#### Key Files:
- `01_bayesian_inference/01_bayes_theorem_basics.py`
- `01_bayesian_inference/02_continuous_bayesian_inference.py`
- `01_bayesian_inference/03_conjugate_priors.py`
- ... (see module README for complete list)

---

### **Module 02: Grid Approximation**
**Duration**: 3-5 days | **Level**: Beginner to Advanced

#### Topics Covered:
- Discretizing continuous parameter spaces
- Computing posteriors numerically on grids
- Multi-dimensional grid approximation
- Computational complexity and optimization
- When to use grid approximation vs. other methods

#### Why This Matters:
Grid approximation provides intuitive understanding of posterior distributions and serves as a computational bridge before learning more sophisticated sampling methods.

#### Key Files:
- `02_grid_approximation/01_grid_approximation_basics.py`
- `02_grid_approximation/02_grid_approximation_intermediate.py`
- `02_grid_approximation/03_grid_approximation_advanced.py`

---

### **Module 03: Importance Sampling**
**Duration**: 1-2 weeks | **Level**: Intermediate

#### Topics Covered:
- **01. Basic Importance Sampling**: Fundamental concept and theory
- **02. Self-Normalized IS**: Practical implementation without normalization constants
- **03. Bayesian Simple Examples**: Application to posterior inference
- **04. Effective Sample Size**: Diagnosing importance sampling quality
- **05. Proposal Diagnostics**: Choosing and evaluating proposal distributions
- **06. Logistic Regression IS**: Real-world Bayesian inference
- **07. Mixture Proposals**: Advanced proposal strategies
- **08. Adaptive Importance Sampling**: Learning proposals from samples
- **09. Sequential Importance Sampling**: Particle filters and SMC
- **10. Defensive Importance Sampling**: Robustness techniques
- **11. Rare Event Simulation**: Specialized applications

#### Key Concepts:
```
Target Distribution p(x) + Proposal Distribution q(x) → 
Importance Weights w(x) = p(x)/q(x) → 
Weighted Estimates E_p[f] ≈ Σ w(x_i)f(x_i) / Σ w(x_i)
```

#### Why This Matters:
Importance sampling is fundamental to modern variational inference, SMC methods, and serves as a building block for understanding more complex algorithms.

---

### **Module 04: MCMC Sampling**
**Duration**: 2 weeks | **Level**: Intermediate to Advanced

#### Topics Covered:
- **01. Gibbs Sampling**: Conditional sampling for multivariate distributions
- **02. Metropolis Algorithm**: Random-walk MCMC with accept/reject
- **03. Metropolis-Hastings**: Generalized proposal distributions
- **04. Langevin Dynamics**: Gradient-informed MCMC (critical for diffusion!)
- **05. Hamiltonian Monte Carlo (HMC)**: Physics-inspired efficient sampling
- **06. No-U-Turn Sampler (NUTS)**: Adaptive HMC implementation
- **07. Riemannian Manifold HMC**: Advanced geometry-aware sampling

#### Mathematical Foundation:
```
Detailed Balance: p(x)T(x'|x) = p(x')T(x|x')
Stationarity: p is the stationary distribution of the Markov chain
Ergodicity: Chain explores entire support of p
```

#### Critical Bridge to Modern ML:
- **Langevin Dynamics** introduces score functions: ∇ log p(x)
- Score functions are fundamental to diffusion models
- Understanding MCMC is essential for modern generative modeling

---

### **Module 05: Variational Inference**
**Duration**: 2-3 weeks | **Level**: Advanced

#### Comprehensive Structure:

##### **Beginner Level**: Foundation Concepts
- **01. Introduction to VI**: Optimization vs. sampling paradigm
- **02. ELBO Derivation**: Evidence Lower BOund and its properties
- **03. Mean-Field Approximation**: Factorized variational families

##### **Intermediate Level**: Practical Methods
- **04. Gaussian Mixture VI**: Multi-modal posterior approximation
- **05. Coordinate Ascent VI**: Classical optimization approach
- **06. Black Box VI**: Monte Carlo gradient estimation
- **07. Reparameterization Trick**: Low-variance gradient estimates
- **08. Bayesian Neural Networks**: VI for deep learning

##### **Advanced Level**: Modern Techniques
- **09. Normalizing Flows**: Flexible variational families
- **10. Importance Weighted VI**: Tighter bounds via IWAE
- **11. Amortized VI**: Inference networks and encoders
- **12. Structured VI**: Beyond mean-field approximations
- **13. Gradient Estimators**: Comprehensive comparison

#### Key Mathematical Relationships:
```
log p(x) = ELBO + KL[q(z)||p(z|x)]
ELBO = E_q[log p(x,z)] - E_q[log q(z)]
     = E_q[log p(x|z)] - KL[q(z)||p(z)]
```

#### Connection to Modern Deep Learning:
- VAEs use variational inference for learning generative models
- Amortized inference enables scalable Bayesian deep learning
- Same mathematical framework as diffusion model training

---

### **Module 06: Score-Based Methods Leading to Diffusion**
**Duration**: 2-3 weeks | **Level**: Advanced

#### Topics Covered:
- Score functions and their estimation
- Score matching (explicit and implicit)
- Denoising score matching
- Langevin dynamics with learned scores
- Connection to diffusion probabilistic models
- Score-based SDEs
- Bridge from Bayesian inference to modern generative models

#### Critical Insights:
```
Bayesian Denoising → Score Matching → Learned Scores → 
Langevin Sampling → Diffusion Models
```

#### Why This Is The Culmination:
This module ties together everything learned:
- Bayesian inference (Module 1) provides the probabilistic foundation
- MCMC (Module 4) introduces gradient-based sampling
- Variational inference (Module 5) provides training objectives
- Score-based methods unify these into modern generative models

---

## 🚀 Getting Started

### Prerequisites

#### Mathematical Background:
- **Probability Theory**: Random variables, distributions, conditional probability
- **Calculus**: Derivatives, integrals, multivariable calculus
- **Linear Algebra**: Matrices, eigenvalues, matrix calculus
- **Statistics**: Maximum likelihood, confidence intervals, hypothesis testing

#### Programming Skills:
- **Python**: Intermediate proficiency
- **NumPy**: Array operations, broadcasting
- **Matplotlib**: Basic plotting
- **Familiarity with**: SciPy, Seaborn (helpful but not required)

### Software Requirements

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0
pip install torch>=1.10.0  # For neural network modules
pip install pandas>=1.3.0

# Optional but recommended
pip install jupyter
pip install tqdm
pip install scikit-learn
```

### Installation

```bash
# Clone or download the package
# Navigate to the directory
cd bayesian_neural_networks

# Verify installation
python -c "import numpy, scipy, matplotlib; print('All packages installed!')"
```

---

## 📚 Recommended Learning Path

### **Path 1: Complete Sequential (Recommended for Courses)**
**Duration**: Full semester (14-16 weeks)

```
Week 1-3:   Module 01 (Bayesian Inference, topics 1-5)
Week 4-5:   Module 01 (Bayesian Inference, topics 6-10)
Week 6:     Module 02 (Grid Approximation)
Week 7-8:   Module 03 (Importance Sampling)
Week 9-10:  Module 04 (MCMC Sampling)
Week 11-13: Module 05 (Variational Inference)
Week 14-16: Module 06 (Score-Based Methods & Diffusion)
```

### **Path 2: Bayesian Foundations Only**
**Duration**: 4-6 weeks

Focus on Module 01 for a complete introduction to Bayesian inference without advanced computational methods.

```
Week 1:   Bayes' Theorem & Continuous Inference
Week 2:   Conjugate Priors & MAP Estimation
Week 3:   Credible Intervals & Hypothesis Testing
Week 4:   Hierarchical Models & Empirical Bayes
Week 5-6: Applications & Advanced Topics
```

### **Path 3: Computational Methods Track**
**Duration**: 6-8 weeks

For students with strong Bayesian background wanting to learn computational techniques.

```
Week 1:     Grid Approximation (Module 02)
Week 2-3:   Importance Sampling (Module 03)
Week 4-5:   MCMC Methods (Module 04)
Week 6-8:   Variational Inference (Module 05)
```

### **Path 4: Fast Track to Generative Models**
**Duration**: 3-4 weeks

Prerequisite: Strong background in probability and deep learning

```
Week 1: Module 01 (topics 1-4) + Module 04 (Langevin)
Week 2: Module 05 (Beginner & Intermediate VI)
Week 3-4: Module 06 (Score-Based Methods)
```

---

## 🎓 Pedagogical Approach

### Documentation Philosophy

Every Python file in this curriculum includes:

1. **Extensive Inline Comments**: Every function, every algorithm step is explained
2. **Mathematical Derivations**: Complete proofs and derivations in docstrings
3. **Conceptual Explanations**: "Why" before "how"
4. **Connections to Literature**: References to key papers and textbooks
5. **Visual Illustrations**: Comprehensive plotting for every concept
6. **Progressive Examples**: Simple → Realistic → Complex

### Code Structure

```python
"""
Module X: Topic Name
Level: Beginner/Intermediate/Advanced
Topics: Specific list of topics covered

This module teaches [main concept] through [approach].
Students will learn to [specific learning objectives].

Mathematical Foundation:
[Key equations and derivations]

Author: Professor Sungchul, Yonsei University
"""

import numpy as np
import matplotlib.pyplot as plt
# ... imports

# =================================================================
# SECTION 1: THEORY AND MATHEMATICAL FOUNDATION
# =================================================================

"""
Detailed explanation of theory with:
- Mathematical notation
- Derivations
- Intuitive explanations
"""

def well_documented_function(params):
    """
    Clear docstring explaining:
    - What the function does
    - Mathematical formulation
    - Parameters and return values
    - Examples of usage
    """
    # Step-by-step implementation with comments
    pass

# =================================================================
# SECTION 2: EXAMPLES AND DEMONSTRATIONS
# =================================================================

# Multiple examples showing different aspects

# =================================================================
# SECTION 3: EXERCISES FOR STUDENTS
# =================================================================

"""
Progressive exercises:
EXERCISE 1: [Basic concept]
EXERCISE 2: [Extension]
EXERCISE 3: [Real application]
"""
```

### Assessment Strategy

#### For Each Module:

**Formative Assessment** (During Learning):
- Inline exercises within code files
- Concept check questions in comments
- Small implementation tasks

**Summative Assessment** (End of Module):
- Comprehensive programming assignments
- Theoretical problem sets
- Real-world data applications

#### Suggested Grading Breakdown:
- Programming Assignments: 40%
- Theoretical Problem Sets: 30%
- Final Project: 20%
- Participation & Exercises: 10%

---

## 🔬 Key Mathematical Concepts

### Fundamental Theorem (Bayes' Rule)

```
p(θ|D) = p(D|θ)p(θ) / p(D)

Posterior = (Likelihood × Prior) / Evidence
```

### Core Computational Challenge

```
p(D) = ∫ p(D|θ)p(θ) dθ

This integral is often intractable!
```

### Solutions Covered in This Course

1. **Conjugate Analysis** (Module 01): Choose priors where integral is analytical
2. **Grid Approximation** (Module 02): Discretize and compute numerically
3. **Importance Sampling** (Module 03): Estimate via weighted samples
4. **MCMC** (Module 04): Sample from posterior without computing normalizing constant
5. **Variational Inference** (Module 05): Optimize an approximation to posterior
6. **Score-Based Methods** (Module 06): Learn and use score functions

---

## 📊 Visualization & Outputs

Each module generates educational visualizations:

### Types of Plots Produced:

1. **Prior-Likelihood-Posterior Relationships**
   - Side-by-side comparisons
   - Evolution of beliefs

2. **Sampling Algorithms**
   - Trace plots showing convergence
   - Autocorrelation analysis
   - Acceptance rate monitoring

3. **Approximation Quality**
   - True vs. approximate distributions
   - Error metrics over iterations
   - Convergence diagnostics

4. **Comparative Analysis**
   - Different methods side-by-side
   - Computational cost vs. accuracy
   - When to use which method

### Example Outputs:
- `medical_test_bayesian_inference.png`
- `sequential_updating.png`
- `map_vs_mle_vs_mean.png`
- `importance_sampling_comparison.png`
- `mcmc_trace_plots.png`
- `variational_approximation.png`

---

## 🔗 Connections to Modern Machine Learning

### How This Curriculum Enables Understanding of:

#### **1. Variational Autoencoders (VAEs)**
- **Module 05** provides the complete mathematical foundation
- ELBO is the VAE training objective
- Reparameterization trick enables gradient-based learning
- Amortized inference is the encoder network

#### **2. Bayesian Neural Networks**
- **Module 01** introduces hierarchical priors on weights
- **Module 04** shows MCMC for posterior sampling over weights
- **Module 05** demonstrates variational inference for BNNs
- Epistemic uncertainty quantification

#### **3. Gaussian Processes**
- **Module 01** (Bayesian Linear Regression) is the starting point
- Kernel methods as infinite-dimensional Bayesian inference
- Predictive distributions with uncertainty

#### **4. Diffusion Models**
- **Module 04** (Langevin Dynamics) introduces score-based sampling
- **Module 06** bridges to score-based generative models
- Denoising as Bayesian inference
- Variational bounds for training

#### **5. Meta-Learning & Few-Shot Learning**
- **Module 07** (Hierarchical Models) provides the framework
- Sharing statistical strength across tasks
- Bayesian approach to transfer learning

---

## 📚 References & Further Reading

### Textbooks

**Bayesian Foundations:**
- Gelman et al., *"Bayesian Data Analysis"* (3rd ed.)
- McElreath, *"Statistical Rethinking"*
- Kruschke, *"Doing Bayesian Data Analysis"*

**Computational Methods:**
- Brooks et al., *"Handbook of Markov Chain Monte Carlo"*
- Bishop, *"Pattern Recognition and Machine Learning"*
- Murphy, *"Machine Learning: A Probabilistic Perspective"*

**Modern Perspectives:**
- Blei et al., *"Variational Inference: A Review for Statisticians"*
- Zhang et al., *"Advances in Variational Inference"*

### Seminal Papers

**Bayesian Computation:**
- Metropolis et al. (1953): "Equation of State Calculations..."
- Hastings (1970): "Monte Carlo Sampling Methods..."
- Geman & Geman (1984): "Stochastic Relaxation, Gibbs..."

**Modern Methods:**
- Kingma & Welling (2014): "Auto-Encoding Variational Bayes"
- Rezende et al. (2014): "Stochastic Backpropagation..."
- Blundell et al. (2015): "Weight Uncertainty in Neural Networks"

**Score-Based Models:**
- Song & Ermon (2019): "Generative Modeling by Estimating Gradients"
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models"
- Song et al. (2021): "Score-Based Generative Modeling through SDEs"

---

## 🛠️ Using This Curriculum

### For Instructors

#### **Course Design Options:**

**Option 1: Full Semester Course (14-16 weeks)**
- Cover all modules sequentially
- Weekly programming assignments
- Midterm after Module 03
- Final project on Modules 05-06

**Option 2: Bayesian Inference Course (8 weeks)**
- Focus on Module 01 entirely
- Add selected topics from Modules 02-03
- Deep dive into applications

**Option 3: Advanced Topics Seminar (8 weeks)**
- Assume Module 01 as prerequisite
- Focus on Modules 04-06
- Research paper discussions
- Implementation projects

#### **Lecture Suggestions:**

**Lecture Format:**
- 45 min: Theory & Mathematical Derivations
- 30 min: Live Coding Demonstration
- 15 min: Q&A and Conceptual Discussion

**Flipped Classroom:**
- Students read code files before class
- Class time for discussion and advanced topics
- Office hours for coding help

### For Self-Study

#### **Recommended Approach:**

1. **Read the Theory First**: Docstrings and comments before running code
2. **Run Code Interactively**: Use Jupyter or IPython
3. **Modify Examples**: Change parameters, try different priors
4. **Complete Exercises**: Don't skip the practice problems
5. **Implement from Scratch**: Try writing your own versions
6. **Visualize Everything**: Plots are not optional—they build intuition

#### **Time Commitment:**

- **Beginner Level**: 4-6 hours per module
- **Intermediate Level**: 6-8 hours per module
- **Advanced Level**: 8-12 hours per module

#### **Study Groups:**
- Form groups of 3-4 students
- Weekly discussion sessions
- Compare implementations
- Explain concepts to each other

### For Researchers

This curriculum provides:
- **Solid Foundation**: Understand classical methods before modern variants
- **Implementation Details**: See how algorithms actually work
- **Connection to Literature**: References to key papers
- **Research Extensions**: Advanced modules include cutting-edge topics

---

## 💡 Unique Features of This Curriculum

### 1. **Complete Mathematical Rigor**
- Every algorithm includes full derivation
- Proofs and intuitions side-by-side
- No "magic"—everything is explained

### 2. **Progressive Difficulty**
- Beginner files: Intuition and simple examples
- Intermediate files: Realistic problems
- Advanced files: Research-level topics

### 3. **Heavy Documentation**
- 50%+ of each file is comments and docstrings
- Mathematical notation explained
- Every design choice justified

### 4. **Executable Education**
- All code runs out-of-the-box
- No dependencies on external datasets
- Self-contained examples

### 5. **Bridging Classical and Modern**
- Shows how VAEs, BNNs, and diffusion models
- Are applications of Bayesian principles
- Historical context meets cutting-edge

### 6. **Visualization-First**
- Every concept illustrated
- Animations for dynamic processes
- Comparative visualizations

---

## 📁 File Organization

```
bayesian_neural_networks/
│
├── MASTER_README.md (this file)
│
├── 01_bayesian_inference/
│   ├── README.md
│   ├── 01_bayes_theorem_basics.py
│   ├── 02_continuous_bayesian_inference.py
│   ├── 03_conjugate_priors.py
│   ├── 04_map_estimation.py
│   ├── 05_credible_intervals.py
│   ├── 06_hypothesis_testing.py
│   ├── 07_hierarchical_models.py
│   ├── 08_empirical_bayes.py
│   ├── 09_bayesian_linear_regression.py
│   └── 10_advanced_applications.py
│
├── 02_grid_approximation/
│   ├── README.md
│   ├── 01_grid_approximation_basics.py
│   ├── 02_grid_approximation_intermediate.py
│   └── 03_grid_approximation_advanced.py
│
├── 03_importance_sampling/
│   ├── README.md
│   ├── 01_basic_importance_sampling.py
│   ├── 02_self_normalized_IS.py
│   ├── 03_bayesian_simple_examples.py
│   ├── 04_effective_sample_size.py
│   ├── 05_proposal_diagnostics.py
│   ├── 06_logistic_regression_IS.py
│   ├── 07_mixture_proposals.py
│   ├── 08_adaptive_importance_sampling.py
│   ├── 09_sequential_importance_sampling.py
│   ├── 10_defensive_importance_sampling.py
│   ├── 11_rare_event_simulation.py
│   └── exercises.py
│
├── 04_mcmc_sampling/
│   ├── README.md
│   ├── 01_gibbs.py
│   ├── 02_metropolis.py
│   ├── 03_metropolis_hastings.py
│   ├── 04_langevin.py
│   ├── 05_hmc.py
│   ├── 06_nuts.py
│   └── 07_rmhmc.py
│
├── 05_variational_inference_approximating_posterior/
│   ├── README.md
│   ├── beginner/
│   │   ├── 01_introduction_to_vi.py
│   │   ├── 02_elbo_derivation.py
│   │   └── 03_mean_field_approximation.py
│   ├── intermediate/
│   │   ├── 04_gaussian_mixture_vi.py
│   │   ├── 05_coordinate_ascent_vi.py
│   │   ├── 06_black_box_vi.py
│   │   ├── 07_reparameterization_trick.py
│   │   └── 08_bayesian_neural_networks_vi.py
│   ├── advanced/
│   │   ├── 09_normalizing_flows.py
│   │   ├── 10_importance_weighted_vi.py
│   │   ├── 11_amortized_inference.py
│   │   ├── 12_structured_vi.py
│   │   └── 13_gradient_estimators_comparison.py
│   ├── exercises/
│   │   └── exercise_01_basic_vi.py
│   └── solutions/
│
└── 06_score_based_methods_leading_to_diffusion/
    ├── README.md
    └── [Score-based model implementations]
```

---

## 🎯 Learning Outcomes by Module

### **After Module 01: Bayesian Inference**
Students can:
- [ ] Derive and apply Bayes' theorem to inference problems
- [ ] Work with conjugate prior families
- [ ] Compute posterior distributions analytically
- [ ] Interpret credible intervals correctly
- [ ] Perform Bayesian hypothesis testing with Bayes factors
- [ ] Implement hierarchical models for multi-level data
- [ ] Apply Bayesian methods to real-world problems

### **After Module 02: Grid Approximation**
Students can:
- [ ] Implement numerical posterior computation on grids
- [ ] Understand computational trade-offs in discretization
- [ ] Know when grid approximation is appropriate
- [ ] Extend to multi-dimensional parameter spaces

### **After Module 03: Importance Sampling**
Students can:
- [ ] Implement basic and self-normalized importance sampling
- [ ] Choose appropriate proposal distributions
- [ ] Diagnose sampling quality with ESS and other metrics
- [ ] Apply to Bayesian inference problems
- [ ] Understand connections to modern variational methods

### **After Module 04: MCMC**
Students can:
- [ ] Implement Metropolis-Hastings from scratch
- [ ] Understand detailed balance and stationarity
- [ ] Diagnose convergence of Markov chains
- [ ] Use gradient information with Langevin dynamics
- [ ] Implement Hamiltonian Monte Carlo
- [ ] Understand score functions and their role

### **After Module 05: Variational Inference**
Students can:
- [ ] Derive the ELBO and understand its properties
- [ ] Implement mean-field variational inference
- [ ] Use the reparameterization trick for gradient estimation
- [ ] Apply VI to Bayesian neural networks
- [ ] Understand connections to VAEs and modern deep learning
- [ ] Compare different gradient estimators

### **After Module 06: Score-Based Methods**
Students can:
- [ ] Understand score matching and its variants
- [ ] Connect Langevin dynamics to score-based sampling
- [ ] See the path from Bayesian inference to diffusion models
- [ ] Understand modern generative modeling frameworks
- [ ] Implement score-based generative models

---

## 🎓 FAQ

### **Q: Can I use this for self-study?**
**A:** Absolutely! The heavy documentation and progressive structure make it ideal for independent learning. Start with Module 01 and work sequentially.

### **Q: Do I need a GPU?**
**A:** No. Most examples run fine on CPU. Only advanced neural network examples benefit from GPU acceleration.

### **Q: What if I get stuck?**
**A:** 
1. Read the inline comments carefully—they contain hints
2. Check the mathematical derivations in docstrings
3. Try simpler parameters first
4. Look at visualization outputs for intuition
5. Consult the references section

### **Q: Can I use this for teaching?**
**A:** Yes! That's exactly what it's designed for. You have full permission to use in educational contexts. Attribution appreciated.

### **Q: How is this different from textbooks?**
**A:** This combines theory with executable code. You don't just read about algorithms—you run them, modify them, and see them work.

### **Q: Do I need to complete all modules?**
**A:** No. Module 01 is comprehensive on its own. Modules 02-06 are for those wanting deeper computational expertise.

### **Q: Can I skip to advanced topics?**
**A:** Not recommended. The foundation in Modules 01-03 is essential for understanding Modules 04-06.

### **Q: Is this suitable for a PhD course?**
**A:** Yes, especially Modules 04-06. The advanced files contain research-level material and connections to current literature.

---

## 🚧 Future Extensions

Planned additions to this curriculum:

1. **Module 07: Gaussian Processes**
   - GP regression and classification
   - Kernel design
   - Sparse approximations

2. **Module 08: Bayesian Deep Learning**
   - Uncertainty in neural networks
   - Bayesian CNN implementations
   - Calibration techniques

3. **Module 09: Advanced Generative Models**
   - Complete diffusion model implementation
   - Score-based SDEs
   - Conditional generation

4. **Module 10: Sequential Decision Making**
   - Bayesian bandits
   - Thompson sampling
   - Bayesian optimization

---

## 📞 Contact & Support

### For Questions:
- **Email**: sungchulyonsei@gmail.com
- **Course**: Yonsei University, Department of Computer Science/Mathematics

### For Bug Reports:
Include:
- Module and file name
- Python version
- Error message
- Minimal reproducible example

### For Contributions:
This is an educational project. Suggestions for improvements are welcome, especially:
- Additional examples or exercises
- Clearer explanations of difficult concepts
- Bug fixes
- Additional visualizations

---

## 📄 License

**Educational Use License**

This curriculum is provided for educational purposes. You are free to:
- Use it for teaching courses
- Use it for self-study
- Modify it for your needs
- Share it with students

Please provide attribution to:
Professor Sungchul, Yonsei University

For commercial use or redistribution in published materials, please contact the author.

---

## 🙏 Acknowledgments

This curriculum was developed for undergraduate education at Yonsei University, with the goal of providing a principled pathway from classical Bayesian methods to modern machine learning techniques.

Special thanks to the students who have provided feedback and helped refine these materials through multiple iterations.

### Inspired By:
- The pedagogical approach of Richard McElreath's *Statistical Rethinking*
- The mathematical rigor of Gelman et al.'s *Bayesian Data Analysis*
- The computational focus of Michael Betancourt's writing
- The deep learning connections in Kingma & Welling's VAE work

---

## 🚀 Quick Start Guide

### **New to Bayesian Methods?**

```bash
# Start here
python 01_bayesian_inference/01_bayes_theorem_basics.py
```

### **Have Bayesian Background?**

```bash
# Jump to computational methods
python 03_importance_sampling/01_basic_importance_sampling.py
```

### **Ready for Advanced Topics?**

```bash
# Explore modern methods
python 05_variational_inference_approximating_posterior/beginner/01_introduction_to_vi.py
```

### **Want to Understand Diffusion Models?**

```bash
# Follow this path
python 04_mcmc_sampling/04_langevin.py  # Learn score-based sampling
python 06_score_based_methods_leading_to_diffusion/  # Connect to diffusion
```

---

## 📈 Curriculum Roadmap

```
Start Here
    ↓
Module 01: Bayesian Inference (Required Foundation)
    ↓
    ├─→ Module 02: Grid Approximation (Optional, Intuition Building)
    ↓
    ├─→ Module 03: Importance Sampling (Recommended)
    ↓
Module 04: MCMC Sampling (Essential for Modern ML)
    ↓
Module 05: Variational Inference (Core for Deep Learning)
    ↓
Module 06: Score-Based Methods (Path to Diffusion Models)
    ↓
Advanced Topics & Research
```

---

**Ready to begin your journey from Bayesian foundations to modern deep learning?**

**Start with: `01_bayesian_inference/01_bayes_theorem_basics.py`**

---

*Version 1.0 | Last Updated: November 2025 | Professor Sungchul, Yonsei University*
