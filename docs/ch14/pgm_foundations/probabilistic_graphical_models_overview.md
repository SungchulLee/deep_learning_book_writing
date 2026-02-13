# Probabilistic Graphical Models (PGMs) - Comprehensive Python Tutorial

## 📚 Module 48: Probabilistic Graphical Models

A complete educational package for understanding and implementing Probabilistic Graphical Models from foundational concepts to advanced applications.

---

## 🎯 Learning Objectives

By completing this tutorial, you will:

1. **Understand PGM fundamentals**: Learn the theoretical foundations of graphical models
2. **Master Bayesian Networks**: Implement and perform inference in directed graphical models
3. **Explore Markov Random Fields**: Work with undirected graphical models
4. **Implement inference algorithms**: Variable elimination, belief propagation, sampling methods
5. **Learn parameter estimation**: Maximum likelihood and Bayesian approaches
6. **Apply structure learning**: Discover graph structures from data
7. **Build real-world applications**: Medical diagnosis, image segmentation, NLP tasks

---

## 📂 Package Structure

```
48_probabilistic_graphical_models/
│
├── README.md                           # This file
├── requirements.txt                    # Package dependencies
│
├── 01_beginner/
│   ├── 01_pgm_fundamentals.py         # Introduction to PGMs, basic concepts
│   ├── 02_bayesian_networks_basics.py  # Simple Bayesian networks
│   ├── 03_conditional_probability.py   # Conditional probability in graphs
│   ├── 04_simple_inference.py         # Basic inference by enumeration
│   └── 05_parameter_representation.py  # CPT representation and manipulation
│
├── 02_intermediate/
│   ├── 01_variable_elimination.py      # Variable elimination algorithm
│   ├── 02_markov_random_fields.py      # Undirected graphical models
│   ├── 03_factor_graphs.py            # Factor graph representation
│   ├── 04_belief_propagation.py       # Sum-product algorithm
│   ├── 05_markov_chains.py            # Temporal models (HMM basics)
│   └── 06_parameter_learning.py       # MLE for PGM parameters
│
├── 03_advanced/
│   ├── 01_advanced_inference.py       # Loopy BP, sampling methods
│   ├── 02_structure_learning.py       # Learning graph structure
│   ├── 03_conditional_random_fields.py # CRFs for structured prediction
│   ├── 04_dynamic_bayesian_networks.py # Temporal models
│   ├── 05_variational_inference.py    # Mean-field approximation
│   └── 06_applications.py             # Real-world case studies
│
├── data/
│   ├── sample_medical_data.csv        # Medical diagnosis dataset
│   ├── sample_text_data.txt           # NER dataset
│   └── sample_image_data.npy          # Image segmentation data
│
├── utils/
│   ├── graph_utils.py                 # Graph manipulation utilities
│   ├── inference_utils.py             # Inference helper functions
│   ├── visualization.py               # Graph visualization tools
│   └── data_generators.py             # Synthetic data generation
│
└── notebooks/
    ├── tutorial_01_introduction.ipynb  # Interactive introduction
    ├── tutorial_02_inference.ipynb     # Inference walkthrough
    └── tutorial_03_applications.ipynb  # Application examples
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Basic understanding of probability theory
- Familiarity with NumPy and PyTorch

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; import numpy; import networkx; print('Setup successful!')"
```

### Required Packages
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
networkx>=3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
scikit-learn>=1.3.0
graphviz>=0.20.0
```

---

## 🚀 Quick Start

### Example 1: Simple Bayesian Network
```python
from beginner.bayesian_networks_basics import SimpleBayesianNetwork

# Create a simple Bayesian network for weather prediction
bn = SimpleBayesianNetwork()
bn.add_node("Cloudy", parents=[])
bn.add_node("Rain", parents=["Cloudy"])
bn.add_node("Sprinkler", parents=["Cloudy"])
bn.add_node("WetGrass", parents=["Rain", "Sprinkler"])

# Define conditional probability tables
bn.set_cpd("Cloudy", [0.5, 0.5])  # P(Cloudy=True)
bn.set_cpd("Rain", [[0.8, 0.2], [0.2, 0.8]])  # P(Rain|Cloudy)

# Perform inference
prob = bn.query("Rain", evidence={"Cloudy": True})
print(f"P(Rain|Cloudy) = {prob}")
```

### Example 2: Variable Elimination
```python
from intermediate.variable_elimination import VariableElimination

# Load pre-defined medical diagnosis network
ve = VariableElimination.from_file("data/medical_network.json")

# Query probability of disease given symptoms
result = ve.query(
    variables=["Disease"],
    evidence={"Fever": True, "Cough": True}
)
print(f"P(Disease|Symptoms) = {result}")
```

### Example 3: Structure Learning
```python
from advanced.structure_learning import StructureLearner

# Learn Bayesian network structure from data
learner = StructureLearner(method="hill_climbing")
data = pd.read_csv("data/sample_medical_data.csv")
learned_structure = learner.fit(data)

# Visualize learned structure
learner.plot_graph()
```

---

## 📖 Curriculum Guide

### **Week 1-2: Foundations (Beginner Level)**
**Goal**: Understand what PGMs are and why they're useful

1. **Day 1-2**: `01_pgm_fundamentals.py`
   - Introduction to probabilistic modeling
   - Graphical model representations
   - Independence and conditional independence
   - D-separation

2. **Day 3-4**: `02_bayesian_networks_basics.py`
   - Directed acyclic graphs (DAGs)
   - Joint probability distribution
   - Local probability models (CPTs)
   - Simple example networks

3. **Day 5-6**: `03_conditional_probability.py`
   - Conditional probability queries
   - Chain rule in Bayesian networks
   - Bayes' theorem in graphs

4. **Day 7-8**: `04_simple_inference.py`
   - Inference by enumeration
   - Forward sampling
   - Rejection sampling

5. **Day 9-10**: `05_parameter_representation.py`
   - Compact parameter representations
   - Context-specific independence
   - Noisy-OR models

**Recommended exercises**: Implement simple networks (alarm system, student grades, weather prediction)

---

### **Week 3-4: Core Algorithms (Intermediate Level)**
**Goal**: Master efficient inference and learning algorithms

1. **Day 1-3**: `01_variable_elimination.py`
   - Variable elimination algorithm
   - Elimination orderings
   - Complexity analysis
   - Practical implementation

2. **Day 4-5**: `02_markov_random_fields.py`
   - Undirected graphical models
   - Potential functions
   - Gibbs distribution
   - Hammersley-Clifford theorem

3. **Day 6-7**: `03_factor_graphs.py`
   - Factor graph representation
   - Converting between representations
   - Factor operations

4. **Day 8-10**: `04_belief_propagation.py`
   - Message passing algorithms
   - Sum-product algorithm
   - Exact inference on trees
   - Loopy belief propagation

5. **Day 11-12**: `05_markov_chains.py`
   - Markov chains
   - Hidden Markov Models (HMMs)
   - Forward-backward algorithm
   - Viterbi algorithm

6. **Day 13-14**: `06_parameter_learning.py`
   - Maximum likelihood estimation
   - Bayesian parameter estimation
   - Expectation-Maximization (EM)
   - Missing data scenarios

**Recommended exercises**: Implement HMM for speech recognition, parameter learning for medical diagnosis

---

### **Week 5-6: Advanced Topics (Advanced Level)**
**Goal**: Tackle complex inference and real-world applications

1. **Day 1-3**: `01_advanced_inference.py`
   - Loopy belief propagation
   - MCMC methods (Gibbs sampling)
   - Importance sampling
   - Particle filtering

2. **Day 4-6**: `02_structure_learning.py`
   - Constraint-based methods (PC algorithm)
   - Score-based methods (BIC, BDe)
   - Hybrid approaches
   - Causal discovery

3. **Day 7-9**: `03_conditional_random_fields.py`
   - Linear-chain CRFs
   - General CRFs
   - Feature engineering
   - Applications to NLP

4. **Day 10-11**: `04_dynamic_bayesian_networks.py`
   - Temporal models
   - 2-Time-Slice BNs (2-TBN)
   - Kalman filters
   - Particle filters

5. **Day 12-13**: `05_variational_inference.py`
   - Mean-field approximation
   - Variational message passing
   - Expectation propagation
   - Modern variational methods

6. **Day 14**: `06_applications.py`
   - Medical diagnosis systems
   - Image segmentation
   - Named entity recognition
   - Robot localization

**Recommended projects**: Build end-to-end diagnostic system, implement CRF for NER

---

## 🎓 Mathematical Prerequisites

### Essential Concepts
- **Probability theory**: Joint, marginal, conditional probabilities
- **Graph theory**: DAGs, undirected graphs, tree structures
- **Linear algebra**: Matrix operations, eigenvalues
- **Calculus**: Gradients, optimization
- **Statistics**: Maximum likelihood, Bayesian inference

### Recommended Review Resources
1. Bishop's "Pattern Recognition and Machine Learning" (Chapters 8, 13)
2. Koller & Friedman's "Probabilistic Graphical Models"
3. Murphy's "Machine Learning: A Probabilistic Perspective" (Chapters 10, 17-20)

---

## 🔬 Key Algorithms Implemented

### Inference Algorithms
- ✅ Inference by enumeration
- ✅ Variable elimination
- ✅ Belief propagation (sum-product)
- ✅ Max-product algorithm
- ✅ Junction tree algorithm
- ✅ Gibbs sampling
- ✅ Importance sampling
- ✅ Particle filtering

### Learning Algorithms
- ✅ Maximum likelihood estimation
- ✅ Bayesian parameter learning
- ✅ EM algorithm
- ✅ PC algorithm (structure learning)
- ✅ Hill climbing (score-based)
- ✅ CRF parameter learning

---

## 💡 Pedagogical Features

### Code Quality
- **Heavily commented**: Every function includes detailed docstrings
- **Type hints**: Full type annotations for clarity
- **Error handling**: Comprehensive input validation
- **Modular design**: Reusable components and utilities

### Learning Aids
- **Visual examples**: Graph visualizations throughout
- **Step-by-step execution**: Detailed algorithm traces
- **Comparative analysis**: Multiple approaches side-by-side
- **Real-world datasets**: Practical applications with actual data

### Progressive Complexity
- **Beginner**: Concrete examples, small networks, visual intuition
- **Intermediate**: Efficient algorithms, medium-scale problems
- **Advanced**: State-of-the-art methods, large-scale applications

---

## 🎯 Applications Covered

### 1. Medical Diagnosis
- Symptom-disease relationships
- Multi-disease inference
- Treatment recommendation
- Risk assessment

### 2. Natural Language Processing
- Part-of-speech tagging
- Named entity recognition
- Dependency parsing
- Sentiment analysis

### 3. Computer Vision
- Image segmentation
- Object recognition
- Scene understanding
- Activity recognition

### 4. Robotics
- Robot localization
- SLAM (Simultaneous Localization and Mapping)
- Motion planning
- Sensor fusion

### 5. Bioinformatics
- Gene regulatory networks
- Protein structure prediction
- Phylogenetic inference

---

## 🧪 Testing & Validation

Each module includes:
- **Unit tests**: Validate individual components
- **Integration tests**: Test complete workflows
- **Numerical tests**: Verify probabilistic computations
- **Benchmark datasets**: Compare against known results

Run all tests:
```bash
python -m pytest tests/
```

---

## 📊 Datasets Included

### 1. Medical Diagnosis Dataset
- 1000 patient records
- 10 symptoms, 5 diseases
- Ground truth diagnoses
- Realistic conditional dependencies

### 2. Text Corpus (NER)
- 5000 sentences
- Named entity annotations
- Multiple entity types
- Train/validation/test splits

### 3. Image Segmentation Data
- 500 images (256x256)
- Pixel-level annotations
- Multiple object classes
- Augmented variations

---

## 🛠️ Utilities Provided

### Graph Utilities (`utils/graph_utils.py`)
- Graph construction and manipulation
- Topological sorting
- Moralization (convert directed to undirected)
- Clique tree construction

### Inference Utilities (`utils/inference_utils.py`)
- Factor operations (product, marginalization)
- Message computation
- Evidence incorporation
- Normalization helpers

### Visualization (`utils/visualization.py`)
- Network structure plotting
- CPT visualization
- Inference result display
- Learning curve plotting

### Data Generators (`utils/data_generators.py`)
- Synthetic Bayesian network data
- Controlled parameter variations
- Missing data injection
- Noise addition

---

## 📈 Performance Considerations

### Computational Complexity
- **Exact inference**: Exponential in tree-width (NP-hard in general)
- **Approximate inference**: Polynomial with approximation guarantees
- **Learning**: Cubic in number of variables (MLE), exponential (structure)

### Scalability Tips
1. Use sparse representations for large networks
2. Choose elimination ordering carefully
3. Consider approximate inference for loopy networks
4. Cache intermediate computations
5. Use batch processing for parameter learning

---

## 🔗 Connections to Other Modules

### Prerequisites
- **Module 06 (MLE)**: Maximum likelihood foundations
- **Module 44 (MCMC)**: Sampling-based inference
- **Module 45 (Bayesian NNs)**: Bayesian learning principles

### Related Modules
- **Module 37 (Language Modeling)**: PGMs for NLP
- **Module 41 (VAEs)**: Latent variable models
- **Module 46 (Diffusion Models)**: Score-based generative models

### Advanced Extensions
- **Module 49 (Score-based Models)**: Energy-based PGMs
- **Module 69 (Graph Neural Networks)**: Neural approaches to graphs
- **Module 71 (RL Basics)**: POMDPs as graphical models

---

## 📚 Further Reading

### Textbooks
1. **Koller & Friedman** - "Probabilistic Graphical Models: Principles and Techniques"
2. **Bishop** - "Pattern Recognition and Machine Learning"
3. **Murphy** - "Machine Learning: A Probabilistic Perspective"
4. **Wainwright & Jordan** - "Graphical Models, Exponential Families, and Variational Inference"

### Papers
1. Pearl (1988) - "Probabilistic Reasoning in Intelligent Systems"
2. Lauritzen & Spiegelhalter (1988) - "Local Computations with Probabilities"
3. Jordan et al. (1999) - "An Introduction to Variational Methods"
4. Lafferty et al. (2001) - "Conditional Random Fields"

### Online Resources
1. Stanford CS228: Probabilistic Graphical Models
2. CMU 10-708: Probabilistic Graphical Models
3. pgmpy documentation (Python library)

---

## 🤝 Contributing

This is an educational package. Suggestions for improvements:
- Additional example networks
- More real-world datasets
- Alternative algorithm implementations
- Additional visualizations
- Tutorial notebooks

---

## 📝 License

Educational use only. All code is provided for learning purposes.

---

## ✅ Learning Checklist

Track your progress through the curriculum:

### Beginner Level
- [ ] Understand graphical model representations
- [ ] Build simple Bayesian networks
- [ ] Perform inference by enumeration
- [ ] Interpret conditional probability tables
- [ ] Visualize and analyze small networks

### Intermediate Level
- [ ] Implement variable elimination
- [ ] Work with Markov random fields
- [ ] Use factor graphs effectively
- [ ] Apply belief propagation
- [ ] Estimate parameters from data
- [ ] Understand HMMs and temporal models

### Advanced Level
- [ ] Implement approximate inference methods
- [ ] Learn network structures from data
- [ ] Apply CRFs to structured prediction
- [ ] Model temporal dynamics
- [ ] Use variational inference
- [ ] Build complete application systems

---

## 💬 Common Questions

**Q: When should I use Bayesian networks vs. Markov random fields?**
A: Use Bayesian networks when you have clear causal relationships (directed). Use MRFs when you have symmetric relationships or constraints (undirected).

**Q: Is exact inference always possible?**
A: Exact inference is tractable for tree-structured graphs but NP-hard in general. For loopy graphs, use approximate methods.

**Q: How do PGMs relate to deep learning?**
A: PGMs provide probabilistic semantics and interpretability. Many deep learning models (VAEs, normalizing flows, attention) have PGM interpretations.

**Q: What's the difference between discriminative and generative models?**
A: Generative models (Bayesian networks) model P(X,Y). Discriminative models (CRFs) model P(Y|X) directly.

---

## 🎓 Assessment

### Beginner Mastery
- Build a Bayesian network from scratch
- Perform inference queries manually
- Explain d-separation
- Interpret CPTs correctly

### Intermediate Mastery
- Implement variable elimination
- Choose good elimination orderings
- Apply belief propagation
- Learn parameters from data

### Advanced Mastery
- Compare inference algorithms
- Perform structure learning
- Apply PGMs to real problems
- Understand theoretical guarantees

---

**Happy Learning! 🚀**

For questions or feedback, please refer to the main curriculum documentation.
