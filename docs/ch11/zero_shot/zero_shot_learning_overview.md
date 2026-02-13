# Module 56: Zero-Shot Learning (ZSL)

## Overview
Zero-shot learning is a machine learning paradigm where models can recognize and classify objects or concepts that were never seen during training. This capability is achieved by leveraging semantic relationships, auxiliary information, and transfer learning from seen to unseen classes.

---

## Theoretical Foundations

### 1. The Zero-Shot Learning Problem

**Standard Supervised Learning:**
- Training set: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} where y ∈ Yˢᵉᵉⁿ
- Test set contains only instances from Yˢᵉᵉⁿ
- Goal: Learn f: X → Yˢᵉᵉⁿ

**Zero-Shot Learning:**
- Training set: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} where y ∈ Yˢᵉᵉⁿ
- Test set contains instances from Yᵘⁿˢᵉᵉⁿ where Yˢᵉᵉⁿ ∩ Yᵘⁿˢᵉᵉⁿ = ∅
- Goal: Learn f: X → Yᵘⁿˢᵉᵉⁿ despite never seeing examples from Yᵘⁿˢᵉᵉⁿ

**Key Insight:** Use auxiliary information (attributes, semantic embeddings) to bridge seen and unseen classes.

---

### 2. Mathematical Framework

#### Semantic Space
Let S be a semantic space where both seen and unseen classes have representations:
- sᶜ ∈ S: semantic representation of class c
- Examples: attribute vectors, word embeddings (Word2Vec, GloVe), knowledge graphs

#### Embedding Functions
Zero-shot learning typically learns two functions:
1. **Visual Embedding**: φ: X → V (maps images to visual feature space)
2. **Compatibility Function**: F: V × S → ℝ (measures compatibility between visual features and semantic representations)

**Prediction Rule:**
```
ŷ = argmax_{c ∈ Yᵘⁿˢᵉᵉⁿ} F(φ(x), sᶜ)
```

---

### 3. Attribute-Based Zero-Shot Learning

#### Attribute Representation
Each class c is described by an attribute vector aᶜ ∈ {0,1}ᴹ or aᶜ ∈ ℝᴹ:
- aᶜ[m] indicates presence/strength of attribute m for class c
- Example: For animals
  - "has_stripes" = 1 for zebra, 0 for elephant
  - "size" = 0.9 for elephant, 0.5 for zebra

#### Direct Attribute Prediction (DAP)
Learn M binary classifiers, one per attribute:
```
P(aₘ = 1 | x) for m = 1, ..., M
```

Prediction:
```
ŷ = argmax_{c ∈ Yᵘⁿˢᵉᵉⁿ} P(c | a₁, ..., aₘ) × ∏ₘ P(aₘ | x)^aᶜₘ
```

#### Indirect Attribute Prediction (IAP)
Learn class-level probabilities using attribute similarity:
```
P(c | x) ∝ exp(-d(a(x), aᶜ))
```
where a(x) is predicted attribute vector and d(·,·) is distance metric.

---

### 4. Semantic Embedding Approaches

#### Visual-Semantic Embedding
Learn a joint embedding space where visual features and semantic vectors are comparable:

**Loss Function (Compatibility Learning):**
```
L = ∑_{(x,y) ∈ Training} max(0, Δ + F(φ(x), s_y') - F(φ(x), s_y))
```
where:
- y is true class, y' is incorrect class
- Δ is margin parameter
- F(v, s) = vᵀWs (bilinear compatibility)

**Variants:**
1. **ConSE (Convex Combination of Semantic Embeddings):**
   ```
   ŷ = argmin_{c ∈ Yᵘⁿˢᵉᵉⁿ} ||φ(x) - ∑ᵢ P(yᵢ|x)·sᵧᵢ||²
   ```

2. **DeViSE (Deep Visual-Semantic Embedding):**
   ```
   L = ∑ₓ max(0, margin + cos(φ(x), s_{y'}) - cos(φ(x), s_y))
   ```

---

### 5. Generalized Zero-Shot Learning (GZSL)

**Problem:** In real-world scenarios, test set may contain both seen and unseen classes.
- Test set: Y^test = Y^seen ∪ Y^unseen
- More challenging due to bias toward seen classes

**Harmonic Mean Metric:**
```
H = 2 × (acc^seen × acc^unseen) / (acc^seen + acc^unseen)
```

**Calibration Techniques:**
1. **Confidence Calibration:** Adjust scores for seen vs unseen
2. **Bias Reduction:** Subtract constant from seen class scores
3. **Gating Mechanisms:** Learn when to predict seen vs unseen

---

### 6. Advanced Architectures

#### Cross-Modal Retrieval
Learn bidirectional mappings:
```
V → S: Map visual features to semantic space
S → V: Map semantic vectors to visual space
```

**Cycle Consistency Loss:**
```
L_cycle = ||x - g(f(x))||² + ||s - f(g(s))||²
```

#### Generative ZSL
Synthesize visual features for unseen classes:
```
G: S × z → V (Generator)
z ~ N(0, I) (Gaussian noise)
```

Train on seen classes, generate features for unseen, then train classifier.

---

### 7. Transductive Zero-Shot Learning

**Setting:** Unlabeled test data available during training (but not labels)
- Can leverage distribution of unseen class instances
- Use techniques like:
  - Semi-supervised learning
  - Self-training
  - Graph-based propagation

---

## Practical Considerations

### 1. Semantic Representations
**Word Embeddings:**
- Word2Vec (CBOW, Skip-gram)
- GloVe (Global Vectors)
- FastText (subword information)

**Attribute Annotations:**
- Expensive to create
- May not capture all discriminative information
- Quality varies with annotator expertise

### 2. Domain Shift Problem
Visual features of seen classes may differ from unseen:
- Distribution mismatch
- Hubness problem (certain points become hubs in high-dimensional spaces)

**Solutions:**
- Domain adaptation techniques
- Feature normalization
- Hubness-aware similarity measures

### 3. Evaluation Protocols
**Conventional ZSL:**
- Train: Y^seen, Test: Y^unseen only

**Generalized ZSL (GZSL):**
- Train: Y^seen, Test: Y^seen ∪ Y^unseen

**Metrics:**
- Top-1 accuracy
- Top-k accuracy (k=5)
- Harmonic mean (for GZSL)
- Area under seen-unseen curve (AUSUC)

---

## Datasets

### Standard ZSL Benchmarks
1. **Animals with Attributes (AwA / AwA2)**
   - 50 animal classes
   - 85 attributes
   - ~30,000 images

2. **Caltech-UCSD Birds (CUB-200-2011)**
   - 200 bird species
   - 312 attributes
   - ~12,000 images

3. **SUN Attribute Database**
   - 717 scene categories
   - 102 attributes
   - ~14,000 images

4. **ImageNet (ILSVRC)**
   - 21,841 classes (WordNet synsets)
   - Can create various seen/unseen splits
   - Word embeddings as semantic space

---

## Module Contents

### Beginner Level
**01_beginner_zero_shot_basics.py**
- Introduction to zero-shot learning concept
- Simple nearest-neighbor in semantic space
- Toy example with synthetic data
- Basic attribute-based classification

### Intermediate Level
**02_intermediate_attribute_based.py**
- Direct Attribute Prediction (DAP)
- Indirect Attribute Prediction (IAP)
- Implementation with real animal dataset
- Evaluation metrics and analysis

**03_intermediate_semantic_embeddings.py**
- Word2Vec embeddings for class representations
- Visual feature extraction (pretrained CNN)
- Compatibility learning with ranking loss
- ConSE (Convex Combination of Semantic Embeddings)

### Advanced Level
**04_advanced_visual_semantic.py**
- Deep visual-semantic embedding (DeViSE)
- Bilinear compatibility models
- Cross-entropy and ranking losses
- Comprehensive training pipeline

**05_advanced_generalized_zsl.py**
- Generalized zero-shot learning (GZSL)
- Seen vs unseen class handling
- Calibration techniques
- Harmonic mean evaluation
- Feature generation approaches

---

## Learning Objectives

After completing this module, students will be able to:
1. ✓ Understand the zero-shot learning problem formulation
2. ✓ Implement attribute-based zero-shot classification
3. ✓ Use semantic embeddings (Word2Vec, GloVe) for ZSL
4. ✓ Build visual-semantic embedding models
5. ✓ Handle generalized zero-shot learning scenarios
6. ✓ Evaluate ZSL models with appropriate metrics
7. ✓ Understand limitations and practical challenges

---

## Prerequisites
- Module 20: Feedforward Networks
- Module 23: Convolutional Neural Networks
- Module 36: Word Embeddings
- Module 53: Transfer Learning

---

## References

### Foundational Papers
1. Lampert et al. (2009) - "Learning to detect unseen object classes by between-class attribute transfer"
2. Palatucci et al. (2009) - "Zero-shot learning with semantic output codes"
3. Socher et al. (2013) - "Zero-shot learning through cross-modal transfer"

### Key Methods
4. Romera-Paredes & Torr (2015) - "An embarrassingly simple approach to zero-shot learning"
5. Frome et al. (2013) - "DeViSE: A deep visual-semantic embedding model"
6. Norouzi et al. (2013) - "Zero-shot learning by convex combination of semantic embeddings"

### Modern Approaches
7. Xian et al. (2018) - "Zero-shot learning—A comprehensive evaluation of the good, the bad and the ugly"
8. Schonfeld et al. (2019) - "Generalized zero- and few-shot learning via aligned variational autoencoders"
9. Verma et al. (2018) - "Generalized zero-shot learning via synthesized examples"

### Surveys
10. Wang et al. (2019) - "A survey on zero-shot learning"
11. Pourpanah et al. (2022) - "A review of generalized zero-shot learning methods"

---

## Installation Requirements

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn gensim --break-system-packages
```

---

## Usage

Each script is standalone and can be run independently:

```bash
# Beginner: Introduction to zero-shot concepts
python 01_beginner_zero_shot_basics.py

# Intermediate: Attribute-based approaches
python 02_intermediate_attribute_based.py

# Intermediate: Semantic embedding methods
python 03_intermediate_semantic_embeddings.py

# Advanced: Visual-semantic embeddings
python 04_advanced_visual_semantic.py

# Advanced: Generalized zero-shot learning
python 05_advanced_generalized_zsl.py
```

---

## Tips for Teaching

1. **Start with Intuition:** Use analogy of describing new animal species by attributes
2. **Visualize Embeddings:** Show t-SNE plots of semantic spaces
3. **Discuss Limitations:** Hubness, domain shift, semantic gap
4. **Real Applications:** Image search, rare disease diagnosis, new product categorization
5. **Compare to Few-Shot:** Highlight relationship and differences

---

## Future Extensions

- Zero-shot learning with vision transformers
- Large language models for zero-shot classification (CLIP, ALIGN)
- Knowledge graph-based zero-shot learning
- Zero-shot learning in NLP (GPT-style models)
- Cross-lingual zero-shot transfer

---

**Author:** Deep Learning Curriculum  
**Module:** 56 - Zero-Shot Learning  
**Level:** Advanced Topics & Applications  
**Estimated Time:** 4-6 hours
