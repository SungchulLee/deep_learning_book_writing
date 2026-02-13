# Module 52: Generative Model Evaluation

## Overview
This module provides a comprehensive guide to evaluating generative models, covering both traditional likelihood-based metrics and modern perceptual quality measures. Understanding how to properly evaluate generative models is crucial for model development, comparison, and deployment.

## Learning Objectives
By completing this module, you will:
- Understand different paradigms for evaluating generative models
- Implement likelihood-based evaluation metrics
- Apply perceptual quality metrics (IS, FID, KID)
- Measure sample diversity and coverage
- Evaluate text generation quality
- Compare different generative model architectures objectively

## Prerequisites
- Module 02: Tensors
- Module 04: Gradients
- Module 20: Feedforward Networks
- Module 23: Convolutional Neural Networks
- Module 34: Variational Autoencoders
- Module 35: Generative Adversarial Networks
- Module 37: Diffusion Models
- Understanding of probability distributions
- Familiarity with statistical distance metrics

## Module Structure

### 1. Beginner Level (`01_beginner/`)
**Focus: Foundational concepts and basic metrics**

#### `01_evaluation_basics.py`
- Why evaluation matters for generative models
- Likelihood vs. sample quality tradeoffs
- Overview of evaluation paradigms
- Simple likelihood computations

#### `02_likelihood_metrics.py`
- Negative log-likelihood (NLL)
- Bits per dimension (BPD)
- Perplexity for language models
- Implementations with toy distributions

#### `03_sample_visualization.py`
- Visual quality assessment
- Grid visualization techniques
- Interpolation in latent space
- Reconstruction quality metrics

### 2. Intermediate Level (`02_intermediate/`)
**Focus: Perceptual metrics and practical implementations**

#### `01_inception_score.py`
- Inception Score (IS) theory and intuition
- Using pre-trained InceptionV3
- Measuring conditional and marginal entropy
- Limitations and failure cases

#### `02_frechet_inception_distance.py`
- Fréchet Inception Distance (FID) derivation
- Feature extraction with InceptionV3
- Computing mean and covariance statistics
- FID vs. IS comparison

#### `03_kernel_inception_distance.py`
- Kernel Inception Distance (KID)
- Unbiased estimator properties
- MMD (Maximum Mean Discrepancy) connection
- Computational efficiency considerations

#### `04_precision_recall_metrics.py`
- Precision and Recall for generative models
- Density and coverage tradeoffs
- K-nearest neighbor approach
- Improved Precision/Recall metrics

### 3. Advanced Level (`03_advanced/`)
**Focus: Comprehensive evaluation suites and specialized metrics**

#### `01_perceptual_path_length.py`
- Perceptual Path Length (PPL) for latent space smoothness
- LPIPS (Learned Perceptual Image Patch Similarity)
- VGG-based perceptual distances
- Evaluating GAN disentanglement

#### `02_text_generation_metrics.py`
- BLEU, ROUGE, METEOR scores
- BERTScore for semantic similarity
- Perplexity for language models
- Diversity metrics (distinct-n, self-BLEU)

#### `03_conditional_generation_evaluation.py`
- Class-conditional generation metrics
- Image-to-image translation evaluation (PSNR, SSIM)
- Text-to-image alignment (CLIPScore)
- Multimodal evaluation frameworks

#### `04_comprehensive_evaluation_suite.py`
- Complete evaluation pipeline
- Multiple metrics computation
- Statistical significance testing
- Comparative analysis tools
- Visualization and reporting

#### `05_evaluation_best_practices.py`
- Common pitfalls and how to avoid them
- Sample size considerations
- Metric selection guidelines
- Reporting standards for papers
- Computational efficiency optimization

## Key Concepts

### 1. Likelihood-Based Metrics
**Mathematical Foundation:**
```
NLL = -E_{x~p_data}[log p_model(x)]
BPD = NLL / (dimensions × log(2))
```

**Use Cases:**
- Models with tractable likelihoods (VAEs, normalizing flows)
- Direct comparison of probabilistic models
- Overfitting detection

**Limitations:**
- May not correlate with perceptual quality
- Sensitive to model capacity and mode coverage

### 2. Inception Score (IS)
**Mathematical Definition:**
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```
where:
- p(y|x): Conditional class distribution (sharpness)
- p(y): Marginal class distribution (diversity)

**Strengths:**
- Single number metric
- Considers both quality and diversity
- Fast to compute

**Weaknesses:**
- Only works for ImageNet-like images
- Cannot detect overfitting to training data
- Ignores within-class diversity

### 3. Fréchet Inception Distance (FID)
**Mathematical Definition:**
```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real × Σ_gen)^(1/2))
```

**Why FID is Better:**
- Compares distributions, not just statistics
- Sensitive to mode dropping
- More robust than IS
- Widely adopted standard

**Implementation Notes:**
- Requires 2048-dimensional InceptionV3 features
- Minimum 10,000 samples recommended
- Use consistent preprocessing

### 4. Precision and Recall
**Conceptual Framework:**
```
Precision = Quality (Are generated samples realistic?)
Recall = Coverage (Does model cover all modes?)
```

**Computation:**
- Build k-NN manifolds from real and generated data
- Precision: % of generated samples within real manifold
- Recall: % of real samples within generated manifold

### 5. Perceptual Metrics
**LPIPS (Learned Perceptual Image Patch Similarity):**
- Uses deep features from VGG/AlexNet
- Better correlation with human judgment
- Weighted by layer importance

**SSIM (Structural Similarity Index):**
- Luminance, contrast, and structure comparison
- Good for image-to-image tasks
- Not ideal for unconditional generation

## Evaluation Workflow

### Step 1: Choose Appropriate Metrics
```
Model Type          | Recommended Metrics
--------------------|--------------------------------------------
VAE                 | ELBO, NLL, FID, Reconstruction MSE
GAN                 | FID, IS, Precision/Recall, PPL
Diffusion           | FID, IS, NFE, Sampling Time
Autoregressive      | NLL, Perplexity, Sample Quality (FID)
Flow                | NLL, BPD, FID
```

### Step 2: Generate Samples
- Sufficient sample size (≥10K for FID)
- Consistent random seeds for reproducibility
- Save samples for later analysis

### Step 3: Compute Metrics
- Use established implementations (torch-fidelity)
- Report confidence intervals
- Multiple evaluation runs

### Step 4: Interpret Results
- No single metric is perfect
- Consider metric limitations
- Human evaluation for final validation

## Common Pitfalls

### 1. Insufficient Sample Size
- FID requires ≥10,000 samples for stability
- Use bootstrap for confidence intervals

### 2. Inconsistent Preprocessing
- Same image resolution and normalization
- Consistent feature extractor versions
- Matched data augmentation

### 3. Cherry-Picking Metrics
- Report multiple complementary metrics
- Acknowledge metric limitations
- Include failure case analysis

### 4. Ignoring Computational Cost
- Balance metric accuracy with computation time
- Consider online vs. offline evaluation

### 5. Overlooking Mode Collapse
- FID alone doesn't catch mode collapse
- Always include diversity metrics
- Visual inspection is crucial

## Practical Implementation Tips

### Using Pre-trained Models
```python
# Load InceptionV3 for FID/IS
from torchvision.models import inception_v3
model = inception_v3(pretrained=True, transform_input=False)
model.eval()
```

### Efficient Batch Processing
```python
# Process in batches to manage memory
batch_size = 50
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    features = extract_features(batch)
```

### Caching Activations
```python
# Pre-compute real data statistics
if not os.path.exists('real_statistics.npz'):
    mu_real, sigma_real = compute_statistics(real_images)
    np.savez('real_statistics.npz', mu=mu_real, sigma=sigma_real)
```

## Recommended Libraries
- `torch-fidelity`: Standard implementation of IS, FID, KID
- `pytorch-fid`: Lightweight FID implementation
- `lpips`: Perceptual similarity metrics
- `clean-fid`: Improved FID with consistent preprocessing

## Further Reading

### Foundational Papers
1. **Inception Score (IS)**
   - Salimans et al., "Improved Techniques for Training GANs" (2016)

2. **Fréchet Inception Distance (FID)**
   - Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017)

3. **Precision and Recall**
   - Sajjadi et al., "Assessing Generative Models via Precision and Recall" (2018)
   - Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models" (2019)

4. **Kernel Inception Distance (KID)**
   - Bińkowski et al., "Demystifying MMD GANs" (2018)

5. **Perceptual Path Length**
   - Karras et al., "A Style-Based Generator Architecture for GANs" (2019)

### Survey Papers
- Borji, "Pros and Cons of GAN Evaluation Measures" (2019)
- Theis et al., "A Note on the Evaluation of Generative Models" (2016)

## Time Estimate
- Beginner: 4-6 hours
- Intermediate: 6-8 hours
- Advanced: 8-10 hours
- **Total: 18-24 hours**

## Assessment
- Implement FID from scratch
- Compare VAE and GAN on same dataset
- Analyze metric correlations with human judgment
- Debug common evaluation mistakes

## Next Steps
After completing this module, proceed to:
- Module 53: Transfer Learning (applying evaluation to fine-tuned models)
- Module 54: Self-Supervised Learning (evaluating learned representations)
- Module 64: Model Deployment (production evaluation pipelines)

---

**Note**: This module emphasizes practical implementation alongside theoretical understanding. All code includes extensive comments explaining the mathematics and intuitions behind each evaluation metric.
