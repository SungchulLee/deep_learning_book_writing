# Usage Guide: Module 52 - Generative Model Evaluation

## Quick Start

### Installation

```bash
# Clone or download the module
cd 52_generative_model_evaluation

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

#### Beginner Level
```bash
# Start with evaluation basics
python 01_beginner/01_evaluation_basics.py

# Learn about likelihood metrics
python 01_beginner/02_likelihood_metrics.py

# Explore sample visualization
python 01_beginner/03_sample_visualization.py
```

#### Intermediate Level
```bash
# Understand Inception Score
python 02_intermediate/01_inception_score.py

# Master Fréchet Inception Distance
python 02_intermediate/02_frechet_inception_distance.py
```

#### Complete Example
```bash
# Run comprehensive evaluation
python examples/complete_evaluation_example.py
```

## Module Structure

```
52_generative_model_evaluation/
├── README.md                    # Comprehensive overview
├── USAGE_GUIDE.md              # This file
├── requirements.txt            # Python dependencies
│
├── 01_beginner/                # Foundational concepts
│   ├── 01_evaluation_basics.py         # Why evaluation matters
│   ├── 02_likelihood_metrics.py        # NLL, BPD, Perplexity
│   └── 03_sample_visualization.py      # Visual assessment
│
├── 02_intermediate/            # Perceptual metrics
│   ├── 01_inception_score.py           # IS implementation
│   ├── 02_frechet_inception_distance.py # FID implementation
│   ├── 03_kernel_inception_distance.py  # (to be created)
│   └── 04_precision_recall_metrics.py   # (to be created)
│
├── 03_advanced/                # Comprehensive evaluation
│   ├── 01_perceptual_path_length.py    # (to be created)
│   ├── 02_text_generation_metrics.py   # (to be created)
│   ├── 03_conditional_generation_evaluation.py
│   ├── 04_comprehensive_evaluation_suite.py
│   └── 05_evaluation_best_practices.py
│
├── examples/                   # Complete workflows
│   └── complete_evaluation_example.py
│
└── utils/                      # Helper functions
    └── (utility modules)
```

## Learning Path

### Week 1: Foundations (Beginner)
**Time: 4-6 hours**

1. **Day 1-2**: Evaluation Basics
   - Run `01_evaluation_basics.py`
   - Understand likelihood vs. sample quality
   - Learn about evaluation tradeoffs
   
2. **Day 3-4**: Likelihood Metrics
   - Run `02_likelihood_metrics.py`
   - Implement NLL, BPD, Perplexity
   - Compare different metrics

3. **Day 5**: Visual Assessment
   - Run `03_sample_visualization.py`
   - Create sample grids
   - Perform latent interpolation

### Week 2: Perceptual Metrics (Intermediate)
**Time: 6-8 hours**

1. **Day 1-2**: Inception Score
   - Run `01_inception_score.py`
   - Understand IS formula
   - Recognize IS limitations

2. **Day 3-5**: Fréchet Inception Distance
   - Run `02_frechet_inception_distance.py`
   - Master FID calculation
   - Interpret FID scores

### Week 3: Advanced Topics
**Time: 8-10 hours**

1. Run comprehensive evaluation example
2. Combine multiple metrics
3. Build evaluation pipelines
4. Apply to your own models

## Code Examples

### Example 1: Compute FID for Your Model

```python
import torch
from pathlib import Path

# Import FID calculator
import sys
sys.path.append('02_intermediate')
from frechet_inception_distance import FIDCalculator

# Load your real data features (2048-dim from InceptionV3)
real_features = load_inception_features('real_data/')

# Generate samples from your model
generated_images = your_model.generate(n_samples=10000)

# Extract features from generated images
generated_features = extract_inception_features(generated_images)

# Compute FID
fid_score = FIDCalculator.calculate_fid(real_features, generated_features)

print(f"FID Score: {fid_score:.2f}")
```

### Example 2: Comprehensive Evaluation

```python
def evaluate_my_model(model, real_data, n_samples=10000):
    """Complete evaluation of your generative model."""
    
    results = {}
    
    # 1. Generate samples
    generated = model.generate(n_samples)
    
    # 2. Likelihood-based (if applicable)
    if hasattr(model, 'log_prob'):
        nll = -model.log_prob(real_data).mean()
        results['nll'] = nll
        results['bpd'] = nll / (model.data_dim * np.log(2))
    
    # 3. FID
    real_feats = extract_inception_features(real_data)
    gen_feats = extract_inception_features(generated)
    results['fid'] = FIDCalculator.calculate_fid(real_feats, gen_feats)
    
    # 4. Inception Score
    gen_probs = get_inception_predictions(generated)
    results['is'], results['is_std'] = InceptionScore.calculate_inception_score(gen_probs)
    
    # 5. Visual inspection
    visualize_samples(generated[:64])
    
    return results
```

### Example 3: Monitor Training Progress

```python
def training_loop(model, dataloader, epochs=100):
    """Training with evaluation monitoring."""
    
    for epoch in range(epochs):
        # Training step
        model.train()
        for batch in dataloader:
            loss = train_step(model, batch)
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Generate samples
                samples = model.generate(1000)
                
                # Quick metrics
                fid = compute_fid(samples, real_data)
                
                print(f"Epoch {epoch}: Loss={loss:.4f}, FID={fid:.2f}")
                
                # Save if improving
                if fid < best_fid:
                    save_model(model, f'best_model_fid{fid:.2f}.pt')
```

## Tips for Different Model Types

### For VAEs
```python
# Evaluate reconstruction quality
recon_mse = compute_mse(original, reconstructed)
recon_psnr = compute_psnr(original, reconstructed)

# Evaluate generation quality
fid = compute_fid(real_data, vae.generate(10000))

# Check latent space
visualize_interpolation(vae.decoder, z1, z2)
```

### For GANs
```python
# Primary metrics
fid = compute_fid(real_data, gan.generate(10000))
is_score, is_std = compute_inception_score(gan.generate(50000))

# Diversity
precision, recall = compute_precision_recall(real_data, generated)

# Mode collapse detection
visualize_sample_grid(gan.generate(100))
```

### For Diffusion Models
```python
# Sample quality
fid = compute_fid(real_data, diffusion.sample(10000))

# Sampling efficiency
nfe = count_function_evaluations(diffusion)

# Quality vs. speed tradeoff
for steps in [10, 50, 100, 1000]:
    samples = diffusion.sample(n_steps=steps)
    fid = compute_fid(real_data, samples)
    print(f"Steps: {steps}, FID: {fid:.2f}")
```

## Common Pitfalls

### 1. Insufficient Samples
```python
# ❌ BAD: Too few samples
fid = compute_fid(real[:100], generated[:100])  # Unreliable!

# ✅ GOOD: Sufficient samples
fid = compute_fid(real[:10000], generated[:10000])  # Stable estimate
```

### 2. Inconsistent Preprocessing
```python
# ❌ BAD: Different preprocessing
real = preprocess_A(real_images)
generated = preprocess_B(generated_images)  # Different!

# ✅ GOOD: Same preprocessing
real = preprocess(real_images)
generated = preprocess(generated_images)
```

### 3. Cherry-Picking Metrics
```python
# ❌ BAD: Only report best metric
print(f"IS: {is_score:.2f}")  # Hiding bad FID?

# ✅ GOOD: Report multiple metrics
print(f"FID: {fid:.2f}, IS: {is_score:.2f}, Precision: {prec:.3f}")
```

## Troubleshooting

### Issue: FID is very high (>100)
**Possible causes:**
- Mode collapse
- Wrong preprocessing
- Insufficient training
- Model architecture issues

**Solutions:**
- Check sample diversity visually
- Verify preprocessing matches training
- Train longer or adjust hyperparameters

### Issue: IS is low (<2.0)
**Possible causes:**
- Blurry images (low quality)
- Limited diversity
- Wrong evaluation setup

**Solutions:**
- Check if images are sharp and clear
- Verify model covers multiple classes
- Ensure using InceptionV3 correctly

### Issue: Metrics don't match visual quality
**Possible:**
- Metrics have limitations
- Visual assessment is subjective
- Need multiple metrics

**Solutions:**
- Use complementary metrics (FID + IS + Precision/Recall)
- Include human evaluation
- Check for known metric failure cases

## Further Resources

### Papers
- FID: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule" (2017)
- IS: Salimans et al. "Improved Techniques for Training GANs" (2016)
- Precision/Recall: Kynkäänniemi et al. "Improved Precision and Recall" (2019)

### Libraries
- `torch-fidelity`: Standard FID/IS implementation
- `pytorch-fid`: Lightweight FID
- `clean-fid`: Improved FID with consistent preprocessing

### Courses
- CS236: Deep Generative Models (Stanford)
- Generative Models (Fast.ai)

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the code comments (heavily documented)
3. Run the examples to understand expected behavior
4. Verify your setup matches the requirements

## Contributing

This is an educational module. Suggested improvements:
- Additional metrics implementations
- More comprehensive examples
- Better visualizations
- Performance optimizations

---

**Happy Evaluating!** 🎉

Remember: No single metric is perfect. Always use multiple complementary metrics and include visual inspection!
