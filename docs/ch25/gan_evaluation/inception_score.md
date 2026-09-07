# Inception Score (IS)
## Overview

The Inception Score (IS) is one of the most widely used metrics for evaluating generative models, particularly Generative Adversarial Networks (GANs). Introduced by Salimans et al. (2016), IS provides a single scalar value that captures both the quality and diversity of generated images.

!!! info "Learning Objectives"
    By the end of this section, you will be able to:
    
    - Understand the mathematical foundation of Inception Score
    - Implement IS computation from scratch in PyTorch
    - Interpret IS values correctly and understand their limitations
    - Apply IS in practical evaluation workflows

## Mathematical Foundation

### Core Formula

The Inception Score is defined as:

$$

\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g}\left[D_{KL}(p(y|x) \| p(y))\right]\right)

$$

where:

- $x$ is a generated image sampled from the generator distribution $p_g$
- $p(y|x)$ is the conditional class distribution given image $x$ (from InceptionV3)
- $p(y) = \mathbb{E}_{x}[p(y|x)]$ is the marginal class distribution
- $D_{KL}$ is the Kullback-Leibler divergence

### Intuition Behind the Components

**Conditional Distribution $p(y|x)$:**

This represents how confident the Inception classifier is about the image's class. A sharp, peaked distribution indicates the classifier is confident—suggesting the image contains a clear, recognizable object.

$$

p(y|x) = \text{softmax}(f_{\text{Inception}}(x))

$$

where $f_{\text{Inception}}(x)$ returns logits for 1000 ImageNet classes.

**Marginal Distribution $p(y)$:**

This is the average class distribution across all generated images:

$$

p(y) = \frac{1}{N}\sum_{i=1}^{N} p(y|x_i)

$$

A uniform marginal distribution indicates the generator produces diverse images covering many classes.

**KL Divergence:**

The KL divergence measures how much the conditional distribution differs from the marginal:

$$

D_{KL}(p(y|x) \| p(y)) = \sum_{c=1}^{C} p(y=c|x) \log\frac{p(y=c|x)}{p(y=c)}

$$

### What IS Actually Measures

| Component | High Value Indicates | Low Value Indicates |
|-----------|---------------------|---------------------|
| $p(y\|x)$ entropy | Uncertain predictions | Confident predictions (quality) |
| $p(y)$ entropy | Diverse classes (diversity) | Mode collapse |
| KL divergence | Both quality AND diversity | Poor quality OR low diversity |

The IS captures both aspects simultaneously:

- **Quality**: Each image should produce a confident classification (low entropy in $p(y|x)$)
- **Diversity**: Generated images should cover many classes (high entropy in $p(y)$)

## Mathematical Derivation

### Expanding the KL Divergence

Starting from the definition:

$$

\begin{aligned}
D_{KL}(p(y|x) \| p(y)) &= \sum_{y} p(y|x) \log\frac{p(y|x)}{p(y)} \\
&= \sum_{y} p(y|x) \log p(y|x) - \sum_{y} p(y|x) \log p(y) \\
&= -H(y|x) + H_{\text{cross}}(p(y|x), p(y))
\end{aligned}

$$

where $H(y|x)$ is the conditional entropy.

### Expected Value

Taking the expectation over generated samples:

$$

\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))] = -\mathbb{E}_x[H(y|x)] + H(y)

$$

The first term represents **average conditional entropy** (lower is better for quality), and the second term is the **marginal entropy** (higher is better for diversity).

### Final Score

$$

\text{IS} = \exp\left(H(y) - \mathbb{E}_x[H(y|x)]\right)

$$

This can be interpreted as the **effective number of classes** the generator can produce with confident predictions.

## PyTorch Implementation

### Complete Implementation from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import stats


class InceptionScoreCalculator:
    """
    자세한 풀이를 곁들인 두루 갖춘 인셉션 점수 셈틀.
    
    인셉션 점수는 미리 익힌 인셉션 그물의 갈래 헤아림을 살펴
    만들어 낸 그림의 품질과 다양함을 함께 잰다.
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        인셉션 점수 셈틀의 첫자리를 잡는다.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device
        self.inception_model = None
        
    def _load_inception(self):
        """ImageNet으로 미리 익힌 InceptionV3 모델을 불러온다."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        # 미리 익힌 InceptionV3을 불러온다
        self.inception_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False  # We'll handle preprocessing ourselves
        )
        self.inception_model.eval()
        self.inception_model.to(self.device)
        
        # 곁들이 내놓음을 끈다
        self.inception_model.aux_logits = False
        
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        InceptionV3에 맞게 그림을 미리 다듬는다.
        
        InceptionV3은 다음을 바란다.
        - 크기가 299×299인 그림
        - ImageNet 평균과 표준편차로 잣대를 맞춘 그림
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            인셉션에 바로 넣을 수 있게 다듬은 그림
        """
        # 필요하면 299×299로 크기를 바꾼다
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images, 
                size=(299, 299), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 필요하면 잿빛을 RGB로 바꾼다
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # ImageNet 통계로 잣대를 맞춘다
        # Note: Inception expects [-1, 1] range internally
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        
        images = (images - mean) / std
        
        return images
    
    def get_predictions(self, 
                       images: torch.Tensor, 
                       batch_size: int = 32) -> np.ndarray:
        """
        그림 묶음에 대한 인셉션 헤아림을 얻는다.
        
        Args:
            images: Generated images [N, C, H, W] in range [0, 1]
            batch_size: 다룰 때 쓰는 묶음 크기
            
        Returns:
            Softmax probabilities [N, 1000]
        """
        if self.inception_model is None:
            self._load_inception()
            
        all_probs = []
        n_images = len(images)
        
        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch = self._preprocess_images(batch)
                
                # 인셉션을 지나 앞으로 걸음
                logits = self.inception_model(batch)
                
                # 소프트맥스를 걸어 확률을 얻는다
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def calculate_inception_score(self,
                                  images: torch.Tensor,
                                  splits: int = 10,
                                  batch_size: int = 32) -> Tuple[float, float]:
        """
        믿음 구간과 함께 인셉션 점수를 셈한다.
        
        Algorithm:
        1. Get p(y|x) from Inception for each image
        2. 흩어짐을 셈하려고 자료를 `splits`개 무리로 나눈다
        3. 조각마다:
           a. Compute marginal p(y) = mean(p(y|x))
           b. Compute KL(p(y|x) || p(y)) for each sample
           c. KL을 평균 내고 지수를 취한다
        4. 조각들의 평균과 표준편차를 돌려준다
        
        Args:
            images: Generated images [N, C, H, W] in range [0, 1]
            splits: 표준편차를 셈할 때 나눌 조각의 수
            batch_size: 인셉션 미룸에 쓰는 묶음 크기
            
        Returns:
            Tuple of (IS mean, IS std)
        """
        # 헤아림을 얻는다
        probs = self.get_predictions(images, batch_size)
        
        # 조각을 나누어 인셉션 점수를 셈한다
        scores = []
        n = len(probs)
        split_size = n // splits
        
        for k in range(splits):
            # 조각을 가져온다
            start = k * split_size
            end = start + split_size if k < splits - 1 else n
            part = probs[start:end]
            
            # Compute marginal: p(y) = (1/N) Σ p(y|x_i)
            p_y = np.mean(part, axis=0, keepdims=True)
            
            # 표본마다 KL 갈림을 셈한다
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            eps = 1e-16
            part = np.clip(part, eps, 1.0)
            p_y = np.clip(p_y, eps, 1.0)
            
            # 로그 비
            log_ratio = np.log(part) - np.log(p_y)
            
            # 표본마다의 KL 갈림
            kl_per_sample = np.sum(part * log_ratio, axis=1)
            
            # KL을 평균 내고 지수를 취한다
            mean_kl = np.mean(kl_per_sample)
            is_score = np.exp(mean_kl)
            
            scores.append(is_score)
        
        return float(np.mean(scores)), float(np.std(scores))


def compute_inception_score_step_by_step(probs: np.ndarray) -> dict:
    """
    배움을 돕고자 중간 결과까지 자세히 내며 인셉션 점수를 셈한다.
    
    이 함수는 인셉션 점수 셈을 읽기 쉬운 걸음으로 나누어
    각 조각이 무엇을 재는지 알기 쉽게 한다.
    
    Args:
        probs: Class probabilities [N, C] from Inception
        
    Returns:
        중간 값과 마지막 인셉션 점수를 담은 사전
    """
    eps = 1e-16
    probs = np.clip(probs, eps, 1.0)
    
    # Step 1: Compute marginal distribution p(y)
    # 이는 모든 표본에 걸친 갈래 분포를 나타낸다
    p_y = np.mean(probs, axis=0)
    
    # Step 2: Compute entropy of marginal H(y)
    # Higher entropy means more diverse samples (covering more classes)
    h_marginal = -np.sum(p_y * np.log(p_y))
    
    # Step 3: Compute conditional entropy H(y|x) for each sample
    # Lower conditional entropy means more confident predictions (higher quality)
    h_conditional_per_sample = -np.sum(probs * np.log(probs), axis=1)
    h_conditional = np.mean(h_conditional_per_sample)
    
    # 걸음 4: KL 갈림을 셈한다
    # KL(p(y|x) || p(y)) = H(y) - H(y|x) in expectation
    # 다만 정확함을 위해 곧바로 셈한다
    kl_per_sample = np.sum(probs * (np.log(probs) - np.log(p_y)), axis=1)
    mean_kl = np.mean(kl_per_sample)
    
    # Step 5: Final IS = exp(mean_kl)
    inception_score = np.exp(mean_kl)
    
    # 덧붙이는 눈썰미
    effective_classes = np.exp(h_marginal)  # Effective number of classes used
    avg_confidence = np.exp(-h_conditional)  # Average prediction confidence
    
    return {
        'inception_score': inception_score,
        'mean_kl_divergence': mean_kl,
        'marginal_entropy': h_marginal,
        'conditional_entropy': h_conditional,
        'effective_classes': effective_classes,
        'avg_confidence': avg_confidence,
        'marginal_distribution': p_y
    }
```

### Practical Usage Example

```python
import torch
import matplotlib.pyplot as plt


def demonstrate_inception_score():
    """
    품질이 다른 여러 상황에서 인셉션 점수 셈을 보인다.
    """
    n_samples = 1000
    n_classes = 10  # Simplified for demonstration
    
    print("=" * 70)
    print("Inception Score Demonstration")
    print("=" * 70)
    
    # Scenario 1: High quality + High diversity (Ideal)
    print("\n📊 Scenario 1: High Quality + High Diversity")
    print("-" * 50)
    
    probs_ideal = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # Uniform coverage
        probs_ideal[i, class_idx] = 0.9
        probs_ideal[i, :] += 0.01  # Small uniform noise
    probs_ideal = probs_ideal / probs_ideal.sum(axis=1, keepdims=True)
    
    results_ideal = compute_inception_score_step_by_step(probs_ideal)
    print(f"  IS: {results_ideal['inception_score']:.2f}")
    print(f"  Effective classes: {results_ideal['effective_classes']:.2f}")
    print(f"  Average confidence: {results_ideal['avg_confidence']:.4f}")
    
    # Scenario 2: Low quality (uncertain predictions)
    print("\n📊 Scenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 50)
    
    probs_uncertain = np.ones((n_samples, n_classes)) / n_classes
    
    results_uncertain = compute_inception_score_step_by_step(probs_uncertain)
    print(f"  IS: {results_uncertain['inception_score']:.2f}")
    print(f"  Effective classes: {results_uncertain['effective_classes']:.2f}")
    print(f"  Average confidence: {results_uncertain['avg_confidence']:.4f}")
    print("  Note: Minimum IS = 1.0 when all predictions are uniform")
    
    # Scenario 3: Mode collapse (only one class)
    print("\n📊 Scenario 3: Mode Collapse (Single Class)")
    print("-" * 50)
    
    probs_collapse = np.zeros((n_samples, n_classes))
    probs_collapse[:, 0] = 0.95
    probs_collapse[:, 1:] = 0.05 / (n_classes - 1)
    
    results_collapse = compute_inception_score_step_by_step(probs_collapse)
    print(f"  IS: {results_collapse['inception_score']:.2f}")
    print(f"  Effective classes: {results_collapse['effective_classes']:.2f}")
    print(f"  Note: Confident but not diverse!")
    
    return {
        'ideal': results_ideal,
        'uncertain': results_uncertain,
        'collapse': results_collapse
    }


# 보임을 돌린다
results = demonstrate_inception_score()
```

## Interpreting IS Values

### Typical Ranges

| IS Value | Quality Level | Interpretation |
|----------|---------------|----------------|
| < 2.0 | Very Poor | Images unrecognizable or highly uncertain predictions |
| 2.0 - 5.0 | Poor to Moderate | Some structure but limited quality or diversity |
| 5.0 - 8.0 | Good | Clear images with reasonable diversity |
| > 8.0 | Excellent | High-quality, diverse image generation |
| ~11.2 | Real ImageNet | Benchmark from real ImageNet images |

### Theoretical Bounds

**Minimum IS = 1.0**: Achieved when $p(y|x) = p(y)$ for all $x$ (uniform predictions).

**Maximum IS**: Theoretically bounded by the number of classes (1000 for ImageNet), achieved when each image is perfectly classified into a unique class.

## Limitations and Pitfalls

### 1. Cannot Detect Memorization

IS cannot distinguish between a model that generates novel images and one that simply memorizes training data:

```python
def demonstrate_memorization_blindness():
    """
    모델이 익힘 자료를 외워도 인셉션 점수는 알아채지 못함을 보인다.
    """
    # 같은 그림 10장을 나무랄 데 없이 만들어 내는 모델도
    # 그 그림들이 자신 있게 갈리면 인셉션 점수가 높게 나온다
    n_unique = 10
    n_total = 1000
    
    probs_memorized = np.zeros((n_total, 10))
    for i in range(n_total):
        class_idx = i % n_unique  # Only 10 unique "images"
        probs_memorized[i, class_idx] = 0.95
        probs_memorized[i, :] += 0.005
    
    probs_memorized = probs_memorized / probs_memorized.sum(axis=1, keepdims=True)
    results = compute_inception_score_step_by_step(probs_memorized)
    
    print(f"IS with memorization: {results['inception_score']:.2f}")
    print("This is HIGH despite only 10 unique images!")
```

### 2. Ignores Within-Class Diversity

IS only measures class-level diversity, not visual diversity within classes:

- 1000 identical cat images → High IS (confident "cat" classification)
- But zero visual diversity!

### 3. Dataset Dependency

IS is only meaningful for ImageNet-like natural images. It may fail for:

- Medical images
- Satellite imagery
- Abstract art
- Domain-specific images

### 4. Can Be Gamed

Adversarial strategies can artificially inflate IS:

```python
def demonstrate_gaming_is():
    """
    맞겨루기 꾀로 인셉션 점수를 어떻게 '주무를' 수 있는지 보인다.
    """
    # 꾀: 갈래마다 그림을 꼭 하나씩만 만든다
    n_classes = 1000
    probs_gamed = np.eye(n_classes)  # Perfect classification for each class
    
    results = compute_inception_score_step_by_step(probs_gamed)
    print(f"Gamed IS: {results['inception_score']:.2f}")
    print("Maximum possible IS with only 1000 unique images!")
```

## Best Practices

### 1. Sample Size

```python
def analyze_sample_size_effect(generator, sample_sizes=[100, 500, 1000, 5000, 10000]):
    """
    표본 수가 인셉션 점수의 든든함에 어떤 영향을 주는지 살핀다.
    """
    calculator = InceptionScoreCalculator()
    
    results = []
    for n in sample_sizes:
        images = generator.generate(n)
        is_mean, is_std = calculator.calculate_inception_score(images)
        results.append({
            'n_samples': n,
            'is_mean': is_mean,
            'is_std': is_std,
            'relative_std': is_std / is_mean
        })
    
    return results
```

**Recommendations:**

- Minimum: 5,000 samples
- Recommended: 10,000+ samples
- Always report confidence intervals

### 2. Splits for Variance Estimation

```python
# 흔한 방식: 조각 10개
is_mean, is_std = calculator.calculate_inception_score(images, splits=10)

# Report as: IS = mean ± std
print(f"IS = {is_mean:.2f} ± {is_std:.2f}")
```

### 3. Combine with Other Metrics

IS should never be used alone. Always combine with:

- **FID**: Detects mode collapse better
- **Precision/Recall**: Measures quality vs coverage tradeoff
- **Visual inspection**: Human judgment remains essential

## Connection to Information Theory

The IS has a beautiful information-theoretic interpretation:

$$

\text{IS} = \exp\left(I(X; Y)\right)

$$

where $I(X; Y)$ is the mutual information between generated images $X$ and their predicted classes $Y$.

**Mutual information decomposes as:**

$$

I(X; Y) = H(Y) - H(Y|X)

$$

- **$H(Y)$**: Entropy of class predictions (diversity)
- **$H(Y|X)$**: Average uncertainty in predictions (quality)

Higher mutual information means:

- The generated images carry more information about class labels
- Both quality and diversity contribute positively

## Summary

!!! success "Key Takeaways"
    
    1. **IS Formula**: $\text{IS} = \exp(\mathbb{E}[D_{KL}(p(y|x) \| p(y))])$
    
    2. **Measures Both**: Quality (confident predictions) and diversity (class coverage)
    
    3. **Range**: 1.0 (minimum) to ~1000 (theoretical max), real ImageNet ≈ 11.2
    
    4. **Limitations**: Cannot detect memorization, ignores within-class diversity, ImageNet-specific
    
    5. **Best Practice**: Use 10,000+ samples, 10 splits, combine with FID and visual inspection

## References

1. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.

2. Barratt, S., & Sharma, R. (2018). "A Note on the Inception Score." *ICML Workshop*.

3. Borji, A. (2019). "Pros and Cons of GAN Evaluation Measures." *Computer Vision and Image Understanding*.

## 익힘 문제

**익힘 1.**
Define the Frechet Inception Distance (FID) and explain why it is preferred over the Inception Score for evaluating GANs.

??? success "익힘 1 풀이"
    FID models the Inception-v3 feature distributions of real and generated images as multivariate Gaussians $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$, then computes:

    $$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

    FID is preferred because: (1) it compares generated images to real images (IS only evaluates generated images), (2) it detects mode collapse (different means/covariances), (3) it is more consistent with human judgment, and (4) lower FID correlates better with perceptual quality.

---

**익힘 2.**
What are the limitations of the Inception Score? Can a model achieve a high IS while producing poor samples?

??? success "익힘 2 풀이"
    IS = $\exp(\mathbb{E}_x [D_{\text{KL}}(p(y|x) \| p(y))])$ where $p(y|x)$ is the Inception classifier's prediction for generated image $x$. Limitations: (1) only measures quality (sharp, classifiable) and diversity (spread across classes), not fidelity to the training data, (2) a model generating one perfect image per class scores high but ignores intra-class diversity, (3) sensitive to the Inception model's biases, (4) does not capture texture/style quality within classes. Yes, a model can achieve high IS by memorizing one representative image per ImageNet class.

---

**익힘 3.**
Explain how precision and recall metrics for generative models differ from their classification counterparts.

??? success "익힘 3 풀이"
    In the generative setting (Kynkaanniemi et al., 2019): **Precision** measures the fraction of generated samples that fall within the support of the real data distribution (quality/fidelity). **Recall** measures the fraction of real data that falls within the support of the generated distribution (diversity/coverage). High precision + low recall = mode collapse (few modes, but realistic). Low precision + high recall = poor quality but diverse. Unlike classification P/R which count discrete matches, generative P/R uses $k$-nearest-neighbor distances in feature space to estimate distribution support.

---

**익힘 4.**
Why should multiple evaluation metrics be used together when assessing generative models?

??? success "익힘 4 풀이"
    No single metric captures all aspects of generation quality. **FID** measures overall distributional similarity but conflates quality and diversity. **IS** captures quality and diversity but ignores fidelity to training data. **Precision/Recall** separates quality from diversity but depends on the choice of feature extractor and $k$. **Perceptual metrics** (LPIPS) measure image-level quality but not diversity. Using metrics together provides a complete picture: a model with low FID, high precision, and low recall has mode collapse; one with high recall but low precision generates diverse but low-quality samples. Human evaluation remains the gold standard for final assessment.
