# Module 52: Inception Score (IS)

This module implements module 52: inception score (is), an important component in deep generative modeling. Understanding this implementation provides insight into the architectural patterns and training procedures used in modern generative models. The code demonstrates practical techniques that are widely adopted in research and production systems.

## 코드

```python
"""
Module 52: Inception Score (IS)
================================

만들어 내는 모델, 특히 GAN을 따지는 데 널리 쓰는 자인
인셉션 점수를 짜고 풀이한다.

배움 목표:
-------------------
1. 인셉션 점수의 수학 바탕을 이해한다
2. 인셉션 점수를 맨바닥부터 짠다
3. 인셉션 점수를 올바로 읽는다
4. 인셉션 점수의 한계를 알아본다

고갱이 식:
-----------
IS = exp(E_x[KL(p(y|x) || p(y))])

where:
- p(y|x): Conditional label distribution (sharpness/quality)
- p(y): Marginal label distribution (diversity)

지은이: 배움용 AI 모둠
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import warnings

# ========================================================================
# 메인
# ========================================================================

torch.manual_seed(42)
np.random.seed(42)


class InceptionScore:
    """
    인셉션 점수 셈틀.
    
    수학 바탕:
    -----------------------
    IS = exp(E_x[KL(p(y|x) || p(y))])
    
    조각을 하나씩 뜯어보면:
    
    1. p(y|x): Conditional class distribution
       - 그림 x에 InceptionV3을 돌린다
       - ImageNet 갈래 1000개에 대한 소프트맥스 확률을 얻는다
       - Sharp distribution (confident predictions) = high quality
    
    2. p(y): Marginal class distribution  
       - Average of p(y|x) over all generated images
       - Uniform distribution = high diversity
       - Peaked distribution = mode collapse
    
    3. KL(p(y|x) || p(y)): KL divergence
       - Measures how much p(y|x) differs from p(y)
       - High KL = images have confident, diverse predictions
       - Low KL = either low quality or low diversity
    
    4. 지수 취하기: 로그 자에서 되돌린다
       - exp(E[KL(...)]) gives final IS
       - 흔한 범위: ImageNet 같은 그림에서 1.0에서 10.0쯤
    
    Intuition:
    ---------
    좋은 만들어 내는 모델은 다음과 같은 그림을 내놓아야 한다.
    - Are clearly recognizable (high p(y|x) entropy → confident)
    - Cover many classes (uniform p(y) → diverse)
    
    인셉션 점수는 KL 갈림으로 두 가지를 함께 잡는다.
    
    Limitations:
    -----------
    1. Only works for ImageNet-like images (uses Inception classifier)
    2. Cannot detect overfitting (memorization)
    3. 갈래 안의 다양함을 놓친다
    4. 갈래마다 그림 하나씩만 만들어도 속일 수 있다
    5. 어떤 인셉션 모델을 쓰느냐에 흔들린다
    """
    
    @staticmethod
    def calculate_inception_score(probs: np.ndarray,
                                  splits: int = 10) -> Tuple[float, float]:
        """
        갈래 확률에서 인셉션 점수를 셈한다.
        
        Args:
            probs: Class probabilities [n_samples, n_classes]
                   InceptionV3 소프트맥스 층의 내놓음
            splits: 표준편차를 셈할 때 나눌 조각의 수
        
        Returns:
            Tuple of (mean IS, std IS)
        
        수학 걸음:
        ------------------
        1. Compute p(y) = (1/N) Σ p(y|x_i) for each x_i
        2. 조각마다:
           a. Compute KL(p(y|x_i) || p(y)) for each sample
           b. Average: E_x[KL(...)]
           c. Exponentiate: exp(E_x[KL(...)])
        3. 조각들의 평균과 표준편차를 돌려준다
        """
        # 확률이 올바른지 다진다
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), \
            "Probabilities must sum to 1"
        
        n_samples = len(probs)
        
        # 표준편차를 셈하려고 묶음으로 나눈다
        split_size = n_samples // splits
        scores = []
        
        for i in range(splits):
            # 조각을 가져온다
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
            part = probs[start_idx:end_idx]
            
            # 1. Compute marginal distribution p(y)
            # 조건 분포의 평균
            # Shape: [n_classes]
            p_y = np.mean(part, axis=0)
            
            # 2. 표본마다 KL 갈림을 셈한다
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            
            # 수치가 든든하도록 작은 엡실론을 더한다
            eps = 1e-16
            p_y = p_y + eps
            part = part + eps
            
            # KL 갈림을 셈한다
            # Shape: [split_size, n_classes]
            kl_div = part * (np.log(part) - np.log(p_y))
            
            # 갈래에 대해 더하고 표본에 대해 평균 낸다
            # E_x[KL(p(y|x) || p(y))]
            kl_mean = np.mean(np.sum(kl_div, axis=1))
            
            # 3. 지수를 취해 인셉션 점수를 얻는다
            is_score = np.exp(kl_mean)
            scores.append(is_score)
        
        # 조각들의 평균과 표준편차를 돌려준다
        return float(np.mean(scores)), float(np.std(scores))
    
    @staticmethod
    def interpret_is(is_score: float) -> str:
        """
        인셉션 점숫값을 풀이한다.
        
        Args:
            is_score: 인셉션 점숫값
        
        Returns:
            풀이 글월
        
        Typical Ranges (ImageNet):
        -------------------------
        - 인셉션 점수 < 2.0: 매우 나쁨
        - 인셉션 점수 2.0~5.0: 나쁨에서 보통
        - IS 5.0-8.0: Good
        - IS > 8.0: Excellent
        
        눈여겨볼 것: ImageNet의 참 그림은 인셉션 점수가 11.2쯤이다
        """
        if is_score < 2.0:
            return "Very Poor"
        elif is_score < 5.0:
            return "Poor to Moderate"
        elif is_score < 8.0:
            return "Good"
        else:
            return "Excellent"


def demonstrate_inception_score_intuition():
    """
    간단한 보기로 인셉션 점수의 느낌을 보인다.
    """
    print("=" * 70)
    print("Inception Score Intuition")
    print("=" * 70)
    
    n_samples = 1000
    n_classes = 10  # Simplified (real IS uses 1000 classes)
    
    # Scenario 1: High quality, high diversity (IDEAL)
    print("\nScenario 1: High Quality + High Diversity (IDEAL)")
    print("-" * 70)
    
    # Each image confidently belongs to one class (sharp p(y|x))
    # All classes equally represented (uniform p(y))
    probs1 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # Cycle through all classes
        probs1[i, class_idx] = 0.9  # Confident prediction
        probs1[i, :] += 0.1 / n_classes  # Small uniform noise
    
    # Normalize
    probs1 = probs1 / probs1.sum(axis=1, keepdims=True)
    
    is_score1, is_std1 = InceptionScore.calculate_inception_score(probs1)
    print(f"IS: {is_score1:.4f} ± {is_std1:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score1)}")
    print("Explanation: Confident predictions + diverse classes = High IS")
    
    # Scenario 2: Low quality (uncertain predictions)
    print("\nScenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 70)
    
    # Each image has uniform distribution (not confident)
    probs2 = np.ones((n_samples, n_classes)) / n_classes
    
    is_score2, is_std2 = InceptionScore.calculate_inception_score(probs2)
    print(f"IS: {is_score2:.4f} ± {is_std2:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score2)}")
    print("Explanation: p(y|x) = p(y) → KL = 0 → IS = exp(0) = 1.0")
    print("Minimum possible IS = 1.0")
    
    # Scenario 3: Mode collapse (only one class)
    print("\nScenario 3: Mode Collapse (Single Class)")
    print("-" * 70)
    
    # 모든 그림이 갈래 0으로 갈린다
    probs3 = np.zeros((n_samples, n_classes))
    probs3[:, 0] = 0.95
    probs3[:, 1:] = 0.05 / (n_classes - 1)
    
    is_score3, is_std3 = InceptionScore.calculate_inception_score(probs3)
    print(f"IS: {is_score3:.4f} ± {is_std3:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score3)}")
    print("Explanation: Confident predictions but no diversity")
    print("p(y) peaked at one class → Low KL → Low IS")
    
    # 상황 4: 품질은 좋으나 다양함이 적다
    print("\nScenario 4: Good Quality but Limited Diversity")
    print("-" * 70)
    
    # 갈래 10개 가운데 3개만 나타난다
    probs4 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % 3  # Only classes 0, 1, 2
        probs4[i, class_idx] = 0.9
        probs4[i, 3:] = 0.1 / (n_classes - 3)
    
    probs4 = probs4 / probs4.sum(axis=1, keepdims=True)
    
    is_score4, is_std4 = InceptionScore.calculate_inception_score(probs4)
    print(f"IS: {is_score4:.4f} ± {is_std4:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score4)}")
    print("Explanation: Confident but not covering all classes")
    
    # Summary
    print("\n" + "=" * 70)
    print("IS Comparison:")
    print("=" * 70)
    print(f"High quality + high diversity:  IS = {is_score1:.2f}")
    print(f"Low quality (uncertain):        IS = {is_score2:.2f} (minimum)")
    print(f"Mode collapse (one class):      IS = {is_score3:.2f}")
    print(f"Good quality + limited diversity: IS = {is_score4:.2f}")
    print("\nKey Insight: IS balances quality (sharpness) and diversity")


def demonstrate_is_limitations():
    """
    인셉션 점수의 한계를 보인다.
    """
    print("\n" + "=" * 70)
    print("Inception Score Limitations")
    print("=" * 70)
    
    n_samples = 1000
    n_classes = 10
    
    # 한계 1: 외워 버린 것을 알아채지 못한다
    print("\nLimitation 1: Cannot Detect Memorization/Overfitting")
    print("-" * 70)
    print("A model that perfectly memorizes training images can achieve")
    print("high IS, even though it's not truly generating novel samples.")
    print("\nExample: Generating 100 real images repeatedly")
    print("         → High IS (confident + diverse)")
    print("         → But not creative/generative!")
    
    # 한계 2: 갈래 안의 다양함을 놓친다
    print("\nLimitation 2: Ignores Within-Class Diversity")
    print("-" * 70)
    print("IS only cares about class labels, not visual diversity.")
    print("\nExample: Generating 1000 identical cat images")
    print("         → Still get high IS if classified as 'cat'")
    print("         → But zero visual diversity!")
    
    # 보기: 모든 그림이 같은 갈래로 갈리는데도 인셉션 점수가 높다
    probs_same = np.zeros((n_samples, n_classes))
    probs_same[:, 0] = 0.95
    probs_same[:, 1:] = 0.05 / (n_classes - 1)
    is_same, _ = InceptionScore.calculate_inception_score(probs_same)
    
    print(f"\n1000 identical images → IS = {is_same:.2f}")
    print("This should be low but IS doesn't capture it!")
    
    # 한계 3: 속일 수 있다
    print("\nLimitation 3: Can Be Fooled by Adversarial Generation")
    print("-" * 70)
    print("Strategy: Generate exactly one image per class")
    print("          → Maximum diversity (uniform p(y))")
    print("          → Confident predictions (sharp p(y|x))")
    print("          → High IS!")
    print("\nBut only 10 unique images for 1000 ImageNet classes is terrible!")
    
    # 한계 4: 인셉션에만 매여 있다
    print("\nLimitation 4: Tied to InceptionV3 Classifier")
    print("-" * 70)
    print("IS depends on InceptionV3's learned representations.")
    print("• Only works well for ImageNet-like natural images")
    print("• May not work for: medical images, satellite imagery,")
    print("  abstract art, non-photorealistic images")
    print("• Different classifiers give different IS values")
    
    print("\n" + "=" * 70)
    print("Recommendation: Use IS alongside other metrics!")
    print("=" * 70)
    print("• Combine with FID (detects mode collapse better)")
    print("• Add precision/recall (measures coverage)")
    print("• Include visual inspection")
    print("• Consider task-specific metrics")


def main():
    """
    으뜸 보임 함수.
    """
    print("\n" + "=" * 70)
    print("MODULE 52: INCEPTION SCORE (IS)")
    print("=" * 70)
    
    # 인셉션 점수의 느낌을 보인다
    demonstrate_inception_score_intuition()
    
    # 한계를 보인다
    demonstrate_is_limitations()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. IS Formula:
       IS = exp(E_x[KL(p(y|x) || p(y))])
       - p(y|x): Conditional distribution (quality/sharpness)
       - p(y): Marginal distribution (diversity)
       - KL 갈림이 둘의 균형을 잡는다
    
    2. 인셉션 점수가 재는 것:
       - 품질: 헤아림이 얼마나 자신 있는가?
       - 다양함: 갈래를 얼마나 두루 덮는가?
       - High IS = Confident predictions + Diverse samples
    
    3. 흔한 값:
       - Minimum: IS = 1.0 (uniform predictions)
       - Good: IS > 5.0
       - Excellent: IS > 8.0
       - 참 ImageNet: 인셉션 점수 ≈ 11.2
    
    4. Strengths:
       - 수 하나로 나타내는 자
       - 셈이 빠르다
       - 품질과 다양함의 절충을 잡아낸다
       - 널리 쓰이고 잘 알려져 있다
    
    5. Limitations:
       - 외워 버림이나 지나치게 맞춰짐을 알아채지 못한다
       - 갈래 안의 다양함을 놓친다
       - ImageNet 같은 그림에서만 쓸모 있다
       - InceptionV3 가름개에 매여 있다
       - 맞겨루기 꾀에 속을 수 있다
    
    6. 모범 사례:
       - Use splits=10 for computing std dev
       - 인셉션 점수 ± 표준편차를 함께 알린다
       - Always combine with other metrics (FID, precision/recall)
       - 눈으로 살펴보는 일을 곁들인다
       - 일에 맞춘 따지기도 함께 생각한다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 논의

The implementation follows established best practices for module 52: inception score (is). The code is organized with clear separation between model definition, training logic, and utility functions. Key design decisions include the choice of activation functions, normalization strategies, and optimization hyperparameters, all of which significantly impact training stability and output quality.

The architecture demonstrates several important patterns common to deep generative models. These include progressive processing of features through multiple network layers, conditioning mechanisms that allow the model to incorporate auxiliary information, and careful initialization to ensure stable gradient flow during training.

Practitioners should pay attention to the hyperparameter choices and training procedures, as these often require careful tuning for new datasets or problem domains. The modular design of the code facilitates experimentation with alternative architectures, loss functions, and training strategies.

## 익힘 문제

**익힘 1.**
Trace through the forward pass of the main model in this module with a concrete input tensor. Document the shape transformations at each layer and verify the output dimensions match expectations.

??? success "익힘 1 풀이"
    Starting from the input tensor, follow each layer's transformation. For convolutional layers, apply the formula $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$ for spatial dimensions. For linear layers, track the feature dimension changes. Document each intermediate shape and verify the final output matches the expected target dimensions for the specific task (image generation, classification, etc.).

---

**익힘 2.**
Identify the key hyperparameters in this implementation (learning rate, batch size, architecture choices). Design an experiment to measure the sensitivity of training to each hyperparameter by varying one at a time while holding others fixed.

??? success "익힘 2 풀이"
    The key hyperparameters include learning rate (typically $10^{-4}$ to $10^{-3}$), batch size (64-256), number of layers/channels, and activation functions. For each hyperparameter, train the model with 3-5 different values and track a relevant metric (loss, sample quality, convergence speed). Plot the results to identify which hyperparameters have the largest impact. Learning rate and architecture depth typically show the strongest effects, while batch size has moderate impact within reasonable ranges.

---

**익힘 3.**
Extend this implementation with a new feature: add gradient clipping, learning rate scheduling, or an alternative loss function. Compare the training dynamics before and after your modification.

??? success "익힘 3 풀이"
    For gradient clipping, add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`. For learning rate scheduling, use `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)` and call `scheduler.step()` each epoch. Compare training loss curves, convergence speed, and final model quality. Gradient clipping typically prevents training spikes, while cosine annealing can improve final performance by allowing finer optimization in later epochs.
