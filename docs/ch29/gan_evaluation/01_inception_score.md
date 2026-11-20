# 인셉션 점수 계산

인셉션 점수(Inception Score, IS)는 생성 이미지에 대한 분류기의 예측 확률을 요약한다. 개별 예측이 확신에 차 있고 전체 예측이 여러 클래스에 분산될수록 커진다. 이 페이지에서는 이미지나 분류기를 사용하지 않고, 직접 만든 확률 행렬로 점수의 계산과 한계를 살펴본다.

## 1. 확률 행렬로 계산하기

입력은 N개 표본과 C개 클래스에 대한 N×C 확률 행렬이다. 각 행은 유한한 비음수 값이며 합이 1이어야 한다. `calculate_inception_score`는 입력의 비음수성과 행 합을 검사하지만 배열 차원, 빈 입력, splits의 유효 범위까지 모두 검사하지는 않는다. 1≤splits≤N인 정수 분할 수가 필요하다.

```python
"""
52단원: 인셉션 점수(IS)
================================

생성 모델, 특히 GAN을 평가하는 데 널리 쓰는 지표인
인셉션 점수를 구현하고 해석한다.

학습 목표:
-------------------
1. 인셉션 점수의 수학적 기초를 이해한다
2. 인셉션 점수를 처음부터 구현한다
3. 인셉션 점수를 올바로 해석한다
4. 인셉션 점수의 한계를 알아본다

핵심 식:
-----------
IS = exp(E_x[KL(p(y|x) || p(y))])

여기서
- p(y|x): 조건부 라벨 분포(뾰족함, 곧 품질)
- p(y): 주변 라벨 분포(다양성)

지은이: 배움용 AI 모둠
날짜: 2025
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
    인셉션 점수 계산기.
    
    수학적 기초:
    -----------------------
    IS = exp(E_x[KL(p(y|x) || p(y))])
    
    항을 하나씩 뜯어보면:
    
    1. p(y|x): 조건부 클래스 분포
       - 그림 x에 InceptionV3을 돌린다
       - ImageNet 클래스 1000개에 대한 소프트맥스 확률을 얻는다
       - 뾰족한 분포(확신 있는 예측) = 높은 품질
    
    2. p(y): 주변 클래스 분포
       - 생성한 모든 이미지에 대한 p(y|x)의 평균
       - 균등한 분포 = 높은 다양성
       - 한쪽으로 치우친 분포 = 모드 붕괴
    
    3. KL(p(y|x) || p(y)): KL 발산
       - p(y|x)이 p(y)와 얼마나 다른지 잰다
       - KL이 크면 이미지의 예측이 확신 있고 다양하다
       - KL이 작으면 품질이 낮거나 다양성이 적다
    
    4. 지수 취하기: 로그 스케일에서 되돌린다
       - exp(E[KL(...)])이 마지막 인셉션 점수다
       - 흔한 범위: ImageNet 같은 이미지에서 1.0에서 10.0쯤
    
    뜻:
    ---------
    좋은 생성 모델은 다음과 같은 이미지를 내놓아야 한다.
    - 또렷이 알아볼 수 있다(p(y|x)이 뾰족하다 → 확신 있다)
    - 클래스를 두루 덮는다(p(y)이 균등하다 → 다양하다)
    
    인셉션 점수는 KL 발산으로 두 가지를 함께 잡는다.
    
    한계:
    -----------
    1. ImageNet 같은 이미지에서만 쓸모 있다(인셉션 분류기를 쓴다)
    2. 과적합(암기)을 탐지하지 못한다
    3. 클래스 내 다양성을 놓친다
    4. 클래스마다 이미지 하나씩만 생성해도 속일 수 있다
    5. 어떤 인셉션 모델을 쓰느냐에 흔들린다
    """
    
    @staticmethod
    def calculate_inception_score(probs: np.ndarray,
                                  splits: int = 10) -> Tuple[float, float]:
        """
        클래스 확률에서 인셉션 점수를 계산한다.
        
        인자:
            probs: 클래스 확률 [n_samples, n_classes]
                   InceptionV3 소프트맥스 레이어의 출력
            splits: 표준편차를 계산할 때 나눌 구간의 수
        
        반환값:
            (인셉션 점수 평균, 표준편차) 튜플
        
        계산 단계:
        ------------------
        1. 각 x_i에 대해 p(y) = (1/N) Σ p(y|x_i)을 계산한다
        2. 구간마다:
           a. 표본마다 KL(p(y|x_i) || p(y))을 계산한다
           b. 평균: E_x[KL(...)]
           c. 지수 취하기: exp(E_x[KL(...)])
        3. 구간들의 평균과 표준편차를 돌려준다
        """
        # 확률이 유효한지 확인한다
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), \
            "확률의 합은 1이어야 한다"
        
        n_samples = len(probs)
        
        # 표준편차를 계산하려고 구간으로 나눈다
        split_size = n_samples // splits
        scores = []
        
        for i in range(splits):
            # 구간을 가져온다
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
            part = probs[start_idx:end_idx]
            
            # 1. 주변 분포 p(y)을 계산한다
            # 조건부 분포의 평균
            # 형태: [n_classes]
            p_y = np.mean(part, axis=0)
            
            # 2. 표본마다 KL 발산을 계산한다
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            
            # 수치 안정성을 위해 작은 엡실론을 더한다
            eps = 1e-16
            p_y = p_y + eps
            part = part + eps
            
            # KL 발산을 계산한다
            # 형태: [split_size, n_classes]
            kl_div = part * (np.log(part) - np.log(p_y))
            
            # 클래스에 대해 더하고 표본에 대해 평균 낸다
            # E_x[KL(p(y|x) || p(y))]
            kl_mean = np.mean(np.sum(kl_div, axis=1))
            
            # 3. 지수를 취해 인셉션 점수를 얻는다
            is_score = np.exp(kl_mean)
            scores.append(is_score)
        
        # 구간들의 평균과 표준편차를 돌려준다
        return float(np.mean(scores)), float(np.std(scores))
    
    @staticmethod
    def interpret_is(is_score: float) -> str:
        """
        인셉션 점숫값을 해석한다.
        
        인자:
            is_score: 인셉션 점숫값
        
        반환값:
            해석 문자열
        
        흔한 범위(ImageNet):
        -------------------------
        - 인셉션 점수 < 2.0: 매우 나쁨
        - 인셉션 점수 2.0~5.0: 나쁨에서 보통
        - 인셉션 점수 5.0~8.0: 좋음
        - 인셉션 점수 8.0 초과: 아주 좋음
        
        참고: ImageNet의 실제 이미지는 인셉션 점수가 11.2쯤이다
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
    간단한 보기로 인셉션 점수의 직관을 보인다.
    """
    print("=" * 70)
    print("Inception Score Intuition")
    print("=" * 70)
    
    n_samples = 1000
    n_classes = 10  # 단출하게 줄였다(실제 인셉션 점수는 클래스 1000개를 쓴다)
    
    # 상황 1: 품질도 다양성도 높다(가장 바람직)
    print("\nScenario 1: High Quality + High Diversity (IDEAL)")
    print("-" * 70)
    
    # 이미지마다 한 클래스에 확신 있게 속한다(p(y|x)이 뾰족하다)
    # 모든 클래스가 고르게 나타난다(p(y)이 균등하다)
    probs1 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # 모든 클래스를 돌아가며 쓴다
        probs1[i, class_idx] = 0.9  # 확신 있는 예측
        probs1[i, :] += 0.1 / n_classes  # 작고 균등한 잡음
    
    # 정규화한다
    probs1 = probs1 / probs1.sum(axis=1, keepdims=True)
    
    is_score1, is_std1 = InceptionScore.calculate_inception_score(probs1)
    print(f"IS: {is_score1:.4f} ± {is_std1:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score1)}")
    print("Explanation: Confident predictions + diverse classes = High IS")
    
    # 상황 2: 품질이 낮다(예측이 불확실하다)
    print("\nScenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 70)
    
    # 이미지마다 분포가 균등하다(확신이 없다)
    probs2 = np.ones((n_samples, n_classes)) / n_classes
    
    is_score2, is_std2 = InceptionScore.calculate_inception_score(probs2)
    print(f"IS: {is_score2:.4f} ± {is_std2:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score2)}")
    print("Explanation: p(y|x) = p(y) → KL = 0 → IS = exp(0) = 1.0")
    print("Minimum possible IS = 1.0")
    
    # 상황 3: 모드 붕괴(클래스가 하나뿐)
    print("\nScenario 3: Mode Collapse (Single Class)")
    print("-" * 70)
    
    # 모든 이미지가 클래스 0으로 분류된다
    probs3 = np.zeros((n_samples, n_classes))
    probs3[:, 0] = 0.95
    probs3[:, 1:] = 0.05 / (n_classes - 1)
    
    is_score3, is_std3 = InceptionScore.calculate_inception_score(probs3)
    print(f"IS: {is_score3:.4f} ± {is_std3:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score3)}")
    print("Explanation: Confident predictions but no diversity")
    print("p(y) peaked at one class → Low KL → Low IS")
    
    # 상황 4: 품질은 좋으나 다양성이 적다
    print("\nScenario 4: Good Quality but Limited Diversity")
    print("-" * 70)
    
    # 클래스 10개 가운데 3개만 나타난다
    probs4 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % 3  # 클래스 0, 1, 2만 쓴다
        probs4[i, class_idx] = 0.9
        probs4[i, 3:] = 0.1 / (n_classes - 3)
    
    probs4 = probs4 / probs4.sum(axis=1, keepdims=True)
    
    is_score4, is_std4 = InceptionScore.calculate_inception_score(probs4)
    print(f"IS: {is_score4:.4f} ± {is_std4:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score4)}")
    print("Explanation: Confident but not covering all classes")
    
    # 요약
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
    
    # 한계 1: 암기를 탐지하지 못한다
    print("\nLimitation 1: Cannot Detect Memorization/Overfitting")
    print("-" * 70)
    print("A model that perfectly memorizes training images can achieve")
    print("high IS, even though it's not truly generating novel samples.")
    print("\nExample: Generating 100 real images repeatedly")
    print("         → High IS (confident + diverse)")
    print("         → But not creative/generative!")
    
    # 한계 2: 클래스 내 다양성을 놓친다
    print("\nLimitation 2: Ignores Within-Class Diversity")
    print("-" * 70)
    print("IS only cares about class labels, not visual diversity.")
    print("\nExample: Generating 1000 identical cat images")
    print("         → Still get high IS if classified as 'cat'")
    print("         → But zero visual diversity!")
    
    # 보기: 모든 이미지가 같은 클래스로 분류되는데도 인셉션 점수가 높다
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
    메인 시연 함수.
    """
    print("\n" + "=" * 70)
    print("MODULE 52: INCEPTION SCORE (IS)")
    print("=" * 70)
    
    # 인셉션 점수의 직관을 보인다
    demonstrate_inception_score_intuition()
    
    # 한계를 보인다
    demonstrate_is_limitations()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 인셉션 점수 식:
       IS = exp(E_x[KL(p(y|x) || p(y))])
       - p(y|x): 조건부 분포(품질, 곧 뾰족함)
       - p(y): 주변 분포(다양성)
       - KL 발산이 둘의 균형을 잡는다
    
    2. 인셉션 점수가 재는 것:
       - 품질: 예측이 얼마나 확신 있는가?
       - 다양성: 클래스를 얼마나 두루 덮는가?
       - 인셉션 점수가 높다 = 확신 있는 예측 + 다양한 표본
    
    3. 흔한 값:
       - 가장 작은 값: 인셉션 점수 = 1.0(균등한 예측)
       - 좋음: 인셉션 점수 5.0 초과
       - 아주 좋음: 인셉션 점수 8.0 초과
       - 실제 ImageNet: 인셉션 점수 ≈ 11.2
    
    4. 좋은 점:
       - 수 하나로 나타내는 지표
       - 계산이 빠르다
       - 품질과 다양성의 절충을 잡아낸다
       - 널리 쓰이고 잘 알려져 있다
    
    5. 한계:
       - 암기나 과적합을 탐지하지 못한다
       - 클래스 내 다양성을 놓친다
       - ImageNet 같은 이미지에서만 쓸모 있다
       - InceptionV3 분류기에 매여 있다
       - 적대적 전략에 속을 수 있다
    
    6. 모범 사례:
       - 표준편차를 계산할 때는 splits=10을 쓴다
       - 인셉션 점수 ± 표준편차를 함께 알린다
       - 늘 다른 지표와 함께 쓴다(FID, 정밀도와 재현율)
       - 눈으로 살펴보는 일을 곁들인다
       - 작업에 맞춘 평가도 함께 고려한다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (135줄)"

    ```

    ======================================================================
    MODULE 52: INCEPTION SCORE (IS)
    ======================================================================
    ======================================================================
    Inception Score Intuition
    ======================================================================

    Scenario 1: High Quality + High Diversity (IDEAL)
    ----------------------------------------------------------------------
    IS: 6.0636 ± 0.0000
    Quality: Good
    Explanation: Confident predictions + diverse classes = High IS

    Scenario 2: Low Quality (Uncertain Predictions)
    ----------------------------------------------------------------------
    IS: 1.0000 ± 0.0000
    Quality: Very Poor
    Explanation: p(y|x) = p(y) → KL = 0 → IS = exp(0) = 1.0
    Minimum possible IS = 1.0

    Scenario 3: Mode Collapse (Single Class)
    ----------------------------------------------------------------------
    IS: 1.0000 ± 0.0000
    Quality: Very Poor
    Explanation: Confident predictions but no diversity
    p(y) peaked at one class → Low KL → Low IS

    Scenario 4: Good Quality but Limited Diversity
    ----------------------------------------------------------------------
    IS: 2.6876 ± 0.0000
    Quality: Poor to Moderate
    Explanation: Confident but not covering all classes

    ======================================================================
    IS Comparison:
    ======================================================================
    High quality + high diversity:  IS = 6.06
    Low quality (uncertain):        IS = 1.00 (minimum)
    Mode collapse (one class):      IS = 1.00
    Good quality + limited diversity: IS = 2.69

    Key Insight: IS balances quality (sharpness) and diversity

    ======================================================================
    Inception Score Limitations
    ======================================================================

    Limitation 1: Cannot Detect Memorization/Overfitting
    ----------------------------------------------------------------------
    A model that perfectly memorizes training images can achieve
    high IS, even though it's not truly generating novel samples.

    Example: Generating 100 real images repeatedly
             → High IS (confident + diverse)
             → But not creative/generative!

    Limitation 2: Ignores Within-Class Diversity
    ----------------------------------------------------------------------
    IS only cares about class labels, not visual diversity.

    Example: Generating 1000 identical cat images
             → Still get high IS if classified as 'cat'
             → But zero visual diversity!

    1000 identical images → IS = 1.00
    This should be low but IS doesn't capture it!

    Limitation 3: Can Be Fooled by Adversarial Generation
    ----------------------------------------------------------------------
    Strategy: Generate exactly one image per class
              → Maximum diversity (uniform p(y))
              → Confident predictions (sharp p(y|x))
              → High IS!

    But only 10 unique images for 1000 ImageNet classes is terrible!

    Limitation 4: Tied to InceptionV3 Classifier
    ----------------------------------------------------------------------
    IS depends on InceptionV3's learned representations.
    • Only works well for ImageNet-like natural images
    • May not work for: medical images, satellite imagery,
      abstract art, non-photorealistic images
    • Different classifiers give different IS values

    ======================================================================
    Recommendation: Use IS alongside other metrics!
    ======================================================================
    • Combine with FID (detects mode collapse better)
    • Add precision/recall (measures coverage)
    • Include visual inspection
    • Consider task-specific metrics

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

        1. 인셉션 점수 식:
           IS = exp(E_x[KL(p(y|x) || p(y))])
           - p(y|x): 조건부 분포(품질, 곧 뾰족함)
           - p(y): 주변 분포(다양성)
           - KL 발산이 둘의 균형을 잡는다
        
        2. 인셉션 점수가 재는 것:
           - 품질: 예측이 얼마나 확신 있는가?
           - 다양성: 클래스를 얼마나 두루 덮는가?
           - 인셉션 점수가 높다 = 확신 있는 예측 + 다양한 표본
        
        3. 흔한 값:
           - 가장 작은 값: 인셉션 점수 = 1.0(균등한 예측)
           - 좋음: 인셉션 점수 5.0 초과
           - 아주 좋음: 인셉션 점수 8.0 초과
           - 실제 ImageNet: 인셉션 점수 ≈ 11.2
        
        4. 좋은 점:
           - 수 하나로 나타내는 지표
           - 계산이 빠르다
           - 품질과 다양성의 절충을 잡아낸다
           - 널리 쓰이고 잘 알려져 있다
        
        5. 한계:
           - 암기나 과적합을 탐지하지 못한다
           - 클래스 내 다양성을 놓친다
           - ImageNet 같은 이미지에서만 쓸모 있다
           - InceptionV3 분류기에 매여 있다
           - 적대적 전략에 속을 수 있다
        
        6. 모범 사례:
           - 표준편차를 계산할 때는 splits=10을 쓴다
           - 인셉션 점수 ± 표준편차를 함께 알린다
           - 늘 다른 지표와 함께 쓴다(FID, 정밀도와 재현율)
           - 눈으로 살펴보는 일을 곁들인다
           - 작업에 맞춘 평가도 함께 고려한다
        
    ======================================================================
    ```

## 2. 점수의 정의와 범위

분류기의 조건부 예측을 $q(y\mid x)$, 생성 샘플 전체에서의 평균 예측을 $\bar q(y)$라고 쓰면 다음과 같다.

$$\mathrm{IS}=\exp\!\left[\mathbb E_x D_{\mathrm{KL}}\big(q(y\mid x)\|\bar q(y)\big)\right],\qquad
\bar q(y)=\mathbb E_x q(y\mid x).$$

로그 점수는 주변 엔트로피에서 평균 조건부 엔트로피를 뺀 값이다.

$$\log\mathrm{IS}=H(\bar q)-\mathbb E_x H(q(\cdot\mid x)).$$

C개 클래스에서 정확한 확률을 쓰면 1≤IS≤C다. 모든 표본의 예측 분포가 같으면 그 분포가 균등하든 한 클래스에 집중되든 IS=1이다. 상한 C는 표본별 예측이 원핫이고 평균 예측은 클래스별로 균등할 때 달성한다. 분류기의 확신은 이미지의 실제 품질과 같지 않고, 클래스 간 분산도 클래스 내부의 다양성과 같지 않다.

예제의 첫 시나리오는 선택한 클래스에 0.91, 나머지에 0.01을 부여한다. 이는 원핫 예측이 아니므로 이상적인 상한 10에 이르지 않는다. 세 번째 시나리오는 모든 행이 같아 IS=1이다. 네 번째는 세 클래스에 대한 주된 예측과 나머지 클래스에 대한 작은 확률을 함께 사용한다.

출력의 '동일한 고양이 이미지 1,000장도 높은 IS를 얻는다'는 설명은 틀렸다. 결정적인 평가 분류기에서 동일 이미지의 예측이 같으면 IS=1이다. 놓치는 사례는 **클래스마다 대표 이미지 하나씩만 반복하면서 클래스 비율은 균등하게 유지하는 경우**다. 또한 출력의 'ImageNet 1,000클래스에 고유 이미지 10개로 최대 점수'라는 설명도 원핫 예측을 전제로 한 최대 1,000의 예와 맞지 않는다.

## 3. 분할, 순서와 오차

이 구현은 입력을 섞지 않고 연속 구간으로 나눈다. 각 구간 안에서 주변 분포를 다시 추정하고, 구간별 점수의 평균과 표준편차를 반환한다. 마지막 구간은 나머지 표본도 포함하므로 표본이 버려지지는 않지만 구간 크기가 다를 수 있다. 전체 행렬에서 한 번 계산한 IS와 분할 점수의 평균은 일반적으로 다르다.

입력이 클래스순으로 정렬되어 있으면 각 구간의 클래스 범위가 좁아져 결과가 달라질 수 있다. 독립적으로 섞인 입력을 쓰거나 미리 정한 시드로 섞은 뒤 동일한 분할 절차를 사용한다. 구간별 표준편차는 구간 사이 변동의 요약이며, 모델 차이의 유의성이나 평균 점수의 신뢰구간을 직접 나타내지 않는다. 이 구현은 기본 `np.std`를 사용하므로 분모는 구간 수다.

아주 작은 확률에 ε를 더하면 log 0과 0×(−∞)를 피할 수 있지만 분포를 미세하게 바꾼다. dtype과 안정화 방식도 명시한다. 분류기 예측의 float32 배열을 이 함수에 전달한다고 자동으로 float64로 변환되는 것은 아니다.

## 4. 평가 결과를 보고하는 조건

이 코드에는 MNIST 분류기 학습, ImageNet 평가, 생성 이미지 평가가 없다. 고정된 `interpret_is` 임계값이나 출력의 ImageNet 기준값은 이 코드로 검증한 보편적인 품질 기준이 아니다. 실제 데이터의 점수도 이론적 상한이 아니며 생성 모델의 점수가 이를 넘는 것만으로 부정한 평가라고 결론 내릴 수 없다.

보고에는 분류기 구조와 체크포인트, 입력 전처리, 클래스 수, 표본 수, 분할 수, 순서·시드, 샘플링·선별 조건을 포함한다. FID, 정밀도·재현율, 표본 및 중복 검사와 함께 해석하되 이들 지표 역시 모든 품질 차이나 암기를 검출하지는 못한다. [인셉션 점수의 한계를 분석한 논문](https://arxiv.org/abs/1801.01973)은 평가 절차와 수치 해석을 함께 검토해야 함을 강조한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
N×C 확률 행렬에서 입력 조건과 각 축의 의미를 설명하라.

</div>

??? success "연습문제 1 풀이"

    행은 평가 표본, 열은 클래스다. 각 행의 값이 유한한 비음수이고 합이 1이어야 한다. 빈 입력이나 클래스 축이 없는 배열은 유효하지 않다. 원핫 정답 라벨 대신 이미지에 대한 평가 분류기의 예측 확률을 사용하는 것이 실제 평가의 목적이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
모든 행이 같은 확률 벡터이면 왜 IS=1인가?

</div>

??? success "연습문제 2 풀이"

    각 행 q가 평균 예측과 같아 KL(q∥q)=0이기 때문이다. 균등 예측뿐 아니라 동일한 원핫 예측에도 성립한다. 따라서 IS=1만으로 불확실성, 단일 클래스 생성, 이미지 자체의 반복 중 어느 원인인지 확정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
고정된 품질 임계값 2, 5, 8을 다른 데이터셋과 분류기에 그대로 적용할 수 있는가?

</div>

??? success "연습문제 3 풀이"

    클래스 수, 평가 분류기, 예측의 보정 정도와 대상 데이터가 달라지므로 적용할 수 없다. 클래스 두 개인 분류기에서는 IS의 상한 자체가 2다. 동일한 평가 조건에서 비교하고 시각적 품질과 분류기 확신을 구분한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
평가에서 eval과 no_grad의 역할은 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"

    eval은 배치 정규화와 드롭아웃 등의 동작을 평가 모드로 바꾼다. no_grad는 그래디언트 계산 그래프의 기록을 끈다. 하나가 다른 하나를 대신하지 않는다. 아래 예시는 별도로 정의된 net과 images를 전제로 하며, 장치와 전처리도 일치해야 한다.

    ```python
    net.eval()
    with torch.no_grad():
        probs = F.softmax(net(images), dim=1)
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
KL 계산에 ε를 더하는 목적과 아래 대안 코드의 한계를 설명하라.

</div>

??? success "연습문제 5 풀이"

    0에 대한 로그와 0×(−∞)를 피하려는 수치 안정화다. 수학적으로는 q=0인 항을 0으로 정의한다. 아래 마스크 예시는 q가 N×C이고 py가 C 또는 1×C일 때 같은 이차원 불리언 마스크로 py를 직접 인덱싱하므로 모양이 맞지 않을 수 있다. 적절한 브로드캐스팅과 축별 합산을 먼저 설계해야 하며 그대로 대체 코드로 사용할 수는 없다. ε나 clipping은 정규화를 미세하게 바꾸므로 그 크기와 정밀도를 기록한다.

    ```python
    kl = (q * (np.log(q + 1e-12) - np.log(py + 1e-12))).sum(1)
    ```
    ```python
    m = q > 0
    kl = (q[m] * (np.log(q[m]) - np.log(py[m]))).sum()
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
아래 분할 함수에서 섞기와 구간별 주변 분포 계산은 어떤 역할을 하는가?

</div>

??? success "연습문제 6 풀이"

    입력 복사본을 시드로 섞어 클래스 정렬에 따른 구간 편향을 줄인다. i*n//splits 경계는 나머지 표본도 포함한다. 각 구간에서 주변 분포를 계산하므로 전체 데이터에서 한 번 계산한 점수와는 다른 추정 절차다. splits는 1 이상 N 이하로 제한해야 하며, 분할 표준편차는 신뢰구간이 아니다. 섞기는 행 순서만 바꾸므로, 같은 표본의 순열에 불변인 FID의 평균·공분산에는 영향을 주지 않는다.

    ```python
    def inception_score(probs, splits=10, seed=0):
        probs = probs.copy()
        np.random.default_rng(seed).shuffle(probs)      # 반드시 섞는다
        scores, n = [], len(probs)
        for i in range(splits):
            q = probs[i*n//splits : (i+1)*n//splits]
            py = q.mean(0, keepdims=True)               # 구간 안의 p(y)
            kl = (q * (np.log(q+1e-12) - np.log(py+1e-12))).sum(1)
            scores.append(np.exp(kl.mean()))
        return np.mean(scores), np.std(scores)
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
아래 시험 코드의 기대값 가운데 분할과 섞기에 따라 달라지는 것은 무엇인가?

</div>

??? success "연습문제 7 풀이"

    동일 원핫 행과 동일 균등 행의 점수는 각 구간에서도 1이다. 반면 원핫 예측이 전체적으로 균형을 이뤄도 무작위 분할 후 각 구간은 정확히 균형을 이루지 않을 수 있다. 그러면 구간별 점수는 클래스 수보다 작아져 아래 perfect의 엄격한 assert가 실패할 수 있다. 상한 확인은 splits=1 또는 각 구간에 같은 수의 클래스가 들어가는 구성을 사용한다. 실제 데이터의 예상 점수 9는 보편적인 시험 기준이 아니다.

    ```python
    C = 10
    one = np.zeros((1000, C)); one[:, 3] = 1.0
    assert abs(inception_score(one)[0] - 1.0) < 1e-6

    perfect = np.eye(C)[np.arange(1000) % C]
    assert abs(inception_score(perfect)[0] - C) < 1e-6

    uniform = np.full((1000, C), 1.0/C)
    assert abs(inception_score(uniform)[0] - 1.0) < 1e-6
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff easy" title="쉬움"></span>
실제 데이터의 점수를 먼저 재는 이유는 무엇이며 그 값이 이론적 상한인가?

</div>

??? success "연습문제 8 풀이"

    같은 분류기와 전처리에서 유용한 경험적 기준을 얻기 위해서다. 실제 클래스 비율이 불균형하거나 예측이 불확실하면 C보다 작을 수 있다. 생성 샘플의 예측이 더 확신에 차거나 더 균형을 이루면 실제 데이터 기준보다 높을 수도 있다. 그것만으로 더 좋은 생성 모델이나 지표 조작을 확정하지 않는다. 이 페이지에는 실제 MNIST 점수를 재는 코드가 없다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
실제 샘플과 생성 샘플의 점수를 비교할 때 무엇을 맞추어야 하는가?

</div>

??? success "연습문제 9 풀이"

    분류기와 체크포인트, 채널·크기·범위·정규화, 추론 모드, 표본 수와 분할·섞기 조건을 맞춘다. IS 자체는 실제 기준 집합을 필요로 하지 않지만 서로 다른 설정의 수치를 직접 비교하면 해석이 어렵다. 아래 probs_of 등은 외부 정의가 필요한 개념적 예시다.

    ```python
    for name, imgs in (("실제 데이터", real), ("모델", fake)):
        print(name, inception_score(probs_of(imgs)))
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
FID를 함께 계산하려면 무엇이 추가되는가?

</div>

??? success "연습문제 10 풀이"

    선택한 피처 추출기의 임베딩과 실제 기준 집합의 피처 평균·공분산이 필요하다. 피처와 분류 확률은 다른 출력이다. 아래 Net과 함수 조각은 완성된 실행 코드가 아니다. 행렬 제곱근에 복소수가 나오면 크기와 수치 오차를 확인해야 하며 무조건 실수부만 취하면 안 된다. FID도 첫 두 모멘트만 비교하므로 모든 클래스 내부 다양성 손실이나 암기를 검출하지 않는다.

    ```python
    class Net(nn.Module):
        def feat(self, x):          # 128차원 피처
            ...
            return torch.relu(self.f1(x.flatten(1)))
        def forward(self, x):
            return self.f2(self.feat(x))
    ```
    ```python
    f = net.feat(batch)
    probs = F.softmax(net.f2(f), dim=1)      # IS 용
    features = f                             # FID 용
    ```


---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff med" title="중간"></span>
tanh 출력의 (x+1)/2 변환만 하면 모든 Inception 전처리가 완료되는가?

</div>

??? success "연습문제 11 풀이"

    아니다. 이 변환은 −1~1을 0~1로 옮길 뿐이다. 평가 구현에 맞는 채널, resize 방식과 추가 정규화가 필요할 수 있다. 최솟값·최댓값은 범위를 확인하는 데 도움이 되지만 이미지 의미나 전처리 전체의 올바름을 검증하지 않는다.

    ```python
    imgs = (imgs + 1) / 2          # tanh 출력이면
    ```
    ```python
    print(imgs.min().item(), imgs.max().item())     # 0 근처와 1 근처여야 한다
    ```


---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff med" title="중간"></span>
평가 계산을 줄이기 위해 재사용할 수 있는 것은 무엇인가?

</div>

??? success "연습문제 12 풀이"

    분류기·전처리·기준 집합이 고정된 FID의 실제 피처 통계는 캐시할 수 있다. IS에는 실제 기준 피처가 필요하지 않다. 배치 크기는 장치 메모리와 추론 시간에 맞추고, no_grad와 eval을 사용한다. 반정밀도는 예측을 바꿀 수 있으므로 속도만 보고 혼용하지 않는다. 아래 코드는 외부 helper를 전제로 한 FID 평가의 개념적 예시다.

    ```python
    real_feats = features_of(real_images)     # 한 번만
    for ep in range(epochs):
        ...
        print(fid(real_feats, features_of(sample(g))))
    ```


---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff easy" title="쉬움"></span>
학습 중 평가 주기와 최종 보고의 표본 수를 어떻게 정하는가?

</div>

??? success "연습문제 13 풀이"

    계산비용과 원하는 정밀도에 맞추어 미리 정한다. 중간 추적에는 고정된 작은 표본 수를 쓸 수 있지만 최종 보고와 같은 조건의 값으로 혼동하지 않는다. 유한 표본 편향과 변동이 있으므로 독립 생성 반복이나 적절한 불확실성 분석도 고려한다. 최종 시험 집합을 반복해서 보고 모델을 선택하면 선택 편향이 생길 수 있다.



## 정리하며

인셉션 점수는 분류 예측의 확신과 주변 클래스 분포를 요약한다. 분할과 순서, 분류기와 전처리를 명시하고, 클래스 내부 다양성과 암기를 별도로 확인한다.
