# 단원 52: 인셉션 점수(IS)

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 단원 52: 인셉션 점수(IS)을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 코드

```python
"""
52단원: 인셉션 점수(IS)
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

여기서
- p(y|x): 조건 이름표 분포(뾰족함, 곧 품질)
- p(y): 가장자리 이름표 분포(다양함)

지은이: 배움용 AI 모둠
때: 2025
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
    
    1. p(y|x): 조건 갈래 분포
       - 그림 x에 InceptionV3을 돌린다
       - ImageNet 갈래 1000개에 대한 소프트맥스 확률을 얻는다
       - 뾰족한 분포(자신 있는 헤아림) = 높은 품질
    
    2. p(y): 가장자리 갈래 분포
       - 만들어 낸 모든 그림에 대한 p(y|x)의 평균
       - 고른 분포 = 높은 다양함
       - 한쪽으로 쏠린 분포 = 봉우리 무너짐
    
    3. KL(p(y|x) || p(y)): KL 갈림
       - p(y|x)이 p(y)와 얼마나 다른지 잰다
       - KL이 크면 그림의 헤아림이 자신 있고 다양하다
       - KL이 작으면 품질이 낮거나 다양함이 적다
    
    4. 지수 취하기: 로그 자에서 되돌린다
       - exp(E[KL(...)])이 마지막 인셉션 점수다
       - 흔한 범위: ImageNet 같은 그림에서 1.0에서 10.0쯤
    
    뜻:
    ---------
    좋은 만들어 내는 모델은 다음과 같은 그림을 내놓아야 한다.
    - 또렷이 알아볼 수 있다(p(y|x)이 뾰족하다 → 자신 있다)
    - 갈래를 두루 덮는다(p(y)이 고르다 → 다양하다)
    
    인셉션 점수는 KL 갈림으로 두 가지를 함께 잡는다.
    
    한계:
    -----------
    1. ImageNet 같은 그림에서만 쓸모 있다(인셉션 가름개를 쓴다)
    2. 지나치게 맞춰짐(외워 버림)을 알아채지 못한다
    3. 갈래 안의 다양함을 놓친다
    4. 갈래마다 그림 하나씩만 만들어도 속일 수 있다
    5. 어떤 인셉션 모델을 쓰느냐에 흔들린다
    """
    
    @staticmethod
    def calculate_inception_score(probs: np.ndarray,
                                  splits: int = 10) -> Tuple[float, float]:
        """
        갈래 확률에서 인셉션 점수를 셈한다.
        
        인자:
            probs: 갈래 확률 [n_samples, n_classes]
                   InceptionV3 소프트맥스 층의 내놓음
            splits: 표준편차를 셈할 때 나눌 조각의 수
        
        돌려주는 값:
            (인셉션 점수 평균, 표준편차) 튜플
        
        수학 걸음:
        ------------------
        1. 각 x_i에 대해 p(y) = (1/N) Σ p(y|x_i)을 셈한다
        2. 조각마다:
           a. 표본마다 KL(p(y|x_i) || p(y))을 셈한다
           b. 평균: E_x[KL(...)]
           c. 지수 취하기: exp(E_x[KL(...)])
        3. 조각들의 평균과 표준편차를 돌려준다
        """
        # 확률이 올바른지 다진다
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), \
            "확률의 합은 1이어야 한다"
        
        n_samples = len(probs)
        
        # 표준편차를 셈하려고 묶음으로 나눈다
        split_size = n_samples // splits
        scores = []
        
        for i in range(splits):
            # 조각을 가져온다
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else n_samples
            part = probs[start_idx:end_idx]
            
            # 1. 가장자리 분포 p(y)을 셈한다
            # 조건 분포의 평균
            # 꼴: [n_classes]
            p_y = np.mean(part, axis=0)
            
            # 2. 표본마다 KL 갈림을 셈한다
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            
            # 수치가 든든하도록 작은 엡실론을 더한다
            eps = 1e-16
            p_y = p_y + eps
            part = part + eps
            
            # KL 갈림을 셈한다
            # 꼴: [split_size, n_classes]
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
        
        인자:
            is_score: 인셉션 점숫값
        
        돌려주는 값:
            풀이 글월
        
        흔한 범위(ImageNet):
        -------------------------
        - 인셉션 점수 < 2.0: 매우 나쁨
        - 인셉션 점수 2.0~5.0: 나쁨에서 보통
        - 인셉션 점수 5.0~8.0: 좋음
        - 인셉션 점수 8.0 초과: 아주 좋음
        
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
    n_classes = 10  # 단출하게 줄였다(참 인셉션 점수는 갈래 1000개를 쓴다)
    
    # 상황 1: 품질도 다양함도 높다(가장 바람직)
    print("\nScenario 1: High Quality + High Diversity (IDEAL)")
    print("-" * 70)
    
    # 그림마다 한 갈래에 자신 있게 든다(p(y|x)이 뾰족하다)
    # 모든 갈래가 고르게 나타난다(p(y)이 고르다)
    probs1 = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # 모든 갈래를 돌아가며 쓴다
        probs1[i, class_idx] = 0.9  # 자신 있는 헤아림
        probs1[i, :] += 0.1 / n_classes  # 작고 고른 잡음
    
    # 고르게 맞춘다
    probs1 = probs1 / probs1.sum(axis=1, keepdims=True)
    
    is_score1, is_std1 = InceptionScore.calculate_inception_score(probs1)
    print(f"IS: {is_score1:.4f} ± {is_std1:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score1)}")
    print("Explanation: Confident predictions + diverse classes = High IS")
    
    # 상황 2: 품질이 낮다(헤아림이 흐릿하다)
    print("\nScenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 70)
    
    # 그림마다 분포가 고르다(자신이 없다)
    probs2 = np.ones((n_samples, n_classes)) / n_classes
    
    is_score2, is_std2 = InceptionScore.calculate_inception_score(probs2)
    print(f"IS: {is_score2:.4f} ± {is_std2:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score2)}")
    print("Explanation: p(y|x) = p(y) → KL = 0 → IS = exp(0) = 1.0")
    print("Minimum possible IS = 1.0")
    
    # 상황 3: 봉우리 무너짐(갈래가 하나뿐)
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
        class_idx = i % 3  # 갈래 0, 1, 2만 쓴다
        probs4[i, class_idx] = 0.9
        probs4[i, 3:] = 0.1 / (n_classes - 3)
    
    probs4 = probs4 / probs4.sum(axis=1, keepdims=True)
    
    is_score4, is_std4 = InceptionScore.calculate_inception_score(probs4)
    print(f"IS: {is_score4:.4f} ± {is_std4:.4f}")
    print(f"Quality: {InceptionScore.interpret_is(is_score4)}")
    print("Explanation: Confident but not covering all classes")
    
    # 간추림
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
    1. 인셉션 점수 식:
       IS = exp(E_x[KL(p(y|x) || p(y))])
       - p(y|x): 조건 분포(품질, 곧 뾰족함)
       - p(y): 가장자리 분포(다양함)
       - KL 갈림이 둘의 균형을 잡는다
    
    2. 인셉션 점수가 재는 것:
       - 품질: 헤아림이 얼마나 자신 있는가?
       - 다양함: 갈래를 얼마나 두루 덮는가?
       - 인셉션 점수가 높다 = 자신 있는 헤아림 + 다양한 표본
    
    3. 흔한 값:
       - 가장 작은 값: 인셉션 점수 = 1.0(고른 헤아림)
       - 좋음: 인셉션 점수 5.0 초과
       - 아주 좋음: 인셉션 점수 8.0 초과
       - 참 ImageNet: 인셉션 점수 ≈ 11.2
    
    4. 좋은 점:
       - 수 하나로 나타내는 자
       - 셈이 빠르다
       - 품질과 다양함의 절충을 잡아낸다
       - 널리 쓰이고 잘 알려져 있다
    
    5. 한계:
       - 외워 버림이나 지나치게 맞춰짐을 알아채지 못한다
       - 갈래 안의 다양함을 놓친다
       - ImageNet 같은 그림에서만 쓸모 있다
       - InceptionV3 가름개에 매여 있다
       - 맞겨루기 꾀에 속을 수 있다
    
    6. 모범 사례:
       - 표준편차를 셈할 때는 splits=10을 쓴다
       - 인셉션 점수 ± 표준편차를 함께 알린다
       - 늘 다른 자와 함께 쓴다(FID, 정밀도와 재현율)
       - 눈으로 살펴보는 일을 곁들인다
       - 일에 맞춘 따지기도 함께 생각한다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 논의

이 짜기는 단원 52: 인셉션 점수(IS)에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 익힘 문제

**익힘 1.**
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

??? success "익힘 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

**익힘 2.**
이 짜기의 핵심 웃매개변수(배움 빠르기, 묶음 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

??? success "익힘 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 묶음 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 묶음 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

**익힘 3.**
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

??? success "익힘 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.
