# 우도 기반 평가 지표

음의 로그 우도(NLL)는 모델이 평가 데이터에 부여한 확률을 요약한다. 차원당 비트 수(bpd)는 이를 데이터 차원과 로그 밑에 맞춰 표현하며, 퍼플렉서티(perplexity)는 언어 모델의 토큰당 NLL을 지수화한 값이다. 세 지표 모두 입력 로그 확률이 무엇을 뜻하는지 먼저 정해야 올바르게 해석할 수 있다.

## 1. 계산 예제

아래 코드는 학습된 생성 모델을 평가하지 않는다. 임의로 만든 로그 확률 배열과 가정한 NLL 값으로 계산식을 보여 준다. 출력의 `Perfect`, `Good`, `Poor`는 예제 이름이며 실제 모델의 품질을 검증한 결과가 아니다.

```python
"""
가능도에 바탕한 잣대
======================================

이 단원은 따지기에 쓰는 바탕이 되는 가능도 잣대를 다룬다
만들어 내는 모델을 위한 자: 음의 로그 가능도(NLL), 차원마다의 비트 수(BPD),
그리고 헷갈림도.

학습 목표:
-------------------
1. 음의 로그 가능도를 이해하고 짠다
2. 고르게 맞춘 견줌을 위해 차원마다 비트를 셈한다
3. 말 모델의 헷갈림도를 셈한다
4. 가능도 잣대를 제대로 풀이한다

핵심 개념:
------------
- 확률 잣대로서의 가능도
- 앎 이론의 풀이
- 어긋 엔트로피와의 이음
- 가능도로 모델 견주기

수학적 바탕:
-----------------------
NLL = -E_{x~p_data}[log p_model(x)]
BPD = NLL / (dimensions × log(2))
Perplexity = exp(NLL per token)

지은이: 가르치기 인공 지능 모둠
날짜: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# ========================================================================
# 메인
# ========================================================================

# 난수 씨앗을 설정한다
torch.manual_seed(42)
np.random.seed(42)


class NegativeLogLikelihood:
    """
    음의 로그 가능도(NLL)의 짜기와 풀이.
    
    수학의 뜻매김:
    -----------------------
    만들어 내는 모델 p_θ(x)에 대해 자료 묶음 D의 음의 로그 가능도는 다음과 같다.
    
        NLL = -1/N * Σ log p_θ(x_i)
    
    여기서 N은 D의 표본 수이다.
    
    해석:
    --------------
    - 음의 로그 가능도는 모델이 자료를 얼마나 잘 설명하는지 잰다
    - 음의 로그 가능도가 낮을수록 자료 분포에 잘 맞는다
    - 음의 로그 가능도가 0이면 나무랄 데 없는 모형이다(실제로는 안 된다)
    - 음의 로그 가능도는 기댓값 로그 가능도의 음수이다
    
    어긋 엔트로피와의 이음:
    ---------------------------
    음의 로그 가능도는 다음 둘 사이의 어긋 엔트로피와 같다.
    - 참 자료 분포 p_data(x)
    - Model distribution p_θ(x)
    
    H(p_data, p_θ) = -E_{x~p_data}[log p_θ(x)] = NLL
    """
    
    @staticmethod
    def compute(log_probs: torch.Tensor) -> float:
        """
        로그 확률에서 음의 로그 가능도를 셈한다.
        
        인수:
            log_probs: 표본마다의 로그 확률 [n_samples]
        
        반환값:
            낱값으로서의 음의 로그 가능도
        
        수학의 걸음:
        ------------------
        1. 로그 확률의 평균: (1/N) Σ log p(x_i)
        2. 음수를 취한다: NLL = -(1/N) Σ log p(x_i)
        """
        # 평균 로그 확률을 셈한다
        mean_log_prob = torch.mean(log_probs)
        
        # 음수를 취해 음의 로그 가능도를 얻는다
        nll = -mean_log_prob
        
        return nll.item()
    
    @staticmethod
    def compute_with_variance(log_probs: torch.Tensor) -> Tuple[float, float]:
        """
        믿음 구간과 함께 음의 로그 가능도를 셈한다.
        
        인수:
            log_probs: 로그 확률 [n_samples]
        
        반환값:
            (음의 로그 가능도, 표준 오차) 튜플
        
        살펴보면:
        ----
        표준 오차 = std(log_probs) / sqrt(n_samples)
        이는 음의 로그 가능도 어림의 흐릿함을 알려 준다.
        """
        # 음의 로그 가능도를 셈한다
        nll = -torch.mean(log_probs).item()
        
        # 표준 오차를 셈한다
        # 표준 오차 = σ / sqrt(N)
        std = torch.std(log_probs).item()
        n = len(log_probs)
        standard_error = std / np.sqrt(n)
        
        return nll, standard_error


class BitsPerDimension:
    """
    가능도를 고르게 견주는 자인 차원마다의 비트 수(BPD).
    
    수학의 뜻매김:
    -----------------------
        BPD = NLL / (D × log(2))
    
    여기서 각 기호는 다음과 같다.
    - D은 자료의 차원이다
    - log(2)이 내트를 비트로 바꾼다
    
    왜 차원마다 비트인가?
    --------
    1. 자료 차원에 맞게 고르게 맞춘다
       - 28×28 그림과 256×256 그림
       - 길이가 다른 차례
    
    2. 앎 이론의 풀이
       - 차원 하나를 부호로 담는 데 드는 평균 비트
       - BPD이 낮을수록 더 잘 눌러 담는다
    
    3. 다음에 걸친 공정한 견줌을 가능하게 한다:
       - 서로 다른 그림 해상도
       - 서로 다른 차례 길이
       - 서로 다른 자료 클래스
    
    보기 풀이:
    ----------------------
    BPD = 3.5 means:
    - 평균으로 화소나 차원마다 3.5비트가 필요하다
    - 8비트 그림에서 고른 분포이면 BPD = 8.0이다
    - 좋은 모델은 자연 그림에서 차원마다 비트 < 4.0을 이룬다
    """
    
    @staticmethod
    def compute(nll: float, dimensions: int) -> float:
        """
        음의 로그 가능도에서 차원마다 비트를 셈한다.
        
        인수:
            nll: 음의 로그 가능도(내트 단위)
            dimensions: 자료의 온 차원 수
                       (보기: MNIST이면 28*28=784, CIFAR이면 32*32*3=3072)
        
        반환값:
            차원마다 비트 값
        
        수학의 걸음:
        ------------------
        1. 음의 로그 가능도는 자연 단위(내트)로 나온다
        2. 차원으로 나누어 고르게 맞춘다
        3. 비트로 바꾼다: log(2) ≈ 0.693으로 나눈다
        """
        # 내트에서 비트로 바꾼다: log(2)으로 나눈다
        # 그런 다음 차원으로 고르게 맞춘다
        bpd = nll / (dimensions * np.log(2))
        
        return bpd
    
    @staticmethod
    def compute_from_log_probs(log_probs: torch.Tensor, dimensions: int) -> float:
        """
        로그 확률에서 곧바로 차원마다 비트를 셈한다.
        
        인수:
            log_probs: 로그 확률 [n_samples]
            dimensions: 자료의 차원
        
        반환값:
            차원마다 비트 값
        """
        # 먼저 음의 로그 가능도를 셈한다
        nll = -torch.mean(log_probs).item()
        
        # 차원마다 비트로 바꾼다
        bpd = nll / (dimensions * np.log(2))
        
        return bpd
    
    @staticmethod
    def interpret_bpd(bpd: float, data_type: str = "image") -> str:
        """
        차원마다 비트 값의 풀이를 준다.
        
        인수:
            bpd: 차원마다 비트 값
            data_type: 자료의 클래스(그림, 글월 따위)
        
        반환값:
            풀이 글자열
        """
        if data_type == "image":
            if bpd > 8.0:
                return "Very Poor (worse than random)"
            elif bpd > 5.0:
                return "Poor"
            elif bpd > 3.5:
                return "Moderate"
            elif bpd > 2.0:
                return "Good"
            else:
                return "Excellent"
        else:
            return f"BPD = {bpd:.3f}"


class Perplexity:
    """
    말 모델의 헷갈림도 잣대.
    
    수학의 뜻매김:
    -----------------------
        Perplexity = exp(NLL per token)
                   = exp(-1/N Σ log p(x_i))
    
    직관의 뜻:
    -----------------
    헷갈림도는 "실제로 쓰이는 낱말 수", 곧 자리마다 모델이
    헷갈려 하는 토큰의 평균 수를 나타낸다.
    
    예제:
    --------
    - 헷갈림도 = 100: 마구잡이로 고르는 것만큼 헷갈린다
                        from 100 equally likely tokens
    - 헷갈림도 = 10:  그럴듯한 토큰을 10개쯤으로 좁혔다
    - 헷갈림도 = 1:   모델이 온전히 확신한다(이론에서만)
    
    어긋 엔트로피와의 이음:
    ---------------------------
    Perplexity = 2^(cross-entropy in bits)
    
    헷갈림도가 낮을수록 좋은 말 모델이다
    """
    
    @staticmethod
    def compute(log_probs: torch.Tensor) -> float:
        """
        로그 확률에서 헷갈림도를 셈한다.
        
        인수:
            log_probs: 토큰마다의 로그 확률 [n_tokens]
        
        반환값:
            헷갈림도 값
        
        수학의 걸음:
        ------------------
        1. 로그 확률의 평균을 셈한다: (1/N) Σ log p(x_i)
        2. 음수를 취해 음의 로그 가능도를 얻는다: -(1/N) Σ log p(x_i)
        3. 지수를 취한다: exp(NLL)
        """
        # 평균 로그 확률을 셈한다
        mean_log_prob = torch.mean(log_probs)
        
        # 음수에 지수를 취한다: exp(-mean_log_prob)
        perplexity = torch.exp(-mean_log_prob)
        
        return perplexity.item()
    
    @staticmethod
    def compute_per_token(log_probs: torch.Tensor) -> torch.Tensor:
        """
        토큰마다의 헷갈림도를 셈한다(살피는 데 쓸모 있다).
        
        인수:
            log_probs: 로그 확률 [n_tokens]
        
        반환값:
            토큰마다의 헷갈림도 [n_tokens]
        """
        # 토큰마다: exp(-log_prob)
        per_token_perplexity = torch.exp(-log_probs)
        
        return per_token_perplexity
    
    @staticmethod
    def interpret_perplexity(perplexity: float, vocab_size: int) -> str:
        """
        헷갈림도 값을 풀이한다.
        
        인수:
            perplexity: 헷갈림도 값
            vocab_size: 낱말 곳간의 크기
        
        반환값:
            풀이 글자열
        """
        # 아무 바탕과 견준다
        random_perplexity = vocab_size
        
        if perplexity >= random_perplexity * 0.9:
            quality = "Very Poor (nearly random)"
        elif perplexity >= vocab_size * 0.5:
            quality = "Poor"
        elif perplexity >= vocab_size * 0.2:
            quality = "Moderate"
        elif perplexity >= vocab_size * 0.05:
            quality = "Good"
        else:
            quality = "Excellent"
        
        return f"{quality} (Random baseline: {random_perplexity})"


def demonstrate_nll_computation():
    """
    구체적인 보기로 음의 로그 가능도 셈하기를 보인다.
    """
    print("=" * 70)
    print("Negative Log-Likelihood (NLL) Demonstration")
    print("=" * 70)
    
    # 보기 1: 흠 없는 모델(실제로는 있을 수 없다)
    print("\nExample 1: Perfect Model")
    print("-" * 70)
    # 모델이 모든 시험 표본에 확률 1.0을 매기면
    # log(1.0) = 0.0이므로 음의 로그 가능도 = 0.0
    perfect_log_probs = torch.zeros(100)
    nll_perfect = NegativeLogLikelihood.compute(perfect_log_probs)
    print(f"Log probabilities: all 0.0 (prob = 1.0)")
    print(f"NLL: {nll_perfect:.6f}")
    print("Interpretation: Model assigns probability 1 to all samples")
    print("               (Only possible if model memorizes all data)")
    
    # 보기 2: 좋은 모델
    print("\n Example 2: Good Model")
    print("-" * 70)
    # 로그 확률이 -2.0 언저리(확률 ≈ 0.135)
    good_log_probs = torch.randn(100) * 0.5 - 2.0
    nll_good, se_good = NegativeLogLikelihood.compute_with_variance(good_log_probs)
    print(f"Mean log probability: {good_log_probs.mean():.4f}")
    print(f"NLL: {nll_good:.4f} ± {se_good:.4f}")
    print(f"Interpretation: Model assigns average probability {np.exp(-nll_good):.4f}")
    
    # 보기 3: 나쁜 모델
    print("\nExample 3: Poor Model")
    print("-" * 70)
    # 로그 확률이 -10.0 언저리(확률 ≈ 0.000045)
    poor_log_probs = torch.randn(100) * 1.0 - 10.0
    nll_poor = NegativeLogLikelihood.compute(poor_log_probs)
    print(f"Mean log probability: {poor_log_probs.mean():.4f}")
    print(f"NLL: {nll_poor:.4f}")
    print(f"Interpretation: Model assigns average probability {np.exp(-nll_poor):.6f}")
    print("               (Very low probability = poor model)")
    
    # 비교
    print("\n" + "=" * 70)
    print("Model Comparison:")
    print("=" * 70)
    print(f"{'Model':<20} {'NLL':<15} {'Avg Probability'}")
    print("-" * 70)
    print(f"{'Perfect':<20} {nll_perfect:<15.4f} {np.exp(-nll_perfect):.6f}")
    print(f"{'Good':<20} {nll_good:<15.4f} {np.exp(-nll_good):.6f}")
    print(f"{'Poor':<20} {nll_poor:<15.4f} {np.exp(-nll_poor):.6f}")
    print("\nLower NLL = Better model")


def demonstrate_bpd_computation():
    """
    차원마다 비트 셈하기와 여러 자료 차원에 걸친 견줌을 보인다.
    """
    print("\n" + "=" * 70)
    print("Bits Per Dimension (BPD) Demonstration")
    print("=" * 70)
    
    # 상황: 그림 크기가 다른 모델을 견준다
    # 모델 A: MNIST(28×28 = 784차원)
    # 모델 B: CIFAR(32×32×3 = 3072차원)
    
    # 두 모델의 표본마다 음의 로그 가능도가 비슷하다
    nll_mnist = 100.0  # 임의 단위
    nll_cifar = 380.0  # 임의 단위
    
    dim_mnist = 28 * 28
    dim_cifar = 32 * 32 * 3
    
    print(f"\nModel A (MNIST):")
    print(f"  Dimensions: {dim_mnist}")
    print(f"  NLL: {nll_mnist:.2f}")
    
    print(f"\nModel B (CIFAR-10):")
    print(f"  Dimensions: {dim_cifar}")
    print(f"  NLL: {nll_cifar:.2f}")
    
    print("\n" + "-" * 70)
    print("Problem: Cannot directly compare NLL across different dimensions!")
    print("-" * 70)
    
    # 공정한 견줌을 위해 차원마다 비트를 셈한다
    bpd_mnist = BitsPerDimension.compute(nll_mnist, dim_mnist)
    bpd_cifar = BitsPerDimension.compute(nll_cifar, dim_cifar)
    
    print(f"\nSolution: Normalize using BPD")
    print("-" * 70)
    print(f"Model A (MNIST):")
    print(f"  BPD: {bpd_mnist:.4f}")
    print(f"  Quality: {BitsPerDimension.interpret_bpd(bpd_mnist, 'image')}")
    
    print(f"\nModel B (CIFAR-10):")
    print(f"  BPD: {bpd_cifar:.4f}")
    print(f"  Quality: {BitsPerDimension.interpret_bpd(bpd_cifar, 'image')}")
    
    # 앎 이론의 풀이
    print("\n" + "=" * 70)
    print("Information-Theoretic Interpretation:")
    print("=" * 70)
    print(f"\nFor 8-bit images, uniform distribution gives BPD = 8.0")
    print(f"(Each pixel can be one of 256 values, requiring 8 bits)")
    print(f"\nModel A achieves {bpd_mnist:.2f} BPD:")
    print(f"  Compression: {(1 - bpd_mnist/8.0)*100:.1f}% compared to uniform")
    print(f"\nModel B achieves {bpd_cifar:.2f} BPD:")
    print(f"  Compression: {(1 - bpd_cifar/8.0)*100:.1f}% compared to uniform")


def demonstrate_perplexity_computation():
    """
    말 모델의 헷갈림도 셈하기를 보인다.
    """
    print("\n" + "=" * 70)
    print("Perplexity Demonstration")
    print("=" * 70)
    
    # 상황: 낱말 수가 10000인 말 모델
    vocab_size = 10000
    
    print(f"\nLanguage Model with vocabulary size: {vocab_size}")
    print("-" * 70)
    
    # 모델 1: 아무 바탕
    # 토큰마다 가능성이 같다: p = 1/vocab_size
    # log p = log(1/vocab_size) = -log(vocab_size)
    random_log_prob = np.log(1.0 / vocab_size)
    random_log_probs = torch.full((1000,), random_log_prob)
    ppl_random = Perplexity.compute(random_log_probs)
    
    print(f"\nModel 1: Random Baseline")
    print(f"  Log prob per token: {random_log_prob:.4f}")
    print(f"  Perplexity: {ppl_random:.1f}")
    print(f"  Interpretation: {Perplexity.interpret_perplexity(ppl_random, vocab_size)}")
    
    # 모델 2: 보통 모델
    # 평균 확률 ~0.01(토큰마다 1%)
    moderate_log_probs = torch.randn(1000) * 0.5 + np.log(0.01)
    ppl_moderate = Perplexity.compute(moderate_log_probs)
    nll_moderate = -moderate_log_probs.mean().item()
    
    print(f"\nModel 2: Moderate Model")
    print(f"  Average log prob: {moderate_log_probs.mean():.4f}")
    print(f"  NLL: {nll_moderate:.4f}")
    print(f"  Perplexity: {ppl_moderate:.1f}")
    print(f"  Interpretation: {Perplexity.interpret_perplexity(ppl_moderate, vocab_size)}")
    
    # 모델 3: 좋은 모델
    # 평균 확률 ~0.2(토큰마다 20%)
    good_log_probs = torch.randn(1000) * 0.3 + np.log(0.2)
    ppl_good = Perplexity.compute(good_log_probs)
    nll_good = -good_log_probs.mean().item()
    
    print(f"\nModel 3: Good Model")
    print(f"  Average log prob: {good_log_probs.mean():.4f}")
    print(f"  NLL: {nll_good:.4f}")
    print(f"  Perplexity: {ppl_good:.1f}")
    print(f"  Interpretation: {Perplexity.interpret_perplexity(ppl_good, vocab_size)}")
    
    # 비교
    print("\n" + "=" * 70)
    print("Perplexity Comparison:")
    print("=" * 70)
    print(f"{'Model':<20} {'Perplexity':<15} {'Effective Choices'}")
    print("-" * 70)
    print(f"{'Random':<20} {ppl_random:<15.1f} All {vocab_size} tokens")
    print(f"{'Moderate':<20} {ppl_moderate:<15.1f} ~{int(ppl_moderate)} likely tokens")
    print(f"{'Good':<20} {ppl_good:<15.1f} ~{int(ppl_good)} likely tokens")
    print("\nLower perplexity = Better language model")
    print("Perplexity ≈ effective vocabulary size at each position")


def visualize_likelihood_metrics():
    """
    가능도 잣대의 그림을 만든다.
    """
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    # 모델 품질별 인공 로그 확률을 만든다
    n_samples = 1000
    
    # 나쁜 모델: 평균 로그 확률 = -10
    poor_log_probs = torch.randn(n_samples) * 2.0 - 10.0
    
    # 보통 모델: 평균 로그 확률 = -5
    moderate_log_probs = torch.randn(n_samples) * 1.5 - 5.0
    
    # 좋은 모델: 평균 로그 확률 = -2
    good_log_probs = torch.randn(n_samples) * 1.0 - 2.0
    
    # 지표를 계산한다
    models = ['Poor', 'Moderate', 'Good']
    log_probs_list = [poor_log_probs, moderate_log_probs, good_log_probs]
    
    nlls = []
    bpds = []
    ppls = []
    
    dimensions = 784  # MNIST 크기
    
    for log_probs in log_probs_list:
        nll = NegativeLogLikelihood.compute(log_probs)
        bpd = BitsPerDimension.compute(nll, dimensions)
        ppl = Perplexity.compute(log_probs)
        
        nlls.append(nll)
        bpds.append(bpd)
        ppls.append(ppl)
    
    # 시각화 만들기
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 그림 1: 로그 확률 분포
    ax = axes[0, 0]
    for i, (model, log_probs) in enumerate(zip(models, log_probs_list)):
        ax.hist(log_probs.numpy(), bins=50, alpha=0.6, label=model,
                density=True)
    ax.set_xlabel('Log Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Log Probability Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (log p = 0)')
    
    # 그림 2: 음의 로그 가능도 견줌
    ax = axes[0, 1]
    bars = ax.bar(models, nlls, color=['#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax.set_title('NLL Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 막대에 값 이름표를 추가한다
    for bar, nll in zip(bars, nlls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{nll:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 그림 3: 차원마다 비트 견줌
    ax = axes[1, 0]
    bars = ax.bar(models, bpds, color=['#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Bits Per Dimension', fontsize=12)
    ax.set_title('BPD Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.axhline(8.0, color='red', linestyle='--', linewidth=2, label='Random (8 bits)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 값 이름표를 추가한다
    for bar, bpd in zip(bars, bpds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{bpd:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 그림 4: 간추린 표
    ax = axes[1, 1]
    ax.axis('off')
    
    # 간추린 표를 만든다
    table_data = [
        ['Model', 'NLL', 'BPD', 'Perplexity'],
        ['', '', '', ''],
    ]
    
    for i, model in enumerate(models):
        table_data.append([
            model,
            f'{nlls[i]:.2f}',
            f'{bpds[i]:.3f}',
            f'{ppls[i]:.1f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 표를 꾸민다
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # 머리글
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1:  # 가름줄
                cell.set_facecolor('#ecf0f1')
            else:
                if table_data[i][0] == 'Good':
                    cell.set_facecolor('#d5f4e6')
                elif table_data[i][0] == 'Moderate':
                    cell.set_facecolor('#fef5e7')
                else:
                    cell.set_facecolor('#fadbd8')
    
    ax.set_title('Likelihood Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('likelihood_metrics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'likelihood_metrics.png'")
    
    return fig


def main():
    """
    가능도 잣대를 보이는 으뜸 함수.
    """
    print("\n" + "=" * 70)
    print("MODULE 52.02: LIKELIHOOD-BASED METRICS")
    print("=" * 70)
    
    # 음의 로그 가능도를 보여 준다
    demonstrate_nll_computation()
    
    # 차원마다 비트를 보여 준다
    demonstrate_bpd_computation()
    
    # 헷갈림도를 보여 준다
    demonstrate_perplexity_computation()
    
    # 시각화 만들기
    visualize_likelihood_metrics()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 음의 로그 가능도(NLL):
       - 모델이 자료에 얼마나 잘 확률을 매기는지 잰다
       - 음의 로그 가능도가 낮을수록 잘 맞는다
       - 익히기 로스 함수로 쓴다
       - 표준 오차로 믿음 구간을 셈할 수 있다
    
    2. 차원마다의 비트 수(BPD):
       - 차원에 걸친 공정한 견줌을 위해 음의 로그 가능도를 고르게 맞춘다
       - 앎 이론의 풀이
       - BPD = NLL / (차원 수 × log(2))
       - 견줌을 가능하게 한다: MNIST와 CIFAR와 ImageNet
    
    3. 헷갈림도:
       - 말 모델에 맞춘 잣대
       - 헷갈림도 = exp(토큰마다의 음의 로그 가능도)
       - 직관: "실제로 쓰이는 낱말 수"
       - 헷갈림도가 낮을수록 헤아림이 자신 있다
    
    4. 잣대 고르기:
       - 변분 오토인코더, 흐름: 음의 로그 가능도나 차원마다 비트를 쓴다
       - 말 모델: 헷갈림도를 쓴다
       - 그림 모델: 공정한 견줌에는 차원마다 비트를 쓴다
       - 늘 믿음 구간을 알린다
    
    5. 한계:
       - 높은 가능도 ≠ 좋은 표본
       - 서로 메우는 표본 바탕 잣대가 필요하다
       - 모델 담이에 민감하다
       - 사람의 느낌과 이어지지 않을 수 있다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (155줄)"

    ```

    ======================================================================
    MODULE 52.02: LIKELIHOOD-BASED METRICS
    ======================================================================
    ======================================================================
    Negative Log-Likelihood (NLL) Demonstration
    ======================================================================

    Example 1: Perfect Model
    ----------------------------------------------------------------------
    Log probabilities: all 0.0 (prob = 1.0)
    NLL: -0.000000
    Interpretation: Model assigns probability 1 to all samples
                   (Only possible if model memorizes all data)

     Example 2: Good Model
    ----------------------------------------------------------------------
    Mean log probability: -1.9701
    NLL: 1.9701 ± 0.0493
    Interpretation: Model assigns average probability 0.1394

    Example 3: Poor Model
    ----------------------------------------------------------------------
    Mean log probability: -9.9636
    NLL: 9.9636
    Interpretation: Model assigns average probability 0.000047
                   (Very low probability = poor model)

    ======================================================================
    Model Comparison:
    ======================================================================
    Model                NLL             Avg Probability
    ----------------------------------------------------------------------
    Perfect              -0.0000         1.000000
    Good                 1.9701          0.139441
    Poor                 9.9636          0.000047

    Lower NLL = Better model

    ======================================================================
    Bits Per Dimension (BPD) Demonstration
    ======================================================================

    Model A (MNIST):
      Dimensions: 784
      NLL: 100.00

    Model B (CIFAR-10):
      Dimensions: 3072
      NLL: 380.00

    ----------------------------------------------------------------------
    Problem: Cannot directly compare NLL across different dimensions!
    ----------------------------------------------------------------------

    Solution: Normalize using BPD
    ----------------------------------------------------------------------
    Model A (MNIST):
      BPD: 0.1840
      Quality: Excellent

    Model B (CIFAR-10):
      BPD: 0.1785
      Quality: Excellent

    ======================================================================
    Information-Theoretic Interpretation:
    ======================================================================

    For 8-bit images, uniform distribution gives BPD = 8.0
    (Each pixel can be one of 256 values, requiring 8 bits)

    Model A achieves 0.18 BPD:
      Compression: 97.7% compared to uniform

    Model B achieves 0.18 BPD:
      Compression: 97.8% compared to uniform

    ======================================================================
    Perplexity Demonstration
    ======================================================================

    Language Model with vocabulary size: 10000
    ----------------------------------------------------------------------

    Model 1: Random Baseline
      Log prob per token: -9.2103
      Perplexity: 10000.0
      Interpretation: Very Poor (nearly random) (Random baseline: 10000)

    Model 2: Moderate Model
      Average log prob: -4.5900
      NLL: 4.5900
      Perplexity: 98.5
      Interpretation: Excellent (Random baseline: 10000)

    Model 3: Good Model
      Average log prob: -1.6211
      NLL: 1.6211
      Perplexity: 5.1
      Interpretation: Excellent (Random baseline: 10000)

    ======================================================================
    Perplexity Comparison:
    ======================================================================
    Model                Perplexity      Effective Choices
    ----------------------------------------------------------------------
    Random               10000.0         All 10000 tokens
    Moderate             98.5            ~98 likely tokens
    Good                 5.1             ~5 likely tokens

    Lower perplexity = Better language model
    Perplexity ≈ effective vocabulary size at each position

    ======================================================================
    Generating Visualizations
    ======================================================================

    ✓ Visualization saved as 'likelihood_metrics.png'

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

        1. 음의 로그 가능도(NLL):
           - 모델이 자료에 얼마나 잘 확률을 매기는지 잰다
           - 음의 로그 가능도가 낮을수록 잘 맞는다
           - 익히기 로스 함수로 쓴다
           - 표준 오차로 믿음 구간을 셈할 수 있다
        
        2. 차원마다의 비트 수(BPD):
           - 차원에 걸친 공정한 견줌을 위해 음의 로그 가능도를 고르게 맞춘다
           - 앎 이론의 풀이
           - BPD = NLL / (차원 수 × log(2))
           - 견줌을 가능하게 한다: MNIST와 CIFAR와 ImageNet
        
        3. 헷갈림도:
           - 말 모델에 맞춘 잣대
           - 헷갈림도 = exp(토큰마다의 음의 로그 가능도)
           - 직관: "실제로 쓰이는 낱말 수"
           - 헷갈림도가 낮을수록 헤아림이 자신 있다
        
        4. 잣대 고르기:
           - 변분 오토인코더, 흐름: 음의 로그 가능도나 차원마다 비트를 쓴다
           - 말 모델: 헷갈림도를 쓴다
           - 그림 모델: 공정한 견줌에는 차원마다 비트를 쓴다
           - 늘 믿음 구간을 알린다
        
        5. 한계:
           - 높은 가능도 ≠ 좋은 표본
           - 서로 메우는 표본 바탕 잣대가 필요하다
           - 모델 담이에 민감하다
           - 사람의 느낌과 이어지지 않을 수 있다
        
    ======================================================================
    ```

## 2. NLL: 표본과 차원의 집계

평가 데이터 $x_1,\ldots,x_N$의 평균 NLL은 다음과 같다.

$$\widehat{\mathrm{NLL}}=-\frac1N\sum_{i=1}^N\log p_\theta(x_i).$$

`NegativeLogLikelihood.compute`에는 표본별 로그 확률을 담은 길이 N의 벡터를 전달한다고 가정한다. 이미지의 픽셀별 로그 확률이라면 먼저 한 이미지 안에서 합산해야 한다. N×D 배열을 그대로 넣으면 모든 원소를 평균 내므로 이미지당 NLL보다 D배 작아진다. 같은 평가 데이터와 같은 확률 정의에서 NLL이 작을수록 평균 로그 우도가 높다.

`compute_with_variance`의 두 번째 반환값은 이름과 달리 분산이 아니라 표준오차 추정값 $s/\sqrt N$이다. 기본 표본 표준편차 계산에는 N≥2가 필요하며, 이 표준오차는 독립 표본을 가정한다. 시계열처럼 관측이 상관되어 있으면 이 식만으로 불확실성을 평가하기 어렵다. 정규 근사가 적절할 때 약 95% 신뢰구간은 NLL±1.96×표준오차로 계산하며, 출력의 ±값 자체를 95% 신뢰구간으로 읽지 않는다.

출력의 `Avg Probability`는 $\exp(-\mathrm{NLL})$이므로 확률의 **기하평균**이다. 산술평균과 다르다. 또한 서로 다른 이산 결과 모두에 확률 1을 부여하는 정규화된 분포는 존재하지 않는다. `perfect_log_probs=0`은 계산상의 예시이며, 여러 데이터를 암기하면 모두 확률 1이 된다는 출력 설명은 잘못이다. 연속 확률 밀도는 1보다 클 수 있어 로그 밀도가 양수이거나 NLL이 음수여도 그 자체로 오류는 아니다.

## 3. 차원당 비트 수

자연로그로 계산한 표본당 NLL과 표본의 스칼라 차원 D를 사용하면 다음과 같다.

$$\mathrm{bpd}=\frac{\mathrm{NLL}}{D\log 2}.$$

흑백 28×28 이미지는 D=784, RGB 32×32 이미지는 D=3072다. 이 정규화는 단위를 맞추지만 서로 다른 데이터셋의 난이도까지 같게 만들지는 않는다. 예제의 NLL 100과 380은 실제 학습 결과가 아니며, 이를 나눈 값만으로 MNIST 모델과 CIFAR-10 모델의 우열을 판단할 수 없다.

이산 확률 질량과 연속 밀도를 구분해야 한다. 8비트 정수 k를 $(k+u)/256$, $u\sim U[0,1)$로 디퀀타이즈한 연속 모델에서는, 통상적인 이산 부호 길이 상한을 보고할 때 연속 NLL/(D log 2)에 **8을 더한다**. 이 보정은 해당 변환과 양자화 구간을 전제로 한다. 이진 픽셀의 Bernoulli 우도에는 같은 +8을 적용하지 않는다. 데이터의 스케일을 바꾸었다면 밀도의 야코비안 보정도 필요하다.

독립적인 8비트 값에 대한 균등 이산 분포의 기준은 8 bpd다. 이는 모든 이미지 표현에 공통인 기준이 아니다. `interpret_bpd`의 고정된 품질 구간은 검증된 보편적 판정 기준이 아니며, 예제의 압축률 출력도 실제 파일을 압축한 결과가 아니다. 실제 부호 길이는 확률 모델, 부호화 방식, 모델 전달 비용 등에 달려 있다.

## 4. 퍼플렉서티

유효한 정답 토큰 T개에 대해 언어 모델의 퍼플렉서티는 다음과 같다.

$$\mathrm{PPL}=\exp\!\left[-\frac1T\sum_{t=1}^T\log p_\theta(x_t\mid x_{<t})\right].$$

패딩은 제외하고, 여러 배치나 문서를 합칠 때는 NLL의 합을 유효 토큰 수의 합으로 나눈다. 길이가 다른 문서의 평균 NLL을 다시 단순 평균하면 다른 지표가 된다. 토크나이저, 어휘, 평가 문맥, 데이터가 같을 때 비교해야 한다.

어휘 크기 V의 균등 모델은 PPL=V다. 그러나 정답에 균등 분포보다 낮은 확률을 주는 모델은 PPL>V가 될 수 있다. PPL을 해당 위치에서 실제로 가능한 단어의 개수라고 해석하지 않는다. `compute_per_token`은 각 정답 확률의 역수 $1/p_t$를 반환하며, 이 값들의 산술평균은 전체 퍼플렉서티와 다르다. 유효한 이산 확률에서는 PPL≥1이지만 연속 밀도를 지수화한 값에 같은 해석을 적용할 수는 없다.

`interpret_perplexity`의 어휘 크기 대비 고정 임계값도 일반적인 품질 판정 기준이 아니다. 시각화는 임의의 로그 확률 배열을 다른 방식으로 요약한 그림이다. 이미지별 로그 밀도를 지수화한 값을 언어 모델의 토큰 퍼플렉서티와 같은 의미로 해석하거나, 세 그래프가 독립적인 품질 증거라고 보아서는 안 된다.

## 5. 생성 모델 비교에서의 범위

일반적인 GAN은 샘플을 생성할 수 있지만 관측 데이터의 우도를 직접 계산하기 어렵다. VAE는 흔히 ELBO나 중요도 가중 추정값을 보고하고, 정규화 흐름은 가역 변환 조건 아래에서 밀도를 계산한다. 정확한 우도와 하한, 몬테카를로 추정값을 구분해 기록해야 한다.

우도는 데이터 분포 적합도의 한 측면이다. 표본의 시각적 품질, 다양성, 암기 여부, 특정 작업의 유용성까지 하나의 수치로 보장하지 않는다. 샘플 기반 평가와 함께 사용하되 각 지표의 데이터 표현과 평가 절차를 명시한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
길이 N의 표본별 로그 확률과 N×D의 좌표별 로그 확률을 이 코드에 전달하면 NLL 계산이 어떻게 달라지는가?

</div>

??? success "연습문제 1 풀이"

    길이 N이면 표본별 로그 확률을 평균 낸다. 좌표별 조건부 로그 확률의 합이 표본 로그 확률인 경우에는 D축을 먼저 합산해야 한다. N×D를 그대로 전달하면 전체 원소를 평균 내므로 표본당 NLL의 1/D가 나온다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
`compute_with_variance`의 두 번째 반환값은 무엇인가? 이를 95% 신뢰구간과 구분하라.

</div>

??? success "연습문제 2 풀이"

    표본 표준편차를 표본 수의 제곱근으로 나눈 표준오차 추정값이다. 분산이나 95% 신뢰구간의 반폭이 아니다. 독립 표본과 적절한 정규 근사 아래에서는 약 95% 구간을 추정값±1.96×표준오차로 계산한다. 표본 간 상관이나 매우 작은 표본 수에서는 별도 검토가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
정답 토큰별 확률이 1/2와 1/8일 때 퍼플렉서티를 구하라. 토큰별 역확률의 산술평균과 같은가?

</div>

??? success "연습문제 3 풀이"

    평균 NLL은 $-(\log(1/2)+\log(1/8))/2=\log4$이므로 PPL=4다. 역확률의 산술평균은 (2+8)/2=5이므로 다르다. 전체 PPL은 정답 역확률의 기하평균이다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
일반적인 GAN에서 우도를 직접 계산하기 어려운 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    생성기와 잠재 분포는 출력 확률분포를 정의한다. 다만 일반적인 GAN에는 관측값의 확률 밀도를 평가하는 식이나 효율적인 알고리즘이 제공되지 않는다. 매끄러운 저차원 잠재 변환의 분포는 주변 공간의 르베그 측도에 대한 밀도가 없을 수도 있다. 이는 분포 자체가 정의되지 않는다는 뜻과 다르다. 가역 생성기 등 특별한 구조에서는 밀도 계산이 가능할 수 있다. 커널 밀도 추정은 생성 표본을 평활화한 다른 밀도를 추정하며, 원래 GAN 우도의 정확한 계산이 아니다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
ELBO를 정확한 로그 우도라고 보고하면 안 되는 이유는 무엇인가? β-VAE 목적함수와도 구분하라.

</div>

??? success "연습문제 5 풀이"

    ELBO는 $\log p_\theta(x)$의 하한이며 그 차이는 $D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x))$이다. 모델을 고정하고 변분 분포를 개선해 하한이 높아져도 모델 우도가 개선된 것은 아니다. 중요도 가중 하한은 표본 수 K와 추정 절차를 명시한다. β≥1인 표준 β-VAE 목적함수는 ELBO에서 비음수인 (β−1)KL을 더 빼므로 여전히 하한이지만 표준 ELBO와 다르다. 0≤β<1에서는 일반적인 하한 보장이 없다. 유한 표본 추정값과 기대값으로 정의된 하한도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
높은 평가 우도가 좋은 시각적 표본을 보장하는가?

</div>

??? success "연습문제 6 풀이"

    보장하지 않는다. 제한된 모델 계열에서 밀도 적합도와 사람이 판단하는 품질이 다르게 움직일 수 있다. VAE의 흐릿한 출력은 디코더 관측 모형, 표현력, 근사 추론 등의 영향을 받으며 우도 최대화가 언제나 흐릿함을 만든다는 뜻은 아니다. 이 페이지에는 학습된 VAE와 GAN을 같은 조건에서 비교한 실험이 없으므로 우열을 나타내는 수치를 보고할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
이진 MNIST의 Bernoulli NLL과 8비트 디퀀타이즈 이미지의 연속 NLL을 bpd로 보고할 때 무엇이 다른가?

</div>

??? success "연습문제 7 풀이"

    둘 다 표본 안에서 로그 확률을 합하고 표본 사이에서 평균 낸 뒤 D log 2로 나눈다. 이진 Bernoulli 값은 이산 확률 질량이므로 +8 보정이 없다. 8비트 값을 (k+u)/256으로 변환한 연속 모델의 이산 부호 길이 상한은 통상 연속 bpd에 8을 더해 보고한다. 양자화, 디퀀타이즈 방식, 스케일과 단위를 명시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
표준 정규분포에서 원점의 밀도가 가장 높다는 사실이 원점이 전형적인 표본이라는 뜻인가?

</div>

??? success "연습문제 8 풀이"

    아니다. 고차원 표준 정규분포의 표본 반지름은 대략 √D 주변에 집중한다. 원점은 점별 밀도가 높아도 그 주변의 작은 영역이 차지하는 확률 질량은 작다. 따라서 낮은 우도만으로 모든 이상 데이터를 탐지할 수 있다고 기대하면 안 된다. 전형성 검사, 배경 모델에 대한 우도비, 별도 검증된 이상 탐지 점수 등을 고려할 수 있으나 어느 방법도 모든 분포에서 성공하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
VAE와 확산 모델의 우도 관련 수치를 비교할 때 어떤 조건을 기록해야 하는가?

</div>

??? success "연습문제 9 풀이"

    같은 평가 데이터와 전처리, 이산·연속 표현, 차원 및 로그 단위를 맞춘다. 정확한 우도인지 변분 하한인지, 사용한 중요도 표본 수와 추정 오차도 기록한다. 서로 다른 하한의 크기만으로 실제 우도의 순서를 확정할 수는 없다. 모델마다 계산법이 달라도 같은 대상을 정확하게 추정한다면 비교할 수 있으므로 동일 알고리즘만 허용되는 것은 아니다. 확산 모델에서도 변분 경계와 다른 우도 평가 절차를 구분한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
우도를 직접 계산하기 어려운 모델에는 어떤 보완 평가를 사용할 수 있는가?

</div>

??? success "연습문제 10 풀이"

    FID는 피처의 평균·공분산 차이를, 인셉션 점수는 분류기 예측의 확신과 주변 클래스 분포를 요약한다. 정밀도·재현율 계열은 피처 공간에서 생성 품질과 분포 포괄성을 나누어 본다. 표본 격자, 사람 평가, 암기·중복 검사, 다운스트림 작업 성능도 목적에 맞게 사용할 수 있다. 생성 데이터로 학습하고 실제 테스트 데이터로 평가하는 방법도 작업·학습기·데이터 누출에 영향을 받으므로 유용성과 다양성을 완전히 증명하지는 않는다.



## 정리하며

NLL, bpd, 퍼플렉서티는 같은 로그 확률을 서로 다른 단위와 집계 방식으로 표현할 수 있다. 비교 전에 표본·좌표·토큰의 축, 이산 확률과 연속 밀도, 전처리와 하한 여부를 확인한다. 이 페이지의 임의 배열과 고정 품질 구간은 실제 생성 모델 평가 결과로 해석하지 않는다.
