# 단원 52.01: 가능도에 바탕한 잣대

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 단원 52.01: 가능도에 바탕한 잣대을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 1. 코드

```python
"""
단원 52.01: 가능도에 바탕한 잣대
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
       - 서로 다른 자료 갈래
    
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
            data_type: 자료의 갈래(그림, 글월 따위)
        
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
    plt.savefig('/home/claude/likelihood_metrics.png', dpi=300, bbox_inches='tight')
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
       - 익히기 손실 함수로 쓴다
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
       - 변분 자기 부호기, 흐름: 음의 로그 가능도나 차원마다 비트를 쓴다
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

## 2. 논의

이 짜기는 단원 52.01: 가능도에 바탕한 잣대에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 연습문제

**연습문제 1.**
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

??? success "연습문제 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

**연습문제 2.**
이 짜기의 핵심 웃매개변수(배움 빠르기, 묶음 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

??? success "연습문제 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 묶음 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 묶음 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

**연습문제 3.**
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

??? success "연습문제 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.

## 정리하며

**다룬 것** — 단원 52.01: 가능도에 바탕한 잣대

이 짜기는 단원 52.01: 가능도에 바탕한 잣대에 대해 자리 잡은 가장 좋은 방식을 따른다.

고갱이 갈래는 `NegativeLogLikelihood`, `BitsPerDimension`, `Perplexity`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
