# 단원 52.01: 만들어 내는 모델의 따지기 바탕

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 단원 52.01: 만들어 내는 모델의 따지기 바탕을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 1. 코드

```python
"""
단원 52.01: 만들어 내는 모델의 따지기 바탕
=====================================================

이 단원은 만들어 내는 모델을 따지는 바탕 개념을 들여온다.
따지기가 왜 중요한지, 여러 따지기 틀, 기본 가능도 셈하기를 살핀다.
가능도 셈하기.

학습 목표:
-------------------
1. 만들어 내는 모델을 따지는 어려움을 이해한다
2. 가능도와 표본 품질을 가려낸다
3. 기본 가능도 셈하기를 짠다
4. 따지기의 맞바꿈을 안다

핵심 개념:
------------
- 가능도에 바탕한 따지기
- 표본에 바탕한 따지기
- 품질과 다양함의 맞바꿈
- 만들어 내는 모델의 지나치게 맞춰짐

지은이: 가르치기 인공 지능 모둠
날짜: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List

# ========================================================================
# 메인
# ========================================================================

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)


class EvaluationParadigms:
    """
    만들어 내는 모델을 따지는 여러 틀을 보인다.
    
    으뜸 틀 셋은 다음과 같다.
    1. 가능도 바탕: 모델이 실제 자료에 얼마나 잘 확률을 매기는가?
    2. 표본 바탕: 만든 표본이 얼마나 좋아 보이는가?
    3. 과제 바탕: 표본이 뒤따르는 일에서 얼마나 잘하는가?
    """
    
    @staticmethod
    def likelihood_evaluation(model, test_data: torch.Tensor) -> float:
        """
        가능도로 만들어 내는 모델을 따진다.
        
        수학적 바탕:
        ----------------------
        만들어 내는 모델 p_θ(x)에 대해 다음을 따진다.
            L = E_{x~p_data}[log p_θ(x)]
        
        가능도가 클수록 모델이 실제 자료 분포에
        더 높은 확률을 매긴다는 뜻이다.
        
        인수:
            model: log_prob 방법이 있는 만들어 내는 모델
            test_data: 참 자료 표본 [batch_size, features]
        
        반환값:
            평균 로그 가능도
        """
        # 시험 표본마다 로그 확률을 셈한다
        # 꼴: [묶음 크기]
        log_probs = model.log_prob(test_data)
        
        # 평균 로그 가능도를 돌려준다
        # 이는 E_{x~p_data}[log p_θ(x)]의 겪어 본 어림이다
        avg_log_likelihood = torch.mean(log_probs).item()
        
        return avg_log_likelihood
    
    @staticmethod
    def sample_quality_evaluation(generated_samples: torch.Tensor,
                                  real_samples: torch.Tensor) -> dict:
        """
        실제 자료와 통계를 견주어 만든 표본을 따진다.
        
        이는 표본에 바탕한 따지기를 단순하게 만든 판이다.
        다음을 견준다.
        1. 평균과 표준 편차
        2. 분포의 모양(히스토그램을 견주어 본다)
        
        인수:
            generated_samples: 모델에서 뽑은 표본 [n_samples, features]
            real_samples: 참 자료 표본 [n_samples, features]
        
        반환값:
            따지기 잣대를 담은 사전
        """
        metrics = {}
        
        # 1. 일차 통계(평균)를 견준다
        mean_diff = torch.mean(torch.abs(
            generated_samples.mean(dim=0) - real_samples.mean(dim=0)
        ))
        metrics['mean_absolute_error'] = mean_diff.item()
        
        # 2. 이차 통계(표준 편차)를 견준다
        std_diff = torch.mean(torch.abs(
            generated_samples.std(dim=0) - real_samples.std(dim=0)
        ))
        metrics['std_absolute_error'] = std_diff.item()
        
        # 3. 해당되면 서로 이어짐을 셈한다(여러 변수 자료)
        if generated_samples.shape[1] > 1:
            # 서로 이어짐 행렬을 셈한다
            gen_corr = torch.corrcoef(generated_samples.T)
            real_corr = torch.corrcoef(real_samples.T)
            
            # 서로 이어짐 짜임을 견준다
            corr_diff = torch.mean(torch.abs(gen_corr - real_corr))
            metrics['correlation_error'] = corr_diff.item()
        
        return metrics


class SimpleGaussianModel:
    """
    보여 주기를 위한 단순한 정규 분포 만들어 내는 모델.
    
    수학의 뜻매김:
    ----------------------
    p_θ(x) = N(x | μ_θ, σ²_θ)
    
    여기서 θ = {μ_θ, σ_θ}은 배울 수 있는 매개변수다.
    
    이것이 가르치는 보기가 되는 까닭은 이렇다.
    1. 정확한 가능도를 셈할 수 있다
    2. 정확한 표본을 만들 수 있다
    3. 가능도와 표본 품질의 맞바꿈을 보여 준다
    """
    
    def __init__(self, dim: int = 1):
        """
        여러 변수 정규 분포 모델을 첫자리매김한다.
        
        인수:
            dim: 자료의 차원
        """
        # 평균과 로그 표준 편차를 첫자리매김한다
        # 수치의 안정과 양수임을 보장하려 로그 표준 편차를 쓴다
        self.mu = nn.Parameter(torch.randn(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        정규 분포 모델에서 로그 확률을 셈한다.
        
        수학의 이끌어 내기:
        ----------------------
        log N(x | μ, σ²) = -0.5 * [(x-μ)/σ]² - log(σ) - 0.5*log(2π)
        
        인수:
            x: 자료 점 [batch_size, dim]
        
        반환값:
            로그 확률 [batch_size]
        """
        # 로그 매개변수화에서 표준 편차를 얻는다
        # 이렇게 하면 σ > 0이 된다
        std = torch.exp(self.log_std)
        
        # 마할라노비스 거리의 제곱을 셈한다: [(x-μ)/σ]²
        # 꼴: [묶음 크기, 차원]
        normalized_diff = (x - self.mu) / std
        squared_distance = normalized_diff ** 2
        
        # 로그 확률의 몫을 셈한다
        # -0.5 * 제곱 거리의 합
        mahalanobis_term = -0.5 * torch.sum(squared_distance, dim=1)
        
        # 차원마다 -log(σ)
        log_normalization = -torch.sum(self.log_std)
        
        # -0.5 * 차원 * log(2π)
        constant_term = -0.5 * x.shape[1] * np.log(2 * np.pi)
        
        # 온 로그 확률
        log_prob = mahalanobis_term + log_normalization + constant_term
        
        return log_prob
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        정규 분포 모델에서 표본을 만든다.
        
        알고리즘:
        ---------
        x = μ + σ * ε, where ε ~ N(0, I)
        
        인수:
            n_samples: 만들 표본의 개수
        
        반환값:
            만들어 낸 표본 [n_samples, dim]
        """
        # 표준 편차를 얻는다
        std = torch.exp(self.log_std)
        
        # 표준 정규 분포에서 뽑는다
        # 꼴: [표본 수, 차원]
        epsilon = torch.randn(n_samples, len(self.mu))
        
        # 목표 분포로 바꾼다
        # 퍼뜨리기가 덧셈을 다룬다
        samples = self.mu + std * epsilon
        
        return samples


def demonstrate_likelihood_sample_tradeoff():
    """
    가능도와 표본 품질 사이의 바탕이 되는 맞바꿈을 보인다.
    
    핵심 통찰:
    -----------
    모델은 다음일 수 있다.
    1. 가능도는 높으나 표본 품질이 나쁘다(외워 버림)
    2. 가능도는 낮으나 표본 품질이 아주 좋다(봉우리를 놓침)
    3. 가능도도 높고 표본도 좋다(가장 바라는 경우)
    
    이 함수는 (1)의 경우를 보인다. 곧 익힘 자료를 외워 버린 모델이다
    익히기 자료에서 흠 없는 가능도를 이루지만 다양함이 나쁜 표본을 만든다.
    """
    print("=" * 70)
    print("Demonstrating Likelihood vs. Sample Quality Tradeoff")
    print("=" * 70)
    
    # 봉우리 둘인 분포를 만든다(정규 분포 둘 섞기)
    # 이는 복잡한 실제 자료 분포를 나타낸다
    n_samples = 1000
    mode1 = torch.randn(n_samples // 2, 1) - 3.0  # 왼쪽 봉우리
    mode2 = torch.randn(n_samples // 2, 1) + 3.0  # 오른쪽 봉우리
    real_data = torch.cat([mode1, mode2], dim=0)
    
    print(f"\nReal data: {n_samples} samples from bimodal distribution")
    print(f"Mode 1 centered at -3.0, Mode 2 centered at +3.0")
    
    # 모델 A: 봉우리 하나만 맞춘다(표본 품질은 높고 가능도는 낮다)
    model_a = SimpleGaussianModel(dim=1)
    model_a.mu.data = torch.tensor([-3.0])  # 왼쪽 봉우리만 담는다
    model_a.log_std.data = torch.tensor([0.0])  # log(1.0)
    
    # 모델 B: 두 봉우리를 어설프게 맞춘다(표본 품질은 낮고 가능도는 높다)
    model_b = SimpleGaussianModel(dim=1)
    model_b.mu.data = torch.tensor([0.0])  # 봉우리 사이를 가운데로
    model_b.log_std.data = torch.tensor([1.5])  # log(e^1.5) ≈ 4.48 - 아주 넓다
    
    # 가능도를 따진다
    print("\n" + "-" * 70)
    print("Likelihood Evaluation:")
    print("-" * 70)
    
    likelihood_a = EvaluationParadigms.likelihood_evaluation(model_a, real_data)
    likelihood_b = EvaluationParadigms.likelihood_evaluation(model_b, real_data)
    
    print(f"Model A (Single Mode):  Log-Likelihood = {likelihood_a:.4f}")
    print(f"Model B (Wide Gaussian): Log-Likelihood = {likelihood_b:.4f}")
    
    if likelihood_b > likelihood_a:
        print("\n⚠️  Model B has HIGHER likelihood despite missing mode structure!")
        print("    This illustrates that likelihood alone doesn't guarantee")
        print("    good sample quality or mode coverage.")
    
    # 표본을 만들어 따진다
    print("\n" + "-" * 70)
    print("Sample Quality Evaluation:")
    print("-" * 70)
    
    samples_a = model_a.sample(n_samples)
    samples_b = model_b.sample(n_samples)
    
    metrics_a = EvaluationParadigms.sample_quality_evaluation(samples_a, real_data)
    metrics_b = EvaluationParadigms.sample_quality_evaluation(samples_b, real_data)
    
    print(f"\nModel A (Single Mode):")
    print(f"  Mean Error: {metrics_a['mean_absolute_error']:.4f}")
    print(f"  Std Error:  {metrics_a['std_absolute_error']:.4f}")
    
    print(f"\nModel B (Wide Gaussian):")
    print(f"  Mean Error: {metrics_b['mean_absolute_error']:.4f}")
    print(f"  Std Error:  {metrics_b['std_absolute_error']:.4f}")
    
    # 핵심 통찰
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("Model B achieves higher likelihood by spreading probability mass")
    print("across both modes, but its samples are less realistic because they")
    print("often fall between modes where real data doesn't exist.")
    print("\nModel A captures one mode perfectly (realistic samples) but")
    print("achieves lower likelihood because it assigns zero probability to")
    print("the other mode (mode collapse).")
    print("\nThis is why we need BOTH likelihood and sample-based evaluation!")
    print("=" * 70)
    
    return {
        'real_data': real_data,
        'samples_a': samples_a,
        'samples_b': samples_b,
        'model_a': model_a,
        'model_b': model_b
    }


def basic_likelihood_computation():
    """
    여러 분포의 기본 가능도 셈하기를 보인다.
    
    다음을 견준다.
    1. 정규 분포
    2. 정규 분포 섞기
    3. 고른 분포
    
    가능도가 무엇을 재는지에 대한 직관을 쌓는 데 도움이 된다.
    """
    print("\n" + "=" * 70)
    print("Basic Likelihood Computation")
    print("=" * 70)
    
    # 시험 자료를 만든다: N(0, 1)에서 표본 100개
    test_data = torch.randn(100, 1)
    
    print(f"\nTest data: 100 samples from N(0, 1)")
    print(f"Mean: {test_data.mean():.4f}, Std: {test_data.std():.4f}")
    
    # 모델 1: 올바른 분포 N(0, 1)
    model1 = SimpleGaussianModel(dim=1)
    model1.mu.data = torch.tensor([0.0])
    model1.log_std.data = torch.tensor([0.0])  # exp(0) = 1
    
    # 모델 2: 틀린 평균 N(5, 1)
    model2 = SimpleGaussianModel(dim=1)
    model2.mu.data = torch.tensor([5.0])
    model2.log_std.data = torch.tensor([0.0])
    
    # 모델 3: 틀린 흩어짐 N(0, 5)
    model3 = SimpleGaussianModel(dim=1)
    model3.mu.data = torch.tensor([0.0])
    model3.log_std.data = torch.tensor([np.log(5.0)])
    
    # 로그가능도들을 계산한다
    ll1 = model1.log_prob(test_data).mean().item()
    ll2 = model2.log_prob(test_data).mean().item()
    ll3 = model3.log_prob(test_data).mean().item()
    
    print("\n" + "-" * 70)
    print("Model Comparison:")
    print("-" * 70)
    print(f"Model 1 N(0, 1)  - Correct:       {ll1:.4f}")
    print(f"Model 2 N(5, 1)  - Wrong mean:    {ll2:.4f}")
    print(f"Model 3 N(0, 25) - Wrong variance: {ll3:.4f}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("Higher log-likelihood means the model better explains the data.")
    print("Model 1 (correct distribution) achieves the highest likelihood.")
    print(f"Likelihood difference (correct vs wrong mean): {ll1 - ll2:.4f}")
    print(f"Likelihood difference (correct vs wrong var):  {ll1 - ll3:.4f}")
    
    # 음의 로그 가능도로 바꾼다 - 실제로 흔히 쓴다
    nll1 = -ll1
    nll2 = -ll2
    nll3 = -ll3
    
    print("\n" + "-" * 70)
    print("Negative Log-Likelihood (NLL) - Lower is Better:")
    print("-" * 70)
    print(f"Model 1: {nll1:.4f}")
    print(f"Model 2: {nll2:.4f}")
    print(f"Model 3: {nll3:.4f}")
    print("\nNLL is often used as a loss function for training generative models.")


def visualize_evaluation_concepts():
    """
    따지기 개념을 보이는 그림을 만든다.
    
    이 함수는 다음을 보이는 그림을 만든다.
    1. 실제 자료와 만든 표본
    2. 가능도 등고선
    3. 표본 품질 견줌
    """
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    # 맞바꿈 보여 주기를 돌린다
    results = demonstrate_likelihood_sample_tradeoff()
    
    # 부분 그림을 갖는 도형 만들기
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 그림 1: 실제 자료 분포
    ax = axes[0, 0]
    ax.hist(results['real_data'].numpy(), bins=50, density=True, alpha=0.7,
            color='blue', edgecolor='black')
    ax.set_title('Real Data Distribution (Bimodal)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Two clear modes\nat x=-3 and x=+3',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 그림 2: 모델 A 표본(봉우리 하나)
    ax = axes[0, 1]
    ax.hist(results['samples_a'].detach().numpy(), bins=50, density=True,
            alpha=0.7, color='green', edgecolor='black', label='Model A Samples')
    ax.hist(results['real_data'].numpy(), bins=50, density=True,
            alpha=0.3, color='blue', edgecolor='black', label='Real Data')
    ax.set_title('Model A: Single Mode (Mode Collapse)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Good sample quality\nbut missing one mode\n(Lower likelihood)',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 그림 3: 모델 B 표본(넓은 정규 분포)
    ax = axes[1, 0]
    ax.hist(results['samples_b'].detach().numpy(), bins=50, density=True,
            alpha=0.7, color='red', edgecolor='black', label='Model B Samples')
    ax.hist(results['real_data'].numpy(), bins=50, density=True,
            alpha=0.3, color='blue', edgecolor='black', label='Real Data')
    ax.set_title('Model B: Wide Gaussian (Mode Averaging)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Covers both modes\nbut poor sample quality\n(Higher likelihood)',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 그림 4: 비교 표
    ax = axes[1, 1]
    ax.axis('off')
    
    # 견줌 표를 만든다
    table_data = [
        ['Metric', 'Model A\n(Single Mode)', 'Model B\n(Wide Gaussian)'],
        ['', '', ''],
        ['Likelihood', '⭐⭐', '⭐⭐⭐'],
        ['Sample Quality', '⭐⭐⭐', '⭐'],
        ['Mode Coverage', '⭐', '⭐⭐⭐'],
        ['', '', ''],
        ['Best For:', 'Sample Quality', 'Coverage'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # 표를 꾸민다
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 머리글
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1 or i == 5:  # 가름 가로줄
                cell.set_facecolor('#E8E8E8')
            else:
                cell.set_facecolor('#F5F5F5')
    
    ax.set_title('Evaluation Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/claude/evaluation_concepts.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'evaluation_concepts.png'")
    
    return fig


def main():
    """
    따지기 바탕을 보이는 으뜸 함수.
    """
    print("\n" + "=" * 70)
    print("MODULE 52.01: EVALUATION BASICS FOR GENERATIVE MODELS")
    print("=" * 70)
    
    # 기본 가능도 셈하기를 보여 준다
    basic_likelihood_computation()
    
    # 가능도와 표본 품질의 맞바꿈을 보여 준다
    demonstrate_likelihood_sample_tradeoff()
    
    # 시각화 만들기
    visualize_evaluation_concepts()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 가능도는 확률 매기기를 잰다:
       - 가능도가 클수록 자료 분포에 잘 맞는다
       - 어떤 모델(가우스, 흐름)에서는 정확히 셈할 수 있다
       - 익힘 목표로 쓴다(가능도를 크게 하는 것은 음의 로그 가능도를 작게 하는 것이다)
    
    2. 표본 품질은 그럴듯함을 잰다:
       - 만든 표본이 얼마나 좋아 보이는가?
       - 사람의 판단이나 배운 잣대가 필요하다
       - 가능도와 이어지지 않을 수 있다
    
    3. 가능도와 표본의 맞바꿈:
       - 가능도가 높다고 좋은 표본이 보장되지 않는다
       - 좋은 표본이 높은 가능도를 보장하지 않는다
       - 두 갈래의 따지기가 모두 필요하다
    
    4. 여러 따지기 틀:
       - 가능도 바탕: 정확한 확률 셈하기
       - 표본 바탕: 보기의 품질과 통계
       - 과제 바탕: 뒤따르는 일에서의 솜씨
    
    5. 흠 없는 잣대 하나는 없다:
       - 잣대마다 센 곳과 여린 곳이 있다
       - 서로 메우는 잣대를 여럿 쓴다
       - 그때의 쓰임새를 살핀다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
```

## 2. 논의

이 짜기는 단원 52.01: 만들어 내는 모델의 따지기 바탕에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

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

**다룬 것** — 단원 52.01: 만들어 내는 모델의 따지기 바탕

이 짜기는 단원 52.01: 만들어 내는 모델의 따지기 바탕에 대해 자리 잡은 가장 좋은 방식을 따른다.

고갱이 갈래는 `EvaluationParadigms`, `SimpleGaussianModel`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
