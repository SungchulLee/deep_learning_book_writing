# 생성 모델 평가의 기초

생성 모델 평가에는 관측 자료의 로그확률·밀도,생성 표본의 품질·다양성,사용 목적의 성능 등 여러 기준이 있다. 각 지표가 무엇을 관측하고 무엇을 놓치는지 구분해야 한다. 이 절의 코드는 이미지 GAN을 학습하지 않고 작은 가우시안 예제로 평가 관점의 차이를 설명한다.

## 1. 예제 코드

```python
"""
만들어 내는 모델의 따지기 바탕
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
        # 꼴: [배치 크기]
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
    
    여기서 θ = {μ_θ, σ_θ}은 배울 수 있는 파라메터다.
    
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
        # 로그 파라메터화에서 표준 편차를 얻는다
        # 이렇게 하면 σ > 0이 된다
        std = torch.exp(self.log_std)
        
        # 마할라노비스 거리의 제곱을 셈한다: [(x-μ)/σ]²
        # 꼴: [배치 크기, 차원]
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
    plt.savefig('evaluation_concepts.png', dpi=300, bbox_inches='tight')
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
       - 두 클래스의 따지기가 모두 필요하다
    
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

EvaluationParadigms.likelihood_evaluation은 model.log_prob의 평균을 반환한다. 따라서 계산 가능한 로그밀도·확률을 제공하는 모델과 표본별 출력 계약이 필요하다. 이 함수 자체는 eval·no_grad를 적용하지 않는다. SimpleGaussianModel은 대각 공분산 가우시안의 log_prob와 sample을 제공하지만 nn.Module을 상속하지 않아 parameters·to·state_dict 같은 모듈 관리가 자동 제공되지는 않는다.

sample_quality_evaluation은 `(N,d)` 벡터 표본의 피처별 평균·표준편차 차이를 계산한다. mean_absolute_error라는 키는 대응 표본 쌍의 MAE가 아니라 **두 평균 벡터의 평균 절대 차이**다. d>1이면 상관행렬 차이도 계산한다. std의 기본 표본 보정에는 표본 수≥2가 필요하고 상수 피처의 상관은 정의되지 않아 NaN이 될 수 있다. 이미지 텐서를 임의로 그대로 넣는 일반적인 이미지 품질 평가 함수는 아니다.

??? note "기존 전체 출력 기록"

    ```

    ======================================================================
    MODULE 52.01: EVALUATION BASICS FOR GENERATIVE MODELS
    ======================================================================

    ======================================================================
    Basic Likelihood Computation
    ======================================================================

    Test data: 100 samples from N(0, 1)
    Mean: 0.0598, Std: 0.9864

    ----------------------------------------------------------------------
    Model Comparison:
    ----------------------------------------------------------------------
    Model 1 N(0, 1)  - Correct:       -1.4023
    Model 2 N(5, 1)  - Wrong mean:    -13.6035
    Model 3 N(0, 25) - Wrong variance: -2.5477

    ----------------------------------------------------------------------
    Interpretation:
    ----------------------------------------------------------------------
    Higher log-likelihood means the model better explains the data.
    Model 1 (correct distribution) achieves the highest likelihood.
    Likelihood difference (correct vs wrong mean): 12.2012
    Likelihood difference (correct vs wrong var):  1.1454

    ----------------------------------------------------------------------
    Negative Log-Likelihood (NLL) - Lower is Better:
    ----------------------------------------------------------------------
    Model 1: 1.4023
    Model 2: 13.6035
    Model 3: 2.5477

    NLL is often used as a loss function for training generative models.
    ======================================================================
    Demonstrating Likelihood vs. Sample Quality Tradeoff
    ======================================================================

    Real data: 1000 samples from bimodal distribution
    Mode 1 centered at -3.0, Mode 2 centered at +3.0

    ----------------------------------------------------------------------
    Likelihood Evaluation:
    ----------------------------------------------------------------------
    Model A (Single Mode):  Log-Likelihood = -10.3827
    Model B (Wide Gaussian): Log-Likelihood = -2.6624

    ⚠️  Model B has HIGHER likelihood despite missing mode structure!
        This illustrates that likelihood alone doesn't guarantee
        good sample quality or mode coverage.

    ----------------------------------------------------------------------
    Sample Quality Evaluation:
    ----------------------------------------------------------------------

    Model A (Single Mode):
      Mean Error: 3.0699
      Std Error:  2.1124

    Model B (Wide Gaussian):
      Mean Error: 0.1079
      Std Error:  1.3129

    ======================================================================
    KEY INSIGHT:
    ======================================================================
    Model B achieves higher likelihood by spreading probability mass
    across both modes, but its samples are less realistic because they
    often fall between modes where real data doesn't exist.

    Model A captures one mode perfectly (realistic samples) but
    achieves lower likelihood because it assigns zero probability to
    the other mode (mode collapse).

    This is why we need BOTH likelihood and sample-based evaluation!
    ======================================================================

    ======================================================================
    Generating Visualizations
    ======================================================================
    ======================================================================
    Demonstrating Likelihood vs. Sample Quality Tradeoff
    ======================================================================

    Real data: 1000 samples from bimodal distribution
    Mode 1 centered at -3.0, Mode 2 centered at +3.0

    ----------------------------------------------------------------------
    Likelihood Evaluation:
    ----------------------------------------------------------------------
    Model A (Single Mode):  Log-Likelihood = -10.2470
    Model B (Wide Gaussian): Log-Likelihood = -2.6584

    ⚠️  Model B has HIGHER likelihood despite missing mode structure!
        This illustrates that likelihood alone doesn't guarantee
        good sample quality or mode coverage.

    ----------------------------------------------------------------------
    Sample Quality Evaluation:
    ----------------------------------------------------------------------

    Model A (Single Mode):
      Mean Error: 3.0316
      Std Error:  2.0913

    Model B (Wide Gaussian):
      Mean Error: 0.0696
      Std Error:  1.3350

    ======================================================================
    KEY INSIGHT:
    ======================================================================
    Model B achieves higher likelihood by spreading probability mass
    across both modes, but its samples are less realistic because they
    often fall between modes where real data doesn't exist.

    Model A captures one mode perfectly (realistic samples) but
    achieves lower likelihood because it assigns zero probability to
    the other mode (mode collapse).

    This is why we need BOTH likelihood and sample-based evaluation!
    ======================================================================

    ✓ Visualization saved as 'evaluation_concepts.png'

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

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
           - 두 클래스의 따지기가 모두 필요하다
        
        4. 여러 따지기 틀:
           - 가능도 바탕: 정확한 확률 셈하기
           - 표본 바탕: 보기의 품질과 통계
           - 과제 바탕: 뒤따르는 일에서의 솜씨
        
        5. 흠 없는 잣대 하나는 없다:
           - 잣대마다 센 곳과 여린 곳이 있다
           - 서로 메우는 잣대를 여럿 쓴다
           - 그때의 쓰임새를 살핀다
        
    ======================================================================
    ```

기존 출력·그림·코드는 보존했으며 이 편집에서는 실행하거나 다시 측정하지 않았다. 특히 출력 안내문의 “다른 모드에 확률 0”이나 “두 모드 사이에는 실제 자료가 없다”는 문장은 이 가우시안 예제에서 엄밀히 맞지 않는다.

## 2. 가우시안 비교가 보여 주는 것

실제 분포는 N(−3,1)과 N(3,1)의 같은 비중 혼합으로 평균 0,분산 10이다. 모델 A는 N(−3,1),B는 평균 0·log_std=1.5,즉 표준편차 e^1.5≈4.48인 가우시안이다. 두 모델은 학습한 것이 아니라 파라미터를 수동 설정한다.

A는 한 성분을 표현하지만 다른 성분의 밀도가 **정확히 0은 아니다**. 두 모드 사이의 참 혼합 밀도도 작지만 양수다. B는 두 모드 주변과 중간에 질량을 두는 과분산 단봉 분포이며 실제 분포의 모드 구조를 복원하지 못한다. 단일 가우시안 중 기대 로그우도를 최대화하는 것은 평균 0·분산 10으로 B와도 다르다.

이 비교는 제한된 후보에서 더 높은 우도를 가진 모델이 특정 시각적 구조까지 잘 재현한다고 할 수 없음을 보여 준다. 무제한의 올바른 분포 계열에서 참 분포가 기대 로그우도를 최대화한다는 성질을 반박하는 예가 아니다. 표본 품질이나 모드 포괄도를 별도 측정하지 않고 그림·별표 표만으로 그 순위를 확정할 수도 없다.

코드는 demonstrate 함수를 직접 호출한 뒤 그림 함수 안에서 다시 호출해 새 표본을 만든다. 두 출력 묶음과 그림의 표본이 같은 배치라고 가정하지 않는다. 전역 시드를 처음 설정해도 반복 호출은 난수 상태를 소비한다.

## 3. 평균·분산과 분포 전체

평균·표준편차·상관이 같아도 분포의 꼬리·모드·형태가 다를 수 있다. 가우시안과 위 혼합이 평균 0·분산 10을 공유하는 경우가 간단한 예다. 반대로 임의의 쌍별 표본 오차는 정답 대응이 없는 무조건부 생성에서는 적절한 평가가 아닐 수 있다.

연속 모델의 log_prob는 점의 확률이 아니라 밀도 로그이며 단위·전처리에 의존한다. 이산 이미지·토큰 확률,연속 밀도,ELBO 같은 하한을 같은 수치로 직접 비교하지 않는다. 일반적 GAN도 표본 분포는 정의하지만 계산 가능한 p_G(x)를 보통 직접 제공하지 않아 위 log_prob 함수를 그대로 적용할 수 없다.

## 4. 이미지 생성 지표의 구분

| 지표 | 입력·비교 대상 | 주로 놓칠 수 있는 것 |
|---|---|---|
| IS | 생성 표본의 평가기 클래스 분포 | 실제 분포와의 직접 비교·클래스 내 다양성 |
| FID | 실제·생성 피처의 평균·공분산 | 같은 두 모멘트를 가진 분포 차이·암기 |
| 생성 정밀도·재현율 | 피처 공간의 표본·근방 | 피처·근방 정의에 따른 오류 |
| 우도·하한 | 모델과 평가 관측 | 지각적 품질·목적별 유용성 |
| 시각·사람 평가 | 표시한 표본·정한 질문 | 선택·표시·평가자 바이어스·전체 포괄도 |

도메인에 맞는 평가 피처가 중요하다. MNIST 분류기 피처로 계산한 Fréchet 통계·분류기 기반 IS를 표준 ImageNet Inception 결과와 같은 수치로 비교하면 안 된다. 높은 분류 정확도 하나도 생성 평가의 완전한 적합성을 보장하지 않는다. 이 코드에는 MNIST 분류기나 그 정확도·참조 점수 측정이 없다.

## 5. 표본 수·선택과 변동

고정된 평가 계약에서도 유한 표본 점수에는 편향과 변동이 있다. [FID·IS의 유한 표본 분석](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)은 바이어스가 모형에 따라 순위에도 영향을 줄 수 있음을 보인다. 표본이 작으면 개별 실행의 FID가 반드시 더 커진다는 단조 법칙이나 보편적인 바닥값은 없다.

표본 수·피처 차원·공분산 정의·시드·IS 분할·전처리와 참조를 기록하고 여러 반복의 변동을 확인한다. 피처 차원의 열 배라는 단일 규칙이 모든 평가의 충분성을 보장하지는 않는다. 모델 선택에 반복 사용한 평가와 독립 최종 보고를 구분하고 암기·중복도 별도로 살핀다. 서로 다른 지표의 불일치는 진단의 시작이며 유일한 실패 원인을 증명하는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SimpleGaussianModel.log_prob와 표본 통계의 입력·출력을 설명하라.

</div>

??? success "풀이"

    x는 `(N,d)`,mu·log_std는 길이 d이고 log_prob는 각 표본의 d차원 로그밀도 합 `(N,)`를 반환한다. likelihood_evaluation은 이를 표본 평균한다. sample은 `(N,d)`를 반환하며 통계 비교는 피처별 평균·표준편차와 선택적 d×d 상관 차이다. 뉴럴 컨볼루션 레이어의 모양 추적이 아니라 확률·통계 축을 확인해야 한다. CPU·자료형·표본 수와 유한값 조건도 맞춘다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 코드의 평가 민감도를 실험 없이 설계하라.

</div>

??? success "풀이"

    모델은 학습 루프 없이 수동 설정되므로 임의의 러닝레이트 탐색을 붙이는 것이 핵심이 아니다. 평균·표준편차·모드 비중·표본 수를 바꾸고 기대 로그밀도·평균·분산 지표가 각각 어떤 차이를 보는지 예측한다. 같은 평균·분산의 단봉·혼합을 비교해 통계 요약의 한계도 확인할 수 있다. 각 반복의 난수·자료 단위·평가 표본과 변동을 정의하고 실제 결과 수치를 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
평가 함수에 그래디언트 클리핑을 추가하는 것보다 어떤 개선이 더 직접적인가?

</div>

??? success "풀이"

    평가 목적에는 eval·no_grad와 장치·자료형·유효 표본 수·상관의 상수 피처 폴리시를 정하는 것이 직접적이다. 개별 표본 로그밀도와 통계의 축·이름도 명확히 한다. 여러 반복과 참 분포를 아는 예제에서 각 지표의 성질을 확인할 수 있다. 학습이 없는 함수에 optimizer 클리핑·스케줄러를 추가한다고 평가 신뢰성이 개선되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
생성 평가가 단일 분류 정확도와 다른 이유를 설명하라.

</div>

??? success "풀이"

    무조건부 새 표본에는 일반적으로 대응하는 단일 정답 이미지가 없다. 그러나 실제 참조 자료·분포·조건·목적별 정답 정보까지 전혀 없는 것은 아니다. 전체 분포의 품질·포괄도·비중·다양성과 암기·사용 목적을 평가해야 하고 한 지표로 모두 식별하기 어렵다. 분류 정확도 역시 목적과 자료 분할의 한계가 있으므로 단순한 좋고 나쁨의 대조로만 설명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
시각 격자가 충분한 전체 평가가 아닌 이유는 무엇인가?

</div>

??? success "풀이"

    유한 격자는 드문 모드·클래스 내 다양성·암기·빈도를 놓치고 표본 선택·표시 범위·보간에 영향을 받는다. 시드와 선택 기준을 밝히고 실제 자료와 동일한 표시를 사용한다. 정량 지표도 완전하지 않아 함께 사용하되 숫자가 있다는 이유만으로 보편적인 품질 순서가 정해지는 것은 아니다. 동일 원형을 반복한 생성기는 높은 클래스 기반 점수를 얻을 수 있다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
IS·FID·정밀도·재현율·우도의 의미를 구분하라.

</div>

??? success "풀이"

    IS는 생성 표본의 평가기 클래스 불확실성과 주변 클래스 분포를 요약하며 직접 실제 표본 통계를 쓰지 않는다. FID는 피처 평균·공분산을 비교해 분포 전체를 완전히 식별하지 못한다. 생성 정밀도·재현율은 정한 피처·근방 아래 품질·포괄도를 나누는 근사 척도다. 우도는 평가 관측에 모델이 주는 확률·밀도이며 GAN에 분포가 없는 것이 아니라 보통 계산 가능한 log_prob가 없다. 어떤 모델은 하한·추정량만 제공하므로 정확도와 단위도 구분한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
MNIST 분류기 피처로 계산한 점수를 어떻게 보고해야 하는가?

</div>

??? success "풀이"

    평가기 학습 자료·구조·체크포인트·정확도·피처 레이어·차원과 전처리·참조를 명시하고 Inception 기반 점수와 구분한다. ImageNet 피처도 수학적으로 계산 가능하지만 도메인 적합성을 따로 검증해야 한다. 이 페이지 코드에는 MNIST 평가기·98.50%·기준 FID·IS를 측정한 과정이 없어 그런 숫자를 이 실행의 결과로 제시할 수 없다. 실제끼리 비교값도 표본·분할·피처에 따라 달라 보편적 기준점이 아니다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
높은 IS와 나쁜 FID는 반드시 다양성 부족을 뜻하는가?

</div>

??? success "풀이"

    클래스마다 같은 원형을 반복하는 경우 그런 패턴이 생길 수 있지만 유일한 원인은 아니다. 전처리·평가기 분포 밖 반응·클래스 비중·평균·공분산·표본 수 등도 영향을 준다. 반대 패턴도 클래스 쏠림 하나로 확정할 수 없다. 서로 다른 지표가 보는 통계를 확인하고 추가 표본·중복·클래스별 분석으로 가설을 검증한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
평가지표를 모델 선택에 사용할 때 무엇을 분리해야 하는가?

</div>

??? success "풀이"

    선택용 검증과 독립 최종 보고 자료·반복 예산을 분리한다. 지표를 최적화하는 것이 그 자체로 잘못은 아니지만 동일 평가를 많이 시도한 뒤 최솟값만 보고하면 선택 바이어스가 생길 수 있다. 훈련 자료 암기는 FID가 단독으로 벌하지 않으며 train·test FID 차이만으로 정확히 탐지할 수도 없다. 최근접·중복·별도 일반화 진단과 대표 표본을 결합하며 사람이 보면 모든 지표 조작을 쉽게 잡는다고 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
평가 표본 수는 어떻게 정하고 보고할 수 있는가?

</div>

??? success "풀이"

    목적·피처 차원·계산 예산에 맞춰 표본 수를 늘렸을 때 점수·변동·순위가 얼마나 달라지는지 확인한다. 같은 모델 비교에서는 참조·생성 표본 수와 평가 절차를 맞추고 수와 반복 변동을 함께 보고한다. 표본이 적으면 개별 FID가 반드시 커지거나 피처 차원의 열 배면 언제나 충분하다는 보장은 없다. 특이한 공분산도 경우에 따라 정의되므로 표본 수 규칙과 수치 알고리즘의 요구를 구분한다.



## 정리하며

생성 평가는 로그확률·표본 통계·지각적 품질·다양성·사용 목적을 나눠 살펴야 한다. 이 가우시안 예제의 수동 비교와 출력 안내문을 실제 이미지 모델 평가 결과로 혼동하지 않는다. 지표의 통계·표본 수·전처리·참조와 모델 선택 절차를 함께 기록한다.
