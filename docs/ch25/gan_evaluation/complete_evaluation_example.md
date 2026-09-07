# 온전한 따지기 보기

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 온전한 따지기 보기을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 코드

```python
"""
온전한 따지기 보기
===========================

이 보기는 만들어 내는 모델의 온전한 따지기 흐름을 보이며
두루 따지려 잣대를 여럿 아우른다.

지은이: 가르치기 인공 지능 모둠
날짜: 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# ========================================================================
# 메인
# ========================================================================

# 씨앗을 둔다
torch.manual_seed(42)
np.random.seed(42)


class MockGenerativeModel:
    """보여 주기를 위한 흉내 만들어 내는 모델."""
    
    def __init__(self, quality_level: str = "good"):
        """
        품질 수준이 다른 흉내 모델을 첫자리매김한다.
        
        인수:
            quality_level: "poor", "moderate", "good" 가운데 하나
        """
        self.quality_level = quality_level
        print(f"Initialized {quality_level} quality generator")
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """
        표본을 만든다.
        
        인수:
            n_samples: 만들 표본의 개수
        
        반환값:
            만들어 낸 그림 [n_samples, 1, 28, 28]
        """
        if self.quality_level == "poor":
            # 높은 잡음, 낮은 짜임
            samples = torch.randn(n_samples, 1, 28, 28) * 0.5
        elif self.quality_level == "moderate":
            # 보통 잡음, 얼마간의 짜임
            samples = torch.randn(n_samples, 1, 28, 28) * 0.3
            # 짜임을 조금 더한다
            samples[:, :, 10:18, 10:18] += 0.5
        else:  # 좋음
            # 낮은 잡음, 또렷한 짜임
            samples = torch.randn(n_samples, 1, 28, 28) * 0.2
            # 또렷한 짜임(십자 결)을 더한다
            samples[:, :, 13:15, :] += 0.8
            samples[:, :, :, 13:15] += 0.8
        
        # [0, 1]로 고르게 맞추기
        samples = torch.sigmoid(samples)
        return samples


def evaluate_generative_model(model: MockGenerativeModel,
                              real_data: torch.Tensor,
                              n_generated: int = 1000) -> Dict:
    """
    만들어 내는 모델을 두루 따진다.
    
    인수:
        model: 따질 만들어 내는 모델
        real_data: 실제 자료 표본
        n_generated: 만들 표본의 수
    
    반환값:
        따지기 잣대 사전
    """
    print("=" * 70)
    print(f"Evaluating {model.quality_level.upper()} quality model")
    print("=" * 70)
    
    metrics = {}
    
    # 1. 표본을 만든다
    print("\n1. Generating samples...")
    generated_data = model.generate(n_generated)
    print(f"   ✓ Generated {n_generated} samples")
    
    # 2. 눈으로 살펴보기
    print("\n2. Visual Quality Assessment:")
    print(f"   Real data shape: {real_data.shape}")
    print(f"   Generated data shape: {generated_data.shape}")
    print(f"   Real data range: [{real_data.min():.3f}, {real_data.max():.3f}]")
    print(f"   Generated range: [{generated_data.min():.3f}, {generated_data.max():.3f}]")
    
    # 3. 통계로 견주기
    print("\n3. Statistical Comparison:")
    real_mean = real_data.mean()
    gen_mean = generated_data.mean()
    real_std = real_data.std()
    gen_std = generated_data.std()
    
    print(f"   Real:      μ={real_mean:.4f}, σ={real_std:.4f}")
    print(f"   Generated: μ={gen_mean:.4f}, σ={gen_std:.4f}")
    print(f"   Mean error: {abs(real_mean - gen_mean):.4f}")
    print(f"   Std error:  {abs(real_std - gen_std):.4f}")
    
    metrics['mean_error'] = abs(real_mean - gen_mean).item()
    metrics['std_error'] = abs(real_std - gen_std).item()
    
    # 4. 흉내 FID(인셉션 특징 대신 단순한 통계를 쓴다)
    print("\n4. Computing Mock FID:")
    # 그림을 펼친다
    real_flat = real_data.reshape(len(real_data), -1)
    gen_flat = generated_data.reshape(len(generated_data), -1)
    
    # 평균과 함께 흩어짐을 셈한다
    mu_real = real_flat.mean(dim=0).numpy()
    mu_gen = gen_flat.mean(dim=0).numpy()
    
    # 단순한 FID 어림: ||μ_real - μ_gen||²
    mock_fid = np.sum((mu_real - mu_gen) ** 2)
    print(f"   Mock FID: {mock_fid:.4f} (lower is better)")
    metrics['mock_fid'] = float(mock_fid)
    
    # 5. 다양함 따지기
    print("\n5. Diversity Assessment:")
    # 짝마다 거리를 셈한다
    gen_flat = generated_data.reshape(n_generated, -1)
    distances = torch.cdist(gen_flat, gen_flat, p=2)
    upper_tri = distances[torch.triu(torch.ones_like(distances), diagonal=1) == 1]
    
    avg_distance = upper_tri.mean().item()
    min_distance = upper_tri.min().item()
    
    print(f"   Average pairwise distance: {avg_distance:.4f}")
    print(f"   Minimum pairwise distance: {min_distance:.4f}")
    
    metrics['avg_diversity'] = avg_distance
    metrics['min_diversity'] = min_distance
    
    # 6. 되짓기 품질(해당되면)
    print("\n6. Sample Quality Metrics:")
    # 견주려 일부만 쓴다
    n_compare = min(100, len(real_data), len(generated_data))
    
    # 화소마다 평균 제곱 어긋남을 셈한다
    # 참고: 이는 되짓기가 아니라 분포를 견주는 것일 뿐이다
    real_subset = real_data[:n_compare]
    gen_subset = generated_data[:n_compare]
    
    sample_mse = torch.mean((real_subset.mean() - gen_subset.mean()) ** 2)
    print(f"   Distribution MSE: {sample_mse:.6f}")
    metrics['distribution_mse'] = sample_mse.item()
    
    return metrics


def compare_models():
    """
    만들어 내는 모델 여럿을 견준다.
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION")
    print("=" * 70)
    
    # 인공 실제 자료를 만든다
    n_real = 1000
    real_data = torch.randn(n_real, 1, 28, 28) * 0.25
    real_data[:, :, 12:16, 12:16] += 0.7  # 짜임을 더한다
    real_data = torch.sigmoid(real_data)
    
    print(f"\nReal dataset: {n_real} samples")
    
    # 품질 수준이 다른 모델을 만든다
    models = {
        "Poor": MockGenerativeModel("poor"),
        "Moderate": MockGenerativeModel("moderate"),
        "Good": MockGenerativeModel("good")
    }
    
    # 모델마다 따진다
    all_metrics = {}
    for name, model in models.items():
        metrics = evaluate_generative_model(model, real_data, n_generated=1000)
        all_metrics[name] = metrics
    
    # 견줌 표를 만든다
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Poor':<15} {'Moderate':<15} {'Good':<15}")
    print("-" * 70)
    
    metric_names = list(all_metrics["Poor"].keys())
    for metric in metric_names:
        values = [all_metrics[name][metric] for name in ["Poor", "Moderate", "Good"]]
        print(f"{metric:<25} {values[0]:<15.6f} {values[1]:<15.6f} {values[2]:<15.6f}")
    
    # 가장 좋은 모델을 가린다
    print("\n" + "-" * 70)
    print("Best Model Analysis:")
    print("-" * 70)
    
    # 이 잣대는 낮을수록 좋다
    poor_fid = all_metrics["Poor"]["mock_fid"]
    mod_fid = all_metrics["Moderate"]["mock_fid"]
    good_fid = all_metrics["Good"]["mock_fid"]
    
    print(f"\nMock FID (lower is better):")
    print(f"  Poor: {poor_fid:.4f}")
    print(f"  Moderate: {mod_fid:.4f}")
    print(f"  Good: {good_fid:.4f}")
    
    if good_fid < mod_fid < poor_fid:
        print("\n✓ Quality ranking matches FID scores!")
    
    print(f"\nDiversity (higher avg distance is better):")
    for name in ["Poor", "Moderate", "Good"]:
        div = all_metrics[name]["avg_diversity"]
        print(f"  {name}: {div:.4f}")


def main():
    """
    온전한 따지기 보기를 돌리는 으뜸 함수.
    """
    print("\n" + "=" * 70)
    print("COMPLETE GENERATIVE MODEL EVALUATION EXAMPLE")
    print("=" * 70)
    print("""
이 보기는 두루 갖춘 따지기 흐름을 보인다.
1. 모델에서 표본을 만든다
2. 눈으로 품질 따지기
3. 통계로 견주기
4. FID 셈하기
5. 다양함 따지기
6. 표본 품질 잣대
7. 견주어 살피기

실제로는 다음도 넣는다.
- 인셉션 점수
- 정밀도와 재현율
- 느낌으로 재는 자(LPIPS)
- 사람이 따지기
    """)
    
    # 견주어 따지기를 돌린다
    compare_models()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM EVALUATION")
    print("=" * 70)
    print("""
    1. 잣대가 여럿 필요하다:
       - 잣대 하나가 모든 것을 담지는 못한다
       - 잣대마다 재는 면이 다르다
       - 수로 따지기 + 결로 따지기를 아우른다
    
    2. 잣대 풀이:
       - FID이 낮을수록 참 분포에 잘 들어맞는다
       - 다양함이 클수록 봉우리 무너짐이 적다
       - 통계 어긋남이 작을수록 적률이 잘 맞는다
    
    3. 품질 수준:
       - 나쁨: 높은 FID, 낮은 다양함, 큰 통계 어긋남
       - 보통: 중간 FID, 보통 다양함
       - 좋음: 낮은 FID, 높은 다양함, 작은 어긋남
    
    4. 가장 좋은 방식:
       - 표본을 넉넉히 만든다(FID이면 1만 개 이상)
       - 서로 메우는 잣대를 여럿 쓴다
       - 눈으로 살펴보기를 넣는다
       - 믿음 구간을 알린다
       - 바탕과 견준다
    
    5. 실제 얼개에서 살필 것:
       - 잣대 셈하기를 자동으로 한다
       - 익히는 동안 잣대를 좇는다
       - 품질 문턱을 정한다
       - 나빠지는지 지켜본다
       - 사람이 꾸준히 따진다
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()```

## 논의

이 짜기는 온전한 따지기 보기에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

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
