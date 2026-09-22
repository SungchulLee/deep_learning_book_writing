# 온전한 따지기 보기

이 단원은 깊은 만들어 내는 모델의 중요한 부품인 온전한 따지기 보기을 짠다. 이 짜기를 이해하면 요즘 만들어 내는 모델에 쓰이는 얼개의 결과 익히기 절차를 꿰뚫어 볼 수 있다. 이 코드는 연구와 실제 얼개에서 널리 쓰이는 쓸모 있는 재주를 보인다.

## 1. 코드

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
       - FID가 낮을수록 참 분포에 잘 들어맞는다
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
    main()
```

**출력:**

```
======================================================================
COMPLETE GENERATIVE MODEL EVALUATION EXAMPLE
======================================================================

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
    

======================================================================
COMPARATIVE EVALUATION
======================================================================

Real dataset: 1000 samples
Initialized poor quality generator
Initialized moderate quality generator
Initialized good quality generator
======================================================================
Evaluating POOR quality model
======================================================================

1. Generating samples...

... (131 lines omitted)

       - 바탕과 견준다
    
    5. 실제 얼개에서 살필 것:
       - 잣대 셈하기를 자동으로 한다
       - 익히는 동안 잣대를 좇는다
       - 품질 문턱을 정한다
       - 나빠지는지 지켜본다
       - 사람이 꾸준히 따진다
    
======================================================================
```

## 2. 논의

이 짜기는 온전한 따지기 보기에 대해 자리 잡은 가장 좋은 방식을 따른다. 코드는 모델 뜻매김, 익히기 논리, 도구 함수를 또렷이 갈라 짜였다. 핵심 설계 결정에는 깨움 함수 고르기, 고르게 맞추기 방책, 가장 좋게 하기 웃매개변수가 들며 모두 익히기의 안정과 내놓기 품질에 크게 영향을 준다.

이 얼개는 깊은 만들어 내는 모델에 흔한 중요한 결 여럿을 보인다. 곧 여러 신경망 층을 지나며 특징을 차츰 다루기, 모델이 곁 앎을 받아들이게 하는 조건 주기 얼개, 익히는 동안 기울기가 안정되게 흐르도록 하는 꼼꼼한 첫자리매김이다.

새 자료 묶음이나 문제 마당에서는 웃매개변수 고르기와 익히기 절차를 꼼꼼히 맞추어야 할 때가 많으므로 다루는 이들은 이에 마음을 써야 한다. 코드가 조각으로 나뉘어 있어 다른 얼개, 손실 함수, 익히기 방책을 실험하기 쉽다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
구체적인 들임 텐서로 이 단원의 으뜸 모델의 앞먹임을 좇아라. 층마다 꼴이 어떻게 바뀌는지 적고 내놓기 차원이 바라던 것과 맞는지 확인하라.

</div>

??? success "연습문제 1 풀이"
    들임 텐서에서 시작해 층마다 바뀜을 따라가라. 겹말기 층에서는 공간 차원에 공식 $H_{out} = \lfloor(H_{in} + 2p - k) / s\rfloor + 1$을 쓴다. 선형 층에서는 특징 차원의 바뀜을 좇는다. 중간 꼴을 하나씩 적고 마지막 내놓기가 그 일(그림 만들어 내기, 가르기 등)에 바라던 목표 차원과 맞는지 확인하라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 짜기의 핵심 웃매개변수(배움 빠르기, 배치 크기, 얼개 고르기)를 가려내라. 다른 것을 붙박아 두고 하나씩 바꾸어 웃매개변수마다 익히기가 얼마나 민감한지 재는 실험을 짜라.

</div>

??? success "연습문제 2 풀이"
    핵심 웃매개변수에는 배움 빠르기(흔히 $10^{-4}$에서 $10^{-3}$), 배치 크기(64-256), 층과 채널의 수, 깨움 함수가 든다. 웃매개변수마다 값을 3~5가지로 바꾸어 모델을 익히고 알맞은 잣대(손실, 표본 품질, 모이는 빠르기)를 좇아라. 결과를 그려 어느 웃매개변수가 가장 큰 영향을 주는지 가려내라. 흔히 배움 빠르기와 얼개 깊이가 가장 세게 영향을 주고, 배치 크기는 알맞은 범위 안에서는 웬만큼 영향을 준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 짜기에 새 기능을 더해 넓혀라. 곧 기울기 자르기, 배움 빠르기 차례표, 다른 손실 함수를 더하라. 고치기 앞뒤의 익히기 움직임을 견주어라.

</div>

??? success "연습문제 3 풀이"
    기울기 자르기는 `optimizer.step()` 앞에 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 더한다. 배움 빠르기 차례표는 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`을 쓰고 바퀴마다 `scheduler.step()`을 부른다. 익히기 손실 곡선, 모이는 빠르기, 마지막 모델 품질을 견주어라. 기울기 자르기는 흔히 익히기가 치솟는 것을 막고, 코사인 식히기는 뒤 바퀴에서 더 곱게 가장 좋게 하여 마지막 솜씨를 높일 수 있다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
한 모델을 값매김하는 온전한 절차를 걸음으로 적어라.

</div>

??? success "연습문제 4 풀이"
    이 장의 잣대들을 모으면 이렇게 된다.

    1. **특징 그물을 준비하고 밝힌다.** 자료에 맞는 것으로. MNIST면 MNIST 분류기
    2. **기준점을 잰다.** 참 자료의 인셉션 점수, 참 대 참 FID
    3. **표본을 뽑는다.** 씨앗 고정, 개수 고정, 온도나 잘라 내기 없이
    4. **범위를 확인한다.** $[0,1]$인지
    5. **잣대를 잰다.** 인셉션 점수, FID, 정밀도와 재현율
    6. **외우기를 확인한다.** 최근접 학습 표본까지의 거리
    7. **눈으로 본다.** 고르지 않은 격자
    8. **설정을 모두 적는다.** 표본 수, 특징 그물, 층, 씨앗

    2번과 6번이 흔히 빠진다. 기준점이 없으면 수를 읽을 수 없고, 외우기는 어느 잣대도
    벌하지 않으므로 따로 보아야 한다.

    7번도 빼면 안 된다. 수가 못 보는 것을 눈이 보는 경우가 있고, 특히 잣대를 겨냥한
    표본은 대개 사람 눈에 이상하다.

    이 순서를 함수 하나로 묶어 두면 모델마다 같은 방식으로 재게 되어 견줄 수 있다.
    설정이 어긋나는 사고가 값매김에서 가장 흔한 잘못이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
이 장에서 잰 두 모델의 값을 정리하고 읽어라.

</div>

??? success "연습문제 5 풀이"
    MNIST 분류기(시험 정확도 98.50%) 기준, 표본 10,000개로 잰 값이다.

    | | 인셉션 점수 | FID | 부류 엔트로피 |
    |---|---|---|---|
    | 참 시험 자료 | **9.467** | 2.52 | — |
    | 적대적 생성망 (60 에포크) | 8.026 | **10.25** | 2.271 |
    | 적대적 생성망 (30 에포크) | 6.964 | 40.68 | 2.206 |
    | 변분 자기 부호기 (20 에포크) | 5.804 | 148.2 | 2.220 |
    | 고른 열 부류 | — | — | 2.303 |

    에포크를 함께 적어 둔 까닭이 있다. 익히기 예산이 이 표에서 가장 큰 손잡이였고,
    적대적 생성망을 30에서 60으로 늘리자 FID가 네 배 나아졌다. 그러므로 아래 두 줄을
    나란히 놓고 "적대적 생성망이 변분 자기 부호기보다 낫다"고 읽을 때, **예산이 맞지
    않다는 것**을 함께 알아야 한다.

    **적대적 생성망이 두 잣대에서 모두 이긴다.** FID로는 세 배 넘는 차이다.

    까닭은 [가능도 잣대](02_likelihood_metrics.md)에서 본 대로다. 변분 자기 부호기는
    자료의 모든 점에 확률을 주어야 하므로 넓게 퍼져 흐릿해지고, 적대적 생성망은 그런
    항이 없어 날카로움을 얻는다.

    부류 엔트로피는 둘이 비슷하다(2.206 대 2.220, 고른 값이 2.303). 곧 **부류 수준의
    다양성에서는 차이가 없고** 그림의 날카로움에서 갈린 것이다. 잣대를 여럿 보아야
    이런 구별이 된다.

    기준점을 함께 보는 것을 잊지 말아야 한다. 참 자료의 FID 2.52가 이 설정의 바닥
    근처이므로, 40.7은 아직 한참 멀다는 뜻이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
무너진 모델의 잣대는 어떤 모습인가?

</div>

??? success "연습문제 6 풀이"
    이 장에서 실제로 겪은 일이다. 배치 정규화도 드롭아웃도 없는 설정으로 익혔더니 완전히
    무너졌다.

    | | 인셉션 점수 | FID | 부류 쏠림 |
    |---|---|---|---|
    | 무너진 모델 (BN·드롭아웃 없음) | **1.000** | **1792.8** | 3 하나에 100% |
    | 제대로 익은 모델 | 6.964 | 40.7 | 두루 |

    두 줄이 배치 정규화와 드롭아웃 말고는 같은 설정이다. 그 둘을 뺀 것만으로 이렇게 된다.

    인셉션 점수 1.000이 특히 또렷한 신호다. 상한이 10이고 하한이 1인데 정확히 하한이다.
    모든 표본이 같은 예측 분포를 내놓는다는 뜻이다.

    부류 쏠림을 보면 3 하나에 100%가 모여 있다. 곧 생성기가 $z$를 무시하고 늘 같은 것을
    내놓고 있다. 어느 부류로 무너지는지는 씨앗에 딸리며, 무너진다는 사실이 요점이다.

    무너짐을 알아채는 신호를 정리하면 이렇다.

    | 신호 | 어디서 보는가 |
    |---|---|
    | 인셉션 점수가 1에 가깝다 | 잣대 |
    | 부류 엔트로피가 0에 가깝다 | 잣대 |
    | 고정된 $z$의 격자가 다 닮았다 | 그림 |
    | 손실이 정상으로 보인다 | **손실로는 못 본다** |

    마지막 줄이 중요하다. 적대적 생성망의 손실은 무너짐을 알려 주지 않는다. 판별기가
    쉽게 이기면 손실이 오히려 안정되어 보인다. 그래서 **잣대와 그림을 반드시 함께
    보아야** 한다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
값매김 결과를 재현할 수 있게 하려면 무엇을 적어야 하는가?

</div>

??? success "연습문제 7 풀이"
    적을 것이 많고 하나라도 빠지면 수가 재현되지 않는다.

    | 무엇 | 왜 |
    |---|---|
    | 특징 그물과 그 정확도 | 잣대의 눈금을 정한다 |
    | 어느 층, 몇 차원 | 값이 달라진다 |
    | 표본 수 (모델과 참 자료 양쪽) | FID에 쏠림이 있다 |
    | 참 자료가 학습인지 시험인지 | 외우기에 민감하다 |
    | 뽑기 씨앗 | 표본이 달라진다 |
    | 인셉션 점수의 조각 수와 섞기 | 값이 크게 달라진다 |
    | 온도나 잘라 내기 | 모델의 표본이 아니게 된다 |
    | 모델의 익히기 설정 | 모델을 다시 만들려면 |

    여섯째가 가장 조용한 함정이다. 섞기를 빠뜨리면 참 자료조차 2.146과 9.290으로 갈린다
    ([인셉션 점수 연습문제 3](inception_score.md)).

    이 목록을 코드가 함께 찍게 해 두면 좋다. 사람이 적으면 빠뜨린다.

    ```python
    meta = dict(feature_net='mnist_cnn_98.50', layer='fc1', dim=128,
                n_fake=10000, n_real=10000, real_split='train[:10000]',
                sample_seed=0, is_splits=10, is_shuffled=True, temperature=1.0)
    ```

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
여러 씨앗으로 재어 편차를 보아야 하는가?

</div>

??? success "연습문제 8 풀이"
    보아야 한다. 그러지 않으면 두 모델의 차이가 뜻 있는지 알 수 없다.

    편차가 오는 곳이 둘이다.

    **뽑기 씨앗.** 같은 모델에서 다른 $z$를 뽑으면 값이 조금 달라진다. 값싸게 여러 번
    재어 볼 수 있다.

    **익히기 씨앗.** 모델을 다시 익히면 더 크게 달라진다. 적대적 생성망은 특히 그렇다.
    익히기가 불안정해 씨앗에 따라 결과가 꽤 다르다.

    둘째가 값이 비싸지만 중요하다. 익히기 씨앗을 하나만 써서 얻은 FID 40.7과 42.3의
    차이를 놓고 모델을 고르면 안 된다.

    이 장의 수치는 익히기 씨앗을 42로 고정하고 한 번씩 익혀 얻은 것이다. 그러므로
    **모델끼리의 큰 차이(40.7 대 148.2)는 믿을 만하고 작은 차이는 그렇지 않다.** 설정
    쓸기에서 40.68과 49.85처럼 가까운 값들은 여러 씨앗으로 확인해야 순서를 말할 수 있다.

    인셉션 점수는 조각으로 나누므로 편차를 공짜로 얻는다. 참 자료가 $9.467 \pm 0.159$
    이니, 0.1 정도의 차이는 뜻이 없다고 보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
이 장의 잣대들이 모두 만족스럽다면 모델이 좋은 것인가?

</div>

??? success "연습문제 9 풀이"
    그렇다고 말할 수 없다. 잣대들이 공통으로 못 보는 것이 남아 있다.

    **외우기.** FID와 정밀도·재현율 모두 학습 자료를 그대로 내놓는 것을 만점으로 준다.
    벌하는 항이 식에 없다([FID 연습문제 19](fid.md)).

    **부류 안의 다양성.** 인셉션 점수는 전혀 못 보고, FID는 보지만 2차 통계량까지만이다.

    **드문 것을 만드는 능력.** 자료의 꼬리에 있는 것들을 못 만들어도 잣대가 거의 떨어지지
    않는다. 표본의 대부분이 흔한 것이면 통계량이 맞기 때문이다.

    **뜻과 쓸모.** 잣대는 자료 분포를 닮았는지만 본다. 만든 것이 쓸모 있는지, 뜻이
    통하는지는 묻지 않는다.

    그러므로 값매김의 옳은 태도가 이렇다. **잣대는 나쁜 모델을 걸러 내는 데 쓰고, 좋다고
    증명하는 데는 쓰지 않는다.** 잣대가 나쁘면 확실히 문제가 있고, 좋아도 좋다는 보장이
    아니다.

    그리고 이 장 전체가 되풀이한 것을 새겨야 한다. 잣대를 고르기에 쓰면 겨냥하게 되고,
    겨냥된 잣대는 더는 심판이 아니다. 마지막에는 사람이 보고, 쓸 자리에서 써 보는 것이
    남는다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
값매김 코드를 짤 때 어떤 구조로 두면 좋은가?

</div>

??? success "연습문제 10 풀이"
    **표본 배치를 받아 모든 잣대를 돌려주는 함수 하나**로 두는 것이 좋다. 그러면 모델마다
    같은 방식으로 재게 된다.

    ```python
    def evaluate(images, ref_stats, net, meta):
        assert images.min() >= 0 and images.max() <= 1, "범위가 [0,1] 이 아니다"
        feats, probs = features_and_probs(images, net)
        return dict(
            inception_score=inception_score(probs, splits=meta['is_splits']),
            fid=fid_from_stats(feats, *ref_stats),
            precision_recall=precision_recall(feats, ref_stats[2]),
            class_entropy=class_entropy(probs),
            nn_distance=nearest_train_distance(feats, ref_stats[2]),
            meta=meta,
        )
    ```

    챙긴 것이 몇 가지다.

    - 범위를 **함수 안에서 확인**한다. 가장 흔한 사고를 막는다
    - 참 자료 통계량을 밖에서 받는다. 되풀이 셈하기를 아낀다
    - `meta`를 결과에 담는다. 설정이 값과 함께 남는다
    - 한 번의 앞먹임으로 특징과 확률을 모두 얻는다

    이렇게 두면 참 자료도 같은 함수에 넣어 기준점을 얻을 수 있다. 기준점과 모델의 값이
    같은 코드에서 나오므로 설정이 어긋날 수 없다.

    ```python
    for name, imgs in (("참 자료", real), ("GAN", gan_s), ("VAE", vae_s)):
        print(name, evaluate(imgs, ref, net, meta))
    ```

## 정리하며

**다룬 것** — 온전한 따지기 보기

이 짜기는 온전한 따지기 보기에 대해 자리 잡은 가장 좋은 방식을 따른다.

고갱이 갈래는 `MockGenerativeModel`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
