# 종합 평가 예제

생성 모델 평가는 샘플을 만드는 단계, 데이터와 출력의 조건을 확인하는 단계, 여러 지표와 표본을 해석하는 단계로 나눌 수 있다. 이 페이지의 코드는 그 흐름을 단순한 인공 패턴으로 보여 준다. 실제 GAN·VAE를 학습하거나 표준 FID와 인셉션 점수를 측정하는 예제는 아니다.

## 1. 인공 패턴과 평가 함수

`MockGenerativeModel`은 학습된 신경망이 아니다. 난수에 중앙 사각형이나 십자 패턴을 더하고 시그모이드를 적용한다. `poor`, `moderate`, `good`은 생성 규칙의 이름이며 평가로 입증된 품질 순위가 아니다. 기준 데이터도 MNIST가 아니라 중앙의 작은 사각형을 강조한 난수 이미지다.

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
    
    5. 실제 구조에서 살필 것:
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

??? note "전체 출력 (175줄)"

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
       ✓ Generated 1000 samples

    2. Visual Quality Assessment:
       Real data shape: torch.Size([1000, 1, 28, 28])
       Generated data shape: torch.Size([1000, 1, 28, 28])
       Real data range: [0.240, 0.845]
       Generated range: [0.091, 0.909]

    3. Statistical Comparison:
       Real:      μ=0.5033, σ=0.0658
       Generated: μ=0.5000, σ=0.1181
       Mean error: 0.0033
       Std error:  0.0523

    4. Computing Mock FID:
       Mock FID: 0.4464 (lower is better)

    5. Diversity Assessment:
       Average pairwise distance: 4.6734
       Minimum pairwise distance: 4.1471

    6. Sample Quality Metrics:
       Distribution MSE: 0.000011
    ======================================================================
    Evaluating MODERATE quality model
    ======================================================================

    1. Generating samples...
       ✓ Generated 1000 samples

    2. Visual Quality Assessment:
       Real data shape: torch.Size([1000, 1, 28, 28])
       Generated data shape: torch.Size([1000, 1, 28, 28])
       Real data range: [0.240, 0.845]
       Generated range: [0.188, 0.855]

    3. Statistical Comparison:
       Real:      μ=0.5033, σ=0.0658
       Generated: μ=0.5098, σ=0.0801
       Mean error: 0.0064
       Std error:  0.0143

    4. Computing Mock FID:
       Mock FID: 0.7318 (lower is better)

    5. Diversity Assessment:
       Average pairwise distance: 2.8887
       Minimum pairwise distance: 2.5466

    6. Sample Quality Metrics:
       Distribution MSE: 0.000039
    ======================================================================
    Evaluating GOOD quality model
    ======================================================================

    1. Generating samples...
       ✓ Generated 1000 samples

    2. Visual Quality Assessment:
       Real data shape: torch.Size([1000, 1, 28, 28])
       Generated data shape: torch.Size([1000, 1, 28, 28])
       Real data range: [0.240, 0.845]
       Generated range: [0.291, 0.914]

    3. Statistical Comparison:
       Real:      μ=0.5033, σ=0.0658
       Generated: μ=0.5266, σ=0.0832
       Mean error: 0.0233
       Std error:  0.0174

    4. Computing Mock FID:
       Mock FID: 3.6429 (lower is better)

    5. Diversity Assessment:
       Average pairwise distance: 1.9243
       Minimum pairwise distance: 1.7126

    6. Sample Quality Metrics:
       Distribution MSE: 0.000544

    ======================================================================
    EVALUATION SUMMARY
    ======================================================================

    Metric                    Poor            Moderate        Good           
    ----------------------------------------------------------------------
    mean_error                0.003308        0.006424        0.023276       
    std_error                 0.052266        0.014296        0.017408       
    mock_fid                  0.446412        0.731808        3.642924       
    avg_diversity             4.673360        2.888695        1.924251       
    min_diversity             4.147131        2.546647        1.712604       
    distribution_mse          0.000011        0.000039        0.000544       

    ----------------------------------------------------------------------
    Best Model Analysis:
    ----------------------------------------------------------------------

    Mock FID (lower is better):
      Poor: 0.4464
      Moderate: 0.7318
      Good: 3.6429

    Diversity (higher avg distance is better):
      Poor: 4.6734
      Moderate: 2.8887
      Good: 1.9243

    ======================================================================
    KEY INSIGHTS FROM EVALUATION
    ======================================================================

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
        
        5. 실제 구조에서 살필 것:
           - 잣대 셈하기를 자동으로 한다
           - 익히는 동안 잣대를 좇는다
           - 품질 문턱을 정한다
           - 나빠지는지 지켜본다
           - 사람이 꾸준히 따진다
        
    ======================================================================
    ```

## 2. 함수가 실제로 계산하는 값

| 결과 키 | 계산 대상 | 해석의 한계 |
|---|---|---|
| mean_error | 두 배치 전체 픽셀 평균의 절대 차이 | 공간 배치를 보지 않는다 |
| std_error | 두 배치 전체 픽셀 표준편차의 절대 차이 | 특징 공분산이나 표준오차가 아니다 |
| mock_fid | 이미지 좌표별 평균 벡터 차이의 제곱합 | 공분산과 Inception 특징이 없어 표준 FID가 아니다 |
| avg_diversity, min_diversity | 생성 이미지 쌍의 픽셀 L2 거리 | 의미적 다양성이나 품질을 보장하지 않는다 |
| distribution_mse | 앞쪽 최대 100개 이미지에서 계산한 두 스칼라 평균 차이의 제곱 | 대응 이미지의 MSE나 분포 전체의 거리가 아니다 |

`mock_fid`는 평균 이미지의 차이만 본다. 픽셀들의 공간 배치가 다른 분포라도 평균 이미지가 같으면 작을 수 있고, 공분산을 전혀 비교하지 않는다. `distribution_mse`는 이름과 달리 이미지별 차이를 계산하지 않아 더욱 제한적인 요약이다. 이를 표준 FID나 재구성 오차로 보고하지 않는다.

다양성 계산은 모든 N×N 거리를 만든 뒤 대각선 위쪽을 선택한다. N≥2가 필요하며 거리·마스크에 O(N²) 메모리가 든다. 잡음 세기가 큰 `poor` 규칙이 더 큰 픽셀 거리를 만들 수 있으므로 '평균 거리가 클수록 좋다'는 출력 설명을 보편적 품질 순위로 읽으면 안 된다. 기준 데이터와 패턴이 다른 `good` 규칙의 mock 점수가 더 나쁠 수도 있다.

## 3. 구현이 포함하지 않는 평가

함수는 데이터 모양과 범위를 출력하지만 이미지 격자를 그리거나 사람이 평가한 결과를 수집하지 않는다. 실제 Inception 특징, IS, 정밀도·재현율, 최근접 학습 이웃, 신뢰구간도 계산하지 않는다. main의 안내문에 나열된 평가 항목과 실제 구현을 구분해야 한다.

데모는 CPU·기울기 추적 없는 텐서를 가정한다. `.numpy()`를 직접 호출하고 생성기도 CPU 텐서를 만들므로 CUDA 입력이나 기울기를 추적하는 입력에 그대로 적용하는 일반 평가 함수가 아니다. 두 데이터 집합의 크기·채널·범위와 유한성에 대한 입력 검증도 별도로 필요하다.

## 4. 실제 모델 평가로 확장할 때

평가 목적과 기준 집합을 정하고, 고정된 특징 모델과 전처리로 실제·생성 이미지를 처리한다. 생성 조건, 표본 수와 선택 규칙을 일치시킨 뒤 FID, 필요하면 IS와 정밀도·재현율을 계산한다. 표본 격자와 중복·최근접 이웃 검사를 함께 보고, 조건부 생성이면 조건의 분포와 조건 충실성도 확인한다.

체크포인트 선택에 쓰는 검증 데이터와 최종 테스트를 구분한다. 고정 모델에서 샘플링을 반복한 변동과 모델을 다시 학습했을 때의 변동도 다르다. 현재 코드는 이런 실험을 수행하지 않으므로 실제 모델 성능이나 학습 시드에 따른 유의성을 이 결과로 주장할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
평가 함수의 mock_fid는 표준 FID와 어떻게 다른가?

</div>

??? success "연습문제 1 풀이"

    28×28 이미지의 좌표별 평균 벡터 차이의 제곱합만 계산한다. Inception 특징 추출과 공분산 비교가 없다. 따라서 표준 FID 수치나 다른 논문의 결과와 직접 비교할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
distribution_mse는 대응된 이미지의 평균 제곱오차인가?

</div>

??? success "연습문제 2 풀이"

    아니다. 앞쪽 최대 100개 이미지에서 각 집합 전체의 평균을 스칼라 하나로 구하고 그 두 수의 차이를 제곱한다. 같은 평균을 가진 전혀 다른 이미지 분포도 0이 될 수 있다. 독립 생성 샘플은 임의의 실제 이미지와 짝지어 재구성 오차로 평가하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 코드에서 이름이 good인 생성기가 반드시 가장 좋은 점수를 얻는가?

</div>

??? success "연습문제 3 풀이"

    그렇지 않다. good은 십자 패턴 생성 규칙이고 기준 데이터는 작은 중앙 사각형 패턴이다. 이름과 평가 대상의 적합도가 일치하지 않을 수 있다. 평균 이미지 거리, 픽셀 거리 등의 결과가 어떤 특성에 반응하는지 확인해야 하며 결과 순서를 미리 사실로 정해 두면 안 된다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
실제 모델의 평가 절차에서 생성·전처리·보고 단계에 필요한 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    기준 집합과 평가 목적, 특징 모델·체크포인트·전처리를 정한다. 고정된 표본 수와 명시한 조건에서 샘플을 만들고 범위·모양·유한성을 확인한다. 지표와 표본 및 중복 검사를 함께 보고 설정과 반복 변동을 저장한다. 온도·절단 샘플링을 쓰는 것도 가능하지만 그 조건의 결과임을 밝히고 비교 조건을 맞춘다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
이 페이지에서 GAN과 VAE의 실측 성능 표를 작성할 수 있는가?

</div>

??? success "연습문제 5 풀이"

    없다. 코드에는 학습된 GAN·VAE, MNIST 데이터, 평가 분류기가 없다. 인공 생성 규칙의 반환값은 비교할 수 있지만 실제 모델 성능으로 바꾸어 해석하면 안 된다. 학습 예산이나 분류기 정확도, FID 등 외부 실험 수치가 필요하면 재현 가능한 별도 근거를 명시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
IS가 1이고 예측 레이블이 하나에 집중되면 생성기가 잠재 변수를 완전히 무시한다고 결론 내릴 수 있는가?

</div>

??? success "연습문제 6 풀이"

    확정할 수 없다. 분류기 관점에서 모든 이미지의 예측 분포가 같거나 비슷할 수 있지만 이미지 자체는 서로 다를 수 있다. 고정·무작위 잠재 코드의 출력, 특징과 픽셀 중복, 잠재 변화에 대한 반응을 함께 확인한다. BN이나 드롭아웃의 제거만으로 모든 GAN이 붕괴한다는 일반 법칙도 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
아래 메타데이터 예시는 평가를 재현하기에 충분한가?

</div>

??? success "연습문제 7 풀이"

    필요한 일부 필드를 보여 주지만 실제 실행 결과나 체크포인트의 증거는 아니다. 특징 가중치 식별자, 전처리·resize·범위, 데이터 버전과 분할, 구현·정밀도·수치 보정, 학습 설정과 생성 조건도 필요하다. is_shuffled라는 필드만 기록해도 실제 계산 코드가 섞지 않으면 평가가 일치하지 않는다.

    ```python
    meta = dict(feature_net='mnist_cnn_98.50', layer='fc1', dim=128,
                n_fake=10000, n_real=10000, real_split='train[:10000]',
                sample_seed=0, is_splits=10, is_shuffled=True, temperature=1.0)
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
샘플링 시드와 학습 시드의 변동을 구분하라. IS 분할 표준편차는 어느 것을 측정하는가?

</div>

??? success "연습문제 8 풀이"

    샘플링 반복은 고정 모델에서 새 샘플을 뽑는 변동이고 학습 반복은 초기화·데이터 순서 등에 따른 모델 자체의 변동이다. IS 분할 표준편차는 한 생성 집합의 구간 사이 변동으로 두 반복을 대신하지 않는다. 단일 학습 결과에서 큰 수치 차이가 나더라도 모델 계열 전체의 우열을 통계적으로 확정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
여러 지표가 만족스러우면 모델의 비암기와 목적 적합성이 입증되는가?

</div>

??? success "연습문제 9 풀이"

    아니다. 특징이 보지 못하는 결함, 드문 패턴의 누락, 암기와 데이터 누출, 조건 위반이 남을 수 있다. 반대로 나쁜 점수도 모델 실패뿐 아니라 전처리나 지표의 도메인 부적합 때문일 수 있다. 평가 구현과 샘플을 확인하고 실제 사용할 작업에서 성능을 검증한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
아래 evaluate 조각의 ref_stats 인터페이스에는 어떤 문제가 있는가?

</div>

??? success "연습문제 10 풀이"

    fid_from_stats에 *ref_stats 전체를 전달하면서 ref_stats[2]를 특징 집합으로 사용하는 구조다. 두 통계량만 받는 일반적인 FID 함수라면 인수가 맞지 않는다. 참 기준 특징과 학습 특징도 같은 인덱스로 쓰고 있어 암기 검사의 대상이 실제 학습 집합인지 불분명하다. 여러 helper와 입력 검증, 추론 모드도 외부 구현이 필요하다. 완성된 실행 함수가 아니라 인터페이스 설계를 설명하는 조각으로 읽고 평균·공분산·평가 특징·학습 특징을 구분해 설계해야 한다.

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
    ```python
    for name, imgs in (("참 자료", real), ("GAN", gan_s), ("VAE", vae_s)):
        print(name, evaluate(imgs, ref, net, meta))
    ```


## 정리하며

종합 평가에는 공통 생성·전처리 조건과 서로 보완하는 지표, 표본·중복 검사가 필요하다. 이 페이지의 mock 값은 인공 패턴의 단순 통계이며 표준 FID나 실제 GAN·VAE 실험 결과가 아니다. 구현한 계산과 설명상의 평가 계획을 구분해 보고한다.
