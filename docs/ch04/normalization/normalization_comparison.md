# 정규화 기법 비교

정규화 층들의 종합 비교. 이 파일은 나란히 놓고 하는 비교와 실용적인 예제를 제공한다

정규화 기법을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
정규화 층의 종합 비교
=================================================

이 파일은 배치 정규화, 층 정규화, 사례 정규화, 그룹 정규화를
나란히 비교하고 실용적인 예제를 제공한다.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib

# ========================================================================
# 메인
# ========================================================================
matplotlib.use('Agg')  # 비대화형 백엔드
import matplotlib.pyplot as plt


class NormalizationComparison:
    """
    여러 정규화 기법을 비교하기 위한 클래스.
    """
    
    def __init__(self):
        self.normalizations = {
            'BatchNorm': nn.BatchNorm2d(3, affine=False),
            'LayerNorm': nn.LayerNorm([3, 4, 4], elementwise_affine=False),
            'InstanceNorm': nn.InstanceNorm2d(3, affine=False),
            'GroupNorm': nn.GroupNorm(3, 3, affine=False),  # 채널 3개에 대한 그룹 3개
        }
        
        # 모두 평가 모드로 두기
        for norm in self.normalizations.values():
            if hasattr(norm, 'eval'):
                norm.eval()
    
    def visualize_normalization_axes(self):
        """
        각 정규화 방법이 어느 축에 작용하는지 시각화한다.
        """
        print("=" * 70)
        print("Normalization Axes Visualization")
        print("=" * 70)
        
        print("\nInput tensor shape: (N, C, H, W) = (Batch, Channels, Height, Width)")
        print("\nNormalization axes (what dimensions are averaged over):")
        print("-" * 70)
        
        visualizations = {
            'BatchNorm':     "Axes: [0, 2, 3] → (N, H, W) | Per channel across batch",
            'LayerNorm':     "Axes: [1, 2, 3] → (C, H, W) | Per sample across features",
            'InstanceNorm':  "Axes: [2, 3]    → (H, W)   | Per sample per channel",
            'GroupNorm':     "Axes: [2, 3]    → (H, W)   | Per sample per group",
        }
        
        for name, desc in visualizations.items():
            print(f"{name:15s}: {desc}")
        
        print("\n" + "=" * 70)
    
    def compare_on_sample_data(self):
        """
        같은 입력에 대해 모든 정규화 방법을 비교한다.
        """
        print("\n" + "=" * 70)
        print("Comparing Normalizations on Sample Data")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        # 예시 입력 만들기: 이미지 2장, 채널 3개, 공간 4x4
        x = torch.randn(2, 3, 4, 4)
        
        # 표본과 채널마다 척도를 다르게 하기
        x[0] *= 5   # 첫 이미지가 더 큰 값을 갖는다
        x[1] *= 0.5  # 둘째 이미지가 더 작은 값을 갖는다
        x[:, 0] *= 2  # 첫 채널이 더 큰 값을 갖는다
        
        print(f"\nInput shape: {x.shape}")
        print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        print("\nOriginal data statistics:")
        for n in range(2):
            for c in range(3):
                mean = x[n, c].mean()
                std = x[n, c].std()
                print(f"  Sample {n}, Channel {c}: mean={mean:7.3f}, std={std:7.3f}")
        
        print("\n" + "-" * 70)
        print("After normalization:")
        print("-" * 70)
        
        for name, norm_layer in self.normalizations.items():
            with torch.no_grad():
                x_norm = norm_layer(x)
            
            print(f"\n{name}:")
            print(f"  Overall: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
            
            # 무엇을 정규화해야 하는지에 맞추어 통계 보이기
            if name == 'BatchNorm':
                print("  Per channel (averaged over batch, H, W):")
                for c in range(3):
                    mean = x_norm[:, c].mean()
                    std = x_norm[:, c].std()
                    print(f"    Channel {c}: mean={mean:.4f}, std={std:.4f}")
            
            elif name == 'LayerNorm':
                print("  Per sample (averaged over C, H, W):")
                for n in range(2):
                    mean = x_norm[n].mean()
                    std = x_norm[n].std()
                    print(f"    Sample {n}: mean={mean:.4f}, std={std:.4f}")
            
            elif name == 'InstanceNorm':
                print("  Per sample per channel (averaged over H, W):")
                for n in range(2):
                    for c in range(3):
                        mean = x_norm[n, c].mean()
                        std = x_norm[n, c].std()
                        print(f"    Sample {n}, Channel {c}: mean={mean:.4f}, std={std:.4f}")
    
    def test_batch_size_sensitivity(self):
        """
        배치 크기가 달라질 때 정규화 방법마다 어떻게 대응하는지 시험한다.
        """
        print("\n" + "=" * 70)
        print("Batch Size Sensitivity Test")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        batch_sizes = [1, 2, 8, 32]
        
        print("\nTesting with different batch sizes:")
        print("(Using the same data distribution)")
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 8, 8)
            
            print(f"\n--- Batch size: {batch_size} ---")
            
            for name, norm_layer in self.normalizations.items():
                # 이동 통계 문제를 피하려고 다시 초기화
                if name == 'BatchNorm':
                    norm_layer = nn.BatchNorm2d(3, affine=False)
                    norm_layer.eval()
                
                with torch.no_grad():
                    x_norm = norm_layer(x)
                
                print(f"{name:15s}: mean={x_norm.mean():7.4f}, std={x_norm.std():7.4f}")
        
        print("\nObservations:")
        print("- BatchNorm is sensitive to batch size (less stable with small batches)")
        print("- LayerNorm, InstanceNorm, GroupNorm are independent of batch size")


def create_comparison_network():
    """
    비교를 위해 정규화 층이 서로 다른 신경망을 만든다.
    """
    print("\n" + "=" * 70)
    print("Example Networks with Different Normalizations")
    print("=" * 70)
    
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, norm_type='batch'):
            super(ConvBlock, self).__init__()
            
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
            # 정규화 고르기
            if norm_type == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'layer':
                # 2차원 데이터에 LayerNorm을 쓰려면 모양을 지정해야 한다
                # 이는 간략한 판본이다
                self.norm = nn.GroupNorm(1, out_channels)  # 표본이 하나일 때는 층 정규화와 같다
            elif norm_type == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == 'group':
                self.norm = nn.GroupNorm(8, out_channels)  # 그룹 8개
            else:
                self.norm = nn.Identity()
            
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.relu(x)
            return x
    
    # 정규화 방식이 다른 신경망 만들기
    networks = {
        'BatchNorm': nn.Sequential(
            ConvBlock(3, 64, 'batch'),
            ConvBlock(64, 128, 'batch'),
        ),
        'InstanceNorm': nn.Sequential(
            ConvBlock(3, 64, 'instance'),
            ConvBlock(64, 128, 'instance'),
        ),
        'GroupNorm': nn.Sequential(
            ConvBlock(3, 64, 'group'),
            ConvBlock(64, 128, 'group'),
        ),
    }
    
    # 예시 입력으로 시험
    x = torch.randn(4, 3, 32, 32)
    
    print("\nTesting networks with input shape:", x.shape)
    
    for name, net in networks.items():
        net.eval()
        with torch.no_grad():
            out = net(x)
        print(f"{name:15s}: output shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")


def performance_comparison():
    """
    정규화 방법들의 계산 성능을 비교한다.
    """
    print("\n" + "=" * 70)
    print("Performance Characteristics")
    print("=" * 70)
    
    characteristics = {
        'BatchNorm': {
            'Speed': 'Fast',
            'Memory': 'Low (stores running stats)',
            'Batch dependency': 'Yes (sensitive to batch size)',
            'Train/Eval difference': 'Yes (uses different stats)',
        },
        'LayerNorm': {
            'Speed': 'Fast',
            'Memory': 'Low',
            'Batch dependency': 'No (batch independent)',
            'Train/Eval difference': 'No (same computation)',
        },
        'InstanceNorm': {
            'Speed': 'Fast',
            'Memory': 'Low',
            'Batch dependency': 'No (batch independent)',
            'Train/Eval difference': 'No (same computation)',
        },
        'GroupNorm': {
            'Speed': 'Fast',
            'Memory': 'Low',
            'Batch dependency': 'No (batch independent)',
            'Train/Eval difference': 'No (same computation)',
        },
    }
    
    for norm_name, chars in characteristics.items():
        print(f"\n{norm_name}:")
        for key, value in chars.items():
            print(f"  {key:25s}: {value}")


def practical_recommendations():
    """
    정규화 방법을 고르기 위한 실용적인 권고를 제시한다.
    """
    print("\n" + "=" * 70)
    print("Practical Recommendations")
    print("=" * 70)
    
    recommendations = """
    과제 종류                    | 권장 정규화       | 이유
    -----------------------------|------------------|-------------------------
    이미지 분류 (CNN)            | BatchNorm         | 큰 배치에서 잘 통한다
    물체 검출                    | GroupNorm/SyncBN  | 작은 배치에 더 낫다
    의미 분할                    | BatchNorm/GroupNorm | 배치 크기에 달렸다
    양식 전이                    | InstanceNorm      | 사례별 정보를 없앤다
    GAN (이미지 대 이미지)       | InstanceNorm      | 표본을 따로따로 처리한다
    트랜스포머 (자연어)          | LayerNorm         | 표준 선택, 배치와 무관하다
    RNN/LSTM                     | LayerNorm         | 길이가 변하는 순차열을 잘 다룬다
    온라인 학습 (batch=1)        | LayerNorm/InstanceNorm | 배치와 무관하다
    작은 배치 학습               | GroupNorm/LayerNorm | 배치 크기에 민감하지 않다
    영상 처리                    | GroupNorm         | 시간 차원을 잘 다룬다
    
    특수한 경우:
    - 배치가 작으면(< 8): 그룹 정규화나 층 정규화를 쓴다
    - 학습이 불안정하면: 그룹 정규화를 시도한다
    - 학습과 평가에서 완전히 같은 거동이 필요하면: 층 정규화나 사례 정규화를 쓴다
    - 다중 GPU 학습이면: SyncBatchNorm을 쓴다 (GPU 사이에서 통계를 맞춘다)
    """
    
    print(recommendations)


def common_mistakes():
    """
    정규화 층을 쓸 때 흔한 실수를 짚는다.
    """
    print("\n" + "=" * 70)
    print("Common Mistakes to Avoid")
    print("=" * 70)
    
    mistakes = """
    1. model.eval() 호출을 잊는 것
       - 배치 정규화는 학습 모드와 평가 모드에서 다르게 작동한다
       - 추론 전에는 언제나 model.eval()을 부르라!
    
    2. 배치 크기가 1인데 배치 정규화를 쓰는 것
       - 배치 정규화는 통계를 내려면 표본이 여럿 필요하다
       - 대신 층 정규화나 사례 정규화를 쓰라
    
    3. 정규화를 활성화 앞에 두는 것
       - 표준: 합성곱 → 정규화 → 활성화
       - 정규화 → 합성곱 → 활성화가 더 낫다는 실험도 있다
       - 구조 안에서 일관되게 하라
    
    4. 배치 정규화의 모멘텀을 조정하지 않는 것
       - 기본 모멘텀(0.1)이 최적이 아닐 수 있다
       - 데이터셋이 작으면 더 작은 모멘텀(0.01)을 시도하라
    
    5. 과제에 맞지 않는 정규화를 쓰는 것
       - 양식 전이에 배치 정규화를 쓰지 마라 (사례 정규화를 쓰라)
       - 분류에 사례 정규화를 쓰지 마라 (배치 정규화를 쓰라)
    
    6. 배치 정규화 층을 잘못 얼리는 것
       - 미세 조정할 때는 배치 정규화 층을 조심하라
       - 평가 모드로 두거나 이동 통계를 갱신해야 할 수 있다
    
    7. 다중 GPU 학습을 고려하지 않는 것
       - 표준 배치 정규화는 GPU마다 따로 통계를 낸다
       - GPU 사이에서 더 좋은 결과를 얻으려면 SyncBatchNorm을 쓰라
    
    8. 아핀 매개변수를 무시하는 것
       - affine=True는 배율과 이동을 학습한다는 뜻이다
       - 보통 True로 두되, 순수한 정규화만 원하면 끈다
    """
    
    print(mistakes)


def quick_reference():
    """
    정규화 층의 간단한 참고 안내.
    """
    print("\n" + "=" * 70)
    print("Quick Reference Guide")
    print("=" * 70)
    
    reference = """
    PyTorch 구현:
    
    # 배치 정규화
    nn.BatchNorm1d(num_features)      # 1차원/선형 층용
    nn.BatchNorm2d(num_channels)      # 2차원/합성곱 층용
    nn.BatchNorm3d(num_channels)      # 3차원 데이터용
    
    # 층 정규화
    nn.LayerNorm(normalized_shape)    # 정규화할 모양 지정
    nn.LayerNorm([C, H, W])          # 2차원 데이터용
    
    # 사례 정규화
    nn.InstanceNorm1d(num_features)   # 1차원 데이터용
    nn.InstanceNorm2d(num_channels)   # 2차원/이미지용
    nn.InstanceNorm3d(num_channels)   # 3차원 데이터용
    
    # 그룹 정규화
    nn.GroupNorm(num_groups, num_channels)  # 채널을 그룹으로 나누기
    
    공통 매개변수:
    - eps: 수치 안정성을 위한 작은 값 (기본값: 1e-5)
    - momentum: 배치 정규화의 이동 통계용 (기본값: 0.1)
    - affine: 학습 가능한 배율/이동 매개변수 (기본값: True)
    - track_running_stats: 배치 정규화용 (기본값: True)
    
    기억할 것:
    - 배치 정규화로 추론할 때는 언제나 model.eval()을 부르라
    - 층 정규화/사례 정규화: 학습과 평가에서 거동이 같다
    - 모드를 바꾸려면 .train()과 .eval()을 쓰라
    """
    
    print(reference)


if __name__ == "__main__":
    comp = NormalizationComparison()
    
    # 모든 비교 실행
    comp.visualize_normalization_axes()
    comp.compare_on_sample_data()
    comp.test_batch_size_sensitivity()
    
    create_comparison_network()
    performance_comparison()
    practical_recommendations()
    common_mistakes()
    quick_reference()
    
    print("\n" + "=" * 70)
    print("For more details, see individual files:")
    print("  - batch_normalization.py")
    print("  - layer_normalization.py")
    print("  - instance_normalization.py")
    print("=" * 70)
```

## 2. 논의

`NormalizationComparison` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 학습 최적화 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`NormalizationComparison`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 번째 합성곱 층의 `in_channels`를 현재 값에서 3으로 바꾼다. 공식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$을 써서 합성곱 층과 풀링 층마다 공간 차원을 다시 계산한다. 마지막 합성곱/풀링 층의 평탄화된 출력에 맞도록 첫 번째 선형 층의 `in_features`를 고친다. 다음으로 확인한다. `model = NormalizationComparison(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `NormalizationComparison`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = NormalizationComparison(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 정규화 기법 비교

`NormalizationComparison` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다.

핵심 클래스는 `NormalizationComparison`, `ConvBlock`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
