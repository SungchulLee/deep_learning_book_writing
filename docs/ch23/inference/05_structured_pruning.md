# 구조적 가지치기

구조적 가지치기는 합성곱의 출력 채널처럼 연산 구조를 이루는 단위를 제거한다. 선택된 필터를 0으로 만드는 것과 작은 계층으로 재구성하는 것은 서로 다르다. 이 예제는 필터의 L1 크기로 후보를 정하려는 코드이며, 실제 채널 수를 줄이는 완성된 압축 구현은 아니다.

## 1. 예제 코드

```python
"""
가운데 수준: 짜임 있는 가지치기

이 각본은 낱낱의 무게가 아니라 짜임 전체(거르개, 채널, 신경 세포)를 없애는
짜임 있는 가지치기를 보인다. 이러면 여느 하드웨어에서도
실제로 빨라진다.

다루는 주제:
- 누비기 층의 거르개 가지치기
- 채널 가지치기
- 온전히 이어진 층의 신경 세포 가지치기
- L1 노름 바탕 중요도 순위
- 정확도를 되찾기 위한 거르개 다시 세우기

수학적 바탕:
- 거르개 중요도: I(F_j) = ||F_j||_1 = Σ|w_ijk|
- L1 노름이 가장 작은 거르개를 없앤다
- 특징 지도가 작아지고 매개변수가 줄어든다

짜임 없는 가지치기에 견준 이점:
- GPU에서 실제로 빨라진다(성긴 연산이 필요 없다)
- 기억 공간 띠너비가 준다
- 이미 있는 하드웨어와 어울린다

먼저 알아야 할 것:
- 가지치기 기본
- 누비기 신경망에 대한 이해
- 특징 지도에 익숙함
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ========================================================================
# 메인
# ========================================================================

from utils import (
    count_parameters,
    evaluate_accuracy,
    compare_model_sizes,
    seed_everything
)


class SimpleCNN(nn.Module):
    """짜임 있는 가지치기 시범을 위한 단순한 누비기 신경망."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 3 * 3, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def compute_filter_importance(layer):
    """
    거르개마다 L1 노름 중요도를 셈한다.
    
    꼴이 (내놓기 채널, 들임 채널, kh, kw)인 누비기 층에서:
    거르개 j의 중요도 = ||F_j||_1 = Σ|w_ijk|
    
    인수:
        layer: 누비기 층
        
    반환값:
        거르개마다의 중요도 점수를 담은 1차원 텐서
    """
    weight = layer.weight.data.abs()
    # 내놓기 채널만 빼고 모든 차원에 걸쳐 더한다
    importance = weight.view(weight.size(0), -1).sum(dim=1)
    return importance


def prune_filters(model, prune_ratio=0.5):
    """
    L1 노름을 바탕으로 누비기 층에서 거르개를 쳐 낸다.
    
    과정:
    1. 거르개마다 중요도를 셈한다
    2. 쳐 낼 거르개를 가려낸다
    3. 더 작은 새 층을 만든다
    4. 살아남은 거르개를 베낀다
    
    인수:
        model: CNN 모델
        prune_ratio: 층마다 쳐 낼 거르개의 비율
        
    반환값:
        층이 작아진, 가지친 모델
    """
    print(f"\nPruning {prune_ratio*100:.0f}% of filters per layer...")
    
    # 층마다 어느 거르개를 남길지 좇는다
    cfg = []
    
    # 1차: 어느 거르개를 남길지 정한다
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            importance = compute_filter_importance(module)
            num_filters = len(importance)
            num_keep = int(num_filters * (1 - prune_ratio))
            
            # 남길 거르개의 번호를 얻는다
            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:num_keep]
            
            cfg.append((name, keep_indices))
            print(f"Layer {name}: Keep {num_keep}/{num_filters} filters")
    
    # 2차: 가지친 층으로 새 모델을 만든다
    # (실전에서는 꼼꼼히 다시 세워야 한다)
    # 이 시범에서는 본디 모델을 그대로 돌려주되
    # 가려낸 거르개를 0으로 만든다
    
    for (name, keep_indices), (module_name, module) in zip(cfg, model.named_modules()):
        if isinstance(module, nn.Conv2d):
            # 가지친 거르개를 0으로 만든다
            mask = torch.zeros(module.weight.size(0))
            mask[keep_indices] = 1.0
            mask = mask.view(-1, 1, 1, 1).to(module.weight.device)
            module.weight.data *= mask
    
    return model


def main():
    """짜임 있는 가지치기 시범의 으뜸 함수."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("STRUCTURED PRUNING DEMONSTRATION")
    print("="*60)
    
    # 데이터를 불러온다
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 모델 생성
    model = SimpleCNN()
    print(f"\nOriginal model parameters: {count_parameters(model):,}")
    
    # 거르개를 쳐 낸다
    pruned_model = prune_filters(model, prune_ratio=0.5)
    
    print("\nStructured pruning achieves:")
    print("- Actual speedup on regular hardware")
    print("- Reduced memory bandwidth")
    print("- Lower sparsity but guaranteed speedup")
    print("- Compatible with all accelerators")
    
    print("\nNote: Full implementation requires layer reconstruction")
    print("      This demo shows filter selection process")


if __name__ == "__main__":
    main()
```

## 2. 모델의 입력과 출력

`SimpleCNN`은 MNIST의 B×1×28×28 입력을 받는다. padding=1인 3×3 합성곱이 공간 크기를 유지하고, 2×2 최대 풀링이 각 단계에서 28→14→7→3으로 줄인다. 출력 채널은 64→128→256이며 마지막 특징은 B×256×3×3이다. 평탄화하면 B×2304, 최종 로짓은 B×10이다.

64×64 RGB 입력으로 바꾸면 첫 합성곱의 입력 채널을 3으로 바꾸고 마지막 공간 크기 8×8에 맞춰 `fc` 입력을 256×8×8로 조정해야 한다. 또는 적응형 풀링으로 분류기 앞의 크기를 고정하는 설계도 가능하다. 이 변경과 채널 가지치기는 별도의 작업이다.

## 3. 필터 중요도와 선택

`compute_filter_importance`는 각 출력 필터의 모든 입력 채널·공간 위치에서 |w|를 합한 L1 점수를 반환한다. 낮은 점수의 필터가 예측에 덜 중요하다는 가정에 기반한 간단한 기준이며 실제 손실 기여도를 직접 측정하지는 않는다. BatchNorm의 스케일과 다음 계층의 가중치도 채널의 역할에 영향을 준다.

`num_keep=int(num_filters*(1-prune_ratio))`는 남길 수를 내림한다. 기본 비율 0.5에서 선택 후보 수는 각각 32, 64, 128개다. 유효한 비율 범위와 최소 한 채널을 남길 조건, 동점 처리도 정의해야 한다.

## 4. 코드에서 선택과 적용이 일치하지 않는 부분

첫 반복문은 합성곱만 모아 `cfg`를 만들지만 두 번째 반복문은 이를 **모든** `named_modules()`와 `zip`으로 묶는다. 루트 모듈과 BatchNorm도 포함되므로 저장한 이름과 실제 적용 모듈이 일치하지 않는다. 예를 들어 두 번째 설정인 `conv2`의 인덱스를 `conv1`의 마스크에 쓰게 될 수 있고, 선택 인덱스가 64 이상이면 범위를 벗어난다. 따라서 이 함수가 각 합성곱을 올바르게 50% 가지치기한다고 가정해서는 안 된다.

설령 대응을 바로잡아 가중치에 마스크를 곱해도 기존 텐서 크기와 밀집 합성곱 연산량은 같다. 필터 바이어스와 BatchNorm의 오프셋 때문에 가중치가 0인 채널의 활성화가 반드시 0인 것도 아니다. 지속적인 마스크나 재매개변수화가 없으므로 이후 학습에서 0인 가중치가 다시 갱신될 수도 있다.

## 5. 실제 구조 축소와 성능 검증

어떤 출력 채널을 제거하면 해당 합성곱의 가중치·바이어스, 연결된 BatchNorm의 파라미터·실행 통계와 다음 합성곱의 입력 채널을 함께 줄여야 한다. 마지막 채널을 줄일 때는 평탄화 순서에 맞추어 분류기의 입력 열도 선택한다. 잔차 연결이나 분기 구조에서는 연결 의존성을 추가로 추적해야 한다.

앞 계층과 뒤 계층의 채널 수를 함께 줄이면 뒤 합성곱의 파라미터와 MAC 수는 두 채널 수의 곱에 따라 감소한다. 다만 실제 속도는 커널의 채널 정렬, 배치와 장치에 달려 있어 보장할 수 없다. [PyTorch 가지치기 튜토리얼](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)에서도 마스크 기반 가지치기와 그 재매개변수화 방식을 확인할 수 있다.

`main`은 데이터로더를 만들지만 학습도 정확도 평가도 하지 않는다. 필터 점수는 무작위 초기화 가중치에서 계산된다. “일반 하드웨어에서 실제 가속”이나 “모든 가속기와 호환”이라는 출력 문자열은 측정 결과가 아니다. 학습된 기준 모델, 구조 재구성, 복구 파인튜닝과 같은 조건의 성능 측정이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치 크기 4인 MNIST 입력의 세 풀링 이후와 분류기 출력 모양을 구하라.

</div>

??? success "연습문제 1 풀이"

    각 풀링 뒤 모양은 4×64×14×14, 4×128×7×7, 4×256×3×3이다. 마지막 풀링은 7을 2로 나눈 값을 내림해 3으로 만든다. 평탄화하면 4×2304이고 로짓은 4×10이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력을 받도록 바꿀 때 입력 채널과 분류기 입력 크기를 계산하라.

</div>

??? success "연습문제 2 풀이"

    첫 합성곱의 입력 채널은 1에서 3으로 바꾼다. 세 풀링 뒤 공간 크기는 64→32→16→8이므로 fc 입력은 256×8×8=16384다. 출력 클래스 수를 유지하면 fc의 출력은 여전히 10이다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
3×3 합성곱의 입력 채널 64개, 출력 채널 128개를 각각 절반으로 줄일 때 바이어스를 제외한 가중치 수를 비교하라.

</div>

??? success "연습문제 3 풀이"

    원래는 64×128×9=73728개다. 두 채널 수를 모두 줄이면 32×64×9=18432개로 1/4이 된다. 같은 출력 공간 크기에서 MAC 수도 1/4이다. 하지만 앞뒤 계층의 연결 차원을 함께 맞추어 실제 텐서를 재구성해야 하며 실행 시간까지 정확히 4배 줄어드는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
깊이를 바꿀 수 있는 CNN에서 채널 가지치기 설정을 각 계층에 안전하게 대응시키는 방법을 설명하라. 아래 코드 조각의 한계도 지적하라.

</div>

??? success "연습문제 4 풀이"

    이름이나 실제 모듈 참조로 각 설정을 연결하고 이전 출력 채널 선택을 다음 계층의 입력 선택에 전달한다. 전체 named_modules 목록과 합성곱 설정을 위치로 zip하는 방식은 피한다. 아래 ModuleList는 모듈 등록의 골격일 뿐 YourBlock이 정의되어 있지 않다. 깊이에 따른 공간 크기와 분류기 차원, BatchNorm과 잔차 연결의 의존성도 별도로 설계해야 한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

구조적 가지치기는 필터 선택뿐 아니라 연결된 계층을 작은 텐서로 재구성하는 과정이다. 이 예제의 적용 루프에는 모듈 대응 오류가 있고 학습·복구·측정도 포함되지 않는다. 채널 의존성을 맞춘 뒤 품질, 실제 저장 공간과 지연 시간을 각각 확인해야 한다.
