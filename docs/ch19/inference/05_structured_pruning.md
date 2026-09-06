# 가운데 수준

가운데 수준: 짜임 있는 가지치기. 이 각본은 짜임 전체를 없애는 짜임 있는 가지치기를 보여 준다.

깊은 배움 모델을 효율적으로 펼치려면 모델 크기, 빠르기, 정확도의 맞바꿈을 조심스레 다듬어야 한다. 여기 짠 것은 실전 환경에서 신경망을 눌러 담고 빠르게 하는 데 쓰는 모델 눌러 담기 재주를 보여 준다.

## 코드

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
- 단원 02: 가지치기 기본
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
    main()```

## 논의

`SimpleCNN` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

여기 보인 무늬는 더 복잡한 장면으로 자연스레 넓혀 쓸 수 있다. 웃매개변수, 얼개의 변종, 서로 다른 자료 뭉치로 실험해 보면 이해가 깊어지고 효율적인 펼치기 일에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`SimpleCNN`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = SimpleCNN(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `SimpleCNN`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = SimpleCNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
