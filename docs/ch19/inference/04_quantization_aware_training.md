# 가운데 수준

가운데 수준: 양자화를 헤아린 익히기(QAT). 이 각본은 양자화를 흉내내며 익히는 법을 보여 준다.

깊은 배움 모델을 효율적으로 펼치려면 모델 크기, 빠르기, 정확도의 맞바꿈을 조심스레 다듬어야 한다. 여기 짠 것은 실전 환경에서 신경망을 눌러 담고 빠르게 하는 데 쓰는 모델 눌러 담기 재주를 보여 준다.

## 코드

```python
"""
가운데 수준: 양자화를 헤아린 익히기(QAT)

이 각본은 익히는 동안 양자화를 흉내내어 모델이 양자화에 튼튼한 무게를
배우게 하는 양자화를 헤아린 익히기를 보인다.

다루는 주제:
- 익히는 동안의 흉내 양자화
- 양자화를 헤아린 익히기 과정
- 익힌 뒤 양자화와의 견줌
- 채널마다와 텐서마다의 양자화 견줌

수학적 바탕:
- 앞먹임: 양자화한 무게를 쓴다
- 뒤먹임: 곧바로 지나가기 어림개를 쓴다
  ∂L/∂w_float ≈ ∂L/∂w_quant (양자화를 항등으로 친다)
- 곧바로 지나가기 어림개는 미분할 수 없는 양자화를 기울기가 지나가게 한다

먼저 알아야 할 것:
- 단원 01: 양자화 기본
- 뒤먹임 퍼뜨리기에 대한 이해
- 익히기 되풀이에 익숙함
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# ========================================================================
# 메인
# ========================================================================

from utils import (
    evaluate_accuracy,
    compare_model_sizes,
    compare_accuracies,
    seed_everything
)


class SimpleResNet(nn.Module):
    """CIFAR-10을 위한 ResNet 꼴 모델."""
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 잔차 블록
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 첫 누비기
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 남는 이음 덩이 1
        identity = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        # 남는 이음 덩이 2(내림 표집 곁들임)
        x = self.relu(self.bn3(self.conv3(x)))
        identity = x
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + identity
        
        # 분류 머리
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_cifar10_dataloaders(batch_size=128):
    """CIFAR-10 데이터셋을 불러온다."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_with_qat(model, train_loader, test_loader, epochs=20, lr=0.01, device='cpu'):
    """
    양자화를 헤아린 익히기로 모델을 익힌다.
    
    양자화를 헤아린 익히기 과정:
    1. 흉내 양자화 모듈을 끼운다
    2. 양자화 흉내와 함께 익힌다
    3. 기울기가 곧바로 지나가기 어림개를 지난다
    4. 익힌 뒤 실제 양자화 모델로 바꾼다
    """
    model = model.to(device)
    
    # 양자화를 헤아린 익히기를 위해 모델을 마련한다
    model.train()
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\nTraining with Quantization-Aware Training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 몸풀기 뒤 묶음 고르게 맞추기를 얼린다
        if epoch > epochs // 2:
            model.apply(torch.quantization.disable_observer)
        if epoch > epochs * 3 // 4:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        
        if epoch % 5 == 0:
            test_acc = evaluate_accuracy(model, test_loader, device)
            print(f"Epoch [{epoch+1}/{epochs}] Test Acc: {test_acc*100:.2f}%")
    
    # 양자화된 모델로 바꾸기
    model.eval()
    model_quantized = quant.convert(model, inplace=False)
    
    return model_quantized


def main():
    """양자화를 헤아린 익히기 시범의 으뜸 함수."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("QUANTIZATION-AWARE TRAINING DEMONSTRATION")
    print("="*60)
    
    # 데이터를 불러온다
    train_loader, test_loader = get_cifar10_dataloaders()
    
    # 양자화를 헤아린 익히기로 모델을 만들고 익힌다
    model = SimpleResNet(num_classes=10)
    model_qat = train_with_qat(model, train_loader, test_loader, epochs=20, device=device)
    
    # 평가한다
    qat_acc = evaluate_accuracy(model_qat, test_loader, device)
    print(f"\nQAT Model Accuracy: {qat_acc*100:.2f}%")
    
    print("\nQAT typically achieves:")
    print("- Better accuracy than PTQ (1-2% improvement)")
    print("- Same size reduction (4x for INT8)")
    print("- Requires retraining (longer time)")
    print("- Best for models sensitive to quantization")


if __name__ == "__main__":
    main()```

## 논의

`SimpleResNet` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기 보인 무늬는 더 복잡한 장면으로 자연스레 넓혀 쓸 수 있다. 웃매개변수, 얼개의 변종, 서로 다른 자료 뭉치로 실험해 보면 이해가 깊어지고 효율적인 펼치기 일에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`SimpleResNet`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = SimpleResNet(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `SimpleResNet`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = SimpleResNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
