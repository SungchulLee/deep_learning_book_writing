# ResNet 구현

완전한 ResNet 구현. ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152를 모두 구현한다.

합성곱 구조는 요즘 컴퓨터 비전 시스템의 뼈대를 이룬다. 이 구현은 PyTorch로 잔차 신경망 설계의 핵심 개념을 보이며, 이미지 데이터에서 공간적인 특징의 위계가 어떻게 학습되는지 드러낸다.

## 코드

```python
"""
완전한 ResNet 구현
===============================
ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152를 모두 구현
바탕: "Deep Residual Learning for Image Recognition" (He 등, 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class BasicBlock(nn.Module):
    """
    기본 잔차 블록 (ResNet-18과 ResNet-34에서 쓴다)
    건너뛰기 연결이 있는 3x3 합성곱 두 개
    """
    expansion = 1  # 출력 채널 = 입력 채널 * 확장 배수
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    병목 블록 (ResNet-50, ResNet-101, ResNet-152에서 쓴다)
    합성곱 세 개: 1x1, 3x3, 1x1 (채널을 줄였다가 늘린다)
    더 깊은 신경망에서 매개변수가 더 효율적이다
    """
    expansion = 4  # 출력 채널 = 입력 채널 * 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 차원을 줄이는 1x1 합성곱
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 합성곱 (주된 계산)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 차원을 늘리는 1x1 합성곱
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # 줄이기
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 3x3 합성곱
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # 늘리기
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet 구조
    
    인수:
        block: BasicBlock 또는 Bottleneck
        layers: 층마다의 블록 수를 담은 리스트
        num_classes: 출력 부류의 수
        in_channels: 입력 채널의 수 (RGB 이미지는 3)
    """
    
    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # 첫 합성곱 (7x7 합성곱, 보폭 2)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 잔차 층들
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 전역 평균 풀링과 완전 연결층
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        잔차 블록 여러 개로 층을 만든다
        """
        downsample = None
        
        # 차원이 바뀌면 하향 표본화 층 만들기
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # 첫 블록 (하향 표본화할 수 있음)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # 뒤따르는 블록을 위해 in_channels 갱신
        self.in_channels = out_channels * block.expansion
        
        # 남은 블록들
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        카이밍 초기화로 가중치를 초기화한다
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 처음 층들
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 잔차 층들
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 분류 머리
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes=1000, in_channels=3):
    """ResNet-18: 블록 [2, 2, 2, 2]"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes=1000, in_channels=3):
    """ResNet-34: 블록 [3, 4, 6, 3]"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50(num_classes=1000, in_channels=3):
    """ResNet-50: 병목 블록 [3, 4, 6, 3]"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def resnet101(num_classes=1000, in_channels=3):
    """ResNet-101: 병목 블록 [3, 4, 23, 3]"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def resnet152(num_classes=1000, in_channels=3):
    """ResNet-152: 병목 블록 [3, 8, 36, 3]"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


def model_summary(model, input_size=(3, 224, 224)):
    """
    층 정보와 함께 모델 요약을 출력한다
    """
    print("=" * 80)
    print(f"Model Summary: {model.__class__.__name__}")
    print("=" * 80)
    
    # 매개변수 개수 세기
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 순전파 시험
    batch_size = 2
    x = torch.randn(batch_size, *input_size)
    
    print(f"\nInput shape: {x.shape}")
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ResNet Architecture Comparison")
    print("=" * 80)
    
    # 여러 ResNet 모델 만들기
    models = {
        "ResNet-18": resnet18(num_classes=10),  # CIFAR-10용
        "ResNet-34": resnet34(num_classes=10),
        "ResNet-50": resnet50(num_classes=10),
        "ResNet-101": resnet101(num_classes=10),
    }
    
    print("\nModel Parameter Counts:")
    print("-" * 80)
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15} {params:>15,} parameters")
    
    print("\n" + "=" * 80)
    print("Detailed Summary for ResNet-18")
    print("=" * 80)
    model_summary(models["ResNet-18"])
    
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    print("=" * 80)
    
    model = resnet18(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    
    print(f"Input: {x.shape}")
    output = model(x)
    print(f"Output: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print("\n✓ ResNet implementation complete!")
    print("=" * 80 + "\n")```

## 논의

이 구현은 클래스 세 개(`BasicBlock`, `Bottleneck`, `ResNet`)를 정의하며, 이들이 어우러져 완전한 잔차 신경망 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 딥러닝 구조 설계에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`BasicBlock`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치에 대해 주요 연산(합성곱, 풀링, 선형층)마다의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`을 지금 값에서 3으로 바꾸어라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 합성곱과 풀링 층마다의 공간 차원을 다시 계산하라. 마지막 합성곱/풀링 층의 펼친 출력에 맞게 첫 선형층의 `in_features`을 고쳐라. `model = BasicBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `BasicBlock`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = BasicBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
