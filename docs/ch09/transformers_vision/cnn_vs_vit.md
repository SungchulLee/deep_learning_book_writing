# 합성곱 신경망과 비전 트랜스포머 견주기

합성곱 신경망(CNN)과 비전 트랜스포머(ViT)는 그림을 이해하는 근본적으로 다른 두 길이다. 그 차이와 강점과 맞바꿈을 이해하는 것은 주어진 과제에 알맞은 구조를 고르는 데 매우 중요하다.

---

## 1. 구조의 바탕

### 합성곱 신경망: 국소에서 전역으로

합성곱 신경망은 국소 연산의 위계로 표현을 쌓아 올린다.

```
Input Image
    ↓ (3×3 conv, stride 1) → Local features, small receptive field
    ↓ (3×3 conv + pool) → Larger receptive field
    ↓ (3×3 conv + pool) → Even larger receptive field
    ↓ ...
    ↓ → Global features through many layers
Output
```

### 비전 트랜스포머: 처음부터 전역

비전 트랜스포머는 전역 주의로 그림을 수열처럼 처리한다.

```
Input Image
    ↓ (patch + linear projection) → Sequence of tokens
    ↓ (self-attention) → All tokens attend to all others
    ↓ (self-attention) → Global context from first layer
    ↓ ...
    ↓ → Rich global representations
Output
```

---

## 2. 구현 견주기

### 간단한 합성곱 신경망

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """견주기 위한 전통적인 합성곱 신경망 구조."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # 블록 1: 224 → 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 2: 112 → 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 3: 56 → 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 4: 28 → 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

---

## 3. 자세히 견주기

### 1. 받는 영역

| 측면 | 합성곱 신경망 | 비전 트랜스포머 |
|--------|-----|-----|
| 1층 | 3×3 (국소) | 224×224 (전역) |
| 커지는 방식 | 차츰 (위계로) | 그대로 (전역) |
| 무늬 | 고정 (핵 기반) | 학습 (주의) |

### 2. 계산 복잡도

**합성곱의 복잡도:**

$$O(\text{Conv}) = O(k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)$$

**자기 주의의 복잡도:**

$$O(\text{Attention}) = O(N^2 \cdot D + N \cdot D^2)$$

여기서 $N = \frac{H \times W}{P^2}$은 조각의 개수이다.

### 3. 귀납 편향

| 편향 | 합성곱 신경망 | 비전 트랜스포머 |
|------|-----|-----|
| 국소성 | 강함 (붙박이) | 없음 (배운다) |
| 옮김 동변성 | 있음 | 없음 (데이터가 필요하다) |
| 크기 불변성 | 일부 (풀링으로) | 없음 |

### 4. 데이터가 얼마나 드는가

| 데이터셋 크기 | 권하는 구조 |
|--------------|-------------------------|
| 작음 (10만 미만) | 합성곱 신경망 |
| 보통 (10만~100만) | 합성곱 신경망이나 섞은 구조 |
| 큼 (100만~1000만) | 데이터를 크게 불린 비전 트랜스포머 |
| 아주 큼 (1000만 초과) | 비전 트랜스포머가 뛰어나다 |

### 5. 학습 설정

**합성곱 신경망:**
```python
cnn_config = {
    'optimizer': 'SGD',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_schedule': 'step_decay',
    'epochs': 90
}
```

**비전 트랜스포머:**
```python
vit_config = {
    'optimizer': 'AdamW',
    'lr': 3e-4,
    'warmup_epochs': 5,
    'weight_decay': 0.1,
    'lr_schedule': 'cosine',
    'augmentation': 'strong',
    'epochs': 300,
    'label_smoothing': 0.1
}
```

---

## 4. 성능 시험 견주기

### ImageNet 분류

| 모형 | 매개변수 | Top-1 정확도 | 학습 데이터 |
|-------|--------|-----------|---------------|
| ResNet-50 | 2500만 | 76.1% | 이미지넷-1K |
| ResNet-152 | 6000만 | 78.3% | 이미지넷-1K |
| EfficientNet-B7 | 6600만 | 84.3% | 이미지넷-1K |
| ViT-B/16 | 8600만 | 77.9% | 이미지넷-1K |
| ViT-B/16 | 8600만 | 84.0% | 이미지넷-21K |
| ViT-L/16 | 307M | 87.8% | JFT-300M |

---

## 5. 언제 무엇을 쓰는가

### 합성곱 신경망을 쓸 때

1. **데이터가 적을 때**: 그림이 10만 장보다 적은 데이터셋
2. **속도가 중요할 때**: 실시간 추론이 필요한 경우
3. **가장자리 배치**: 자원이 빠듯한 기기
4. **국소 짜임이 뚜렷할 때**: 과제가 국소성 편향의 덕을 보는 경우

### 비전 트랜스포머를 쓸 때

1. **데이터가 많을 때**: 그림이 수백만 장 있는 경우
2. **전이 학습**: 사전 학습된 모형을 쓰는 경우
3. **전역 맥락**: 과제에 먼 거리 의존이 필요한 경우
4. **최고 수준**: 성능을 최대로 뽑아야 하는 경우

### 섞은 구조를 쓸 때

1. **데이터가 보통일 때**: 그림 10만~100만 장
2. **균형이 필요할 때**: 국소 추론과 전역 추론이 모두 필요한 경우
3. **빽빽한 예측**: 분할이나 탐지 과제

---

## 6. 주의와 합성곱 깊이 들여다보기

자기 주의는 데이터에 따라 달라지는 합성곱으로 볼 수 있다.

- **합성곱**: 고정된 가중치, 국소 범위
- **주의**: 학습된 가중치(내용에 따라 달라진다), 전역 범위

둘 다 가중합을 셈하지만 주의의 가중치는 그때그때 달라진다.
```
Convolution: y = sum(w_fixed * x_local)
Attention:   y = sum(w_dynamic(q,k) * x_global)
```

---

## 7. 요약표

| 측면 | 합성곱 신경망 | 비전 트랜스포머 |
|--------|-----|-----|
| 받는 영역 | 국소 → 전역 | 언제나 전역 |
| 귀납 편향 | 강함 (국소성) | 약함 (자유롭다) |
| 데이터 효율 | 높음 | 낮음 |
| 키우기 | 보통 | 좋음 |
| 해석 가능성 | 활성 지도 | 주의 지도 |
| 학습 안정성 | 높음 | 조심해야 함 |
| 추론 속도 | 빠름 | 보통 |
| 기억 사용 | 크기에 비례 | 조각 수의 제곱에 비례 |

---

## 연습문제

**연습문제 1.**
여섯 가지 측면에서 합성곱 신경망과 비전 트랜스포머를 견주는 표를 만들어라.

??? success "연습문제 1 풀이"
    | 측면 | 합성곱 신경망 | 비전 트랜스포머 |
    |---|---|---|
    | 귀납 편향 | 국소성, 옮김 동변성 | 없음 (데이터에서 배운다) |
    | 받는 영역 | 깊어질수록 넓어진다 | 1층부터 전역 |
    | 데이터 효율 | 데이터가 적을 때 낫다 | 많은 데이터나 데이터 불리기가 필요하다 |
    | 키우기 | 규모가 커지면 보람이 준다 | 규모가 커질수록 계속 나아진다 |
    | 복잡도 | $O(N \cdot k^2)$ | $O(N^2)$ |
    | 잘 맞는 곳 | 가장자리·모바일, 작은 데이터셋 | 큰 규모, 넉넉한 계산 |

---

**연습문제 2.**
언제 비전 트랜스포머 대신 합성곱 신경망을 골라야 하고 언제 그 반대인지 설명하라.

??? success "연습문제 2 풀이"
    합성곱 신경망을 고를 때는 학습 데이터가 적을 때(그림 1만 장 미만), 가장자리 기기에 올릴 때(효율), 공간에 대한 강한 사전 지식이 필요한 과제(작은 데이터셋의 의료 영상)이다. 비전 트랜스포머를 고를 때는 데이터가 많을 때(그림 100만 장 초과), 사전 학습된 모형을 쓸 수 있을 때, 전역 맥락이 필요한 과제일 때, 계산 예산이 넉넉할 때이다.

---

**연습문제 3.**
DeiT는 무엇이며 어떻게 비전 트랜스포머의 데이터 효율을 높이는가?

??? success "연습문제 3 풀이"
    DeiT(Touvron 외, 2021)는 (1) 강한 데이터 불리기(RandAugment, Mixup, CutMix, 무작위 지우기), (2) 합성곱 신경망 스승에게서의 지식 증류, (3) [CLS]와 나란한 증류 토큰을 써서 (그림 120만 장, 추가 데이터 없이) ImageNet만으로 비전 트랜스포머를 잘 학습시킨다. 이로써 비전 트랜스포머와 합성곱 신경망의 데이터 효율 격차를 메운다.

---

**연습문제 4.**
섞은 구조(합성곱 신경망과 트랜스포머)가 순수 비전 트랜스포머를 앞서는가? 근거를 들어 논하라.

??? success "연습문제 4 풀이"
    작거나 보통 크기의 데이터셋에서는 섞은 구조가 이기는 경우가 많다. 합성곱 신경망이 국소 특징을 효율적으로 뽑고 트랜스포머가 전역 맥락을 잡아낸다. CoAtNet, LeViT가 그 보기이다. 아주 큰 규모(그림 수십억 장)에서는 순수 비전 트랜스포머가 합성곱 신경망이 귀납 편향으로 주는 국소 특징을 스스로 배울 수 있어 섞은 구조와 맞먹거나 앞선다. 흐름은 이렇다. 데이터가 늘수록 합성곱 신경망 귀납 편향의 값어치가 줄어든다.

## 정리하며

합성곱 신경망과 비전 트랜스포머는 서로 다른 맞바꿈을 나타낸다.

- **합성곱 신경망**은 데이터가 적을 때 도움이 되는 쓸모 있는 편향을 품지만 자유로움을 제한할 수 있다
- **비전 트랜스포머**는 모든 것을 데이터에서 배워 데이터가 넉넉할 때 뛰어나다
- **섞은 방식**은 둘의 강점을 아우른다

**참고 문헌**

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
2. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Raghu, M., et al. "Do Vision Transformers See Like CNNs?" NeurIPS 2021.
