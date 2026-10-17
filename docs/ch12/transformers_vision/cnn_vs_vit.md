# 컨볼루션 뉴럴 넷트웍과 비전 트랜스포머 비교

CNN과 ViT는 국소 연결·파라미터 공유와 전역 정보 결합을 다르게 설계한다. 구조의 차이를 살펴보되 데이터 수·모델 이름만으로 속도나 정확도의 우열을 정하지 않는다. 사전 학습·어그멘테이션·입력 해상도와 계산 예산을 함께 비교한다.

## 1. 구조의 기초

### 컨볼루션 뉴럴 넷트웍: 국소에서 전역으로

컨볼루션 뉴럴 넷트웍은 국소 연산의 위계로 표현을 쌓아 올린다.

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

비전 트랜스포머는 전역 어텐션으로 이미지를 시퀀스처럼 처리한다.

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

## 2. 구현 비교

### 간단한 컨볼루션 뉴럴 넷트웍

SimpleCNN은 RGB 이미지에 컨볼루션·BatchNorm·최대 풀링 네 블록을 적용하고 전역 평균으로 분류한다. 기본 출력은 (B,10) 로짓이다. 최소 공간 크기, 배치·공간의 정규화 통계와 학습·평가 모드를 확인해야 한다. 이 코드만으로 ViT와 실제 학습·성능 비교를 수행하지 않는다.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """견주기 위한 전통적인 컨볼루션 뉴럴 넷트웍 구조."""
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

## 3. 자세히 비교

### 1. 리셉티브 필드

| 기준 | CNN | 전역 어텐션 ViT |
|---|---|---|
| 국소 레이어의 연결 | 커널 범위 | 첫 어텐션에서 모든 패치 연결 가능 |
| 범위 확장 | 깊이·풀링·팽창 등에 의존 | 마스크가 허용하는 패치 관계 |
| 웨이트 | 학습한 커널을 위치별로 공유 | 학습한 투영과 입력에 따른 점수 |

첫 패치 투영 자체는 P×P 국소 연산이다. ViT의 첫 연산부터 모든 픽셀을 동일하게 직접 처리하는 것은 아니다.

### 2. 계산 복잡도

**컨볼루션의 복잡도:**

$$O(\text{Conv}) = O(k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)$$

**셀프 어텐션의 복잡도:**

$$O(\text{Attention}) = O(N^2 \cdot D + N \cdot D^2)$$

여기서 $N = \frac{H \times W}{P^2}$은 패치의 개수이다.

### 3. 귀납 바이어스

| 설계 | CNN | ViT |
|---|---|---|
| 국소성 | 커널에서 강하게 반영 | 패치 내부 투영 등에도 존재 |
| 위치 공유 | 컨볼루션 계수 공유 | 패치 투영·FFN 등 공유 |
| 이동 성질 | 패딩·스트라이드·경계에 의존 | 패치 격자·위치 표현·경계에 의존 |
| 크기 변화 | 풀링·구조와 학습에 의존 | 패치·위치 표·학습에 의존 |

CNN의 완전한 이동·스케일 불변성이나 ViT의 “귀납 바이어스 없음”을 일반적으로 주장할 수는 없다.

### 4. 데이터가 얼마나 드는가

사전 학습 웨이트를 쓸 수 있는지, 목표 데이터의 분포·레이블 수와 어그멘테이션·학습 예산을 먼저 확인한다. 작은 데이터라도 사전 학습 ViT 미세 조정이 유용할 수 있고, 큰 데이터라고 모든 ViT가 CNN보다 좋은 것은 아니다. 10만·100만 장 같은 고정 경계를 선택 법칙으로 쓰지 않는다.

### 5. 학습 설정

아래 사전은 선택 가능한 설정 예시이며 학습 루프가 아니다. CNN 90에포크와 ViT 300에포크·다른 어그멘테이션은 계산·데이터 노출 예산이 다르다. 그 결과를 구조 하나만의 효과로 해석해서는 안 된다.

**컨볼루션 뉴럴 넷트웍:**
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

## 4. 성능 시험 비교

체크포인트별 ImageNet Top-1은 데이터셋·사전 학습·입력 해상도·어그멘테이션·평가 크롭에 따라 달라진다. 서로 다른 조건의 수치를 출처 없이 같은 표에 놓아 구조를 순위 매기지 않는다.

이 페이지에는 ImageNet을 학습·평가하는 코드가 없으므로 제시된 모델의 성능을 측정한 것으로 읽어서는 안 된다. 비교에는 원 논문·모델 카드의 체크포인트와 평가 규약을 함께 명시하고, 목표 환경에서 품질·처리량·지연·메모리를 보고한다.

## 5. 언제 무엇을 쓰는가

- 국소 피처와 가벼운 배포가 중요하면 적절한 CNN을 출발점으로 비교할 수 있다.
- 전역 정보 결합과 사용 가능한 사전 학습 표현이 유리하면 ViT를 후보로 둘 수 있다.
- 국소·전역 경로를 함께 설계하는 혼합 구조도 후보가 된다.

최종 선택은 같은 데이터 분할과 검증 예산의 품질·지연·메모리로 정한다. 구조 이름이 모바일 적합성·최고 성능을 보장하거나 학습 이미지 수만으로 선택을 결정하지는 않는다.

## 6. 어텐션과 컨볼루션 깊이 들여다보기

어텐션과 컨볼루션은 모두 정보를 가중합하지만 웨이트의 생성·공유·적용 범위가 다르다. 어텐션을 입력에 따른 동적 결합으로 비유할 수 있으나 일반 컨볼루션과 수학적으로 같은 연산이라는 뜻은 아니다.

- **컨볼루션**: 학습한 위치 공유 커널로 지정된 국소 범위를 결합한다.
- **어텐션**: 학습한 투영 뒤 입력 내용·위치·마스크로 웨이트를 계산한다.

둘 다 가중합을 셈하지만 어텐션 웨이트는 그때그때 달라진다.
```
Convolution: y = sum(w_fixed * x_local)
Attention:   y = sum(w_dynamic(q,k) * x_global)
```

---

## 7. 요약표

| 비교 항목 | 함께 고정하거나 보고할 조건 |
|---|---|
| 품질 | 데이터 분할·사전 학습·정답·평가 규약 |
| 연산 | 패치·해상도·피처 폭·레이어 수와 실제 FLOP |
| 속도 | 장치·배치·정밀도·커널·워밍업·동기화 |
| 메모리 | 계수·활성·옵티마이저·캐시와 측정 범위 |
| 학습 안정성 | 초기화·러닝레이트 탐색·정규화·어그멘테이션 |
| 해석 | 활성·어텐션 지도는 예측의 완전한 인과 설명이 아님 |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
CNN과 ViT 비교표에 연결 구조 외에 포함할 여섯 평가 조건을 제시하라.

</div>

??? success "연습문제 1 풀이"
    과제 품질, 사전 학습·데이터 노출량, 해상도·토큰 수, 계수·학습 계산, 실제 지연·처리량과 메모리를 함께 보고한다. 귀납 바이어스와 리셉티브 필드는 구조적 차이이며 성능·효율의 고정 순위를 뜻하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
작은 데이터나 제한된 장치에서 구조 선택을 어떻게 검증해야 하는가?

</div>

??? success "연습문제 2 풀이"
    사용 가능한 체크포인트를 같은 전처리·분할과 합리적인 튜닝 예산에서 비교한다. 목표 품질에 도달하는 학습·추론 비용을 측정하고 장치의 커널 지원도 확인한다. CNN·ViT 이름이나 이미지 수의 고정 임계만으로 정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
DeiT가 제시한 학습·증류 접근을 설명하라.

</div>

??? success "연습문제 3 풀이"
    DeiT는 강한 어그멘테이션·정규화·학습 설정으로 ImageNet 규모에서 ViT를 학습하는 방법을 보여 주었다. 증류 변형은 교사 신호와 별도 증류 토큰도 사용한다. 모든 DeiT 설정에 교사·증류 토큰이 필요한 것은 아니며 그 결과를 모든 데이터의 효율 격차가 사라졌다는 주장으로 일반화하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
CNN·어텐션 혼합 구조가 순수 ViT보다 항상 좋은가?

</div>

??? success "연습문제 4 풀이"
    혼합은 국소 처리와 전역 결합을 함께 설계하는 선택이지만 항상 좋은 것은 아니다. 해상도·데이터·사전 학습·계수·튜닝 예산을 맞춰 과제와 장치에서 평가한다. 데이터가 늘면 국소 바이어스의 밸류가 반드시 단조 감소한다는 보장도 없다.

## 정리하며

CNN과 ViT의 연결·공유·위치 설계를 이해하되 품질·속도·메모리를 구조 이름의 고정 특성으로 보지 않는다. 비교는 체크포인트·학습·평가 조건을 맞춰야 하며 시각화만으로 해석 가능성을 확정하지 않는다.

**참고 문헌**

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
2. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Raghu, M., et al. "Do Vision Transformers See Like CNNs?" NeurIPS 2021.
