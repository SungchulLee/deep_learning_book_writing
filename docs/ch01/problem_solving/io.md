# 입력과 출력

모든 알고리즘과 신경망은 입력에서 출력으로 가는 대응을 정의한다. 입출력 계약을 정확히 이해하는 것이 모델이나 데이터 파이프라인 설계의 첫걸음이다.

---

## 1. 정의

알고리즘(또는 모델)은 하나의 함수이다.

$$
f: \mathcal{X} \rightarrow \mathcal{Y}
$$

여기서 $\mathcal{X}$는 입력 공간, $\mathcal{Y}$는 출력 공간이다. 딥러닝에서 $\mathcal{X}$는 보통 텐서 공간이고(예: 이미지의 경우 $\mathbb{R}^{B \times C \times H \times W}$), $\mathcal{Y}$는 확률 분포, 스칼라, 또는 다른 텐서이다.

---

## 2. 설명

입출력 계약을 정확히 명시하면 버그를 막고 모델 설계가 분명해진다.

- **입력 모양과 자료형**: $(B, 3, 224, 224)$ float32 텐서를 기대하는 모델에 $(B, 224, 224, 3)$(채널 마지막 형식)을 주면 조용히 실패하거나 쓸모없는 결과를 낸다. 텐서 모양은 항상 문서화하고 검증해야 한다.
- **출력의 의미**: 분류 모델은 로짓(정규화 전), 확률(소프트맥스 후), 또는 로그 확률(로그 소프트맥스 후)을 출력한다. 이들을 혼동하면 손실 계산이 망가진다.
- **전처리 계약**: 원본 데이터(텍스트, 이미지, 표 형태의 행)를 모델이 받을 수 있는 텐서로 바꾸는 대응 역시 입력 명세의 일부이다. 정규화된 데이터로 학습한 모델은 정규화되지 않은 입력에서 실패한다.

PyTorch의 데이터 적재는 명확한 입출력 사슬을 따른다. 원본 파일이 (텐서를 반환하는 `__getitem__`을 정의하는) `Dataset`을 거치고, 이어서 (배치를 만들고 섞는) `DataLoader`를 거쳐 마지막으로 모델에 들어간다.

---

## 3. 예제

```python
import torch
import torch.nn as nn

# 명시적인 입출력 계약을 가진 모델 정의
class Classifier(nn.Module):
    """입력: (batch, 10) float32. 출력: (batch, 3) 로짓."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[1] == 10, f"Expected (B, 10), got {x.shape}"
        return self.net(x)

model = Classifier()
x = torch.randn(4, 10)
logits = model(x)
probs = torch.softmax(logits, dim=1)
print(f"Input shape:  {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Probs sum:    {probs.sum(dim=1)}")  # [1, 1, 1, 1]이어야 한다

# 전처리를 일치시키는 것의 중요성을 보여준다
x_raw = torch.randn(4, 10) * 100  # 크기 조정 전
x_normalized = (x_raw - x_raw.mean(0)) / (x_raw.std(0) + 1e-8)
print(f"Raw logits range:   [{model(x_raw).min():.1f}, {model(x_raw).max():.1f}]")
print(f"Norm logits range:  [{model(x_normalized).min():.1f}, {model(x_normalized).max():.1f}]")
```

---

## 연습문제

**연습문제 1.**
어떤 모델은 채널 우선 형식의 $(B, 3, 224, 224)$ 모양 입력 텐서를 기대하지만, 데이터 로더는 채널 마지막 형식의 $(B, 224, 224, 3)$을 반환한다. 두 형식 사이를 변환하는 PyTorch 연산을 작성하라.

??? success "연습문제 1 풀이"
    `x = x.permute(0, 3, 1, 2)`는 $(B, H, W, C)$를 $(B, C, H, W)$로 변환한다. 또는 `x = x.movedim(-1, 1)`을 쓸 수 있다. 역변환은 `x.permute(0, 2, 3, 1)`이다. 기대하는 형식과 맞추지 않으면 모델이 행을 채널로 해석하여, 아무런 오류 메시지 없이 의미 없는 출력을 만들어낸다. 조용한 실패 유형이다.

---

**연습문제 2.**
모델 출력으로서의 로짓, 확률, 로그 확률의 차이를 설명하라. `nn.CrossEntropyLoss`에는 어느 것을 써야 하며 그 이유는 무엇인가?

??? success "연습문제 2 풀이"
    **로짓**: 정규화되지 않은 원시 점수 $z \in \mathbb{R}^K$. **확률**: $p_i = e^{z_i}/\sum_j e^{z_j}$이며 합이 1이다. **로그 확률**: $\log p_i = z_i - \log\sum_j e^{z_j}$. `nn.CrossEntropyLoss`에는 로짓을 써야 한다. 이 손실이 내부적으로 수치적 안정성을 위한 log-sum-exp 기법을 써서 `log_softmax + nll_loss`를 적용하기 때문이다. 확률을 넘기면 `log`를 다시 취해야 하는데 이는 0 근처에서 수치적으로 불안정하다. 로그 확률을 넘긴다면 대신 `nn.NLLLoss`를 써야 한다.

---

**연습문제 3.**
`Dataset.__getitem__`이 튜플 `(image, label)`을 반환한다. 이미지 크기가 서로 다를 때 `batch_size > 1`인 표준 `DataLoader`가 실패하는 이유를 설명하고 해결책을 제안하라.

??? success "연습문제 3 풀이"
    `DataLoader`는 `torch.stack`을 사용해 개별 표본을 배치 텐서로 쌓는데, 이는 모든 텐서가 같은 모양일 것을 요구한다. 크기가 다른 이미지는 모양이 다른 텐서를 만들어 실행 오류를 일으킨다. 해결책: (1) 전처리 단계에서 모든 이미지를 고정 크기로 조정한다(가장 흔함). (2) 배치 안의 최대 크기에 맞추어 이미지를 채워 넣는 사용자 정의 `collate_fn`을 작성한다. (3) `batch_size=1`인 `DataLoader`로 이미지를 개별 처리한다(학습에는 비현실적).

---

**연습문제 4.**
ImageNet 정규화된 입력(평균 $[0.485, 0.456, 0.406]$, 표준편차 $[0.229, 0.224, 0.225]$)으로 학습한 모델이 추론 시점에 원시 $[0, 255]$ uint8 이미지를 받는다. 필요한 전처리 파이프라인을 기술하고, 정규화를 건너뛰면 어떤 일이 생기는지 설명하라.

??? success "연습문제 4 풀이"
    파이프라인: (1) uint8을 float32로 변환한다. (2) 255로 나누어 $[0, 1]$ 범위로 만든다. (3) 채널별로 ImageNet 평균을 빼고 ImageNet 표준편차로 나눈다. 정규화가 없으면 모델은 학습 때와 완전히 다른 분포의 입력을 받는다. 첫 층의 활성값이 몇 자릿수만큼 지나치게 커지고(입력이 $\sim 0$이 아니라 $\sim 100$), 활성값이 포화되고 경사가 넘쳐 사실상 무작위 예측이 나온다. 모델 내부의 특징 검출기들은 정규화된 입력 범위에 맞추어 조정되어 있기 때문이다.

---

**연습문제 5.**
의미론적 분할(semantic segmentation) 모델의 입출력 명세를 설계하라. 크기 $512 \times 512$ 이미지와 20개 객체 클래스에 대해 입력 텐서 모양과 자료형, 출력 텐서 모양, 손실 함수, 평가 지표를 정의하라.

??? success "연습문제 5 풀이"
    **입력**: $(B, 3, 512, 512)$, float32, 데이터셋 통계로 정규화. **출력**: $(B, 20, 512, 512)$ 로짓 — 클래스마다 채널 하나이며 입력과 같은 공간 해상도를 가진다. **손실**: 픽셀별 교차 엔트로피 $\ell = \frac{1}{HW}\sum_{h,w}\text{CE}(\text{logits}_{:,h,w},\; y_{h,w})$이며 $y_{h,w} \in \{0, \ldots, 19\}$이다. **지표**: 평균 IoU(mIoU) $\text{mIoU} = \frac{1}{K}\sum_k \frac{|P_k \cap G_k|}{|P_k \cup G_k|}$이며 $P_k, G_k$는 클래스 $k$의 예측 픽셀 집합과 정답 픽셀 집합이다. 손실과 지표의 간극(CE 대 mIoU)은 학습 시 클래스 가중 손실이나 Dice 손실을 사용하여 완화한다.

## 정리하며

이 마당은 정의、설명、예제을 차례로 짚었다.
