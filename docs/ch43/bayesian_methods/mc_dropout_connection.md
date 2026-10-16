# MC 드롭아웃: 근사 베이즈 추론
**몬테카를로 드롭아웃(MC 드롭아웃)**은 정규화 재주인 드롭아웃을 근사 베이즈 추론으로 다시 본다. 시험할 때도 드롭아웃을 켜 둔 채 앞으로 걸음을 여러 번 돌리면, 여느 드롭아웃으로 학습된 네트워크에서 값이 더 들거나 구조를 고치지 않고도 불확실성 근사를 얻는다.

---

## 1. 핵심 깨침

### 익히는 동안의 드롭아웃

여느 드롭아웃은 익히는 동안 살아남을 아무렇게나 0으로 만든다.

$$
\tilde{h}_j = z_j \cdot h_j, \quad z_j \sim \text{Bernoulli}(1-p)
$$

여기서 $p$은 드롭아웃 확률이다.

### 근사 사후분포로서의 드롭아웃

갈과 가라마니(2016)는 드롭아웃으로 익히는 일이 다음 둘 사이의 KL 갈림을 근사로 가장 작게 함을 밝혔다.

- 드롭아웃 분포로 세운 근사 사후분포 $q(\theta)$
- 참 사후분포 $p(\theta \mid \mathcal{D})$

**변이 분포**:

$$
q(W) = M \cdot \text{diag}(z), \quad z_i \sim \text{Bernoulli}(1-p)
$$

여기서 $M$은 배운 웨이트 행렬이고 $z$은 베르누이 아무 변수의 벡터다.

### 정규화에서 불확실성으로

**여느 추론**: 드롭아웃을 끄고 잣대 잡은 웨이트를 쓴다 → 붙박인 추론 하나

**MC 드롭아웃 추론**: 드롭아웃을 켜 두고 앞으로 걸음을 $T$번 돌린다 → 걸음 사이의 분산이 불확실성을 준다

$$
\boxed{p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid x^*, \hat{W} \cdot \text{diag}(z_t))}
$$

---

## 2. 이론 기초

### 변분 추론으로 본 눈

MC 드롭아웃은 ELBO를 거쳐 참 사후분포에 대한 KL 갈림을 가장 작게 한다.

$$
\mathcal{L} = -\mathbb{E}_{q(\theta)}[\log p(\mathcal{D} \mid \theta)] + \text{KL}(q(\theta) \| p(\theta))
$$

**드롭아웃에서는**:

- 바라는 로그 우도를 여느 드롭아웃 학습 로스로 추정한다
- KL 항은 L2 정규화(웨이트 줄이기)에 맞물린다

### 같음

엇갈린 엔트로피 로스와 L2 정규화를 쓰는 **드롭아웃 학습**은 다음을 지닌 **변분 추론**과 같다.

- 베르누이 근사 사후분포
- 가우시안 사전분포 $p(\theta) = \mathcal{N}(0, \sigma^2 I)$

**사전분포의 촘촘함**은 웨이트 줄이기와 이렇게 이어진다.

$$
\lambda = \frac{p \cdot l^2}{2N\tau}
$$

---

## 3. 구현

```python
"""
MC 드롭아웃: 베이즈 어림으로서의 드롭아웃
"""

import numpy as np
from typing import Tuple, List

class MCDropoutMLP:
    """아리송함을 어림하는 MC 드롭아웃 MLP."""

    def __init__(self, layer_sizes: List[int], dropout_prob: float = 0.5):
        self.layer_sizes = layer_sizes
        self.dropout_prob = dropout_prob

        # 짐의 첫자리를 잡는다
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """드롭아웃을 켜거나 끈 앞으로 걸음."""
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:  # 마지막 켜가 아니면
                h = np.maximum(0, h)  # ReLU
                if training:
                    mask = (np.random.rand(*h.shape) > self.dropout_prob)
                    h = h * mask / (1 - self.dropout_prob)
        return h

    def predict_with_uncertainty(
        self, x: np.ndarray, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """아리송함을 곁들인 MC 드롭아웃 미루어 봄."""
        predictions = [self.forward(x, training=True) for _ in range(n_samples)]
        predictions = np.array(predictions)

        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std
```

---

## 4. 불확실성 쪼개기

### 회귀에서는

$$
\text{Var}[y^*] = \underbrace{\frac{1}{T}\sum_{t=1}^T (f_t(x^*) - \bar{f}(x^*))^2}_{\text{앎의}} + \underbrace{\sigma^2_{\text{noise}}}_{\text{타고난}}
$$

### 분류에서는

**온 불확실성**: $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$

**우연적 것**: $\mathbb{E}[\mathbb{H}[p]] = \frac{1}{T}\sum_t \mathbb{H}[p_t]$

**인식적 것(서로 나눈 소식)**: $\mathbb{I}[y; \theta] = \mathbb{H}[\bar{p}] - \mathbb{E}[\mathbb{H}[p]]$

---

## 5. 참으로 헤아릴 것

### 표본의 수

| 표본 $T$ | 쓰일 자리 |
|-------------|----------|
| 10~30 | 빠른 근사 |
| 50~100 | 여느 추론 |
| 100~1000 | 목숨이 걸린 판단 |

### 드롭아웃 확률

- $p = 0.1$~$0.3$: 가벼운 드롭아웃
- $p = 0.5$: 여느 값
- $p$이 클수록: 불확실성이 크나 맞음이 떨어질 수 있다

---

## 6. 다른 방법과 비교

| 특성 | MC 드롭아웃 | 깊은 앙상블 | 변분 베이즈 뉴럴 넷트웍 |
|--------|------------|----------------|-----------------|
| 학습 값 | 네트워크 1개 | 네트워크 M개 | 네트워크 1개(고침) |
| 추론 값 | 걸음 T번 | 걸음 M번 | 걸음 T번 |
| 짜기 | 아주 쉬움 | 쉬움 | 가운데 |
| 불확실성 성능 | 좋음 | 흔히 더 좋음 | 좋음 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 두 켜 뉴럴 넷트웍에서, 이 마디에서 밝힌 방법에 따른 근사 사후분포의 꼴을 이끌어 내어라.

</div>

??? success "연습문제 1 풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 근사 방법은 다룰 수 있는 꼴을 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 가운데로 삼고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 넌지시 세워진다. 근사마다 참 사후분포 모습의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 방법에서 얻은 불확실성 근사의 캘리브레이션을 MC 드롭아웃, 깊은 앙상블과 비교하는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

</div>

??? success "연습문제 2 풀이"
    자: (1) 통 15개의 바라는 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음수 로그 우도(NLL), (4) 분포 외 알아내기의 AUROC. 그림: 방법마다 본 잦기를 예측한 신뢰도에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 데이터에서 캘리브레이션을, SVHN에서 분포 외 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 베이스라인으로 쓴다. 아무렇게나 하는 시드 5개에 걸친 평균과 표준편차를 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 넷트웍의 추론 분산이 인식적 불확실성과 우연적 불확실성으로 쪼개짐을 증명하여라. 학습 데이터 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

</div>

??? success "연습문제 3 풀이"
    추론 분산은 온 분산 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 몫은 데이터를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 몫은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 우연적 불확실성만 남는다. 이 쪼갬은 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 마디의 불확실성 재기 방법을 거래 구조의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

</div>

??? success "연습문제 4 풀이"
    판단 규칙: 자리 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측한 돌아옴, $\sigma_e^2$을 인식적 분산이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 위험 꺼림 값이다. 인식적 불확실성이 크면(낯선 시장 상황) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 인식적 불확실성 위끝을 두어 삼갈 수 있다. 이 틀은 모델의 신뢰도로 잣대를 잡은 켈리 잣대 특성의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

### 핵심 식

**사후분포 근사**: $q(W) = M \cdot \text{diag}(z)$, $z \sim \text{Bernoulli}(1-p)$

**예측하는 분포**: $p(y^*) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid W_t)$

### 나은 점과 한계

| 나은 점 | 한계 |
|------------|-------------|
| 구조를 고치지 않는다 | 드롭아웃이 있어야 한다 |
| 이미 있는 모델을 쓴다 | 불확실성을 낮게 볼 수 있다 |
| 짜기가 단순하다 | 근사가 거칠다 |

### 핵심 살펴볼 거리

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning? *NeurIPS*.
