# MC 드롭아웃: 근사 베이즈 추론
**몬테카를로 드롭아웃(MC 드롭아웃)**은 정칙화 기법인 드롭아웃을 근사 베이즈 추론으로 다시 본다. 테스트할 때도 드롭아웃을 켜 둔 채 순전파를 여러 번 돌리면, 보통의 드롭아웃으로 학습한 네트워크에서 학습 비용을 더 들이거나 구조를 고치지 않고도 불확실성 근사를 얻는다.

---

## 1. 핵심 통찰

### 학습하는 동안의 드롭아웃

보통의 드롭아웃은 학습하는 동안 은닉 유닛의 활성화를 무작위로 0으로 만든다.

$$
\tilde{h}_j = z_j \cdot h_j, \quad z_j \sim \text{Bernoulli}(1-p)
$$

여기서 $p$는 드롭아웃 확률이다.

### 근사 사후분포로서의 드롭아웃

갈과 가라마니(2016)는 드롭아웃으로 학습하는 일이 다음 둘 사이의 KL 발산을 근사로 가장 작게 함을 밝혔다.

- 드롭아웃 분포로 정의한 근사 사후분포 $q(\theta)$
- 참 사후분포 $p(\theta \mid \mathcal{D})$

**변분 분포**:

$$
q(W) = M \cdot \text{diag}(z), \quad z_i \sim \text{Bernoulli}(1-p)
$$

여기서 $M$은 학습한 웨이트 행렬이고 $z$는 베르누이 확률 변수의 벡터다.

### 정칙화에서 불확실성으로

**보통의 추론**: 드롭아웃을 끄고 스케일을 맞춘 웨이트를 쓴다 → 결정적 예측 하나

**MC 드롭아웃 추론**: 드롭아웃을 켜 두고 순전파를 $T$번 돌린다 → 순전파 결과 사이의 분산이 불확실성을 준다

$$
\boxed{p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid x^*, \hat{W} \cdot \text{diag}(z_t))}
$$

---

## 2. 이론 기초

### 변분 추론의 관점

MC 드롭아웃은 ELBO를 거쳐 참 사후분포에 대한 KL 발산을 가장 작게 한다.

$$
\mathcal{L} = -\mathbb{E}_{q(\theta)}[\log p(\mathcal{D} \mid \theta)] + \text{KL}(q(\theta) \| p(\theta))
$$

**드롭아웃에서는**:

- 기대 로그 우도를 보통의 드롭아웃 학습 로스로 추정한다
- KL 항은 L2 정칙화(웨이트 감쇠)에 대응한다

### 동치

크로스 엔트로피 로스와 L2 정칙화를 쓰는 **드롭아웃 학습**은 다음을 지닌 **변분 추론**과 같다.

- 베르누이 근사 사후분포
- 가우시안 사전분포 $p(\theta) = \mathcal{N}(0, \sigma^2 I)$

**사전분포의 정밀도**는 웨이트 감쇠와 이렇게 이어진다.

$$
\lambda = \frac{p \cdot l^2}{2N\tau}
$$

여기서 $l$은 사전분포의 길이 스케일, $N$은 데이터 수, $\tau$는 모델의 정밀도(관측 잡음 분산의 역수)다. 이 식의 $p$는 갈의 표기를 따라 유닛을 **남길** 확률이다.

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
        """드롭아웃을 켜거나 끈 앞으로 스텝."""
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

## 4. 불확실성 분해

### 회귀에서는

$$
\text{Var}[y^*] = \underbrace{\frac{1}{T}\sum_{t=1}^T (f_t(x^*) - \bar{f}(x^*))^2}_{\text{앎의}} + \underbrace{\sigma^2_{\text{noise}}}_{\text{타고난}}
$$

### 분류에서는

**전체 불확실성**: $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$

**우연적 불확실성**: $\mathbb{E}[\mathbb{H}[p]] = \frac{1}{T}\sum_t \mathbb{H}[p_t]$

**인식적 불확실성(상호 정보량)**: $\mathbb{I}[y; \theta] = \mathbb{H}[\bar{p}] - \mathbb{E}[\mathbb{H}[p]]$

---

## 5. 실전에서 고려할 것

### 표본의 수

| 표본 $T$ | 쓰일 자리 |
|-------------|----------|
| 10~30 | 빠른 근사 |
| 50~100 | 일반적인 추론 |
| 100~1000 | 안전이 중요한 판단 |

### 드롭아웃 확률

- $p = 0.1$~$0.3$: 가벼운 드롭아웃
- $p = 0.5$: 흔히 쓰는 값
- $p$가 클수록: 불확실성이 크나 정확도가 떨어질 수 있다

---

## 6. 다른 방법과 비교

| 특성 | MC 드롭아웃 | 딥 앙상블 | 변분 베이즈 뉴럴 네트워크 |
|--------|------------|----------------|-----------------|
| 학습 비용 | 네트워크 1개 | 네트워크 M개 | 네트워크 1개(변분 레이어로 고친 것) |
| 추론 비용 | 순전파 T번 | 순전파 M번 | 순전파 T번 |
| 구현 | 아주 쉬움 | 쉬움 | 중간 |
| 불확실성 품질 | 좋음 | 흔히 더 좋음 | 좋음 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 2-레이어 뉴럴 네트워크에서, 평균장 변분 추론 · 라플라스 근사 · MC 드롭아웃이 저마다 내놓는 근사 사후분포의 형태를 견주어라.

</div>

??? success "풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 세 근사 방법은 저마다 다룰 수 있는 형태를 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 중심으로 하고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 암묵적으로 정의된다. 근사마다 참 사후분포 형태의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MC 드롭아웃에서 얻은 불확실성 근사의 캘리브레이션을 딥 앙상블과 비교하는 실험을 설계하여라. 쓸 지표와 그림을 밝혀라.

</div>

??? success "풀이"
    지표: (1) 버킷 15개의 기대 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음의 로그 우도(NLL), (4) 분포 외 탐지의 AUROC. 그림: 방법마다 관측 빈도를 예측 신뢰도에 대고 그린 신뢰도 다이어그램(reliability diagram). 절차: 모든 방법을 CIFAR-10(분포 내)에서 학습하고, CIFAR-10 테스트 데이터에서 캘리브레이션을, SVHN에서 분포 외 탐지를 평가한다. 사후(post-hoc) 온도 스케일링을 베이스라인으로 쓴다. 무작위 시드 5개에 걸친 평균과 표준편차를 보고한다. 캘리브레이션이 잘 된 방법은 신뢰도 다이어그램에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 네트워크의 예측 분산이 인식적 불확실성과 우연적 불확실성으로 분해됨을 증명하여라. 학습 데이터 크기 $N \to \infty$일 때 두 항이 어떻게 되는지 보여라.

</div>

??? success "풀이"
    예측 분산은 전분산 법칙으로 분해된다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 항은 데이터 생성 과정의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 항은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 부근으로 수렴하므로 $O(1/N)$으로 준다. 극한에서는 우연적 불확실성만 남는다. 이 분해는 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 절의 불확실성 추정 방법을 트레이딩 시스템의 포지션 크기 결정에 어떻게 쓸 수 있는지 논하여라. 실용적인 판단 규칙을 제시하여라.

</div>

??? success "풀이"
    판단 규칙: 포지션 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측 수익률, $\sigma_e^2$을 인식적 분산이라 하자. 포지션은 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$는 위험 회피 계수다. 인식적 불확실성이 크면(낯선 시장 상황) 포지션을 줄이고, 작으면(익숙한 국면) 더 과감하게 거래한다. 여기에 더해 인식적 불확실성의 상한을 두고 그 위로는 거래하지 않게 해 기권(abstain)할 수 있다. 이 틀은 모델의 신뢰도로 스케일한 켈리 기준 방식의 포지션 크기 결정을 자연스럽게 구현한다. 워크포워드(walk-forward) 분석으로 백테스트해 $\lambda$를 보정한다. $\square$

## 정리하며

### 핵심 식

**사후분포 근사**: $q(W) = M \cdot \text{diag}(z)$, $z \sim \text{Bernoulli}(1-p)$

**예측 분포**: $p(y^*) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid W_t)$

### 장점과 한계

| 장점 | 한계 |
|------------|-------------|
| 구조를 고치지 않는다 | 드롭아웃이 있어야 한다 |
| 이미 있는 모델을 쓴다 | 불확실성을 낮게 볼 수 있다 |
| 구현이 단순하다 | 근사가 거칠다 |

### 핵심 살펴볼 거리

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning? *NeurIPS*.
