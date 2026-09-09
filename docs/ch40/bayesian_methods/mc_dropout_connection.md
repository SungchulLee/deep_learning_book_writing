# MC 드롭아웃: 어림 베이즈 미루어 봄으로서의 드롭아웃
**몬테카를로 드롭아웃(MC 드롭아웃)**은 정칙화 재주인 드롭아웃을 어림 베이즈 미루어 봄으로 다시 본다. 시험할 때도 드롭아웃을 켜 둔 채 앞으로 걸음을 여러 번 돌리면, 여느 드롭아웃으로 익힌 그물에서 값이 더 들거나 얼개를 고치지 않고도 아리송함 어림을 얻는다.

---

## 1. 고갱이 깨침

### 익히는 동안의 드롭아웃

여느 드롭아웃은 익히는 동안 살아남을 아무렇게나 0으로 만든다.

$$
\tilde{h}_j = z_j \cdot h_j, \quad z_j \sim \text{Bernoulli}(1-p)
$$

여기서 $p$은 드롭아웃 낌새다.

### 어림 뒷분포로서의 드롭아웃

갈과 가라마니(2016)는 드롭아웃으로 익히는 일이 다음 둘 사이의 KL 갈림을 어림으로 가장 작게 함을 밝혔다.

- 드롭아웃 분포로 세운 어림 뒷분포 $q(\theta)$
- 참 뒷분포 $p(\theta \mid \mathcal{D})$

**변이 분포**:

$$
q(W) = M \cdot \text{diag}(z), \quad z_i \sim \text{Bernoulli}(1-p)
$$

여기서 $M$은 배운 짐 행렬이고 $z$은 베르누이 아무 변수의 벡터다.

### 정칙화에서 아리송함으로

**여느 미루어 봄**: 드롭아웃을 끄고 잣대 잡은 짐을 쓴다 → 붙박인 미루어 봄 하나

**MC 드롭아웃 미루어 봄**: 드롭아웃을 켜 두고 앞으로 걸음을 $T$번 돌린다 → 걸음 사이의 흩어짐이 아리송함을 준다

$$
\boxed{p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid x^*, \hat{W} \cdot \text{diag}(z_t))}
$$

---

## 2. 이론 밑바탕

### 변이 미루어 봄으로 본 눈

MC 드롭아웃은 ELBO를 거쳐 참 뒷분포에 대한 KL 갈림을 가장 작게 한다.

$$
\mathcal{L} = -\mathbb{E}_{q(\theta)}[\log p(\mathcal{D} \mid \theta)] + \text{KL}(q(\theta) \| p(\theta))
$$

**드롭아웃에서는**:

- 바라는 로그 그럴듯함을 여느 드롭아웃 익힘 잃음으로 어림한다
- KL 항은 L2 정칙화(짐 줄이기)에 맞물린다

### 같음

엇갈린 엔트로피 잃음과 L2 정칙화를 쓰는 **드롭아웃 익힘**은 다음을 지닌 **변이 미루어 봄**과 같다.

- 베르누이 어림 뒷분포
- 가우스 앞선 분포 $p(\theta) = \mathcal{N}(0, \sigma^2 I)$

**앞선 분포의 촘촘함**은 짐 줄이기와 이렇게 이어진다.

$$
\lambda = \frac{p \cdot l^2}{2N\tau}
$$

---

## 3. 짜보기

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

## 4. 아리송함 쪼개기

### 되돌이에서는

$$
\text{Var}[y^*] = \underbrace{\frac{1}{T}\sum_{t=1}^T (f_t(x^*) - \bar{f}(x^*))^2}_{\text{앎의}} + \underbrace{\sigma^2_{\text{noise}}}_{\text{타고난}}
$$

### 가름에서는

**온 아리송함**: $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$

**타고난 것**: $\mathbb{E}[\mathbb{H}[p]] = \frac{1}{T}\sum_t \mathbb{H}[p_t]$

**앎의 것(서로 나눈 소식)**: $\mathbb{I}[y; \theta] = \mathbb{H}[\bar{p}] - \mathbb{E}[\mathbb{H}[p]]$

---

## 5. 참으로 헤아릴 것

### 표본의 수

| 표본 $T$ | 쓰일 자리 |
|-------------|----------|
| 10~30 | 빠른 어림 |
| 50~100 | 여느 미루어 봄 |
| 100~1000 | 목숨이 걸린 판단 |

### 드롭아웃 낌새

- $p = 0.1$~$0.3$: 가벼운 드롭아웃
- $p = 0.5$: 여느 값
- $p$이 클수록: 아리송함이 크나 맞음이 떨어질 수 있다

---

## 6. 다른 방법과 견주기

| 결 | MC 드롭아웃 | 깊은 모둠 | 변이 베이즈 신경 그물 |
|--------|------------|----------------|-----------------|
| 익힘 값 | 그물 1개 | 그물 M개 | 그물 1개(고침) |
| 미루어 봄 값 | 걸음 T번 | 걸음 M번 | 걸음 T번 |
| 짜기 | 아주 쉬움 | 쉬움 | 가운데 |
| 아리송함 됨됨이 | 좋음 | 흔히 더 좋음 | 좋음 |

---

## 연습문제

**연습문제 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "연습문제 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**연습문제 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "연습문제 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**연습문제 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "연습문제 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**연습문제 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "연습문제 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

### 고갱이 식

**뒷분포 어림**: $q(W) = M \cdot \text{diag}(z)$, $z \sim \text{Bernoulli}(1-p)$

**미루어 보는 분포**: $p(y^*) \approx \frac{1}{T}\sum_{t=1}^T p(y^* \mid W_t)$

### 나은 점과 한계

| 나은 점 | 한계 |
|------------|-------------|
| 얼개를 고치지 않는다 | 드롭아웃이 있어야 한다 |
| 이미 있는 모형을 쓴다 | 아리송함을 낮게 볼 수 있다 |
| 짜기가 단순하다 | 어림이 거칠다 |

### 고갱이 살펴볼 거리

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning? *NeurIPS*.
