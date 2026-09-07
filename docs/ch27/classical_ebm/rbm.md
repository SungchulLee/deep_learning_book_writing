# 제한 볼츠만 기계



## 학습 목표

이 절을 마치면 다음을 할 수 있다:

1. 제한 볼츠만 기계의 두 쪽 얼개를 이해한다
2. 다룰 만한 조건 분포를 이끌어 낸다
3. 맞댐 벌어짐 익히기를 짠다
4. 실제 자료(MNIST)로 제한 볼츠만 기계를 익힌다
5. 배운 특징을 그려 보고 되짓기를 한다

## 들어가며

제한 볼츠만 기계(RBM)는 에너지 바탕 배움의 가장 성공한 실제 쓰임새이다. "제한"은 이음이 드러난 층과 숨은 층 사이에만 있고 같은 층 안에는 없다는 제약을 가리킨다. 이 두 쪽 얼개 덕에 덩이 깁스 뽑기로 추론이 다룰 만해지고 맞댐 벌어짐으로 효율 좋게 익힐 수 있다.

## 구조

### 두 쪽 그래프 얼개

제한 볼츠만 기계는 정해진 이음 무늬를 가진다:

```
Visible Layer (v):    ○  ○  ○  ○  ○  ○
                       \\ | // \\ | //
                        \\|//   \\|//
Hidden Layer (h):        ○       ○       ○
```

**핵심 제약**:

- 드러난 단위끼리의 이음 없음($W_{vv} = 0$)
- 숨은 단위끼리의 이음 없음($W_{hh} = 0$)
- 드러난 단위와 숨은 단위 사이의 이음만 있음($W_{vh}$)

### 에너지 함수

제한 볼츠만 기계의 에너지 함수는 다음과 같다:

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h} - \mathbf{v}^T \mathbf{W} \mathbf{h}$$

여기서 각 기호는 다음과 같다.

- $\mathbf{v} \in \{0, 1\}^{n_v}$: 드러난 단위(자료)
- $\mathbf{h} \in \{0, 1\}^{n_h}$: 숨은 단위(특징)
- $\mathbf{W} \in \mathbb{R}^{n_v \times n_h}$: 무게 행렬
- $\mathbf{a} \in \mathbb{R}^{n_v}$: 드러난 단위의 치우침
- $\mathbf{b} \in \mathbb{R}^{n_h}$: 숨은 단위의 치우침

## 다룰 만한 조건 분포

### 핵심 통찰

두 쪽 얼개 덕에 조건 분포가 인수로 나뉜다:

$$P(\mathbf{h} | \mathbf{v}) = \prod_j P(h_j | \mathbf{v})$$

$$P(\mathbf{v} | \mathbf{h}) = \prod_i P(v_i | \mathbf{h})$$

### 유도

숨은 단위 $j$에 대해:

$$P(h_j = 1 | \mathbf{v}) = \sigma(b_j + \sum_i W_{ij} v_i) = \sigma(\mathbf{W}_{:,j}^T \mathbf{v} + b_j)$$

드러난 단위 $i$에 대해:

$$P(v_i = 1 | \mathbf{h}) = \sigma(a_i + \sum_j W_{ij} h_j) = \sigma(\mathbf{W}_{i,:} \mathbf{h} + a_i)$$

여기서 $\sigma(x) = 1/(1 + e^{-x})$은 시그모이드 함수이다.

**이 인수 나눔 덕에 한 층의 모든 단위를 나란히 뽑을 수 있다!**

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class RestrictedBoltzmannMachine(nn.Module):
    """
    이진 드러난 단위와 숨은 단위를 가진 제한 볼츠만 기계.
    
    얼개: 드러난 단위와 숨은 단위 사이의 이음만 있는 두 쪽 그래프.
    익히기: 견줌 갈림(CD-k)
    
    매개변수
    ----------
    n_visible : int
        드러난 단위의 개수
    n_hidden : int
        숨은 단위의 개수
    k : int
        CD-k에서 깁스 걸음의 수(기본값 1)
    learning_rate : float
        매개변수 고침의 배움 빠르기
    """
    
    def __init__(self, 
                 n_visible: int, 
                 n_hidden: int,
                 k: int = 1,
                 learning_rate: float = 0.01):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = learning_rate
        
        # 매개변수
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))   # visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))    # hidden bias
    
    def sample_hidden(self, v: torch.Tensor) -> tuple:
        """
        드러난 단위가 주어질 때 숨은 단위를 뽑는다.
        
        P(h_j = 1 | v) = σ(W_j · v + b_j)
        """
        activation = F.linear(v, self.W, self.b)
        prob_h = torch.sigmoid(activation)
        sample_h = torch.bernoulli(prob_h)
        return prob_h, sample_h
    
    def sample_visible(self, h: torch.Tensor) -> tuple:
        """
        숨은 단위가 주어질 때 드러난 단위를 뽑는다.
        
        P(v_i = 1 | h) = σ(W^T_i · h + a_i)
        """
        activation = F.linear(h, self.W.t(), self.a)
        prob_v = torch.sigmoid(activation)
        sample_v = torch.bernoulli(prob_v)
        return prob_v, sample_v
    
    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute energy E(v, h) = -a^T v - b^T h - v^T W^T h"""
        visible_term = torch.einsum('bi,i->b', v, self.a)
        hidden_term = torch.einsum('bj,j->b', h, self.b)
        interaction_term = torch.einsum('bi,ji,bj->b', v, self.W, h)
        return -(visible_term + hidden_term + interaction_term)
    
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        자유 힘 F(v) = -log Σ_h exp(-E(v,h))을 셈한다
        
        F(v) = -a^T v - Σ_j log(1 + exp(b_j + W_j · v))
        """
        visible_term = torch.einsum('bi,i->b', v, self.a)
        wx_b = F.linear(v, self.W, self.b)
        hidden_term = F.softplus(wx_b).sum(dim=1)
        return -(visible_term + hidden_term)
    
    def contrastive_divergence(self, v0: torch.Tensor) -> float:
        """
        맞댐 벌어짐 CD-k 익히기 걸음.
        """
        batch_size = v0.shape[0]
        
        # 양의 국면
        prob_h0, h0 = self.sample_hidden(v0)
        positive_grad = torch.matmul(prob_h0.t(), v0) / batch_size
        
        # 음의 국면(깁스 k걸음)
        vk, hk = v0, h0
        for _ in range(self.k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        
        negative_grad = torch.matmul(hk.t(), vk) / batch_size
        
        # 매개변수 갱신
        self.W.data += self.lr * (positive_grad - negative_grad)
        self.a.data += self.lr * (v0 - vk).mean(dim=0)
        self.b.data += self.lr * (prob_h0 - hk).mean(dim=0)
        
        return ((v0 - vk) ** 2).sum(dim=1).mean().item()
    
    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """Reconstruct: v → h → v'"""
        _, h = self.sample_hidden(v)
        prob_v, _ = self.sample_visible(h)
        return prob_v
```

## 맞댐 벌어짐

### CD-k 알고리즘

**정확한 기울기**:

$$\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

**CD-k 어림**:

$$\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} \approx \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{k}$$

### 맞댐 벌어짐이 통하는 까닭

- 자료에서 시작하면 사슬이 더 빨리 섞인다
- 좋은 결과를 얻는 데 흔히 $k=1$이면 넉넉하다
- 이 어림은 치우쳐 있지만 흩어짐이 작다

## 핵심 정리

!!! success "핵심 개념"

    1. 제한 볼츠만 기계는 이음을 드러난 단위와 숨은 단위의 짝으로만 제한한다
    2. 두 쪽 얼개가 다룰 만한 조건 분포를 가능하게 한다
    3. 맞댐 벌어짐이 효율 좋은 어림 기울기를 준다
    4. 자유 에너지는 드러난 단위에 대한 실제 에너지이다
    5. 제한 볼츠만 기계는 쓸모 있는 특징 나타냄을 배운다

## 참고 문헌

- Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence.
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A Fast Learning Algorithm for Deep Belief Nets.

## 연습문제

**연습문제 1.**
결합 에너지 $E(\mathbf{v}, \mathbf{h})$에서 숨은 단위를 닫힌 꼴로 가장자리로 몰아내어 이진 제한 볼츠만 기계의 자유 에너지 $F(\mathbf{v})$을 이끌어 내라. $\mathbf{h}$에 대한 합의 걸음마다 보여라.

??? success "연습문제 1 풀이"
    결합 분포에서 시작한다:

    $$P(\mathbf{v}) = \frac{1}{Z}\sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$$

    제한 볼츠만 기계의 에너지를 넣으면:

    $$\sum_{\mathbf{h}} \exp\!\left(\mathbf{a}^T \mathbf{v} + \mathbf{b}^T \mathbf{h} + \mathbf{v}^T \mathbf{W} \mathbf{h}\right)$$

    $\mathbf{v}$이 주어지면 숨은 단위가 서로 얽매이지 않으므로 합이 인수로 나뉜다:

    $$= \exp(\mathbf{a}^T \mathbf{v}) \prod_{j=1}^{n_h} \sum_{h_j \in \{0,1\}} \exp\!\left((b_j + \mathbf{W}_{:,j}^T \mathbf{v}) h_j\right)$$

    인수마다 $1 + \exp(b_j + \mathbf{W}_{:,j}^T \mathbf{v})$이 되어 다음을 얻는다:

    $$F(\mathbf{v}) = -\mathbf{a}^T \mathbf{v} - \sum_{j=1}^{n_h} \log\!\left(1 + \exp(b_j + \mathbf{W}_{:,j}^T \mathbf{v})\right)$$

    이는 짜기에서 쓴 `softplus` 적기와 맞는다. $\square$

---

**연습문제 2.**
CD-1 고침 규칙이 로그 가능도 기울기의 치우친 어림개임을 보여라. 자세히는 음의 국면 표본 $\langle v_i h_j \rangle_1$이 왜 모델 기댓값 $\langle v_i h_j \rangle_{\text{model}}$과 같지 않은지 밝히고, 치우침이 줄어드는 조건을 적어라.

??? success "연습문제 2 풀이"
    정확한 기울기는 $\langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$이며 모델 기댓값에는 평형 분포의 표본이 필요하다. CD-1은 마르코프 사슬을 자료 점에서 첫자리매김하고 깁스 걸음을 한 번만 돌려 평형에 이르지 못한 분포 $q^{(1)}$의 표본을 만든다. 따라서 $\langle v_i h_j \rangle_1 \neq \langle v_i h_j \rangle_{\text{model}}$이며 이 어림개는 치우쳐 있다. $k$이 커질수록 사슬이 평형에 가까워지므로 치우침이 줄어든다. 실제로는 자료 점에서 첫자리매김하면 사슬이 확률 높은 자리 가까이에 놓이므로 치우침이 작다. 게다가 이어지는 맞댐 벌어짐(PCD)은 고침에 걸쳐 사슬을 이어 가며 익히는 동안 섞이게 하여 치우침을 더 줄인다. $\square$

---

**연습문제 3.**
드러난 단위가 조건 분포 $P(v_i | \mathbf{h}) = \mathcal{N}(\mu_i, \sigma_i^2)$을 가진 실수인 정규-베르누이 제한 볼츠만 기계를 짜라. 고친 에너지 함수와 $\mu_i$, $\sigma_i$의 고침 규칙을 이끌어 내라.

??? success "연습문제 3 풀이"
    정규 드러난 단위에서 에너지는 다음과 같이 된다:

    $$E(\mathbf{v}, \mathbf{h}) = \sum_i \frac{(v_i - a_i)^2}{2\sigma_i^2} - \sum_j b_j h_j - \sum_{i,j} \frac{v_i}{\sigma_i} W_{ij} h_j$$

    숨은 단위가 주어졌을 때 드러난 단위의 조건 분포는 다음과 같다:

    $$P(v_i | \mathbf{h}) = \mathcal{N}\!\left(a_i + \sigma_i \sum_j W_{ij} h_j,\; \sigma_i^2\right)$$

    숨은 단위의 조건 분포는 여전히 시그모이드이다: $P(h_j = 1 | \mathbf{v}) = \sigma\!\left(b_j + \sum_i \frac{v_i}{\sigma_i} W_{ij}\right)$. 맞댐 벌어짐 고침 규칙은 얼개가 같지만(양의 국면 통계에서 음의 국면 통계를 뺀다) 드러난 단위 뽑기는 이제 베르누이가 아니라 정규 분포에서 뽑는다. 흩어짐 $\sigma_i^2$은 로그 가능도에 기울기 내려가기를 하여 배우거나 간단히 1로 붙박이할 수 있다. $\square$

---

**연습문제 4.**
$F$이 자유 에너지일 때 제한 볼츠만 기계의 나눔 함수가 $Z = \sum_{\mathbf{v}} \exp(-F(\mathbf{v}))$을 만족함을 밝혀라. 그다음 흔한 제한 볼츠만 기계 크기에서 $Z$을 정확히 셈하는 것이 왜 다룰 수 없는지 밝혀라.

??? success "연습문제 4 풀이"
    뜻매김에 따라 $Z = \sum_{\mathbf{v}, \mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$이다. 이를 다음과 같이 고쳐 쓸 수 있다:

    $$Z = \sum_{\mathbf{v}} \sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h})) = \sum_{\mathbf{v}} \exp(-F(\mathbf{v}))$$

    여기서 $F(\mathbf{v}) = -\log \sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$은 자유 에너지이다. $F(\mathbf{v})$이 숨은 단위를 닫힌 꼴로 가장자리로 몰아내므로(두 쪽 얼개 덕에 다룰 만하다) 이는 정확하다. 그러나 $Z$을 셈하려면 드러난 단위의 자리 얽이 $2^{n_v}$가지 모두에 대해 $\exp(-F(\mathbf{v}))$을 더해야 한다. $n_v = 784$(MNIST)인 흔한 제한 볼츠만 기계에서 이 합은 항이 $2^{784}$개여서 정확한 셈이 다룰 수 없다. $\square$
