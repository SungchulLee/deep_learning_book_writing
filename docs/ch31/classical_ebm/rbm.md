# 제한 볼츠만 기계(RBM)

일반 볼츠만 기계의 추론을 단순화하려면 연결 구조를 제한할 수 있다. RBM은 가시레이어와 은닉 레이어 사이에만 연결을 두어, 한 레이어를 고정했을 때 다른 레이어의 각 단위가 조건부 독립이 되게 한다. 이 성질은 병렬 블록 깁스 갱신을 가능하게 하지만 전체 정규화 상수를 쉽게 만들어 주지는 않는다.

## 1. 이분 그래프와 변수 관례

```
Visible Layer (v):    ○  ○  ○  ○  ○  ○
                       \\ | // \\ | //
                        \\|//   \\|//
Hidden Layer (h):        ○       ○       ○
```

가시변수 vᵢ와 은닉변수 hⱼ는 0 또는 1을 사용한다. 이 페이지에서는 **코드에 맞추어 W의 모양을 n_h×n_v**로 둔다. 가시 바이어스는 a, 은닉 바이어스는 b이며 온도는 1이다.

$$E(v,h)=-a^Tv-b^Th-h^TWv$$

가시·가시 또는 은닉·은닉 연결은 없다. ±1 스핀 에너지의 조건부 확률에서 나타나는 계수 2를 이 0·1 정의에 그대로 붙이지 않는다. 연결 행렬을 전치한 n_v×n_h 관례와도 혼동하지 않는다.

## 2. 조건부 확률의 인수분해

v를 고정하면 에너지는 hⱼ별 선형 항으로 나뉜다. 각 0·1 선택의 비정규화 웨이트는 1과 exp(bⱼ+ΣᵢWⱼᵢvᵢ)이므로 다음을 얻는다.

$$p(h_j=1\mid v)=\operatorname{sigmoid}\left(b_j+\sum_iW_{ji}v_i\right)$$

같은 방법으로 가시 단위의 조건부를 계산한다.

$$p(v_i=1\mid h)=\operatorname{sigmoid}\left(a_i+\sum_jW_{ji}h_j\right)$$

주어진 반대 레이어 안에서 각 단위가 독립이므로 한 레이어 전체를 동시에 뽑을 수 있다. 그러나 주변분포 p(v)의 가시 단위들이 서로 독립이라는 뜻은 아니다. 은닉변수를 합하면 가시 단위 사이의 의존성을 표현할 수 있다.

## 3. 구현과 텐서 모양

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
    
    구조: 드러난 단위와 숨은 단위 사이의 이음만 있는 두 쪽 그래프.
    익히기: 견줌 갈림(CD-k)
    
    파라메터
    ----------
    n_visible : int
        드러난 단위의 개수
    n_hidden : int
        숨은 단위의 개수
    k : int
        CD-k에서 깁스 스텝의 수(기본값 1)
    learning_rate : float
        파라메터 고침의 러닝 빠르기
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
        
        # 파라메터
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
        맞댐 벌어짐 CD-k 익히기 스텝.
        """
        batch_size = v0.shape[0]
        
        # 양의 국면
        prob_h0, h0 = self.sample_hidden(v0)
        positive_grad = torch.matmul(prob_h0.t(), v0) / batch_size
        
        # 음의 국면(깁스 k스텝)
        vk, hk = v0, h0
        for _ in range(self.k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        
        negative_grad = torch.matmul(hk.t(), vk) / batch_size
        
        # 파라메터 갱신
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

입력 v는 B×n_v, 은닉 확률·샘플은 B×n_h다. F.linear(v,W,b)는 vWᵀ+b를 계산한다. 가시 복원에는 Wᵀ를 웨이트로 넘겨 hW+a를 계산한다. 에너지와 자유 에너지는 각 관측치의 길이 B 벡터를 반환한다.

free_energy는 은닉 상태를 정확히 합한 다음 식을 사용한다.

$$F(v)=-a^Tv-\sum_j\operatorname{softplus}\left(b_j+\sum_iW_{ji}v_i\right)$$

이는 가시 상태에 의존하는 자유 에너지이며, 모델 전체의 단일 스칼라 −log Z와 다른 양이다. p(v)=exp(−F(v))/Z를 계산하려면 여전히 Z가 필요하다.

## 4. CD-k의 업데이트

관측 v에서 시작해 h|v와 v|h를 k번 교대로 뽑고, 데이터의 충분통계에서 짧은 사슬의 충분통계를 뺀다. 정확한 우도 그래디언트의 negative phase는 평형 모델 분포를 요구하지만 CD-k는 일반적으로 거기에 도달하지 않은 분포를 사용한다.

코드의 positive phase는 은닉 확률을 사용해 prob_h0ᵀv0/B를 계산한다. negative phase는 마지막 **은닉 샘플 hk**를 사용해 hkᵀvk/B를 계산한다. 마지막 은닉 확률로 대체하는 다른 구현과 표본 분산이 다를 수 있다. a와 b도 데이터·사슬 평균 차이로 직접 갱신한다.

nn.Parameter를 사용하지만 optimizer와 backward를 호출하는 학습 루프는 아니다. .data에 러닝레이트를 곱한 통계 차이를 더한다. 자동 미분 문맥을 끄지는 않아 중간 연산의 그래프가 불필요하게 만들어질 수 있다. k는 1 이상의 정수여야 하지만 함수가 검사하지 않는다. k=0이면 negative phase에 초기 은닉 샘플을 사용하는 다른 계산이 된다.

CD는 유한 k에서 바이어스된 우도 그래디언트 근사다. 데이터 시작이 모든 모드를 탐색한다는 보장이나 k=1이면 언제나 충분하다는 결론은 없다. 적절한 고정 모델 사슬에서 k→∞의 수렴을 말할 수 있어도 유한 k의 바이어스가 매 단계 단조롭게 줄거나, 학습 중 persistent chain이 자동으로 평형을 유지하는 것은 아니다.

## 5. 재구성과 생성 평가

contrastive_divergence의 반환값은 (v0−vk)²를 가시 차원에서 합한 뒤 배치 평균한 재구성 오차다. 이진 입력이면 비트 불일치 개수의 평균이며, 정확한 NLL이나 정규화된 우도는 아니다. 낮은 재구성 오차만으로 새로운 표본의 분포 품질을 보장하지 않는다.

reconstruct는 은닉 상태를 **한 번 샘플링**하고 가시 확률을 반환한다. 은닉 샘플이 달라지면 재구성 확률도 달라질 수 있으므로 결정적인 오토인코더와 다르다. 테스트 입력을 주고 한 번 복원하는 것은 모델 평형에서 독립 생성한 이미지와도 다르다.

이 페이지에는 클래스만 정의되어 있다. DataLoader와 MNIST를 import하지만 실제 데이터 로드·훈련 반복·생성 그림이나 평가 결과는 없다. 0~1의 회색조 입력을 그대로 넣는 관행과 이산 Bernoulli 관측모델의 가정을 구분하고, 실제 이진 샘플링 또는 별도 관측분포가 필요한지 결정해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
코드의 행렬 방향을 사용해 은닉변수를 합하고 F(v)를 유도하라.

</div>

??? success "연습문제 1 풀이"

    v를 고정하면 exp(−E)=exp(aᵀv)∏ⱼexp(hⱼ[bⱼ+ΣᵢWⱼᵢvᵢ])다. 각 hⱼ∈{0,1}를 합하면 1+exp(bⱼ+ΣᵢWⱼᵢvᵢ)를 얻는다. 곱의 음의 로그를 취하면 F(v)=−aᵀv−Σⱼsoftplus(bⱼ+ΣᵢWⱼᵢvᵢ)이며 W는 n_h×n_v다. Z를 구한 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
CD-k의 negative phase가 정확한 우도 그래디언트의 모델 기댓값과 일반적으로 다른 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    데이터 분포 q₀에서 k번 전이한 qₖ를 사용하기 때문이다. qₖ가 평형 pθ와 다르면 해당 충분통계의 기대도 달라질 수 있다. 고정 모델의 적절한 사슬이 충분히 혼합하면 차이가 사라지지만, 짧은 사슬·다중 모드·움직이는 파라미터에서는 보장되지 않는다. 재구성 오차 감소만으로 이 차이가 작다고 판단할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
가우스 가시변수를 사용하는 RBM의 조건부분포와 파라미터 그래디언트를 유도하라. σᵢ=exp(ρᵢ)>0으로 두고 에너지를 Σᵢ(vᵢ−aᵢ)²/(2σᵢ²)−bᵀh−Σᵢⱼ(vᵢ/σᵢ)Wⱼᵢhⱼ로 정하라.

</div>

??? success "연습문제 3 풀이"

    vᵢ의 제곱을 완성하면 vᵢ|h∼N(aᵢ+σᵢΣⱼWⱼᵢhⱼ,σᵢ²)다. hⱼ|v의 성공확률은 sigmoid(bⱼ+ΣᵢWⱼᵢvᵢ/σᵢ)다. 일반식 ∇log p(v)=E_model[∇E]−E_{h|v}[∇E]를 사용한다. Wⱼᵢ의 그래디언트는 (vᵢ/σᵢ)hⱼ의 positive phase 평균에서 모델 평균을 뺀 값이고, aᵢ는 (vᵢ−aᵢ)/σᵢ²의 같은 차이, bⱼ는 hⱼ의 같은 차이다. ρᵢ에 대해서는 Aᵢ=(vᵢ−aᵢ)²/σᵢ²−(vᵢ/σᵢ)ΣⱼWⱼᵢhⱼ를 정의하면 E_positive[Aᵢ]−E_model[Aᵢ]다. 로그 우도를 최대화하려면 이 그래디언트 방향으로 더하며, NLL을 최소화할 때는 반대 부호를 쓴다. 이 조건부 평균은 별도 자유 모수가 아니라 a·σ·W와 h의 함수다. 여기의 Bernoulli 클래스에는 이 가우스 샘플링이 구현되어 있지 않다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
은닉변수 합이 닫힌 형태이면 Z도 쉽게 계산되는가?

</div>

??? success "연습문제 4 풀이"

    Z=Σ_vΣ_h exp(−E)=Σ_v exp(−F(v))다. F(v)는 쉽게 계산해도 가시 상태 전체는 2^n_v개다. 은닉 쪽을 대신 열거해 더 작은 레이어를 사용할 수도 있지만 일반적으로 작은 쪽도 크면 지수적 계산이 남는다. 조건부 추론의 단순함과 전체 정규화의 단순함을 구분한다.



## 정리하며

RBM은 이분 그래프로 조건부 독립성과 닫힌 형태의 가시 자유 에너지를 얻는다. 코드의 W 방향과 확률·샘플의 사용 위치를 맞추어 읽고, CD 통계 업데이트와 정확한 우도 그래디언트, 재구성과 평형 생성을 구분한다.

**참고 문헌**

- Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence.
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A Fast Learning Algorithm for Deep Belief Nets.
