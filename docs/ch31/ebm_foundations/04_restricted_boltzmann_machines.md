# MNIST를 이용한 RBM의 CD 학습

[RBM의 조건부 확률](../classical_ebm/rbm.md)을 실제 이진 이미지 학습 루프로 연결한다. 예제는 MNIST 픽셀을 이진화하고 CD-1 통계 차이로 파라미터를 갱신한다. 로그 우도 대신 재구성 오차를 기록하므로 학습 지표가 무엇을 측정하는지 구분해야 한다.

## 1. 코드와 보존된 출력

```python
"""
제한 볼츠만 기계: 쓸모 있는 에너지 바탕 모델
==========================================================

제한 볼츠만 기계는 에너지 바탕 러닝의 가장 성공한 실제 쓰임새이다.
이어짐을 보이는 레이어와 숨은 레이어 사이로만 옭아맨다(두 쪽 그래프),
맞댐 벌어짐으로 익히기를 다룰 만하게 만든다.

학습 목표:
-------------------
1. 제한 볼츠만 기계의 구조와 에너지 함수를 이해한다
2. 맞댐 벌어짐(CD-k) 알고리즘을 짠다
3. 참 자료(MNIST)로 제한 볼츠만 기계를 익힌다
4. 배운 피처를 그려 본다
5. 제한 볼츠만 기계를 되짓기와 만들어 내기에 쓴다

핵심 개념:
------------
- 두 쪽 그래프: v끼리나 h끼리의 이음이 없다
- Energy: E(v,h) = -aᵀv - bᵀh - vᵀWh
- Conditionals factor: P(h|v) = ∏ P(hⱼ|v), P(v|h) = ∏ P(vᵢ|h)
- 맞댐 벌어짐: 어림 그래디언트
- 덩이 깁스 뽑기

걸리는 시간: 90~120분
미리 알 것: 단원 01-03
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================

torch.manual_seed(42)
np.random.seed(42)

class RestrictedBoltzmannMachine(nn.Module):
    """
    제한 볼츠만 기계 짜기.
    
    제한 볼츠만 기계는 다음을 지닌 두 쪽 방향 없는 그림 모델이다.
    - Visible layer v ∈ {0,1}ⁿ
    - Hidden layer h ∈ {0,1}ᵐ
    - 같은 레이어 안의 이음이 없다
    
    Energy function:
    E(v,h) = -aᵀv - bᵀh - vᵀWh
    
    여기서 W은 무게 행렬, a은 드러난 치우침, b은 숨은 치우침이다.
    """
    
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=1):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k  # CD-k steps
        
        # 파라메터를 초기화한다
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))   # hidden bias
        
        self.lr = learning_rate
        
    def sample_hidden(self, v):
        """보이는 알갱이가 주어질 때 숨은 알갱이를 뽑는다: P(h=1|v) = σ(Wv + b)"""
        activation = F.linear(v, self.W, self.b)
        prob = torch.sigmoid(activation)
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def sample_visible(self, h):
        """숨은 알갱이가 주어질 때 보이는 알갱이를 뽑는다: P(v=1|h) = σ(Wᵀh + a)"""
        activation = F.linear(h, self.W.t(), self.a)
        prob = torch.sigmoid(activation)
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def energy(self, v, h):
        """Compute energy E(v,h) = -aᵀv - bᵀh - vᵀWh"""
        return -(v @ self.a + h @ self.b + (v @ self.W.t() * h).sum(dim=1))
    
    def free_energy(self, v):
        """
        자유 힘 F(v) = -log Σₕ exp(-E(v,h))을 셈한다
        F(v) = -aᵀv - Σⱼ log(1 + exp(bⱼ + Wⱼv))
        """
        wx_b = F.linear(v, self.W, self.b)
        visible_term = (v * self.a).sum(dim=1)
        hidden_term = wx_b.exp().add(1).log().sum(dim=1)
        return -(visible_term + hidden_term)
    
    def contrastive_divergence(self, v0):
        """
        맞댐 벌어짐 k스텝(CD-k) 익히기.
        
        어림한 그래디언트: ∇L ≈ E_data[vh] - E_model_k[vh]
        """
        batch_size = v0.shape[0]
        
        # 양의 국면: 자료에서 뽑는다
        ph0, h0 = self.sample_hidden(v0)
        
        # 음의 국면: 깁스 뽑기 k스텝
        vk, hk = v0, h0
        for _ in range(self.k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        
        # 양의 그래디언트와 음의 그래디언트를 셈한다
        positive_grad = torch.matmul(ph0.t(), v0)
        negative_grad = torch.matmul(hk.t(), vk)
        
        # 파라메터 갱신
        self.W.data += self.lr * (positive_grad - negative_grad) / batch_size
        self.a.data += self.lr * (v0 - vk).mean(dim=0)
        self.b.data += self.lr * (ph0 - hk).mean(dim=0)
        
        # 살피기 위해 되짓기 어긋남을 셈한다
        recon_error = ((v0 - vk)**2).sum(dim=1).mean()
        
        return recon_error.item()
    
    def reconstruct(self, v):
        """보이는 알갱이를 되살린다: v → h → v'"""
        _, h = self.sample_hidden(v)
        _, v_recon = self.sample_visible(h)
        return v_recon

def train_rbm_mnist():
    """MNIST 자료 묶음으로 제한 볼츠만 기계를 익힌다."""
    print("\n" + "="*70)
    print("TRAINING RBM ON MNIST")
    print("="*70)
    
    # MNIST 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 제한 볼츠만 기계를 만든다
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(n_visible, n_hidden, learning_rate=0.01, k=1)
    
    print(f"\nRBM Architecture:")
    print(f"  Visible units: {n_visible}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  CD-k steps: {rbm.k}")
    
    # 학습 루프
    n_epochs = 10
    errors = []
    
    for epoch in range(n_epochs):
        epoch_error = 0
        n_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.view(-1, 784)
            error = rbm.contrastive_divergence(data)
            epoch_error += error
            n_batches += 1
        
        avg_error = epoch_error / n_batches
        errors.append(avg_error)
        print(f"Epoch {epoch+1}/{n_epochs}, Reconstruction Error: {avg_error:.4f}")
    
    print("\n✓ RBM training complete")
    return rbm

def main():
    print("="*70)
    print("RESTRICTED BOLTZMANN MACHINES")
    print("="*70)
    
    train_rbm_mnist()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ RBMs use bipartite architecture for tractable inference")
    print("  ✓ Contrastive Divergence enables practical training")
    print("  ✓ RBMs learn useful feature representations")
    print("\nNext: 05_contrastive_divergence.py")

if __name__ == "__main__":
    main()
```

```
======================================================================
RESTRICTED BOLTZMANN MACHINES
======================================================================

======================================================================
TRAINING RBM ON MNIST
======================================================================

RBM Architecture:
  Visible units: 784
  Hidden units: 256
  CD-k steps: 1
Epoch 1/10, Reconstruction Error: 122.7035
Epoch 2/10, Reconstruction Error: 86.7822
Epoch 3/10, Reconstruction Error: 75.9758
Epoch 4/10, Reconstruction Error: 69.3618
Epoch 5/10, Reconstruction Error: 64.7623
Epoch 6/10, Reconstruction Error: 61.2752
Epoch 7/10, Reconstruction Error: 58.5078
Epoch 8/10, Reconstruction Error: 56.2475
Epoch 9/10, Reconstruction Error: 54.3384
Epoch 10/10, Reconstruction Error: 52.7134

✓ RBM training complete

======================================================================
MODULE COMPLETE
======================================================================

Key Takeaways:
  ✓ RBMs use bipartite architecture for tractable inference
  ✓ Contrastive Divergence enables practical training
  ✓ RBMs learn useful feature representations

Next: 05_contrastive_divergence.py
```

코드의 실행 설정과 기존 출력은 이번 편집에서 실행하거나 재측정하지 않았다. 아래에서는 모델·데이터·업데이트의 연결을 설명하며, 필터나 생성 표본의 품질을 출력만으로 단정하지 않는다.

## 2. 데이터와 행렬 방향

ToTensor 뒤 x>0.5를 적용해 픽셀을 0 또는 1로 만든다. 이는 픽셀 값에 비례한 확률로 Bernoulli 샘플을 뽑는 전처리와 다르다. 28×28 이미지를 784차원으로 펼치고 숫자 라벨은 사용하지 않는다.

가시 784개, 은닉 256개이며 W는 **256×784**다. 입력 B×784에 Wᵀ를 곱해 B×256 은닉 확률을 계산한다. 역방향에는 은닉 샘플에 W를 곱해 B×784 가시 확률을 만든다. 바이어스까지 포함한 파라미터 수는 201,744개다.

이 방향에서 에너지는 E(v,h)=−aᵀv−bᵀh−hᵀWv다. W가 n_v×n_h인 교재 표기와 비교할 때 전치 위치를 바꾸어야 한다. 확률 출력과 실제 이진 샘플은 각각 다른 반환값이다.

## 3. CD-1 통계 갱신

데이터 v₀에서 은닉 확률 ph₀와 샘플 h₀를 만들고, 가시·은닉 조건부 샘플링을 한 번씩 수행한다. positive phase는 ph₀ᵀv₀, negative phase는 hₖᵀvₖ를 사용한다. 두 통계 차이를 배치 크기로 나누고 러닝레이트 0.01을 곱해 W에 직접 더한다. 바이어스도 평균 차이로 갱신한다.

파라미터는 nn.Parameter지만 .data를 직접 갱신하며 optimizer나 backward를 사용하지 않는다. 반환하는 로스는 이미지별 제곱 오차 합의 평균이다. 이진 입력·샘플에서 값은 평균 해밍 거리이며 이미지당 0~784 범위다. 정확한 NLL이나 좌표당 평균 MSE와 같은 값이 아니다.

CD-1은 짧은 데이터 시작 사슬의 통계를 사용한다. 사슬이 모델 평형에 도달했다고 가정할 수 없으므로 일반적으로 정확한 최대 우도 그래디언트의 바이어스된 근사다. 학습이 유용할 수 있다는 사실과 바이어스·혼합 문제가 사라진다는 주장은 구분한다.

## 4. 에포크 평균과 구현상의 제한

60,000개를 배치 64로 나누고 마지막 32개까지 사용하므로 한 에포크는 938회, 10에포크는 9,380회 업데이트다. epoch_error는 배치 평균들을 같은 비중으로 평균한다. 마지막 작은 배치 때문에 샘플 수로 가중한 전체 평균과는 조금 다르다.

free_energy는 log(1+exp(z))를 직접 계산한다. 이론적으로 softplus지만 큰 양의 z에서는 exp가 overflow할 수 있다. [고전 RBM 구현](../classical_ebm/rbm.md)의 F.softplus와 수치 안정성이 같다고 설명하지 않는다. 자유 에너지의 낮은 값만으로 다른 모델과 우도를 비교하려면 Z가 빠진 문제도 남는다.

이 파일의 reconstruct는 은닉과 가시 상태를 모두 샘플링해 **이진 이미지**를 반환한다. 가시 확률을 반환하는 앞 페이지의 동명 함수와 다르다. main은 학습만 호출하며 필터 시각화, 생성 평가나 체크포인트 저장은 구현하지 않는다.

## 5. CD-5와 persistent chain

k=5는 한 업데이트마다 블록 깁스 갱신을 다섯 번 수행한다. 고정 모델에서 충분한 혼합은 평형 근사를 개선할 수 있지만, 유한 k를 늘릴 때 모든 통계 오차가 단조롭게 줄거나 생성 품질이 반드시 좋아지는 것은 아니다.

연습문제의 PersistentRBM은 negative phase를 이전 사슬 상태에서 시작한다. 현재 배치와 장기 사슬의 상태를 비교한 제곱 오차는 **같은 입력을 복원한 오차가 아니다**. CD-1의 재구성 지표와 직접 비교해 우열을 판단하면 안 된다. 자세한 상태 유지 조건은 아래 풀이에서 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
W=[[1,−1,0,1],[0,1,1,−1]], a=b=0인 RBM에서 v=(1,0,1,0)의 자유 에너지를 계산하라.

</div>

??? success "연습문제 1 풀이"

    Wv+b=(1,1)이므로 F(v)=−2log(1+e)≈−2.627이다. 이 값은 숨은 상태를 합한 비정규화 가시 에너지이며 log p(v)=−F(v)−log Z다. F만 계산해 확률을 구한 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
조건부 독립성과 한 에포크의 업데이트 수를 설명하라.

</div>

??? success "연습문제 2 풀이"

    가시 상태를 고정하면 은닉 에너지가 단위별 선형 항으로 나뉘어 은닉 Bernoulli가 조건부 독립이다. 반대 방향도 같다. 하지만 주변 가시 변수는 일반적으로 독립이 아니다. 60,000=937×64+32이므로 한 에포크는 938회 업데이트한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 PersistentRBM의 사슬 유지와 반환 지표를 검토하라. 마지막 배치 크기가 달라지면 어떻게 되는가?

</div>

??? success "연습문제 3 풀이"

    k=5이면 매 업데이트에 다섯 번 전이하고 이전 가시 사슬을 재사용한다. 단, 배치 크기가 달라지면 현재 데이터로 사슬을 다시 초기화한다. 배치 64의 MNIST에서 마지막 32개와 다음 에포크 첫 64개에서 연속성이 끊기므로 항상 같은 입자를 유지하는 구현은 아니다. persistent_chain은 buffer로 등록되지 않아 모델의 장치 이동·상태 저장에 자동 포함되지 않는다. 반환값은 데이터 v0와 독립적인 장기 사슬 vk 사이의 차이여서 입력별 재구성 품질이나 NLL이 아니다. persistent chain도 파라미터 변화에 비해 충분히 혼합해야 하며 더 좋은 그래디언트를 자동으로 보장하지 않는다.

    ```python
    class PersistentRBM(RestrictedBoltzmannMachine):
        def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=5):
            super().__init__(n_visible, n_hidden, learning_rate, k)
            self.persistent_chain = None
        
        def contrastive_divergence(self, v0):
            batch_size = v0.shape[0]
            ph0, h0 = self.sample_hidden(v0)
            
            # 자료에서 다시 시작하는 대신 이어지는 사슬을 쓴다
            if self.persistent_chain is None or self.persistent_chain.shape[0] != batch_size:
                self.persistent_chain = v0.clone()
            
            vk = self.persistent_chain
            for _ in range(self.k):
                _, hk = self.sample_hidden(vk)
                _, vk = self.sample_visible(hk)
            
            self.persistent_chain = vk.detach()
            _, hk = self.sample_hidden(vk)
            
            positive_grad = torch.matmul(ph0.t(), v0)
            negative_grad = torch.matmul(hk.t(), vk)
            
            self.W.data += self.lr * (positive_grad - negative_grad) / batch_size
            self.a.data += self.lr * (v0 - vk).mean(dim=0)
            self.b.data += self.lr * (ph0 - hk).mean(dim=0)
            
            return ((v0 - vk)**2).sum(dim=1).mean().item()
    ```


## 정리하며

이 예제는 이진 MNIST와 784→256 RBM의 CD-1 통계 업데이트를 연결한다. 기록값은 이미지별 비트 불일치 수이며 우도가 아니다. 행렬 방향, 자유 에너지의 수치 안정성과 persistent chain의 배치 크기·상태 유지 조건을 확인한다.
