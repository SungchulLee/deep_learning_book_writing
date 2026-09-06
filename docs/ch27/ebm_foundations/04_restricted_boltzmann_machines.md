# 제한 볼츠만 기계

제한 볼츠만 기계(RBM)는 에너지 바탕 배움의 가장 성공한 실제 쓰임새 가운데 하나이다. 이음을 드러난 층과 숨은 층 사이의 두 쪽 얼개로 제한하여 맞댐 벌어짐 알고리즘으로 익히기를 다룰 만하게 만든다. 제한 볼츠만 기계는 쓸모 있는 특징 나타냄을 배우고 더 깊은 만들어 내는 모델의 벽돌 노릇을 한다.

## 코드

```python
"""
제한 볼츠만 기계: 쓸모 있는 에너지 바탕 모델
==========================================================

제한 볼츠만 기계는 에너지 바탕 배움의 가장 성공한 실제 쓰임새이다.
They restrict connections to be between visible and hidden layers only (bipartite),
맞댐 벌어짐으로 익히기를 다룰 만하게 만든다.

학습 목표:
-------------------
1. 제한 볼츠만 기계의 얼개와 에너지 함수를 이해한다
2. 맞댐 벌어짐(CD-k) 알고리즘을 짠다
3. Train RBMs on real data (MNIST)
4. 배운 특징을 그려 본다
5. 제한 볼츠만 기계를 되짓기와 만들어 내기에 쓴다

핵심 개념:
------------
- 두 쪽 그래프: v끼리나 h끼리의 이음이 없다
- Energy: E(v,h) = -aᵀv - bᵀh - vᵀWh
- Conditionals factor: P(h|v) = ∏ P(hⱼ|v), P(v|h) = ∏ P(vᵢ|h)
- 맞댐 벌어짐: 어림 기울기
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
    
    RBM is a bipartite undirected graphical model with:
    - Visible layer v ∈ {0,1}ⁿ
    - Hidden layer h ∈ {0,1}ᵐ
    - 같은 층 안의 이음이 없다
    
    Energy function:
    E(v,h) = -aᵀv - bᵀh - vᵀWh
    
    여기서 W은 무게 행렬, a은 드러난 치우침, b은 숨은 치우침이다.
    """
    
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=1):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k  # CD-k steps
        
        # 매개변수를 초기화한다
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))   # hidden bias
        
        self.lr = learning_rate
        
    def sample_hidden(self, v):
        """Sample hidden units given visible units: P(h=1|v) = σ(Wv + b)"""
        activation = F.linear(v, self.W, self.b)
        prob = torch.sigmoid(activation)
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def sample_visible(self, h):
        """Sample visible units given hidden units: P(v=1|h) = σ(Wᵀh + a)"""
        activation = F.linear(h, self.W.t(), self.a)
        prob = torch.sigmoid(activation)
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def energy(self, v, h):
        """Compute energy E(v,h) = -aᵀv - bᵀh - vᵀWh"""
        return -(v @ self.a + h @ self.b + (v @ self.W.t() * h).sum(dim=1))
    
    def free_energy(self, v):
        """
        Compute free energy F(v) = -log Σₕ exp(-E(v,h))
        F(v) = -aᵀv - Σⱼ log(1 + exp(bⱼ + Wⱼv))
        """
        wx_b = F.linear(v, self.W, self.b)
        visible_term = (v * self.a).sum(dim=1)
        hidden_term = wx_b.exp().add(1).log().sum(dim=1)
        return -(visible_term + hidden_term)
    
    def contrastive_divergence(self, v0):
        """
        맞댐 벌어짐 k걸음(CD-k) 익히기.
        
        Approximate gradient: ∇L ≈ E_data[vh] - E_model_k[vh]
        """
        batch_size = v0.shape[0]
        
        # 양의 국면: 자료에서 뽑는다
        ph0, h0 = self.sample_hidden(v0)
        
        # 음의 국면: 깁스 뽑기 k걸음
        vk, hk = v0, h0
        for _ in range(self.k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        
        # 양의 기울기와 음의 기울기를 셈한다
        positive_grad = torch.matmul(ph0.t(), v0)
        negative_grad = torch.matmul(hk.t(), vk)
        
        # 매개변수 갱신
        self.W.data += self.lr * (positive_grad - negative_grad) / batch_size
        self.a.data += self.lr * (v0 - vk).mean(dim=0)
        self.b.data += self.lr * (ph0 - hk).mean(dim=0)
        
        # 살피기 위해 되짓기 어긋남을 셈한다
        recon_error = ((v0 - vk)**2).sum(dim=1).mean()
        
        return recon_error.item()
    
    def reconstruct(self, v):
        """Reconstruct visible units: v → h → v'"""
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

## 논의

제한 볼츠만 기계는 두 쪽 에너지 함수 $E(v,h) = -a^\top v - b^\top h - v^\top W h$ 위에 세운 바탕이 되는 만들어 내는 모델이다. 여기서 $v$과 $h$은 드러난 이진 단위와 숨은 이진 단위를 나타내고 $W$, $a$, $b$은 배울 수 있는 매개변수이다. 두 쪽 제한, 곧 드러난 층 안이나 숨은 층 안에 이음이 없다는 것이 제한 볼츠만 기계를 다룰 만하게 만든다. 조건 분포 $P(h|v)$과 $P(v|h)$이 서로 얽매이지 않은 베르누이 분포로 나뉘어 효율 좋은 덩이 깁스 뽑기가 가능해진다.

최대 가능도로 제한 볼츠만 기계를 익히려면 로그 나눔 함수의 기울기를 셈해야 하는데 이는 보통 다룰 수 없다. 맞댐 벌어짐(CD-k)은 사슬을 평형까지 돌리는 대신 깁스 뽑기를 $k$걸음만 써서 기울기의 음의 국면을 어림하여 이를 비켜 간다. 양의 국면은 자료 분포의 통계를 담고 음의 국면은 모델이 되지은 것의 통계를 담는다. CD-1(깁스 한 걸음)조차 실제로 놀랍도록 잘 들어, 배운 거르개로 그려 볼 수 있는 쓸모 있는 특징 찾개를 만들어 낸다.

자유 에너지 $F(v) = -a^\top v - \sum_j \log(1 + \exp(b_j + W_j v))$은 모델이 주어진 드러난 자리 얽이를 얼마나 잘 설명하는지 간추린 다룰 만한 낱값을 준다. MNIST으로 익힌 제한 볼츠만 기계는 무게 행렬에 모서리 찾개와 획 무늬를 배우며, 이는 스승 없는 에너지 바탕 배움이 뜻있는 얼개를 찾아낼 수 있음을 보여 준다. 제한 볼츠만 기계는 깊은 믿음 신경망의 벽돌이자 깊은 얼개의 앞익히기 조각 노릇도 한다.

## 연습문제

**연습문제 1.**
드러난 단위 4개와 숨은 단위 2개를 가지고 무게가 $W = \begin{pmatrix} 1 & -1 & 0 & 1 \\ 0 & 1 & 1 & -1 \end{pmatrix}$, 드러난 치우침이 $a = (0, 0, 0, 0)$, 숨은 치우침이 $b = (0, 0)$인 제한 볼츠만 기계에서 $v = (1, 0, 1, 0)$의 자유 에너지 $F(v)$을 셈하라.

??? success "연습문제 1 풀이"
    먼저 숨은 단위마다 $W v + b$을 셈한다:
    
    $$
    W v + b = \begin{pmatrix} 1 \cdot 1 + (-1) \cdot 0 + 0 \cdot 1 + 1 \cdot 0 \\ 0 \cdot 1 + 1 \cdot 0 + 1 \cdot 1 + (-1) \cdot 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
    $$
    
    자유 에너지는 다음과 같다:
    
    $$
    F(v) = -a^\top v - \sum_{j=1}^{2} \log(1 + e^{(Wv+b)_j}) = -0 - \log(1 + e^1) - \log(1 + e^1) = -2\log(1 + e) \approx -2.627
    $$

---

**연습문제 2.**
제한 볼츠만 기계의 두 쪽 얼개가 왜 조건 분포 $P(h|v)$과 $P(v|h)$을 인수로 나뉘게 하는지, 그리고 이것이 온전히 이어진 볼츠만 기계에 견주어 어떤 셈의 이점을 주는지 밝혀라.

??? success "연습문제 2 풀이"
    온전히 이어진 볼츠만 기계에서는 숨은 단위가 같은 층 안의 이음으로 서로 얽혀 있어 $P(h|v)$을 셈하려면 숨은 단위의 모든 자리 얽이를 더해야 한다. 그래서 정확한 추론이 숨은 단위 개수에 대해 지수가 된다.
    
    제한 볼츠만 기계에는 숨은 단위끼리의 이음이 없다. 드러난 단위 $v$이 주어지면 숨은 단위 $h_j$마다 제 무게 $W_j$과 치우침 $b_j$을 거쳐 $v$에만 매인다:
    
    $$
    P(h_j = 1 | v) = \sigma(b_j + W_j v)
    $$
    
    $v$이 주어지면 숨은 단위가 조건부로 서로 얽매이지 않으므로 다음을 얻는다:
    
    $$
    P(h|v) = \prod_j P(h_j | v)
    $$
    
    이 인수 나눔 덕에 모든 숨은 단위를 지수 시간이 아니라 $O(nm)$ 시간에 나란히 뽑을 수 있어 덩이 깁스 뽑기가 효율 좋아지고 맞댐 벌어짐이 쓸 만해진다.

---

**연습문제 3.**
제한 볼츠만 기계 짜기를 CD-1 대신 CD-5을 쓰도록 고치고, 고칠 때마다 깁스 사슬을 자료에서 다시 시작하지 않고 앞의 음의 표본에서 이어 가는 이어지는 맞댐 벌어짐(PCD)을 더하라. 익히기 움직임에서 예상되는 차이를 적어라.

??? success "연습문제 3 풀이"
    CD-5에는 생성자에서 `k=5`으로 바꾼다. PCD에는 이어지는 사슬을 지닌다:
    
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
    
    깁스 사슬이 더 오래 돌아 모델 분포에 더 가까운 음의 표본을 만들므로 CD-5은 CD-1보다 나은 기울기 어림을 준다. PCD은 고침에 걸쳐 사슬을 이어 가며 모델 분포를 더 꼼꼼히 살피게 하여 이를 더 낫게 한다. PCD은 보통 더 나은 만들어 내는 모델을 배우지만 음의 표본이 지금의 자료 묶음과 덜 얽혀 있어 익히는 동안 되짓기 어긋남이 클 수 있다.
