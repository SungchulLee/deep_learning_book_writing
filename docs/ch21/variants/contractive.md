# 오그리는 자기 부호기
들임의 흔들림에 대한 부호기의 민감함에 대놓고 벌을 주어 튼튼한 나타냄을 배운다.

---

## 개요

**배울 것:**

- 부호기의 야코비 행렬과 민감함을 재는 데서의 노릇
- 오그림 벌주기: 부호기 야코비 행렬의 프로베니우스 노름
- 다양체 배움으로 보는 기하 풀이
- 잡음 없애는 자기 부호기와의 이음
- PyTorch에서 효율 좋고 정확하게 야코비 행렬 셈하기

---

## 수학적 바탕

### 오그림 벌주기

**오그리는 자기 부호기(CAE)**는 여느 다시 세우기 손실에 부호기 야코비 행렬의 프로베니우스 노름 벌주기를 더한다:

$$\mathcal{L} = \underbrace{\|x - g(f(x))\|^2}_{\text{다시 세우기}} + \underbrace{\lambda \|J_f(x)\|_F^2}_{\text{오그림 벌주기}}$$

여기서 각 기호는 다음과 같다.

- $f$: 들임을 숨은 나타냄에 옮기는 부호기 함수
- $g$: 숨은 것을 들임 공간으로 되돌리는 풀개 함수
- $J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{k \times d}$: 부호기의 야코비 행렬
- $\|J_f\|_F^2 = \sum_{ij} J_{ij}^2$: 프로베니우스 노름의 제곱(모든 편미분의 제곱합)
- $\lambda$: 다시 세우기와 오그림의 맞바꿈을 다스리는 벌주기 세기

### 직관

| 조각 | 효과 |
|-----------|--------|
| 다시 세우기 손실 | 들임을 충실히 다시 세우는 법을 배운다 |
| 야코비 벌주기 | 부호기가 들임의 흔들림에 무디게 한다 |

야코비 벌주기는 다음을 이끈다:

- **국소 불변:** 들임이 조금 바뀌면 숨은 부호도 조금만 바뀐다
- **튼튼한 나타냄:** 부호기가 요긴한 짜임은 잡고 잡음은 무시하는 법을 배운다
- **평평한 다양체:** 숨은 공간이 잡음 방향으로는 그때그때 일정하고 자료 다양체를 따라서는 바뀐다

### 맞바꿈

$$\text{작은 } \lambda \to \text{다시 세우기는 낫고 튼튼함은 덜하다}$$

$$\text{큰 } \lambda \to \text{튼튼함은 더하고 다시 세우기는 못하다}$$

다시 세우기 항은 (정확히 다시 세우려) $f$이 들임의 모든 흔들림에 민감하기를 바라고 오그림 항은 무디기를 바란다. 이 저울질이 부호기를 다시 세우는 데 중요한 방향, 곧 자료 다양체를 따라서만 민감하게 만든다.

---

## 잡음 없애는 자기 부호기와의 이음

### 이론의 이음

흩어짐이 $\sigma^2$인 작은 정규 잡음에서 잡음 없애는 자기 부호기의 목표는 대략 다음을 가장 작게 한다:

$$\mathcal{L}_{DAE} \approx \|x - g(f(x))\|^2 + \sigma^2 \|J_f(x)\|_F^2$$

**핵심 눈썰미:** 정규 잡음으로 잡음을 없애는 것은 넌지시 오그림 벌주기를 쓰는 것이며 잡음 흩어짐 $\sigma^2$이 $\lambda$ 노릇을 한다.

### 견줌

| 갈래 | 잡음 없애는 자기 부호기 | 오그리는 자기 부호기 |
|--------|--------------|----------------|
| 벌주기 | 망가뜨린 들임으로 | 또렷한 야코비 벌주기로 |
| 셈하기 | 잡음을 곁들인 여느 앞먹임 | 야코비 셈하기가 필요하다 |
| 융통성 | 여러 잡음 갈래를 쓸 수 있다 | 오그림 세기를 곧바로 다스린다 |
| 풀이 | 잡음 없애는 법을 배운다 | 부호기의 민감함을 가장 작게 한다 |
| 익히기 값 | 여느 것 | 더 높다(야코비가 비싸다) |

---

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class ContractiveAutoencoder(nn.Module):
    """
    야코비 벌주기를 갖춘 오그리는 자기 부호기.
    
    내놓기를 가두려 부호기에 에스자 깨어남을 쓰며,
    그래야 야코비가 얌전하다.
    """
    
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 내놓기를 가두려 에스자를 쓴 부호기
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()  # 야코비가 안정되도록 [0,1]로 가둔다
        )
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def compute_jacobian_penalty(model, x):
    """
    부호기 야코비 행렬의 프로베니우스 노름 제곱을 셈한다.
    
    J_f(x)_ij = ∂z_i / ∂x_j
    ||J_f||²_F = Σ_ij (∂z_i / ∂x_j)²
    
    저절로 미분하기로 세로줄마다 셈한다.
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    jacobian_norm_sq = 0.0
    
    for i in range(z.shape[1]):
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0
        
        jacobian_col = grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        jacobian_norm_sq = jacobian_norm_sq + torch.sum(jacobian_col ** 2)
    
    return jacobian_norm_sq / x.shape[0]  # 묶음에 걸친 평균
```

---

## 학습

```python
def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    오그리는 자기 부호기를 익힌다.
    
    손실 = 다시 세우기 + λ × ||J_f||²_F
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_criterion = nn.MSELoss()
    
    history = {'recon_loss': [], 'contractive_loss': [], 'total_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_recon = 0
        epoch_contractive = 0
        epoch_total = 0
        
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            recon, z = model(images)
            
            # 되살림 손실
            recon_loss = recon_criterion(recon, images)
            
            # 오그림 벌주기
            contractive_loss = compute_jacobian_penalty(model, images)
            
            # 전체 손실
            total_loss = recon_loss + lambda_contractive * contractive_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_recon += recon_loss.item()
            epoch_contractive += contractive_loss.item()
            epoch_total += total_loss.item()
        
        n_batches = len(train_loader)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['contractive_loss'].append(epoch_contractive / n_batches)
        history['total_loss'].append(epoch_total / n_batches)
        
        print(f"Epoch {epoch+1}: Recon={epoch_recon/n_batches:.6f}, "
              f"Contract={epoch_contractive/n_batches:.6f}")
    
    return history
```

---

## 민감함 살피기

```python
def analyze_contraction(model, test_loader, device, noise_std=0.1):
    """
    숨은 값의 바뀜과 들임의 바뀜의 비를 재어 배운 나타냄이
    얼마나 오그라드는지 살핀다.
    
    오그리는 부호기는 민감함 비가 1보다 훨씬 작다.
    """
    model.eval()
    
    sensitivity_scores = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            
            z_original = model.encode(images)
            
            noise = torch.randn_like(images) * noise_std
            z_noisy = model.encode(images + noise)
            
            input_change = torch.norm(noise, dim=1)
            latent_change = torch.norm(z_noisy - z_original, dim=1)
            
            # 민감함 = ||Δz|| / ||Δx||
            sensitivity = latent_change / (input_change + 1e-8)
            sensitivity_scores.extend(sensitivity.cpu().numpy())
            
            if len(sensitivity_scores) > 1000:
                break
    
    return np.array(sensitivity_scores)


def compare_with_standard_ae(train_loader, test_loader, device):
    """
    부호기의 민감함으로 오그리는 자기 부호기와 여느 자기 부호기를 견준다.
    """
    # 여느 자기 부호기(얼개는 같고 오그림 벌주기는 없다)
    standard_ae = ContractiveAutoencoder().to(device)
    optimizer = optim.Adam(standard_ae.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training Standard AE...")
    for epoch in range(15):
        standard_ae.train()
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, _ = standard_ae(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
    
    # 오그리는 자기 부호기
    contractive_ae = ContractiveAutoencoder().to(device)
    
    print("\nTraining Contractive AE...")
    train_contractive_autoencoder(
        contractive_ae, train_loader, device, 
        lambda_contractive=0.1, num_epochs=15
    )
    
    # 민감함을 견준다
    print("\nAnalyzing sensitivity to noise...")
    
    std_sensitivity = analyze_contraction(standard_ae, test_loader, device)
    cae_sensitivity = analyze_contraction(contractive_ae, test_loader, device)
    
    print(f"Standard AE sensitivity: {np.mean(std_sensitivity):.4f} "
          f"± {np.std(std_sensitivity):.4f}")
    print(f"Contractive AE sensitivity: {np.mean(cae_sensitivity):.4f} "
          f"± {np.std(cae_sensitivity):.4f}")
    
    # 시각화한다
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(std_sensitivity, bins=50, alpha=0.7, label='Standard AE')
    axes[0].hist(cae_sensitivity, bins=50, alpha=0.7, label='Contractive AE')
    axes[0].set_xlabel('Sensitivity (||Δz|| / ||Δx||)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Encoder Sensitivity Distribution')
    axes[0].legend()
    
    axes[1].boxplot([std_sensitivity, cae_sensitivity], 
                    labels=['Standard', 'Contractive'])
    axes[1].set_ylabel('Sensitivity')
    axes[1].set_title('Sensitivity Comparison')
    
    plt.tight_layout()
    plt.savefig('contractive_comparison.png', dpi=150)
    plt.show()
    
    return standard_ae, contractive_ae
```

---

## 기하학적 해석

### 다양체 배움의 관점

오그림 벌주기는 부호기가 다음과 같은 대응을 배우도록 이끈다:

1. **자료 다양체 방향 지킴:** 부호기가 자료가 실제로 놓인 방향을 따라 바뀐다
2. **잡음 방향 오그림:** 다양체 밖 방향(잡음)은 숨은 공간에서 거의 0의 바뀜으로 옮겨진다
3. **평평한 숨은 다양체:** 숨은 나타냄이 잡음 방향으로는 그때그때 일정하다

이것이 바로 차원 높은 공간에 묻힌 자료 다양체의 속 기하를 배우는 데 바라는 몸가짐이다.

---

## 계량 금융에서의 응용

오그리는 자기 부호기는 금융에서 **안정된 요인 나타냄**을 배우는 데 값지다:

- **튼튼한 위험 요인:** 오그림 벌주기가 시장 자료의 작은 흔들림(호가 튐, 미시 짜임 잡음)이 뽑은 요인을 바꾸지 않게 해 더 안정된 위험 쪼개기를 낸다
- **국면에 흔들리지 않는 특징:** 민감함에 벌을 주므로 배운 특징이 잠깐의 시장 어긋남에 덜 흔들린다
- **벌주기를 갖춘 공분산 어림:** 오그리는 부호기가 배운 공분산 짜임에 넌지시 벌을 주어 차원 높은 상황에서 어림 어긋남을 줄인다

---

## 요약

| 갈래 | 여느 자기 부호기 | 오그리는 자기 부호기 |
|--------|-------------|----------------|
| 손실 | 다시 세우기만 | 다시 세우기 + $\|J\|_F^2$ |
| 민감함 | 높음(제약 없음) | 낮음(설계상) |
| 튼튼함 | 제한됨 | 나아짐 |
| 셈하기 | 빠름 | 느림(야코비 셈하기) |
| 다양체 배움 | 넌지시 | 벌주기로 또렷이 |

**핵심 눈썰미:** 오그리는 자기 부호기는 부호기의 민감함에 곧바로 벌을 주어 들임의 흔들림에 튼튼한 나타냄을 배우는 원칙 있는 길을 준다. (작은 정규 잡음에서) 잡음 없애는 자기 부호기와 이론상 같다는 점이 겉보기에 다른 두 벌주기 전략을 하나의 얼거리로 아우른다.

---

## 참고 문헌

- Rifai, S., et al. (2011). "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction." *ICML*.
- Alain, G., & Bengio, Y. (2014). "What Regularized Auto-Encoders Learn from the Data-Generating Distribution." *JMLR*.

## 연습문제

### 연습 1: λ 다듬기
$\lambda \in \{0.001, 0.01, 0.1, 1.0\}$으로 오그리는 자기 부호기를 익혀라. 맞바꿈의 앞머리를 그리려 다시 세우기 어긋남과 민감함을 그려라.

### 연습 2: 잡음 없애기와 견줌
오그리는 자기 부호기($\lambda = 0.1$)와 잡음 없애는 자기 부호기(잡음 $\sigma = 0.3$)를 견주어라. 배운 나타냄이 비슷한가? 숨은 공간 거리 상관으로 재어라.

### 연습 3: 야코비 행렬 그려 보기
익힌 오그리는 자기 부호기에서 들임 숫자마다 야코비 행렬을 그려 보아라. 어느 들임 방향이 가장 많이 오그라드는가?

---
