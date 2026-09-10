# 자기 부호기의 손실 함수
자기 부호기 변형의 다시 세우기 손실, 벌주기 목표, 특화된 익히기 잣대.

---

**배울 것:**

- 다시 세우기 손실 함수: 평균 제곱 어긋남, 두 값 엇갈린 엔트로피, 평균 절대 어긋남
- 성김 벌주기: L1 벌주기와 KL 벌어짐
- 오그림 벌주기: 야코비 노름 벌주기
- 잡음 없애기 목표: 망가뜨린 들임에서 배우기
- 손실 함수를 고르는 것이 배운 나타냄에 미치는 영향

---

## 1. 1부: 다시 세우기 손실 함수

모든 자기 부호기는 들임 $x$과 다시 세운 $\hat{x}$의 차를 가장 작게 한다는 공통 바탕을 나눠 갖는다. 손실 함수를 고르는 것은 자료 분포에 대한 가정을 담는 일이다.

### 평균 제곱 어긋남(MSE)

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \hat{x}_i\|^2$$

**성질:**

- 작은 어긋남보다 큰 어긋남에 더 벌을 준다(이차)
- 정규 잡음 모델을 가정한다: $p(x|\hat{x}) \propto \exp(-\|x - \hat{x}\|^2 / 2\sigma^2)$
- **흐릿한** 다시 세우기를 내기 쉽다(여러 결을 평균 낸다)
- 어떤 내놓기 깨어남과도 통한다

### 두 값 엇갈린 엔트로피(BCE)

$$\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [x_i \log(\hat{x}_i) + (1-x_i) \log(1-\hat{x}_i)]$$

**성질:**

- $[0, 1]$의 두 값 자료나 고르게 한 자료에 자연스럽다
- 내놓기 깨어남으로 **에스자**가 필요하다
- 화소 값을 베르누이 확률로 본다
- 흔히 평균 제곱 어긋남보다 **또렷한** 다시 세우기를 낸다

### 평균 절대 어긋남(MAE / L1)

$$\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \hat{x}_i|$$

**성질:**

- 평균 제곱 어긋남보다 동떨어진 값에 튼튼하다(선형 벌주기)
- 더 성긴 기울기를 낸다(크기가 일정하다)
- 다시 세운 것에서 모서리가 더 또렷해질 수 있다
- 라플라스 잡음 모델을 가정한다

### 견줌과 짜기

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AutoencoderWithAnalysis(nn.Module):
    """숨은 공간 살피기 방법을 갖춘 자기 부호기."""
    
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
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
        return self.decode(z), z

def compare_loss_functions(model_class, train_loader, test_loader, device):
    """여러 다시 세우기 손실 함수를 견준다."""
    
    losses = {
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss(),
        'L1': nn.L1Loss()
    }
    
    results = {}
    
    for loss_name, criterion in losses.items():
        print(f"\nTraining with {loss_name} loss...")
        
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        for epoch in range(10):
            model.train()
            epoch_loss = 0
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
        
        # 평가한다
        model.eval()
        with torch.no_grad():
            test_images, _ = next(iter(test_loader))
            test_images = test_images[:10].view(10, -1).to(device)
            recon, _ = model(test_images)
        
        results[loss_name] = {
            'model': model,
            'train_losses': train_losses,
            'reconstructions': recon.cpu().numpy()
        }
    
    return results
```

### 가장 좋은 숨은 차원 찾기

```python
def find_optimal_latent_dim(train_loader, test_loader, device, 
                            dims=[2, 4, 8, 16, 32, 64, 128, 256]):
    """다시 세우기 어긋남으로 가장 좋은 숨은 차원을 찾는다."""
    
    results = []
    
    for dim in dims:
        model = AutoencoderWithAnalysis(latent_dim=dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 학습
        for epoch in range(15):
            model.train()
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
        
        # 평가한다
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(device)
                recon, _ = model(images)
                test_loss += criterion(recon, images).item()
        
        test_loss /= len(test_loader)
        results.append({'dim': dim, 'error': test_loss})
        print(f"Latent dim {dim}: Test MSE = {test_loss:.6f}")
    
    return results
```

---

## 2. 2부: 성김 벌주기

성긴 자기 부호기는 숨은 깨어남 대부분이 0이 되도록 벌을 주어 더 풀이하기 쉽고 넘치게 갖춘 나타냄을 얻는다.

### 여느 자기 부호기와 성긴 자기 부호기의 손실

| 갈래 | 손실 함수 |
|------|---------------|
| 여느 것 | $\mathcal{L} = \|x - f(x)\|^2$ |
| 성김(L1) | $\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$ |
| 성김(KL) | $\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$ |

### L1 벌주기

$$\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$$

여기서:

- $h_j$은 숨은 층에서 신경 세포 $j$의 깨어남이다
- $\lambda$은 성김 벌주기의 세기이다
- $\sum_j |h_j|$은 많은 깨어남이 딱 0이 되도록 이끈다

**성김이 도움이 되는 까닭:**

1. **골라 깨우는 특징** — 들임마다 관련 있는 특징만 깨운다
2. **풀이할 수 있는 나타냄** — 특징이 뜻 있는 결에 맞닿는다
3. **잡음에 튼튼함** — 성긴 부호가 더 안정되다
4. **더 나은 두루 통함** — 지나치게 맞춰지는 것을 막는다

### KL 벌어짐 성김

$$\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$$

여기서:

- $\rho$은 목표 성김 수준이다(예컨대 0.05은 평균 깨어남 5%를 뜻한다)
- $\hat{\rho}_j = \frac{1}{n}\sum_{i=1}^n h_j(x_i)$은 신경 세포 $j$의 평균 깨어남이다
- $\text{KL}(\rho \| \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho) \log\frac{1-\rho}{1-\hat{\rho}_j}$

KL 벌어짐은 $\hat{\rho}_j = \rho$일 때 가장 작아지므로 목표 깨어남 수준을 **정밀히 다스릴** 수 있다.

### 구현

```python
class SparseAutoencoder_L1(nn.Module):
    """
    숨은 깨어남에 L1 벌주기를 쓴 성긴 자기 부호기.
    
    손실 = 다시 세우기 손실 + λ × L1(숨은 깨어남)
    
    L1 벌주기가 숨은 깨어남 여럿을 딱 0이 되도록 이끈다.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_L1, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 부호기: 넘치게 갖춘 나타냄을 위해 흔히 latent_dim을 크게 둔다
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()  # 정류 선형이 저절로 성김을 북돋운다
        )
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def l1_loss(latent: torch.Tensor) -> torch.Tensor:
    """
    숨은 깨어남의 L1 벌주기를 셈한다.
    
    L1(h) = Σᵢⱼ |hᵢⱼ|
    
    0이 아닌 깨어남에 벌을 주어 성김을 이끈다.
    """
    return torch.mean(torch.abs(latent))

class SparseAutoencoder_KL(nn.Module):
    """
    KL 벌어짐 성김 제약을 쓴 성긴 자기 부호기.
    
    신경 세포마다의 평균 깨어남이 목표 성김 수준 ρ(예컨대 0.05)에
    가깝도록 옭아맨다.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_KL, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # KL 벌어짐을 위해 에스자를 쓴 부호기(내놓기가 [0,1])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()  # KL 벌어짐에 필요하다
        )
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def kl_divergence_loss(latent: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    """
    KL 벌어짐 성김 벌주기를 셈한다.
    
    신경 세포 j마다 평균 깨어남이 ρ̂ⱼ ≈ ρ이기를 바란다.
    
    KL(ρ || ρ̂ⱼ) = ρ log(ρ/ρ̂ⱼ) + (1-ρ) log((1-ρ)/(1-ρ̂ⱼ))
    
    ρ̂ⱼ = ρ일 때 가장 작다.
    """
    # 묶음에 걸친 신경 세포마다의 평균 깨어남
    rho_hat = torch.mean(latent, dim=0)
    
    # log(0)을 피한다
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    
    # 신경 세포마다의 KL 벌어짐
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    return torch.sum(kl)

def train_sparse_autoencoder(
    model, train_loader, optimizer, device, epoch,
    sparsity_type='l1', sparsity_weight=0.001, rho=0.05
):
    """
    성긴 자기 부호기를 한 바퀴 익힌다.
    
    전체 손실 = 다시 세우기 손실 + 성김 벌주기
    """
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    num_batches = 0
    
    recon_criterion = nn.MSELoss()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        
        # 순전파
        reconstructed, latent = model(images)
        
        # 되살림 손실
        recon_loss = recon_criterion(reconstructed, images)
        
        # 성김 벌주기
        if sparsity_type == 'l1':
            sparsity_loss = l1_loss(latent)
        elif sparsity_type == 'kl':
            sparsity_loss = kl_divergence_loss(latent, rho)
        
        # 전체 손실
        total_loss = recon_loss + sparsity_weight * sparsity_loss
        
        # 역전파
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        sparsity_loss_sum += sparsity_loss.item()
        num_batches += 1
    
    return (total_loss_sum / num_batches, 
            recon_loss_sum / num_batches, 
            sparsity_loss_sum / num_batches)
```

### 성김 잣대

| 지표 | 정의 |
|--------|------------|
| **표본별 성김** | 표본마다 신경 세포의 몇 몫이 깨어 있는가? |
| **평생 성김** | 신경 세포마다 표본의 몇 몫이 그것을 깨우는가? |

```python
def analyze_sparsity(model, test_loader, device, num_samples=1000):
    """배운 나타냄의 성김을 살핀다."""
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            if len(all_activations) * test_loader.batch_size >= num_samples:
                break
            images = images.view(images.size(0), -1).to(device)
            _, latent = model(images)
            all_activations.append(latent.cpu().numpy())
    
    all_activations = np.concatenate(all_activations, axis=0)[:num_samples]
    
    # 깨어남 > 문턱을 "깨어 있음"이라 한다
    threshold = 0.1
    active = all_activations > threshold
    
    # 표본별 성김: 표본마다 깨어 있는 신경 세포의 평균 몫
    population_sparsity = np.mean(np.mean(active, axis=1))
    
    # 평생 성김: 신경 세포마다 그것을 깨우는 표본의 몫
    lifetime_sparsity = np.mean(active, axis=0)
    
    return population_sparsity, lifetime_sparsity
```

### L1과 KL 견줌

| 방법 | 장치 | 좋은 점 | 나쁜 점 |
|--------|-----------|------|------|
| **L1** | $\|h\|_1$에 벌주기 | 단순하고 빠르다 | 목표 성김에 딱 이르지 못할 수 있다 |
| **KL** | 목표 $\rho$에서 벌어진 만큼 벌주기 | 성김을 정밀히 다스린다 | 에스자 깨어남이 필요하다 |

---

## 3. 3부: 오그림 벌주기

**오그리는 자기 부호기(CAE)**는 부호기 야코비 행렬의 프로베니우스 노름에 벌을 주어 부호기가 들임의 흔들림에 무디도록 이끈다.

### 오그림 벌주기

$$\mathcal{L} = \|x - g(f(x))\|^2 + \lambda \|J_f(x)\|_F^2$$

여기서 각 기호는 다음과 같다.

- $J_f(x) = \frac{\partial f(x)}{\partial x} \in \mathbb{R}^{k \times d}$: 부호기의 야코비 행렬
- $\|J_f\|_F^2 = \sum_{ij} J_{ij}^2$: 프로베니우스 노름의 제곱

### 직관

| 조각 | 효과 |
|-----------|--------|
| 다시 세우기 손실 | 들임을 다시 세우는 법을 배운다 |
| 야코비 벌주기 | 부호기가 들임의 흔들림에 무디게 한다 |

야코비 벌주기는 **국소 불변**(들임이 조금 바뀌면 숨은 값도 조금만 바뀜), **튼튼한 나타냄**(잡음은 무시하고 요긴한 짜임을 잡음), **평평한 다양체**(잡음 방향으로는 숨은 공간이 그때그때 일정함)를 이끈다.

### 잡음 없애는 자기 부호기와의 이음

흩어짐이 $\sigma^2$인 작은 정규 잡음에서 잡음 없애는 자기 부호기는 대략 다음을 가장 작게 한다:

$$\mathcal{L}_{DAE} \approx \|x - g(f(x))\|^2 + \sigma^2 \|J_f(x)\|_F^2$$

**핵심 눈썰미:** 정규 잡음으로 잡음을 없애는 것은 넌지시 오그림 벌주기를 쓰는 것이다!

| 갈래 | 잡음 없애는 자기 부호기 | 오그리는 자기 부호기 |
|--------|--------------|----------------|
| 벌주기 | 망가뜨린 들임으로 | 또렷한 야코비 벌주기로 |
| 셈하기 | 잡음을 곁들인 앞먹임 | 야코비 셈하기가 필요하다 |
| 융통성 | 여러 잡음 갈래 | 오그림을 곧바로 다스린다 |
| 풀이 | 잡음 없애는 법을 배운다 | 부호기의 민감함을 가장 작게 한다 |

### 구현

```python
from torch.autograd import grad

class ContractiveAutoencoder(nn.Module):
    """야코비 벌주기를 갖춘 오그리는 자기 부호기."""
    
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
    ||J_f||_F^2 = Σ_ij (∂z_i / ∂x_j)^2
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    
    # 야코비 행렬을 세로줄마다 셈한다
    jacobian_norm_sq = 0.0
    
    for i in range(z.shape[1]):
        # x에 대한 z_i의 기울기
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0
        
        jacobian_col = grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 이 세로줄의 제곱합
        jacobian_norm_sq = jacobian_norm_sq + torch.sum(jacobian_col ** 2)
    
    return jacobian_norm_sq / x.shape[0]  # 묶음에 걸친 평균

def train_contractive_autoencoder(
    model, train_loader, device, 
    lambda_contractive=0.1, num_epochs=15
):
    """
    오그리는 자기 부호기를 익힌다.
    
    손실 = 다시 세우기 + λ × ||J_f||_F^2
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
            
            # 역전파
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

### 기하학적 해석

오그림 벌주기는 다음을 이끈다:

1. **평평한 숨은 다양체:** 부호기가 내놓는 것이 들임에 따라 천천히 바뀐다
2. **잡음 방향 오그림:** 다양체 밖 방향이 눌린다
3. **자료 다양체 지킴:** 중요한 흔들림이 남는다

맞바꿈:

$$\text{작은 } \lambda \to \text{다시 세우기는 낫고 튼튼함은 덜하다}$$

$$\text{큰 } \lambda \to \text{튼튼함은 더하고 다시 세우기는 못하다}$$

---

## 4. 4부: 잡음 없애기 목표

잡음 없애는 자기 부호기는 근본이 다른 익히기 목표를 쓴다. 곧 **망가뜨린** 들임에서 **깨끗한** 자료를 다시 세운다.

### 여느 목표와 잡음 없애기 목표

| 자기 부호기 갈래 | 익히기 목표 |
|------------------|-------------------|
| 여느 것 | $\|x - f(x)\|^2$을 가장 작게 |
| 잡음 없애기 | $\|x - f(\tilde{x})\|^2$을 가장 작게 |

여기서 $\tilde{x} = \text{corrupt}(x)$은 잡음 낀 들임이고 손실은 **깨끗한** 본디 것에 대해 셈한다. 그러면 그물이 망가진 관측에서 바탕 신호를 되찾을 수 있는 튼튼한 특징을 배우게 된다.

### 망가뜨리는 전략

| 전략 | 식 | 설명 | 쓰임새 |
|----------|---------|-------------|----------|
| 정규 | $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | 더하는 흰 잡음 | 두루 쓰임 |
| 소금과 후추 | 아무 화소 → 0이나 1 | 충격 잡음 | 문서와 감지기 자료 |
| 가리기 | 아무 화소 → 0 | 떨구기와 비슷 | 가려짐에 튼튼함 |
| 짜임 있는 것 | 덩이나 자리 가리기 | 공간에서 이어진 잡음 | 메워 그리기 |

### 잡음 짜기

```python
def add_noise(images: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
    """
    그림에 정규 잡음을 더한다.
    
    망가뜨린 그림: x̃ = x + ε, 여기서 ε ~ N(0, σ²)
    """
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)

def add_salt_pepper_noise(images: torch.Tensor, 
                          noise_prob: float = 0.2) -> torch.Tensor:
    """그림에 소금과 후추 잡음을 더한다."""
    noisy_images = images.clone()
    
    noise_mask = torch.rand_like(images) < noise_prob
    salt_mask = torch.rand_like(images) > 0.5
    
    noisy_images[noise_mask & salt_mask] = 1.0   # 소금
    noisy_images[noise_mask & ~salt_mask] = 0.0  # 후추
    
    return noisy_images

def add_masking_noise(images: torch.Tensor, 
                      mask_prob: float = 0.3) -> torch.Tensor:
    """
    화소를 아무렇게나 0으로 두어 가리기 잡음을 더한다.
    요즘 보기 변환기의 가린 자기 부호기(MAE)와 맞닿는다.
    """
    mask = (torch.rand_like(images) > mask_prob).float()
    return images * mask
```

### 점수 맞추기와의 이음

흩어짐이 $\sigma^2$인 작은 정규 잡음에서 잡음 없애는 자기 부호기는 넌지시 **점수 함수**를 어림한다:

$$\nabla_x \log p(x) \approx \frac{1}{\sigma^2}(f(\tilde{x}) - \tilde{x})$$

이는 잡음 없애는 자기 부호기를 점수 바탕 만들어 내는 모델, 퍼짐 모델, 에너지 바탕 모델과 잇는다.

---

## 5. 5부: 벌주기를 갖춘 온전한 익히기

모든 손실 조각을 아우르면:

```python
def train_regularized_autoencoder(
    model, train_loader, optimizer, device, epoch,
    regularization='none',       # 'none', 'l1', 'kl', 'contractive', 'denoising'
    reg_weight=0.001,            # 벌주기 항의 무게
    rho=0.05,                    # KL의 목표 성김
    noise_factor=0.1             # 잡음 없애기의 잡음 수준
):
    """
    모든 벌주기 갈래를 받치는 아우른 익히기 함수.
    
    전체 손실 = 다시 세우기 손실 + reg_weight × 벌주기 항
    """
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    reg_loss_sum = 0.0
    num_batches = 0
    
    recon_criterion = nn.MSELoss()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        
        # 들임을 마련한다(잡음 없애기면 망가뜨리고 아니면 깨끗이)
        if regularization == 'denoising':
            input_images = add_noise(images, noise_factor)
        else:
            input_images = images
        
        # 순전파
        reconstructed, latent = model(input_images)
        
        # 다시 세우기 손실(늘 깨끗한 그림에 대해)
        recon_loss = recon_criterion(reconstructed, images)
        
        # 벌주기 항
        if regularization == 'l1':
            reg_loss = l1_loss(latent)
        elif regularization == 'kl':
            reg_loss = kl_divergence_loss(latent, rho)
        elif regularization == 'contractive':
            reg_loss = compute_jacobian_penalty(model, input_images)
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        # 전체 손실
        total_loss = recon_loss + reg_weight * reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        reg_loss_sum += reg_loss.item()
        num_batches += 1
    
    return (total_loss_sum / num_batches,
            recon_loss_sum / num_batches,
            reg_loss_sum / num_batches)
```

---

## 6. 배운 특징 그려 보기

성긴 자기 부호기는 빽빽한 자기 부호기보다 흔히 더 풀이하기 쉬운 특징을 배운다:

```python
def visualize_learned_features(model, num_features=64):
    """
    하나만 뜨거운 숨은 벡터를 풀어 배운 특징을 그려 본다.
    
    성긴 자기 부호기에서는 특징이 흔히 더 풀이하기 쉬우며
    국소한 결을 보인다.
    """
    model.eval()
    
    latent_dim = model.latent_dim
    num_features = min(num_features, latent_dim)
    
    features = []
    with torch.no_grad():
        for i in range(num_features):
            # 하나만 뜨거운 벡터를 만든다
            latent = torch.zeros(1, latent_dim)
            latent[0, i] = 1.0  # 신경 세포 i만 깨운다
            
            # 그림 공간으로 푼다
            feature = model.decoder(latent)
            features.append(feature.cpu().numpy().reshape(28, 28))
    
    # 격자로 그려 본다
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_features):
        axes[i].imshow(features[i], cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Learned Features (Decoder Basis)')
    plt.savefig('learned_features.png', dpi=150)
    plt.show()
```

---

## 연습문제

### 연습 1: 손실 함수 견줌

평균 제곱 어긋남, 두 값 엇갈린 엔트로피, L1 손실로 자기 부호기를 익혀라. 다시 세우기 품질을 눈으로도 수치로도 견주어라. 어느 것이 가장 또렷한 내놓기를 내는가?

### 연습 2: 성김 무게 다듬기

성김 무게를 달리해 모델을 익혀라:

```python
l1_weights = [0.0001, 0.001, 0.01, 0.1, 1.0]
kl_weights = [0.001, 0.01, 0.1, 1.0, 10.0]
```

성김 무게가 다시 세우기 품질과 성김의 맞바꿈에 어떤 영향을 주는가? 제약이 셀수록 특징이 더 풀이하기 쉬워지는가?

### 연습 3: L1과 KL 견줌

실효 성김이 비슷한 모델 둘을 익혀라. 곧 $\lambda = 0.01$인 L1과 $\beta = 0.1$, $\rho = 0.05$인 KL이다. 익히기의 움직임, 마지막 성김 수준, 특징 품질을 견주어라.

### 연습 4: 오그림 λ 다듬기

$\lambda \in \{0.001, 0.01, 0.1, 1.0\}$으로 오그리는 자기 부호기를 익혀라. 다시 세우기 어긋남과 부호기의 민감함을 그려라. 가장 좋은 맞바꿈은 무엇인가?

### 연습 5: 잡음 수준 살피기(잡음 없애기)

잡음 수준을 달리해 잡음 없애는 자기 부호기를 익혀라:

```python
noise_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
```

익히기에 가장 좋은 잡음 수준이 있는가? 큰 잡음으로 익힌 모델이 작은 잡음도 없앨 수 있는가?

### 연습 6: 잡음 갈래에 대한 튼튼함

잡음 갈래(정규, 소금과 후추, 가리기)마다 따로 모델 셋을 익혀라. 모델마다 모든 잡음 갈래로 시험하라. 한 갈래로 익힌 것이 다른 갈래에도 두루 통하는가?

---

## 정리하며

| 손실 / 벌주기 | 식 | 효과 | 쓰임새 |
|----------------|---------|--------|----------|
| **평균 제곱 어긋남** | $\|x - \hat{x}\|^2$ | 정규 가정, 흐릿함 | 이어진 자료 |
| **두 값 엇갈린 엔트로피** | $-[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ | 또렷함, 두 값 가정 | 고르게 한 그림 |
| **평균 절대 어긋남** | $\|x - \hat{x}\|_1$ | 동떨어진 값에 튼튼 | 잡음 낀 자료 |
| **L1 성김** | $\lambda\sum\|h_j\|$ | 깨어남을 0으로 몬다 | 풀이할 수 있는 특징 |
| **KL 성김** | $\beta\sum\text{KL}(\rho\|\hat{\rho}_j)$ | 성김을 정밀히 다스림 | 넘치게 갖춘 자기 부호기 |
| **오그림** | $\lambda\|J_f\|_F^2$ | 들임에 무딘 부호기 | 튼튼한 다양체 배우기 |
| **잡음 없애기** | 들임을 망가뜨리고 깨끗한 것을 다시 세움 | 넌지시 벌주기 | 튼튼한 특징 |

**핵심 눈썰미:** 손실 함수와 벌주기를 고르는 것이 자기 부호기가 무엇을 배우는지 근본에서 정한다. 다시 세우기 손실은 자료 잡음에 대한 가정을 담고, 벌주기 항은 숨은 공간의 기하와 풀이 가능함을 빚는다.
