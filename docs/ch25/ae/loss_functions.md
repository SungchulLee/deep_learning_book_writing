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


<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
MNIST 화소가 $[0,1]$ 사이 값일 때 평균 제곱 어긋남과 두 값 엇갈린 엔트로피 가운데 무엇을 써야 하는가? 실제로 익혀 견주어라. L1 손실도 함께 넣어 보라.

</div>

??? success "연습문제 1 풀이"
    병목 16, 15 에포크로 익히고 시험 자료에서 재면 이렇다. MAE와 평균 밝기를 함께
    적어 둔다(자료 자체의 평균 밝기는 0.1325다).

    | 익힐 때 쓴 손실 | BCE | MSE | L1 |
    |---|---|---|---|
    | 시험 MSE | **0.00926** | 0.00958 | 0.11331 |
    | 시험 MAE | **0.03141** | 0.03217 | 0.13187 |
    | 평균 밝기 | 0.1320 | 0.1282 | **0.0179** |

    **BCE로 익힌 쪽이 MSE로 재어도 더 낫다.** 3% 차이로 크지는 않지만 방향이
    일정하다.

    L1은 아주 나쁘다. 눈여겨볼 것은 **자기 자신의 잣대인 MAE에서도 진다**는 점이다.
    손실이 안 맞는 것을 재서 생긴 착시가 아니라는 뜻이다.

    평균 밝기가 까닭을 알려 준다. 0.0179는 자료의 0.1325에 견주어 턱없이 어두우므로,
    이 모델은 **거의 검은 그림을 내놓고 있다.** L1을 최소화하는 값은 평균이 아니라
    **중앙값**이고, MNIST 화소의 대부분이 배경 0이므로 화소마다의 중앙값이 0이다.
    L1은 바로 그 답으로 끌고 간다.

    학습률 탓이 아닌지 확인해 볼 만하다. 확인해 보면 아니다.

    | L1 학습률 | 1e-3 | 3e-4 | 1e-4 | 3e-5 |
    |---|---|---|---|---|
    | 시험 MAE | 0.13187 | **0.06282** | 0.07371 | 0.10318 |
    | 평균 밝기 | 0.0179 | 0.0925 | 0.0860 | 0.0582 |

    가장 좋은 자리를 골라도 MAE가 0.06282로 BCE의 0.03141에 두 배 뒤지고 밝기도
    여전히 모자란다. **손실을 잘못 고른 것이며 걸음 크기로 고칠 수 있는 문제가
    아니다.**

    까닭은 MNIST 화소가 대부분 0이거나 1에 가깝기 때문이다. BCE는 0과 1 근처에서
    기울기가 커서 그 자리를 정확히 맞히도록 세게 민다. MSE는 어디서나 기울기가
    일정하므로 회색 언저리를 적당히 맞추는 데 만족한다.

    다만 BCE를 쓰려면 출력이 반드시 $[0,1]$이어야 한다. 그래서 풀개 끝에 시그모이드가
    붙는다. 화소가 $[0,1]$이 아닌 자료(표준화한 자료, 음수가 있는 신호)에는 BCE를
    쓸 수 없고 MSE가 자연스럽다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
다시 세우기 손실을 화소마다 더할 것인가 평균 낼 것인가? `reduction='sum'`과 `reduction='mean'`이 학습에 어떤 차이를 만드는가?

</div>

??? success "연습문제 2 풀이"
    값만 놓고 보면 784배 차이다. 그런데 경사 하강법에서 중요한 것은 손실의 절댓값이
    아니라 **기울기의 크기**이므로, 이 차이는 학습률을 784배 바꾼 것과 같다.

    | | 손실 크기 | 실효 학습률 |
    |---|---|---|
    | `sum` | 화소 784개의 합 | 크다 |
    | `mean` | 그 784분의 1 | 작다 |

    어느 쪽을 써도 되지만 **바꿀 때 학습률을 함께 손봐야 한다.** `mean`으로 바꾸고
    학습률을 그대로 두면 손실이 거의 내려가지 않아 "모델이 안 배운다"고 오해하기 쉽다.

    표본 축은 사정이 다르다. 묶음 안에서는 **반드시 평균**을 내야 묶음 크기를 바꿔도
    실효 학습률이 유지된다. 이 장의 코드가 `reduction='sum'`으로 더한 뒤
    `/ xb.size(0)`으로 표본 수만 나누는 까닭이 이것이다. 화소는 더하고 표본은 평균 낸다.

    변분 자기 부호기에서는 이 선택이 더 중요해진다. 다시 세우기 항과 KL 항의 크기
    비가 곧 $\beta$ 노릇을 하기 때문이다([26장](../../ch26/index.md)).

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
평균 절대 어긋남(L1)으로 익히면 다시 세운 그림이 어떻게 달라지는가?

</div>

??? success "연습문제 3 풀이"
    L1은 이상값에 덜 휘둘린다. 제곱이 아니라 절댓값이므로 크게 틀린 화소 하나가
    손실을 지배하지 않는다.

    그림에서는 **덜 흐릿하고 더 또렷한** 결과로 나타난다. MSE는 확신이 없을 때 여러
    가능성의 **평균**을 그리는 것이 유리하다. 획이 있을지 없을지 모르겠으면 회색을
    칠하는 편이 제곱 오차를 줄이기 때문이다. L1은 평균이 아니라 **중앙값**을 그리는
    것이 유리하므로, 애매하면 있는 쪽이나 없는 쪽으로 결정한다.

    그래서 MSE로 익힌 자기 부호기의 출력은 특유의 뿌연 느낌을 갖는다. 이 흐릿함은
    자기 부호기와 변분 자기 부호기가 공통으로 겪는 문제이며,
    [맞겨루기 만들개](../../ch29/index.md)가 손실 자체를 버리고 판별기로 바꾼 까닭
    가운데 하나다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
성긴 자기 부호기는 코드 대부분이 0이 되도록 벌점을 준다. 병목을 좁히는 것과 무엇이 다른가?

</div>

??? success "연습문제 4 풀이"
    둘 다 "적게 쓰라"는 제약이지만 **적음의 뜻이 다르다.**

    | | 제약 | 자료마다 |
    |---|---|---|
    | 좁은 병목 | 코드가 $k$개뿐 | 모든 자료가 **같은** $k$개 축을 쓴다 |
    | 성김 벌점 | 코드는 많되 대부분 0 | 자료마다 **다른** 몇 개를 골라 쓴다 |

    성김의 값어치가 여기 있다. 코드를 128개 두고 자료마다 그중 다섯 개만 켜게 하면,
    자료 종류마다 다른 다섯 개를 고를 수 있다. 1은 이 다섯 개, 8은 저 다섯 개를
    쓰는 식이다. 좁은 병목은 모든 자료를 같은 16차원 공간에 밀어 넣어야 하므로
    그런 분업이 불가능하다.

    그래서 성긴 코드는 대개 **더 읽기 쉽다.** 켜진 단위가 무엇에 반응하는지 하나씩
    살펴볼 수 있기 때문이다. 좁은 병목의 코드는 모든 것이 모든 차원에 섞여 있다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
성김을 L1으로 줄 때와 KL 벌어짐으로 줄 때가 어떻게 다른가?

</div>

??? success "연습문제 5 풀이"
    **L1**은 코드의 절댓값 합에 벌점을 준다. 값을 그냥 0 쪽으로 민다.

    $$\Omega = \lambda \sum_j \lvert z_j \rvert$$

    **KL**은 단위마다의 평균 깨어남 $\hat\rho_j$를 목표 $\rho$(이를테면 0.05)에
    맞춘다.

    $$\Omega = \sum_j \rho \log\frac{\rho}{\hat\rho_j}
      + (1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}$$

    차이가 둘이다.

    첫째, **KL은 양쪽으로 벌한다.** $\hat\rho_j$가 목표보다 크면 물론이고 **작아도**
    벌점을 받는다. 그래서 아예 죽어 버리는 단위가 생기지 않는다. L1은 0으로 갈수록
    좋으므로 단위가 영영 죽을 수 있다.

    둘째, **KL은 평균을 본다.** 자료 전체에 걸친 평균 깨어남을 맞추므로, 한 자료에서
    세게 켜지고 나머지에서 꺼지는 것을 허용한다. 이것이 바로 우리가 원하던 성김이다.
    L1은 자료마다 값을 누르므로 어디서든 작아지라는 압력이 된다.

    KL 쪽이 뜻에 더 맞지만 목표 $\rho$를 정해 주어야 한다는 번거로움이 있다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
성김 벌점에 쓰는 KL은 [26장](../../ch26/index.md)의 KL 항과 같은 것인가?

</div>

??? success "연습문제 6 풀이"
    **식의 이름만 같고 하는 일이 다르다.**

    | | 성긴 자기 부호기 | 변분 자기 부호기 |
    |---|---|---|
    | 무엇의 KL인가 | 단위마다의 **평균 깨어남** $\hat\rho_j$와 목표 $\rho$ | 자료마다의 **사후 분포** $q(z \mid x)$와 사전 분포 $p(z)$ |
    | 확률인가 | 깨어남을 확률처럼 **본** 것 | 진짜 확률 분포 |
    | 자료마다 다른가 | 아니다, 자료 전체의 평균 | 그렇다, $x$마다 다르다 |
    | 뽑을 수 있게 되는가 | **아니다** | 그렇다 |

    성긴 자기 부호기의 KL은 두 베르누이 분포 사이의 거리를 빌려 와 "이 단위가 켜지는
    빈도를 5%로 맞추라"고 말하는 **정칙화 장치**다. 깨어남 값은 확률이 아니며 아무도
    그것을 분포로 다루지 않는다.

    변분 자기 부호기의 KL은 실제로 부호기가 내놓은 분포와 사전 분포 사이의 거리이고,
    그 항이 있어야 증거 하한이 성립한다. 정칙화로 덧붙인 것이 아니라 **유도에서 나온
    것**이다.

    그래서 성긴 자기 부호기는 벌점을 아무리 세게 주어도
    [뽑을 수 있게 되지 않는다](../limits/latent_sampling.md). 코드가 0 근처에 몰리게
    할 수는 있어도, 그 몰린 모양이 우리가 뽑을 줄 아는 분포라는 보장이 없기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
오그림 벌점은 무엇을 작게 만드는가? 식으로 적어라.

</div>

??? success "연습문제 7 풀이"
    부호기의 **야코비 행렬**의 프로베니우스 노름 제곱이다.

    $$\Omega = \lambda \left\lVert \frac{\partial z}{\partial x} \right\rVert_F^2
      = \lambda \sum_{i,j} \left( \frac{\partial z_j}{\partial x_i} \right)^2$$

    뜻은 "입력이 조금 흔들려도 코드가 덜 흔들리게 하라"이다. 야코비가 작다는 것은
    부호기가 입력의 작은 변화를 **무시한다**는 뜻이다.

    재어 보면 효과가 뚜렷하다.

    | $\lambda$ | 0 | 0.1 |
    |---|---|---|
    | 야코비 노름 | 24.329 | **0.538** |
    | 시험 MSE | 0.01014 | 0.01067 |

    노름이 45분의 1로 줄어드는 대신 다시 세우기 오차는 5% 나빠진다. 여기서도 맞바꿈이며,
    무엇을 버리라고 강제하느냐만 다를 뿐 [병목](48_autoencoder.md)과 같은 이야기다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
오그림 벌점이 모든 방향의 민감도를 눌러 버린다면 자료를 구별하는 힘까지 사라지지 않는가?

</div>

??? success "연습문제 8 풀이"
    그렇게 되지 않는 까닭은 **손실이 둘이기 때문**이다.

    오그림 항만 있다면 부호기가 상수 함수가 되는 것이 최선이다. 야코비가 0이 되어
    벌점이 완전히 사라진다. 그러나 그러면 다시 세우기가 불가능해져 첫째 항이
    폭발한다.

    두 힘이 맞서면서 부호기는 **자료가 실제로 놓인 방향**은 살리고 **놓이지 않은
    방향**만 누르게 된다. 다양체를 따라가는 방향으로는 민감해야 되돌릴 수 있고,
    다양체에서 벗어나는 방향으로는 둔감해도 잃을 것이 없기 때문이다.

    그래서 오그리는 자기 부호기가 배우는 것은 **접평면**이다. 야코비의 특이값을
    보면 큰 것 몇 개와 거의 0인 나머지로 갈리는데, 큰 쪽이 다양체의 접방향이다.
    [주다양체](../architecture/04_ae_principal_manifold.md)에서 본 것과 같은 대상을
    다른 길로 잡은 셈이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
잡음 없애는 자기 부호기와 오그리는 자기 부호기는 무엇이 같고 무엇이 다른가?

</div>

??? success "연습문제 9 풀이"
    **같은 것**은 목표다. 둘 다 부호기가 입력의 작은 흔들림에 둔감해지기를 바란다.

    **다른 것**은 그 일을 시키는 방법이다.

    | | 잡음 없애는 것 | 오그리는 것 |
    |---|---|---|
    | 방법 | 입력을 실제로 더럽힌다 | 야코비에 벌점을 준다 |
    | 비용 | 공짜 (자료만 손본다) | 야코비를 셈해야 한다 |
    | 정확도 | 잡음 표본으로 어림 | 정확히 벌한다 |

    실은 둘이 이어져 있다. 잡음의 세기를 0으로 보내면서 전개하면, 잡음 없애기의
    손실이 **다시 세우기 + 오그림 벌점**으로 근사된다는 것이 알려져 있다. 곧
    잡음을 더하는 것은 야코비 벌점을 몬테카를로로 어림하는 셈이다.

    실무에서는 잡음 쪽을 훨씬 많이 쓴다. 구현이 한 줄이고 야코비를 셈하지 않아도
    되기 때문이다. 야코비 계산은 코드 차원마다 역전파를 한 번씩 해야 해서 비싸다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
여기 나온 벌점들(성김, 오그림, 잡음, 좁은 병목)을 하나의 생각으로 묶을 수 있는가?

</div>

??? success "연습문제 10 풀이"
    묶을 수 있다. 모두 **무엇을 버릴지 강제하는** 장치다.

    다시 세우기 손실만 두면 자기 부호기는 아무것도 버리지 않으려 한다. 극단적으로
    병목을 입력 크기로 두면 항등 함수를 배워 오차가 0이 되지만 배운 것은 없다
    ([주다양체 연습문제 9](../architecture/04_ae_principal_manifold.md)). 그러므로
    쓸모 있는 나타냄을 얻으려면 **버리라고 시켜야** 한다.

    | 장치 | 무엇을 버리게 하는가 |
    |---|---|
    | 좁은 병목 | 차원 자체를 줄여 담을 수 있는 양을 제한 |
    | 성김 | 자료마다 쓸 수 있는 차원의 **개수**를 제한 |
    | 잡음 | 잡음에 묻히는 자잘한 것을 버리게 함 |
    | 오그림 | 자료가 놓이지 않은 **방향**을 버리게 함 |

    정보 이론으로 보면 하나의 물음이다. 코드가 입력에 대해 가질 수 있는 정보량에
    상한을 두고, 그 한정된 예산 안에서 무엇을 담을지 고르게 하는 것이다. 예산을
    어떻게 매기느냐(차원 수, 켜진 개수, 잡음 대비, 민감도)만 다르다.

    [변분 자기 부호기](../../ch26/index.md)의 KL 항도 이 목록에 들어간다. 그것은
    코드가 사전 분포에서 얼마나 벗어날 수 있는지에 예산을 매기며, 실제로 그 항은
    정보량의 상한으로 해석된다. 다만 KL만이 **뽑을 수 있는 숨은 공간**을 덤으로 준다.

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
