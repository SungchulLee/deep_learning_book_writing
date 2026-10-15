# 상각 변분 추론

상각 변분 추론(amortized variational inference)은 여러 관측에 공통인 추론 함수를 학습해 사후 근사의 계산을 공유한다. 학습에 비용을 들이는 대신 새 관측의 변분 매개변수를 순전파로 빠르게 얻는다. 계산을 공유하는 방식과 각 근사분포의 표현력은 별개의 선택이다.

## 1. 개별 최적화에서 공유 추론으로

생성모델의 θ를 고정하고 관측 x의 잠재변수 사후를 q_λ(z)로 근사하자. 개별 추론은 λ★(x)∈argmax_λ Lₓ(λ;θ)를 구한다. 상각 추론은 λ=f_ψ(x)를 학습해 관측마다 최적화를 반복하는 비용을 줄인다. 인코더의 매개변수 ψ와 관측별 분포의 매개변수 λ를 구분한다.

모든 VI가 관측별 독립 최적화 문제로 분해되는 것은 아니다. 여기서는 조건부로 분리되는 지역 잠재변수와 고정된 전역 모델을 기준으로 비교한다. 새 관측에도 인코더를 적용할 수 있지만 훈련 분포 밖에서의 정확성이나 일반화가 자동으로 보장되지는 않는다.

## 2. 근사 차이와 상각 차이

같은 θ와 같은 분포족에서 Lₓ★=sup_λ Lₓ(λ;θ)라고 정의하면 다음처럼 나눌 수 있다.

$$\log p_\theta(x)-L_x(f_\psi(x);\theta)=\underbrace{\log p_\theta(x)-L_x^\star}_{\text{근사 차이}}+\underbrace{L_x^\star-L_x(f_\psi(x);\theta)}_{\text{상각 차이}}.$$

첫 차이는 분포족의 표현 제약, 둘째는 공유 함수가 관측별 최적값을 달성하지 못한 정도다. 인코더 학습의 최적화 오차도 실제 둘째 차이에 포함될 수 있다. 이론적 차이는 음이 아니지만, 유한한 지역 최적화와 잡음 있는 ELBO로 구한 차이가 그 정확한 값이라는 뜻은 아니다.

비교할 때 생성모델 θ를 고정해야 한다. θ까지 바뀌면 사후와 기준 최적 ELBO도 달라진다. 정규화 흐름으로 분포족을 확장하는 것은 근사 차이를 줄일 가능성을 만들지만, 공유 추론의 상각 차이까지 반드시 줄이는 것은 아니다. 이 구분은 [Cremer 외의 분석](https://arxiv.org/abs/1801.03558)에서 다룬다.

## 3. VAE의 생성모델과 추론모델

VAE는 z∼p(z), x|z∼p_θ(x|z)의 생성모델과 q_ψ(z|x)의 인코더를 함께 학습한다. 인코더는 정확한 사후 p_θ(z|x)를 근사하며, 디코더는 잠재값에서 관측 **분포의 매개변수**를 출력한다.

대각 가우스 인코더는 q_ψ(z|x)=N(μ_ψ(x),diag σ_ψ²(x))다. 신경망이 여러 층이어도 이 조건부 근사는 블록 간 상관을 갖는 전 공분산 가우스가 되는 것은 아니다. 표준정규 사전에 대한 KL은 다음과 같다.

$$D_{\mathrm{KL}}(q_\psi\Vert p)=\frac12\sum_j\left[\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right].$$

관측별 ELBO는 다음이며 학습 목적은 데이터에 대한 합 또는 평균이다.

$$L_x=E_{q_\psi(z|x)}[\log p_\theta(x|z)]-D_{\mathrm{KL}}(q_\psi(z|x)\Vert p(z)).$$

ε∼N(0,I), z=μ+exp(log σ²/2)⊙ε로 표본을 생성하면 인코더와 디코더까지 경로별 경사를 계산할 수 있다. 확률적 경사 갱신은 유한 반복마다 목적이 단조롭게 증가하는 알고리즘은 아니다.

## 4. 예제 손실과 생성 표본의 의미

아래 구현은 재구성 제곱오차를 특징 차원에서 합하고 배치 크기로 나눈 뒤, 같은 관측당 단위의 KL을 더한다. β=1일 때 가우스 관측분산 σ_x²=1/2를 고정한 음의 ELBO와 **관측 정규화 상수만큼** 차이가 난다. 표준편차 1의 가우스 우도를 의도한다면 제곱오차 계수 1/2와 단위가 달라진다. 생략한 상수를 복원하지 않은 절대 손실을 로그 증거와 직접 비교하지 않는다.

`sample`은 z를 사전에서 뽑아 디코더 출력을 반환한다. 디코더가 가우스 평균을 출력한다면 이는 생성 평균의 표본이며, 관측 잡음까지 포함한 p_θ(x)의 완전한 표본은 아니다. `reconstruct`도 디코더에 잠재 평균을 넣는 결정론적 재구성으로, 비선형 디코더의 사후 예측 평균 E_q[p_θ의 평균]과 일반적으로 다르다.

`measure_amortization_gap`은 생성모델을 갱신하지 않고 지역 μ·logvar를 유한 회수 최적화한다. 그러나 각 목적값에 몬테카를로 잡음이 있고 마지막 `loss`는 마지막 갱신 전 값이다. ‘optimal ELBO’는 참 최적값이 아니며 추정 차이가 음수가 될 수도 있다. 디코더 매개변수의 경사는 누적될 수 있으나 지역 옵티마이저가 그 매개변수를 갱신하는 것은 아니다.

배치별 평균 손실을 다시 단순 평균하므로 마지막 배치 크기가 다르면 전체 관측당 평균과 다를 수 있다. 시각화는 최소 두 잠재 차원을 전제하고, 입력 장치·모델 장치의 일치도 필요하다. 다음 코드와 출력은 재실행하지 않고 유지했다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    """
    VAE 인코더: 들임 x을 변분 매개변수 (μ, log σ²)로 잇는다.
    
    구조: x -> [숨은 층] -> (μ, log σ²)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        # 인코더 층 세우기
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 평균과 로그 흩어짐을 내는 층
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x을 변분 매개변수로 부호화하기.
        
        반환값:
            mu: q(z|x)의 평균
            logvar: q(z|x)의 로그 흩어짐
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    """
    VAE 디코더: 숨은 z을 되살림 매개변수로 잇는다.
    
    구조: z -> [숨은 층] -> x_recon
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        
        # 디코더 층 세우기
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z을 풀어 되살리기.
        """
        return self.decoder(z)

class VAE(nn.Module):
    """
    변분 자동인코더.
    
    인코더(추론 망)와 디코더(낳는 모형)를 어우르며
    ELBO를 가장 크게 하여 함께 익힌다.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """들임을 변분 매개변수로 부호화하기."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """숨은 값을 풀어 되살리기."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        매개변수 바꾸기 재주: z = μ + σ ⊙ ε
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        온전한 앞먹임.
        
        반환값:
            x_recon: 되살린 들임
            mu: 변분 평균
            logvar: 변분 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss(self, x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE 손실 = -ELBO = 되살림 손실 + β * KL 벌어짐 셈하기
        
        인수:
            x: 들임 자료
            x_recon: 되살린 자료
            mu: 변분 평균
            logvar: 변분 로그 흩어짐
            beta: KL 항의 무게(β-VAE)
        
        반환값:
            loss: 전체 손실
            recon_loss: 되살림 항
            kl_loss: KL 벌어짐 항
        """
        # 되살림 손실(음의 로그 가능도)
        # 이어진 자료에는 평균 제곱 오차를 쓴다(가우스 가능도)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        # KL 벌어짐: KL(N(μ,σ²) || N(0,1))
        # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 전체 손실
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        모형에서 표본 만들기.
        
        z ~ p(z) = N(0, I)
        x ~ p(x|z)
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """인코더와 디코더를 거쳐 들임 되살리기."""
        mu, _ = self.encode(x)  # 평균 쓰기(표집 없음)
        return self.decode(mu)

def train_vae(model: VAE, train_loader: DataLoader,
              n_epochs: int = 100, lr: float = 1e-3,
              beta: float = 1.0, verbose: bool = True) -> Dict:
    """
    ELBO를 가장 크게 하여 VAE 익히기.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch in train_loader:
            x = batch[0]
            
            optimizer.zero_grad()
            
            # 순전파
            x_recon, mu, logvar = model(x)
            
            # 손실을 계산한다
            loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar, beta)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        n_batches = len(train_loader)
        history['loss'].append(epoch_loss / n_batches)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kl_loss'].append(epoch_kl / n_batches)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {history['loss'][-1]:.4f}, "
                  f"Recon = {history['recon_loss'][-1]:.4f}, "
                  f"KL = {history['kl_loss'][-1]:.4f}")
    
    return history

def measure_amortization_gap(model: VAE, x: torch.Tensor,
                              n_opt_steps: int = 1000,
                              lr: float = 0.01) -> Tuple[float, float]:
    """
    나눠 갚는 최적화와 낱낱 최적화를 견주어 나눠 갚기 틈 재기.
    
    틈 = ELBO(낱낱 최적) - ELBO(나눠 갚음)
    """
    # 나눠 갚는 ELBO
    with torch.no_grad():
        x_recon, mu_amort, logvar_amort = model(x)
        _, recon_amort, kl_amort = model.loss(x, x_recon, mu_amort, logvar_amort)
        elbo_amortized = -(recon_amort + kl_amort).item()
    
    # 낱낱 최적화
    mu_opt = mu_amort.clone().detach().requires_grad_(True)
    logvar_opt = logvar_amort.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([mu_opt, logvar_opt], lr=lr)
    
    for _ in range(n_opt_steps):
        optimizer.zero_grad()
        
        # 최적화한 매개변수로 z 표집
        std = torch.exp(0.5 * logvar_opt)
        z = mu_opt + std * torch.randn_like(std)
        
        # 디코딩
        x_recon = model.decode(z)
        
        # 손실
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar_opt - mu_opt.pow(2) - logvar_opt.exp())
        loss = recon_loss + kl_loss
        
        loss.backward()
        optimizer.step()
    
    elbo_optimal = -loss.item() / x.size(0)
    gap = elbo_optimal - elbo_amortized
    
    return elbo_amortized, elbo_optimal, gap

def visualize_vae_results(model: VAE, history: Dict, test_data: torch.Tensor):
    """VAE 결과 두루 그려 보기."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 그림 1: 학습 곡선
    ax = axes[0, 0]
    ax.plot(history['loss'], label='Total Loss', linewidth=2)
    ax.plot(history['recon_loss'], label='Reconstruction', linewidth=2)
    ax.plot(history['kl_loss'], label='KL Divergence', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('(a) Training Curves', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 2: 본디와 되살린 것(처음 몇 표본)
    ax = axes[0, 1]
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(test_data[:10])
    
    n_show = min(5, len(test_data))
    for i in range(n_show):
        ax.plot(test_data[i].numpy(), 'b-', alpha=0.5)
        ax.plot(recon[i].numpy(), 'r--', alpha=0.5)
    ax.plot([], [], 'b-', label='Original')
    ax.plot([], [], 'r--', label='Reconstructed')
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(b) Reconstruction Quality', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 3: 숨은 공간(2차원일 때)
    ax = axes[0, 2]
    with torch.no_grad():
        mu, _ = model.encode(test_data)
    
    if model.latent_dim == 2:
        ax.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.5, s=10)
        ax.set_xlabel('z₁', fontsize=11)
        ax.set_ylabel('z₂', fontsize=11)
    else:
        # 처음 두 차원 보이기
        ax.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.5, s=10)
        ax.set_xlabel('z₁', fontsize=11)
        ax.set_ylabel('z₂', fontsize=11)
    ax.set_title('(c) Latent Space (first 2 dims)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 4: 만들어 낸 표본
    ax = axes[1, 0]
    with torch.no_grad():
        samples = model.sample(20)
    
    for i in range(min(10, len(samples))):
        ax.plot(samples[i].numpy(), alpha=0.5)
    ax.set_xlabel('Dimension', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(d) Generated Samples', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 5: 차원마다의 KL
    ax = axes[1, 1]
    with torch.no_grad():
        mu, logvar = model.encode(test_data)
        # 숨은 차원마다의 KL
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    
    ax.bar(range(len(kl_per_dim)), kl_per_dim.numpy())
    ax.set_xlabel('Latent Dimension', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('(e) KL per Latent Dimension', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 6: 나눠 갚기 틈 살피기
    ax = axes[1, 2]
    gaps = []
    n_test = min(50, len(test_data))
    for i in range(n_test):
        _, _, gap = measure_amortization_gap(model, test_data[i:i+1], n_opt_steps=200)
        gaps.append(gap)
    
    ax.hist(gaps, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(gaps), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {np.mean(gaps):.4f}')
    ax.set_xlabel('Amortization Gap', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(f) Amortization Gap Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# 사용 예
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 인공 자료 만들기(차원 높은 공간의 가우스 섞음)
    n_samples = 2000
    input_dim = 20
    
    # 단순한 모형에서 만들기: z -> x = Wz + 잡음
    true_latent_dim = 3
    W = torch.randn(input_dim, true_latent_dim)
    z_true = torch.randn(n_samples, true_latent_dim)
    data = z_true @ W.T + 0.1 * torch.randn(n_samples, input_dim)
    
    # 데이터 나누기
    train_data = data[:1600]
    test_data = data[1600:]
    
    # 자료 실개 만들기
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print("=" * 60)
    print("Variational Autoencoder Training")
    print("=" * 60)
    print(f"\nData shape: {data.shape}")
    print(f"True latent dim: {true_latent_dim}")
    
    # VAE 만들고 익히기
    latent_dim = 5  # 참 숨은 차원을 살짝 높여 잡음
    model = VAE(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        latent_dim=latent_dim
    )
    
    print(f"\nVAE architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: [64, 32]")
    print(f"  Latent dim: {latent_dim}")
    
    history = train_vae(model, train_loader, n_epochs=100, lr=1e-3, verbose=True)
    
    # 결과를 그려 본다
    visualize_vae_results(model, history, test_data)
    
    # 나눠 갚기 틈 재기
    print("\n--- Amortization Gap Analysis ---")
    elbo_amort, elbo_opt, gap = measure_amortization_gap(model, test_data[:1])
    print(f"Amortized ELBO: {elbo_amort:.4f}")
    print(f"Optimal ELBO: {elbo_opt:.4f}")
    print(f"Gap: {gap:.4f}")
```

```
============================================================
Variational Autoencoder Training
============================================================

Data shape: torch.Size([2000, 20])
True latent dim: 3

VAE architecture:
  Input dim: 20
  Hidden dims: [64, 32]
  Latent dim: 5
Epoch  10: Loss = 9.7852, Recon = 3.2732, KL = 6.5120
Epoch  20: Loss = 7.6809, Recon = 2.2110, KL = 5.4699
Epoch  30: Loss = 7.4358, Recon = 1.9500, KL = 5.4857
Epoch  40: Loss = 7.3350, Recon = 1.8965, KL = 5.4385
Epoch  50: Loss = 7.3403, Recon = 1.9382, KL = 5.4021
Epoch  60: Loss = 7.3298, Recon = 1.8380, KL = 5.4918
Epoch  70: Loss = 7.2790, Recon = 1.8411, KL = 5.4380
Epoch  80: Loss = 7.2825, Recon = 1.8445, KL = 5.4380
Epoch  90: Loss = 7.3502, Recon = 1.8260, KL = 5.5242
Epoch 100: Loss = 7.3394, Recon = 1.8611, KL = 5.4783

--- Amortization Gap Analysis ---
Amortized ELBO: -7.6733
Optimal ELBO: -8.2068
Gap: -0.5335
```

## 5. 조건부 VAE

조건 c를 사용하는 CVAE는 p_θ(z|c), p_θ(x|z,c), q_ψ(z|x,c)를 정의한다. 사전은 c에 의존하도록 학습하거나 공통 표준정규로 고정할 수 있다.

$$L(x,c)=E_{q_\psi(z|x,c)}[\log p_\theta(x|z,c)]-D_{\mathrm{KL}}(q_\psi(z|x,c)\Vert p_\theta(z|c)).$$

학습에서는 x와 c를 인코더에 주지만 조건부 생성에서는 c만 주고 조건부 사전에서 z를 뽑는다. 클래스 조건부 생성, 입력에 따른 출력 생성 등에서 다중 가능한 출력을 모델링하는 데 사용한다. 조건을 넣었다는 사실만으로 조건 준수나 모든 모드의 포괄이 보장되지는 않는다.

## 6. 지역 보정과 반복 추론

인코더의 출력을 초기값으로 지역 변분 매개변수를 더 최적화하면 유한 계산 예산에서 상각 오차를 줄일 수 있다. 같은 θ·목적을 고정하고 실제 개선을 확보해야 하며, 임의의 Adam 반복이 매번 개선한다는 보장은 없다. 반복 인코더는 현재 근사나 경사 정보를 입력받아 갱신을 학습하는 다른 방식이다.

아래 코드는 지역 보정의 의도를 보여 주지만 그대로 완성된 구현은 아니다. `mu.clone().requires_grad_(True)`는 원래 인코더 출력의 계산 그래프를 유지하므로 일반적인 Adam이 요구하는 leaf 텐서가 아니다. 독립적인 지역 최적화에는 분리한 leaf 변수가 필요하다. `compute_loss`도 외부 정의이며, 디코더·인코더의 학습 경로와 고정 여부를 명시해야 한다.

보정 후 `.detach()`한 결과만 반환하는 추론 절차와, 보정 단계를 미분해 인코더를 학습하는 정식 semi-amortized 학습은 구분한다. 후자는 추가적인 미분과 메모리·계산비용이 든다.

```python
def semi_amortized_inference(model, x, n_refine_steps=10):
    # 나눠 갚아 얻은 첫 매개변수 가져오기
    mu, logvar = model.encode(x)
    
    # 기울기 내리기로 다듬기
    mu = mu.clone().requires_grad_(True)
    logvar = logvar.clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([mu, logvar], lr=0.01)
    
    for _ in range(n_refine_steps):
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        x_recon = model.decode(z)
        loss = compute_loss(x, x_recon, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return mu.detach(), logvar.detach()
```

## 7. 평가할 항목

재구성 오차가 작다는 사실만으로 사후 근사나 생성분포 전체가 정확한 것은 아니다. 관측당 단위의 목적, 사후 예측, 잠재변수 사용 정도와 조건부 생성 품질을 함께 살핀다. 작은 KL은 사전과 가까운 근사를 의미하지만 잠재정보의 유용성까지 결정하지 않는다. posterior collapse에서는 디코더가 잠재변수를 거의 사용하지 않을 수 있다.

생성모델을 고정한 뒤 같은 분포족의 충분한 지역 최적화와 비교하면 공유 인코더의 한계를 살필 수 있다. 다만 정확한 최적값을 모를 때는 ‘지역 보정에 의한 추정 개선량’처럼 계산한 양을 구체적으로 보고한다.

## 연습문제

### 연습 1: 두 차이의 분해

고정 θ에서 로그 주변우도와 상각 ELBO의 차이를 근사 차이와 상각 차이로 분해하라. 인코더를 크게 만드는 것과 변분분포족을 확장하는 것의 역할을 구분하라.

### 연습 2: VAE 손실의 단위

관측 가우스 분산을 σ_x²로 둘 때 제곱오차 앞 계수와 정규화 상수를 쓰라. 특징별 합·관측별 평균을 특징까지 평균하는 손실로 바꾸면 KL과의 상대 가중치가 어떻게 달라지는가?

### 연습 3: 조건부 생성

CVAE의 학습과 생성에서 사용할 수 있는 입력을 구분하라. 생성 시 x를 요구하지 않고 c만으로 z와 x를 순서대로 생성하는 분포를 쓰라.

### 연습 4: 보정과 평가

유한한 지역 최적화 결과를 참 최적 ELBO로 부를 수 없는 이유를 설명하라. 모델 고정, leaf 변수, 목적값 재평가와 몬테카를로 오차를 고려한 평가 절차를 설계하라.

## 정리하며

상각 추론은 공유 인코더로 관측별 추론 비용을 줄인다. 고정된 생성모델과 분포족에서 근사 차이와 상각 차이를 구분하고, 지역 보정의 유한·확률적 결과를 참 최적값으로 해석하지 않는다. VAE의 손실 단위와 디코더의 확률적 의미도 함께 명시한다.

**참고 문헌**

1. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."

3. Sohn, K., Lee, H., & Yan, X. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models."

4. Cremer, C., Li, X., & Duvenaud, D. (2018). "Inference Suboptimality in Variational Autoencoders."

5. Higgins, I., et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework."

6. Kim, Y., et al. (2018). "Semi-Amortized Variational Autoencoders."
