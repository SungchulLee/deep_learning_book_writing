# 나눠 갚는 변분 추론
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. 변분 추론에서 나눠 갚기의 개념 이해하기
2. 뒤확률 어림을 위한 추론 망 구현하기
3. 변분 자동부호기(VAE) 세우고 익히기
4. 나눠 갚기 틈과 그것이 뜻하는 바 살피기
5. 큰 문제에 나눠 갚는 변분 추론 쓰기

## 낱낱 추론에서 나눠 갚는 추론으로

전통 변분 추론은 **관측마다 따로** 변분 매개변수를 최적화한다:

$$
\phi_n^* = \arg\max_\phi \text{ELBO}(q_\phi(z_n | x_n))
$$

관측이 $N$개이면 따로 된 최적화 문제가 $N$개 필요하다!

**나눠 갚는 변분 추론**은 관측을 변분 매개변수로 잇는 **함수 하나**를 배운다:

$$
\phi = f_\psi(x) \quad \text{(inference network)}
$$

여기서 $\psi$은 모든 관측이 함께 쓰는 망의 매개변수이다.

### 나눠 갚기의 이로움

1. **규모 키우기**: 모든 자료 점에 망 하나
2. **넓혀 나감**: 처음 보는 새 자료의 뒤확률도 추론할 수 있다
3. **빠르기**: 시험할 때 최적화가 필요 없다(앞먹임만 하면 된다)
4. **어우러짐**: 깊은 낳는 모형과 자연스럽게 맞는다

### 그 값: 나눠 갚기 틈

나눠 갚기 틈은 망 하나를 함께 씀으로써 생기는, 최적에 못 미치는 정도이다:

$$
\text{Gap}(x) = \max_\phi \text{ELBO}(q_\phi(z|x)) - \text{ELBO}(q_{f_\psi(x)}(z|x))
$$

추론 망이 자료 점마다의 가장 좋은 변분 매개변수를 완벽히 담아내지는 못할 수 있다.

## 변분 자동부호기(VAE)

**변분 자동부호기**는 나눠 갚는 변분 추론의 대표 보기로 다음을 어우른다:

1. **부호기**(추론 망): 관측을 변분 매개변수로 잇는다
2. **풀개**(낳는 모형): 숨은 변수를 관측으로 잇는다
3. **ELBO 목표**: 부호기와 풀개를 함께 익힌다

### VAE의 낳는 모형

$$
\begin{aligned}
\text{Prior: } & z \sim p(z) = \mathcal{N}(0, I) \\
\text{Likelihood: } & x | z \sim p_\theta(x|z)
\end{aligned}
$$

풀개 신경망 $p_\theta(x|z)$은 숨은 부호를 자료 분포로 잇는다.

### VAE의 추론 모형

부호기는 다룰 수 없는 뒤확률 $p(z|x)$을 어림한다:

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))
$$

여기서 $\mu_\phi(x)$과 $\sigma_\phi(x)$은 부호기 망의 내임이다.

### VAE의 ELBO

VAE의 목표는 자료 전체에 걸쳐 평균 낸 ELBO이다:

$$
\mathcal{L}(\theta, \phi) = \frac{1}{N} \sum_{n=1}^N \left[\mathbb{E}_{q_\phi(z|x_n)}[\log p_\theta(x_n|z)] - \text{KL}(q_\phi(z|x_n) \| p(z))\right]
$$

**되살림 항**: 풀개가 표집한 $z$으로 $x$을 얼마나 잘 되살릴 수 있나?

**KL 항**: 어림 뒤확률이 앞확률에 얼마나 가까운가?

### VAE의 매개변수 바꾸기

표집 연산을 거쳐 되짚으려면:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

이러면 기울기가 $\mu_\phi$과 $\sigma_\phi$을 지나 흐를 수 있다.

## PyTorch 구현

### 온전한 VAE 구현

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
    VAE 부호기: 들임 x을 변분 매개변수 (μ, log σ²)로 잇는다.
    
    얼개: x -> [숨은 층] -> (μ, log σ²)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        # 부호기 층 세우기
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
    VAE 풀개: 숨은 z을 되살림 매개변수로 잇는다.
    
    얼개: z -> [숨은 층] -> x_recon
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        
        # 풀개 층 세우기
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
    변분 자동부호기.
    
    부호기(추론 망)와 풀개(낳는 모형)를 어우르며
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
        """부호기와 풀개를 거쳐 들임 되살리기."""
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

## 조건부 VAE(CVAE)

**조건부 VAE**은 VAE을 넓혀 $p(x|c)$을 본뜬다. 여기서 $c$은 조건이 되는 변수이다:

$$
\begin{aligned}
\text{Prior: } & z | c \sim p(z|c) \\
\text{Likelihood: } & x | z, c \sim p_\theta(x|z, c) \\
\text{Inference: } & q_\phi(z|x, c)
\end{aligned}
$$

### CVAE의 ELBO

$$
\mathcal{L}(\theta, \phi; x, c) = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - \text{KL}(q_\phi(z|x,c) \| p(z|c))
$$

### 쓰임새

- **갈래를 조건으로 한 만들어 내기**: 정해진 갈래에서 표본을 만든다
- **그림에서 그림으로 옮기기**: 근원이 주어졌을 때 과녁을 만든다
- **짜임새 있는 미리봄**: 조건부 분포를 본뜬다

## 나눠 갚기 틈 줄이기

나눠 갚는 최적화와 낱낱 최적화 사이의 틈을 줄이는 기법이 여럿 있다:

### 1. 반쯤 나눠 갚는 추론

나눠 갚는 추론으로 시작한 뒤 최적화를 몇 걸음 더 해 다듬는다:

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

### 2. 되풀이하며 나눠 갚는 추론

부호기를 여러 번 되풀이해 쓴다:

$$
(\mu^{(t+1)}, \sigma^{(t+1)}) = f_\psi(x, \mu^{(t)}, \sigma^{(t)})
$$

### 3. 표현력이 더 좋은 추론 망

고르게 하는 흐름으로 $q_\phi(z|x)$의 유연함을 높인다.

## 요약

**나눠 갚는 변분 추론**은 함께 쓰는 추론 망을 배운다:

$$
\phi = f_\psi(x)
$$

**VAE의 핵심 부품:**

- **부호기**: $q_\phi(z|x)$ - 어림 뒤확률
- **풀개**: $p_\theta(x|z)$ - 낳는 모형
- **ELBO**: $\mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$
- **매개변수 바꾸기**: $z = \mu + \sigma \odot \epsilon$

**주고받음:**

- 시험할 때 추론이 빠르다
- 나눠 갚기 틈이 질을 떨어뜨린다
- 반쯤 나눠 갚는 방법으로 누그러뜨릴 수 있다

## 참고 문헌

1. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."

3. Sohn, K., Lee, H., & Yan, X. (2015). "Learning Structured Output Representation using Deep Conditional Generative Models."

4. Cremer, C., Li, X., & Duvenaud, D. (2018). "Inference Suboptimality in Variational Autoencoders."

5. Higgins, I., et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework."

6. Kim, Y., et al. (2018). "Semi-Amortized Variational Autoencoders."

## 연습문제

### 연습 1: 합성곱 VAE

그림 자료를 위한 합성곱 부호기와 풀개를 갖춘 VAE을 구현하여라.

### 연습 2: β-VAE

β-VAE을 구현하고 β이 서로 풀림에 주는 영향을 살펴라.

### 연습 3: 조건부 만들어 내기를 위한 CVAE

MNIST에서 갈래를 조건으로 한 만들어 내기를 위해 CVAE을 구현하여라.
