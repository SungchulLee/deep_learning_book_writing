# 단순한 2차원 맞겨루기 만들개

2차원 장난감 자료 묶음으로 맞겨루기 만들개를 익히면 맞겨루기 익히기의 움직임에 대한 값진 직관을 얻는다. 두 차원에서 다루면 만들개가 자료 분포를 어떻게 어림해 가는지, 익히는 동안 가름개의 결정 가장자리가 어떻게 바뀌는지 곧바로 그려 볼 수 있다. 이 방식은 봉우리 무너짐, 익히기의 불안정, 두 신경망의 주고받음을 이해하는 데 더없이 값지다.

## 코드

```python
"""
단순한 2차원 맞겨루기 만들개

2차원 장난감 자료에서 맞겨루기 만들개 익히기를 그려 본다.
맞겨루기 익히기의 움직임을 이해하기에 알맞다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================


class SimpleGenerator(nn.Module):
    """2차원 자료를 위한 단순한 만들개."""
    
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, z):
        return self.model(z)


class SimpleDiscriminator(nn.Module):
    """2차원 자료를 위한 단순한 가름개."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def generate_data(n_samples=1000, dataset='moons'):
    """2차원 장난감 자료 묶음을 만든다."""
    if dataset == 'moons':
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
    elif dataset == 'circles':
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
    elif dataset == 'gaussian':
        # 정규 분포 둘
        data1 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([2, 2])
        data2 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([-2, -2])
        data = np.vstack([data1, data2])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return torch.tensor(data, dtype=torch.float32)


def visualize_training_step(generator, discriminator, real_data, epoch, 
                           fixed_noise, filename=None):
    """만들개 분포와 가름개 결정 가장자리를 그려 본다."""
    generator.eval()
    discriminator.eval()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    with torch.no_grad():
        # 가짜 표본을 만든다
        fake_data = generator(fixed_noise).cpu().numpy()
    
    real_data_np = real_data.cpu().numpy()
    
    # 그림 1: 실제 자료
    axes[0].scatter(real_data_np[:, 0], real_data_np[:, 1], alpha=0.5, s=20)
    axes[0].set_title('Real Data')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].grid(True, alpha=0.3)
    
    # 그림 2: 만든 자료
    axes[1].scatter(fake_data[:, 0], fake_data[:, 1], alpha=0.5, s=20, color='red')
    axes[1].set_title(f'Generated Data (Epoch {epoch})')
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].grid(True, alpha=0.3)
    
    # 그림 3: 가름개 결정 가장자리
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                         dtype=torch.float32)
    
    with torch.no_grad():
        d_scores = discriminator(points).cpu().numpy()
    
    d_scores = d_scores.reshape(200, 200)
    
    contour = axes[2].contourf(X, Y, d_scores, levels=20, cmap='RdYlBu')
    axes[2].scatter(real_data_np[:, 0], real_data_np[:, 1], 
                   alpha=0.3, s=10, color='blue', label='Real')
    axes[2].scatter(fake_data[:, 0], fake_data[:, 1], 
                   alpha=0.3, s=10, color='red', label='Fake')
    axes[2].set_title('Discriminator Decision Boundary')
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].legend()
    plt.colorbar(contour, ax=axes[2], label='D(x)')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    generator.train()
    discriminator.train()


def train_gan_2d(data, latent_dim=2, n_epochs=1000, batch_size=256, 
                lr=0.0002, device='cpu'):
    """2차원 자료에서 단순한 맞겨루기 만들개를 익힌다."""
    
    # 신경망을 첫자리매김한다
    generator = SimpleGenerator(latent_dim=latent_dim).to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # 가장 좋게 하개
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 손실
    criterion = nn.BCELoss()
    
    # 그려 보기를 위한 붙박이 잡음
    fixed_noise = torch.randn(500, latent_dim, device=device)
    
    # 학습 루프
    g_losses = []
    d_losses = []
    
    print("Training 2D GAN...")
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        # 묶음을 뽑는다
        indices = torch.randint(0, len(data), (batch_size,))
        real_batch = data[indices].to(device)
        
        # 이름표
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # 가름개를 익힌다
        discriminator.zero_grad()
        
        # 실제 자료
        d_real = discriminator(real_batch)
        real_loss = criterion(d_real, real_labels)
        
        # 가짜 자료
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = generator(noise)
        d_fake = discriminator(fake_batch.detach())
        fake_loss = criterion(d_fake, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # 만들개를 익힌다
        generator.zero_grad()
        
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = generator(noise)
        d_fake = discriminator(fake_batch)
        g_loss = criterion(d_fake, real_labels)
        
        g_loss.backward()
        g_optimizer.step()
        
        # 손실 기록
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        pbar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
        
        # 나아감을 그려 본다
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            visualize_training_step(
                generator, discriminator, data, epoch,
                fixed_noise, filename=f'2d_gan_epoch_{epoch:04d}.png'
            )
    
    return generator, discriminator, g_losses, d_losses


def plot_loss_curves(g_losses, d_losses):
    """익히기 손실 곡선을 그린다."""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator', alpha=0.7)
    plt.plot(d_losses, label='Discriminator', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('2d_gan_losses.png', dpi=150)
    plt.close()
    print("Saved loss curves")


def main():
    """2차원 맞겨루기 만들개 보여 주기의 으뜸 함수."""
    print("=" * 60)
    print("2D GAN Visualization Demo")
    print("=" * 60)
    
    # 설정
    dataset = 'moons'  # 시험해 보라: 'moons', 'circles', 'gaussian'
    n_samples = 2000
    latent_dim = 2
    n_epochs = 1000
    
    print(f"\nDataset: {dataset}")
    print(f"Samples: {n_samples}")
    print(f"Epochs: {n_epochs}\n")
    
    # 데이터를 생성한다
    data = generate_data(n_samples, dataset)
    
    # 본디 자료를 그린다
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.5, s=20)
    plt.title('Original Data')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)
    plt.savefig('2d_gan_original_data.png', dpi=150)
    plt.close()
    
    # 맞겨루기 만들개를 익힌다
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    generator, discriminator, g_losses, d_losses = train_gan_2d(
        data, latent_dim=latent_dim, n_epochs=n_epochs, device=device
    )
    
    # 손실 곡선을 그린다
    plot_loss_curves(g_losses, d_losses)
    
    print("\n" + "=" * 60)
    print("Demo complete! Generated files:")
    print("  - 2d_gan_original_data.png: Original dataset")
    print("  - 2d_gan_epoch_*.png: Training progress")
    print("  - 2d_gan_losses.png: Loss curves")
    print("=" * 60)


if __name__ == "__main__":
    main()```

## 논의

SimpleGenerator과 SimpleDiscriminator은 2차원 자료를 위해 만든 작은 여러 층 신경망이다. 만들개는 2차원 숨은 벡터를 ReLU 깨움을 갖춘 숨은 층으로 옮겨 2차원 내놓기 점을 만든다. 가름개는 2차원 점을 받아 LeakyReLU 깨움과 마지막 시그모이드로 확률을 내놓는다. 2차원 숨은 공간은 자료 차원과 맞아 옮김을 더 풀이하기 쉽게 한다.

익히기 되풀이는 여느 맞겨루기 만들개 익히기 절차를 짠다. 곧 먼저 두값 어긋 엔트로피 손실로 실제 자료와 가짜 자료 모두에서 가름개를 익히고, 이어 가짜 자료를 실제 이름표와 함께 가름개에 넣어 만들개를 익힌다. $\beta_1 = 0.5$인 Adam 가장 좋게 하개는 맞겨루기 만들개 익히기에 대한 DCGAN의 권고를 따른다. 그려 보기 함수는 실제 자료, 만든 자료, 가름개의 결정 가장자리를 보이는 세 칸을 만든다.

`generate_data` 함수는 반달, 겹동그라미, 두 정규 분포 섞기 같은 여러 장난감 자료 묶음을 받쳐 준다. 이 자료 묶음은 만들개 배움의 여러 면을 시험한다. 곧 달 모양은 비선형 다양체 배우기를, 동그라미는 돌림 대칭을, 정규 분포는 봉우리 덮기를 시험한다. 결정 가장자리 그림은 특히 알려 주는 바가 많아, 익히기가 나아가면서 가름개가 실제 표본과 만든 표본을 어떻게 갈라내는지 보여 준다.

## 연습문제

**연습문제 1.**
세 자료 묶음(달 모양, 동그라미, 정규 분포)마다 2차원 맞겨루기 만들개를 1000바퀴 돌려라. 마지막 만들개 분포와 가름개 결정 가장자리를 견주어라. 어느 자료 묶음이 배우기 가장 어려운가? 왜인가?

??? success "연습문제 1 풀이"
    정규 분포 섞기는 떨어진 대칭 봉우리 둘로 이루어져 흔히 가장 쉽다. 달 모양 자료 묶음은 굽고 서로 끼워진 짜임 때문에 웬만큼 어렵다. 동그라미는 안쪽 동그라미가 바깥 동그라미에 온전히 둘러싸여 만들개가 서로 다른 반지름 둘에서 표본을 내야 하므로 가장 어렵기 쉽다. 동그라미의 가름개 결정 가장자리는 닫힌 고리 모양 자리를 이루어야 하는데, 이는 다른 자료 묶음에 필요한 선형이나 부드럽게 굽은 가장자리보다 복잡하다.

---

**연습문제 2.**
만들개를 한 번 고칠 때마다 가름개를 $k$번 고치도록 익히기 되풀이를 고쳐라($k = 1, 3, 5$을 시험하라). 이 비율이 만든 표본의 품질과 가름개 결정 가장자리의 매끄러움에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    $k = 1$(기본)이면 가름개와 만들개가 균형을 이루지만 가름개가 좋은 기울기를 줄 만큼 세지 않을 수 있다. $k = 3$이면 가름개가 더 정확해져 만들개에 더 나은 기울기 신호를 주며 흔히 표본 품질이 나아진다. $k = 5$이면 가름개가 너무 세져 만들개의 기울기가 사라질 수 있다. 가장 좋은 비율은 자료 묶음의 복잡함에 달렸다. 단순한 2차원 자료 묶음에서는 흔히 $k = 1$이나 $k = 2$이 잘 듣고, $k$이 크면 결정 가장자리가 매끄러워지지만 만들개가 느리게 모일 수 있다.

---

**연습문제 3.**
짝마다 평균 거리로 만든 표본의 다양함을 재는 봉우리 무너짐 알아내개를 짜라. 이 잣대를 익히기 되풀이에 더하고 만들개와 가름개 손실과 함께 그려라.

??? success "연습문제 3 풀이"
    ```python
    def mode_collapse_metric(generator, latent_dim, n_samples=500, device='cpu'):
        noise = torch.randn(n_samples, latent_dim, device=device)
        with torch.no_grad():
            fake = generator(noise).cpu().numpy()
        dists = np.sqrt(((fake[:, None] - fake[None, :]) ** 2).sum(-1))
        return np.mean(dists[np.triu_indices(n_samples, k=1)])
    ```
    익히기 바퀴에 따라 이 잣대를 그려라. 짝마다 평균 거리가 갑자기 떨어지면 봉우리 무너짐이며, 만들개가 들임과 상관없이 거의 같은 것을 내놓는다는 뜻이다. 건강한 익히기에서는 이 잣대가 실제 자료의 짝 거리와 비슷한 값에서 안정된다.
