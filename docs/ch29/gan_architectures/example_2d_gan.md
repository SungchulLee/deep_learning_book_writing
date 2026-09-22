# 단순한 2차원 맞겨루기 만들개

2차원 장난감 자료 묶음으로 맞겨루기 만들개를 익히면 맞겨루기 익히기의 움직임에 대한 값진 직관을 얻는다. 두 차원에서 다루면 만들개가 자료 분포를 어떻게 어림해 가는지, 익히는 동안 가름개의 결정 가장자리가 어떻게 바뀌는지 곧바로 그려 볼 수 있다. 이 방식은 봉우리 무너짐, 익히기의 불안정, 두 신경망의 주고받음을 이해하는 데 더없이 값지다.

## 1. 코드

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
        # 배치를 뽑는다
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
    main()
```

**출력:**

```
============================================================
2D GAN Visualization Demo
============================================================

Dataset: moons
Samples: 2000
Epochs: 1000

Using device: cpu

Training 2D GAN...
Saved loss curves

============================================================
Demo complete! Generated files:
  - 2d_gan_original_data.png: Original dataset
  - 2d_gan_epoch_*.png: Training progress
  - 2d_gan_losses.png: Loss curves
============================================================
```

## 2. 논의

SimpleGenerator과 SimpleDiscriminator은 2차원 자료를 위해 만든 작은 여러 층 신경망이다. 만들개는 2차원 숨은 벡터를 ReLU 깨움을 갖춘 숨은 층으로 옮겨 2차원 내놓기 점을 만든다. 가름개는 2차원 점을 받아 LeakyReLU 깨움과 마지막 시그모이드로 확률을 내놓는다. 2차원 숨은 공간은 자료 차원과 맞아 옮김을 더 풀이하기 쉽게 한다.

익히기 되풀이는 여느 맞겨루기 만들개 익히기 절차를 짠다. 곧 먼저 두값 어긋 엔트로피 손실로 실제 자료와 가짜 자료 모두에서 가름개를 익히고, 이어 가짜 자료를 실제 이름표와 함께 가름개에 넣어 만들개를 익힌다. $\beta_1 = 0.5$인 Adam 가장 좋게 하개는 맞겨루기 만들개 익히기에 대한 DCGAN의 권고를 따른다. 그려 보기 함수는 실제 자료, 만든 자료, 가름개의 결정 가장자리를 보이는 세 칸을 만든다.

`generate_data` 함수는 반달, 겹동그라미, 두 정규 분포 섞기 같은 여러 장난감 자료 묶음을 받쳐 준다. 이 자료 묶음은 만들개 배움의 여러 면을 시험한다. 곧 달 모양은 비선형 다양체 배우기를, 동그라미는 돌림 대칭을, 정규 분포는 봉우리 덮기를 시험한다. 결정 가장자리 그림은 특히 알려 주는 바가 많아, 익히기가 나아가면서 가름개가 실제 표본과 만든 표본을 어떻게 갈라내는지 보여 준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
세 자료 묶음(달 모양, 동그라미, 정규 분포)마다 2차원 맞겨루기 만들개를 1000바퀴 돌려라. 마지막 만들개 분포와 가름개 결정 가장자리를 견주어라. 어느 자료 묶음이 배우기 가장 어려운가? 왜인가?

</div>

??? success "연습문제 1 풀이"
    정규 분포 섞기는 떨어진 대칭 봉우리 둘로 이루어져 흔히 가장 쉽다. 달 모양 자료 묶음은 굽고 서로 끼워진 짜임 때문에 웬만큼 어렵다. 동그라미는 안쪽 동그라미가 바깥 동그라미에 온전히 둘러싸여 만들개가 서로 다른 반지름 둘에서 표본을 내야 하므로 가장 어렵기 쉽다. 동그라미의 가름개 결정 가장자리는 닫힌 고리 모양 자리를 이루어야 하는데, 이는 다른 자료 묶음에 필요한 선형이나 부드럽게 굽은 가장자리보다 복잡하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
만들개를 한 번 고칠 때마다 가름개를 $k$번 고치도록 익히기 되풀이를 고쳐라($k = 1, 3, 5$을 시험하라). 이 비율이 만든 표본의 품질과 가름개 결정 가장자리의 매끄러움에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    $k = 1$(기본)이면 가름개와 만들개가 균형을 이루지만 가름개가 좋은 기울기를 줄 만큼 세지 않을 수 있다. $k = 3$이면 가름개가 더 정확해져 만들개에 더 나은 기울기 신호를 주며 흔히 표본 품질이 나아진다. $k = 5$이면 가름개가 너무 세져 만들개의 기울기가 사라질 수 있다. 가장 좋은 비율은 자료 묶음의 복잡함에 달렸다. 단순한 2차원 자료 묶음에서는 흔히 $k = 1$이나 $k = 2$이 잘 듣고, $k$이 크면 결정 가장자리가 매끄러워지지만 만들개가 느리게 모일 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
짝마다 평균 거리로 만든 표본의 다양함을 재는 봉우리 무너짐 알아내개를 짜라. 이 잣대를 익히기 되풀이에 더하고 만들개와 가름개 손실과 함께 그려라.

</div>

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
2차원 자료로 적대적 생성망을 보이는 것이 왜 좋은가?

</div>

??? success "연습문제 4 풀이"
    **눈으로 분포를 볼 수 있다.** 784차원에서는 못 하는 일이다.

    그림 한 장에 참 자료의 점들과 생성기의 점들을 함께 찍으면 두 분포가 얼마나 겹치는지
    한눈에 보인다. 그래서 잣대 없이도 익히기의 진행을 판단할 수 있다.

    특히 잘 보이는 것들이 있다.

    | 무엇 | 2차원에서 |
    |---|---|
    | 무너짐 | 점들이 한 자리에 뭉친다 |
    | 봉우리 빠뜨림 | 자료의 어떤 덩이에 점이 없다 |
    | 퍼짐이 잘못됨 | 점들이 너무 넓거나 좁다 |
    | 판별기의 결정 경계 | 배경에 색으로 그릴 수 있다 |

    마지막 줄이 특히 값지다. 판별기가 무엇을 보고 있는지 그릴 수 있어, 두 그물이 서로
    어떻게 밀고 당기는지 이해하게 된다. MNIST에서는 상상만 할 수 있는 일이다.

    그래서 이 예가 값매김 잣대보다 먼저 와야 한다. 잣대가 재는 것이 무엇인지 알려면
    분포가 겹친다는 것이 무슨 뜻인지 먼저 보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
봉우리 여러 개인 2차원 자료에서 무너짐이 어떤 모습으로 나타나는가?

</div>

??? success "연습문제 5 풀이"
    생성기가 **봉우리 하나(또는 몇 개)에만 점을 찍는다.** 나머지 봉우리는 비어 있다.

    이것을 봉우리 빠뜨림(mode dropping)이라 하고, 더 심하면 한 점으로 모인다.

    왜 일어나는가를 2차원에서 보면 이해가 쉽다. 생성기가 봉우리 하나를 잘 흉내 내면
    그 자리에서는 판별기를 속일 수 있다. 다른 봉우리로 옮겨 갈 이유가 **당장은** 없다.
    손실이 옮겨 가라고 밀어 주지 않기 때문이다.

    더 고약한 모습도 있다. 생성기가 봉우리 사이를 **돌아다니는** 것이다. 판별기가 지금
    있는 자리를 배우면 생성기가 다른 자리로 옮기고, 판별기가 따라오면 또 옮긴다. 수렴하지
    않고 순환한다.

    이 순환이 [내시 균형](../gan_foundations/nash.md)에서 다루는 문제이며, 2차원 그림으로
    보면 아주 또렷하다. 점들의 덩이가 봉우리를 차례로 옮겨 다니는 것이 보인다.

    MNIST에서는 이 순환을 보기 어렵다. 부류 쏠림을 에포크마다 기록하면 비슷한 것을 볼
    수 있다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
판별기의 결정 경계를 그리면 무엇을 알 수 있는가?

</div>

??? success "연습문제 6 풀이"
    두 그물이 어떻게 맞물려 있는지 보인다.

    격자 점마다 $D(x)$를 셈해 색으로 그리면 판별기가 어디를 참으로 보고 어디를 거짓으로
    보는지 나온다.

    읽을 것이 몇 가지다.

    **경계가 또렷한가.** 아주 날카로우면 판별기가 이기고 있다는 뜻이고, 기울기가 사라질
    위험이다([GAN 기초 연습문제 2](45_gan.md)).

    **생성기 점들이 어디 있는가.** 판별기가 거짓으로 보는 자리에 있으면 밀려날 것이고,
    참으로 보는 자리에 있으면 머물 것이다. 점들이 움직이는 방향을 예측할 수 있다.

    **참 자료 영역 안에 거짓으로 보는 자리가 있는가.** 생성기가 아직 덮지 못한 봉우리다.

    잘 익어 가면 경계가 흐려지고 $D \approx 0.5$가 넓게 퍼진다. 곧 판별기가 두 분포를
    구별하지 못하는 상태이며, 이론이 말하는 균형에 가까운 모습이다.

    이 그림을 에포크마다 저장해 이어 보면 두 그물이 서로 좇는 과정이 애니메이션처럼
    보인다. 적대적 익히기를 이해하는 데 가장 도움이 되는 그림이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
2차원 예에서 잣대를 어떻게 재겠는가?

</div>

??? success "연습문제 7 풀이"
    MNIST의 잣대들이 그대로 맞지 않는다. 인셉션 점수와 FID는 특징 그물이 필요하고,
    2차원 자료에는 그런 것이 없다.

    대신 쓸 수 있는 것들이 있다.

    | 잣대 | 어떻게 |
    |---|---|
    | 참 자료와의 거리 | 2차원이므로 직접 재도 뜻이 있다 |
    | 봉우리 덮음 | 자료의 봉우리마다 생성기 점이 있는지 센다 |
    | 두 표본 검정 | 두 표본이 같은 분포인지 통계적으로 검정한다 |
    | 눈으로 | 2차원이므로 가장 곧다 |

    둘째가 특히 쓸 만하다. 봉우리 위치를 알고 있으므로(우리가 만든 자료다) 각 봉우리
    근처에 점이 몇 개인지 셀 수 있다. 무너짐이 정확히 수로 나온다.

    **참값을 아는 자료의 값어치**가 여기 있다. 실제 자료에서는 봉우리가 몇인지, 어디인지
    모르므로 덮음을 정확히 잴 수 없다. 그래서 잣대들이 어림에 기대게 된다.

    이것이 인공 자료로 먼저 실험하는 까닭이다. 잣대가 무엇을 잘 재고 무엇을 놓치는지
    확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
2차원에서 잘되는 설정이 MNIST에서도 잘되는가?

</div>

??? success "연습문제 8 풀이"
    반드시 그렇지 않다. 옮겨 갈 때 달라지는 것이 여럿이다.

    **차원.** 2차원에서는 두 분포가 겹치기 쉽지만 784차원에서는 거의 겹치지 않는다.
    그래서 옌센–섀넌 벌어짐의 기울기가 사라지는 문제가 높은 차원에서 훨씬 심하다
    ([GAN 기초 연습문제 5](45_gan.md)).

    **얼개.** 2차원에는 공간 짜임이 없으니 완전 연결로 충분하다. 그림에서는 누비기가
    쓸모 있다.

    **자료의 복잡도.** 봉우리 여덟 개인 인공 자료는 실제 그림 자료보다 훨씬 단순하다.

    그래도 2차원 실험이 값진 까닭이 있다. **왜 그런 일이 벌어지는지**를 보여 주기 때문이다.
    MNIST에서 무너짐을 만나면 그것이 어떤 모습인지 이미 2차원에서 보았으므로 진단이 빠르다.

    옮겨 가는 것과 안 옮겨 가는 것을 가르면 이렇다.

    | 옮겨 간다 | 안 옮겨 간다 |
    |---|---|
    | 무너짐의 성질과 원인 | 구체적인 웃매개변수 값 |
    | 판별기와 생성기의 균형이라는 생각 | 몇 층, 몇 채널인지 |
    | 손실이 신호가 아니라는 사실 | 학습률의 값 |

    그러므로 2차원은 **이해를 위한 것**이고 설정은 자료마다 다시 찾아야 한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
이 예에서 잡음 차원을 자료 차원보다 크게 두어야 하는가?

</div>

??? success "연습문제 9 풀이"
    2차원 자료에 잡음 차원을 2로 두어도 되고, 크게 두는 것이 흔하다.

    잡음 차원이 자료의 속 차원보다 **작으면** 생성기가 자료를 다 덮을 수 없다. 그것은
    분명하다.

    같거나 크면 원칙적으로 충분한데, 실무에서 크게 두는 편이 낫다고 이야기된다. 최적화가
    쉬워지기 때문이라고 설명된다. 여유 차원이 있으면 생성기가 더 매끄러운 사상을 고를
    수 있다.

    남는 차원이 어떻게 되는지는 재미있는 물음이다. 생성기가 그 차원을 **무시하는** 쪽으로
    배우는 것이 보통이며, 그 방향으로 $z$를 바꾸어도 출력이 거의 안 변하는 것을 확인할
    수 있다.

    [26장의 변분 자기 부호기](../../ch26/architecture/prior.md)에서 차원이 죽는 것과
    비슷해 보이지만 뜻이 다르다. 거기서는 KL 항이 차원을 끄도록 밀었고, 여기서는 벌하는
    항이 없는데도 쓰지 않는 것이다. 그리고 적대적 생성망에는 그것을 진단할 KL 같은 값이
    없어 재기가 더 어렵다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 예를 애니메이션으로 만들려면 무엇을 저장해야 하는가?

</div>

??? success "연습문제 10 풀이"
    에포크마다 세 가지를 저장한다.

    ```python
    snapshots.append(dict(
        fake=g(fixed_z).detach().cpu().numpy(),      # 고정된 z 의 출력
        grid=decision_grid(d),                       # 판별기 경계
        epoch=ep,
    ))
    ```

    **$z$를 고정하는 것**이 핵심이다. 매번 새로 뽑으면 점들이 무작위로 튀어 흐름이 안
    보인다. 고정하면 같은 점들이 움직여 가는 것이 보인다.

    판별기 경계를 함께 저장하면 두 그물이 서로 좇는 것이 보인다. 격자를 너무 촘촘히
    하면 저장이 커지므로 $100\times100$ 정도면 넉넉하다.

    이 책의 그림 관례로는 SVG를 쓰는데, 애니메이션은 예외다. 프레임이 많으므로 GIF나
    연속된 PNG가 맞다. 정적인 그림 두세 장(초기, 중간, 끝)을 골라 SVG로 싣는 것이
    책에는 더 낫다.

## 정리하며

**다룬 것** — 단순한 2차원 맞겨루기 만들개

SimpleGenerator과 SimpleDiscriminator은 2차원 자료를 위해 만든 작은 여러 층 신경망이다.

고갱이 갈래는 `SimpleGenerator`, `SimpleDiscriminator`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
