# 2차원 데이터로 살펴보는 GAN 학습

2차원 예제는 실제·생성 표본과 판별기 점수를 같은 좌표에 그려 학습을 이해하는 데 유용하다. 산점도에서 모양·스프레드·누락을 살펴볼 수 있지만 유한한 그림만으로 두 분포의 일치나 내시 수렴을 증명할 수는 없다. 이 절은 작은 MLP GAN의 코드 동작과 평가 한계를 설명한다.

## 1. 구현

```python
"""
단순한 2차원 맞겨루기 생성기

2차원 장난감 자료에서 맞겨루기 생성기 익히기를 그려 본다.
맞겨루기 익히기의 액션을 이해하기에 알맞다.
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
    """2차원 자료를 위한 단순한 생성기."""
    
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
    """2차원 자료를 위한 단순한 판별기."""
    
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
    """생성기 분포와 판별기 결정 가장자리를 그려 본다."""
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
    
    # 그림 3: 판별기 결정 가장자리
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
    """2차원 자료에서 단순한 맞겨루기 생성기를 익힌다."""
    
    # 뉴럴 넷트웍을 첫자리매김한다
    generator = SimpleGenerator(latent_dim=latent_dim).to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # 가장 좋게 하개
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 로스
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
        
        # 판별기를 익힌다
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
        
        # 생성기를 익힌다
        generator.zero_grad()
        
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = generator(noise)
        d_fake = discriminator(fake_batch)
        g_loss = criterion(d_fake, real_labels)
        
        g_loss.backward()
        g_optimizer.step()
        
        # 로스 기록
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
    """익히기 로스 곡선을 그린다."""
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
    """2차원 맞겨루기 생성기 보여 주기의 으뜸 함수."""
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
    
    # 맞겨루기 생성기를 익힌다
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    generator, discriminator, g_losses, d_losses = train_gan_2d(
        data, latent_dim=latent_dim, n_epochs=n_epochs, device=device
    )
    
    # 로스 곡선을 그린다
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

기본 G는 2→128→128→2와 ReLU,D는 2→128→128→1과 LeakyReLU·Sigmoid다. 학습 파라미터는 각각 17,154개와 17,025개다. 생성 출력은 Tanh 같은 범위 제한이 없고 실제 자료와 같은 2차원 좌표 단위를 사용한다. 입력·출력 차원이 같다고 G가 가역적이거나 밀도를 쉽게 계산할 수 있는 것은 아니다.

**기존 출력 기록:**

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

기존 기록은 보존했다. 이 편집에서는 학습·시각화를 실행하거나 새 품질 수치를 측정하지 않았다. 아래에서 설명하는 “epoch”의 의미와 장치 계약도 함께 읽어야 한다.

## 2. 데이터 생성과 반복 횟수

generate_data는 잡음 있는 반달·동심원 또는 평균 (2,2), (−2,−2)의 가우시안 두 성분을 만든다. 가우시안 분기의 각 성분 수는 n_samples//2이므로 홀수 요청이면 총 n_samples−1개다. 반달·원에도 0.05 잡음이 있어 이상적인 1차원 곡선에만 놓인 자료와 다르다. NumPy·PyTorch 시드를 고정하지 않아 초기화·자료·잡음에 변동이 있다.

main은 반달 표본 2,000개,잠재 2차원과 n_epochs=1,000을 사용한다. 그러나 train_gan_2d의 바깥 반복마다 실제 자료에서 복원추출한 배치 **하나**로 D 한 번·G 한 번을 갱신한다. 전체 자료를 한 번 순회하는 통상적 에포크가 아니라 1,000개 학습 step이다. 같은 “1,000에포크”라는 이름으로 다른 로더 실험과 예산을 비교하지 않는다.

D는 실제·생성 평균 BCE의 합,G는 실제 레이블 1의 비포화 BCE를 최소화한다. D 단계의 생성 표본은 detach하고 G 단계는 새 잡음을 사용한다. G 단계의 D grad는 계산되지만 D optimizer를 step하지 않으며 다음 D 초기화에서 지운다. 두 optimizer는 Adam 0.0002,betas=(0.5,0.999)를 쓴다.

## 3. 그림의 장치·모드와 의미

고정 잠재 500개를 첫 step 뒤와 100step 간격·종료 시점에 표시한다. 실제 표본,생성 표본과 200×200 격자의 D 점수를 그린다. 세 번째 패널은 생성 밀도가 아니라 **판별기 분류 점수**다. 높은 D 값이 곧 높은 데이터 밀도나 좋은 생성 품질을 뜻하지 않는다.

격자 points는 CPU float32로 만들어 D에 직접 전달한다. 학습 모델이 CUDA이면 첫 시각화에서 장치가 맞지 않아 실패한다. 실제 모델과 격자·잠재 표본의 장치·자료형을 맞춰야 하며 현재 코드를 CUDA에서도 완전 실행 가능하다고 설명할 수 없다.

시각화는 두 모델을 eval·no_grad로 평가한 뒤 무조건 train으로 바꾸므로 원래 평가 모드를 그대로 보존하는 일반 도구는 아니다. 그림의 −4~4 범위 밖 질량도 보이지 않을 수 있다. FuncAnimation은 import하지만 실제 애니메이션 생성은 구현하지 않는다.

## 4. 분포 비교와 모드 진단

모드가 알려진 자료에서는 영역별 생성 비중·누락과 실제 비중을 비교할 수 있다. 쌍별 거리·최근접 거리·표본 검정도 도움되지만 단일 지표는 분포의 모든 차이를 식별하지 못한다. 출력이 넓게 흩어지거나 노이즈가 커져도 쌍별 거리는 증가한다.

D 점수의 경계가 날카롭거나 평균이 0.5라는 사실만으로 G의 다음 이동·포화·균형을 확정할 수 없다. G 업데이트는 D의 입력 그래디언트와 G 파라미터의 야코비안·배치 목적에 달려 있다. 로스·그래디언트·표본 분포와 여러 초기화의 경로를 함께 기록해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
세 데이터셋의 난이도를 비교하는 실험에서 무엇을 통제해야 하는가?

</div>

??? success "풀이"

    같은 표본 수·좌표 스케일·구조·평가·업데이트 예산과 여러 초기화를 사용한다. 현재 n_epochs는 전체 자료 순회가 아닌 배치 step 수다. 가우시안 두 성분,반달,동심원의 모양만으로 보편적인 난이도 순서를 정할 수 없다. 모드 누락·생성 비중·스프레드와 비용을 함께 비교하며 판별기 경계가 반드시 특정 곡선 모양이어야 한다고 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
D를 k번 갱신하는 비교는 어떤 예산 차이를 만드는가?

</div>

??? success "풀이"

    G step 수를 고정하면 k가 클수록 D 계산·실제 표본 추출 수가 늘어난다. 총 시간 또는 전체 업데이트 수를 맞춘 비교와는 다른 질문이다. 같은 배치를 재사용할지 새로 뽑을지도 정한다. k=3이면 늘 개선되고 k=5이면 반드시 포화한다는 예측은 할 수 없다. 비포화 G의 그래디언트·판별 진단·품질·다양성을 공통 조건에서 평가한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 평균 쌍별 거리의 계산비용과 진단 한계를 설명하라.

</div>

??? success "풀이"

    n개 2차원 표본의 n×n 거리를 만들므로 시간·메모리는 n²에 비례한다. n≥2가 필요하고 단위·표본 수·모드 상태를 맞춰야 한다. 값 감소는 집중의 신호일 수 있지만 참 분포에 더 잘 맞아 분산이 줄어드는 경우도 있고,잘못된 넓은 노이즈는 값을 키울 수 있다. 같은 거리 평균이 분포 일치나 모드 포괄도를 뜻하지 않는다. 함수는 eval을 자동 설정하지 않으므로 일반 생성기의 모드도 정해야 한다.

    ```python
    def mode_collapse_metric(generator, latent_dim, n_samples=500, device='cpu'):
        noise = torch.randn(n_samples, latent_dim, device=device)
        with torch.no_grad():
            fake = generator(noise).cpu().numpy()
        dists = np.sqrt(((fake[:, None] - fake[None, :]) ** 2).sum(-1))
        return np.mean(dists[np.triu_indices(n_samples, k=1)])
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
2차원 산점도의 이점과 한계를 설명하라.

</div>

??? success "풀이"

    좌표 전체를 직접 그려 위치·스프레드·모드 누락을 쉽게 볼 수 있고 판별기 점수장을 함께 표시할 수 있다. 그러나 표시 범위·표본 수·투명도·축 비율 때문에 밀도 차이를 놓칠 수 있다. 겹쳐 보이는 점군만으로 통계적으로 같은 분포라고 결론 내리지 않는다. 독립 표본과 정량 평가를 함께 사용한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
모드 누락·한 점 집중·모드 간 순환을 어떻게 구분하라?

</div>

??? success "풀이"

    여러 실제 영역의 생성 비중을 시간에 따라 기록한다. 일부 영역이 지속적으로 비면 누락,전체 출력이 거의 같은 점이면 강한 집중,여러 영역을 시간에 따라 번갈아 방문하면 순환 가능성이 있다. 단일 최종 산점도로 세 학습 경로를 완전히 구분할 수는 없다. 본문에는 이를 입증하는 모드별 시간 통계가 없어 특정 실행에서 실제 순환을 관찰했다고 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
판별기 색상 패널에서 직접 읽을 수 있는 것은 무엇인가?

</div>

??? success "풀이"

    선택한 격자와 현재 D가 평가 모드에서 내는 분류 점수다. 생성 밀도·데이터 밀도나 G의 정확한 다음 이동은 아니다. 실제 표본 주변 D가 낮다고 그곳이 반드시 생성 누락 모드인 것도 아니다. G 파라미터 업데이트와 점수 그래디언트를 함께 보아야 원인을 분석할 수 있다. D≈0.5 영역이 넓어져도 약한 D일 수 있어 균형의 충분조건이 아니다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
2차원 모형에 적합한 평가를 설계하라.

</div>

??? success "풀이"

    직접 좌표에서 표본 거리 기반 통계·영역별 비중·두 표본 비교를 정의할 수 있다. 실제 생성 과정의 모드·잡음 분포를 아는 경우 명확한 기준 영역을 정하되 겹치는 가우시안은 샘플별 모드 판정 자체에 불확실성이 있다. 검정에서 귀무가설을 기각하지 못했다고 두 분포의 동일성이 증명되는 것은 아니다. Inception 같은 이미지 피처는 직접 맞지 않지만 일반적인 피처 통계 비교 자체가 2차원에 불가능한 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
2차원 예제에서 얻은 이해를 MNIST에 어떻게 적용할 수 있는가?

</div>

??? success "풀이"

    목적·그래프 분리·상대 업데이트·모드 진단이라는 개념을 적용할 수 있지만 하이퍼파라미터·피처·평가 기준은 다시 정해야 한다. 차원만으로 분포 지지집합의 겹침이나 JS 그래디언트 크기가 자동 결정되지는 않는다. 2차원에 잘 맞는 설정이 이미지에서도 우세하다는 보장은 없고,원인 분석 역시 추가 근거가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
잠재 차원을 데이터 차원과 비교할 때 어떤 조건을 고려해야 하는가?

</div>

??? success "풀이"

    일반적인 매끄럽거나 국소 Lipschitz인 결정적 G에서 잠재 차원이 작으면 출력 분포가 더 낮은 차원에 놓여 가득 찬 연속 밀도를 정확히 표현하지 못할 수 있다. 그러나 유한 표본 근사·평가 기준과 정확한 분포 표현은 다르다. 잠재 차원이 같거나 크다고 표현·학습 성공이 보장되지 않으며 큰 차원이 항상 최적화에 유리하거나 남는 좌표를 완전히 무시하는 것도 아니다. 잡음 있는 반달·원은 정확한 1차원 곡선 자료가 아니라는 점도 고려한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
애니메이션 재료와 평가 표본을 어떻게 기록할 수 있는가?

</div>

??? success "풀이"

    같은 fixed_z의 생성 좌표,동일 격자의 D 점수,학습 step을 저장하면 같은 잠재 점의 액션을 비교할 수 있다. 아래 snapshots·g·fixed_z·decision_grid·ep는 외부에서 정의해야 한다. 이 골격이나 FuncAnimation import만으로 애니메이션이 저장되지는 않는다. 장치·모드·표시 범위·격자 해상도를 맞추고,고정 표본 추적 외에 새 표본의 분포 진단도 별도로 기록한다.

    ```python
    snapshots.append(dict(
        fake=g(fixed_z).detach().cpu().numpy(),      # 고정된 z 의 출력
        grid=decision_grid(d),                       # 판별기 경계
        epoch=ep,
    ))
    ```


## 정리하며

2차원 GAN은 학습 경로를 보기 쉽게 하지만 산점도·D 점수·쌍별 거리는 각각 제한된 진단이다. 현재 반복은 배치 step이며 CUDA 시각화에는 CPU 격자 불일치가 있다. 같은 좌표·평가·예산에서 품질·모드 비중·다양성을 함께 비교한다.
