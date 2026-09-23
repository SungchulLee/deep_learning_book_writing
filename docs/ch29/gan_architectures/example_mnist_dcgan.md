# MNIST에서 DCGAN 익히기

이 단원은 MNIST 손글씨 숫자 자료 묶음에서 DCGAN을 익히는 온전한 물길을 보인다. 자료 불러오기와 미리 다듬기부터 알맞은 맞겨루기 생성기 재주(얼개에 은근히 담긴 이름표 부드럽게 하기, 특정 베타 값을 쓴 Adam)로 익히기를 거쳐 표본 만들기와 되짚을 자리 관리까지 온 흐름을 보인다. 이는 그림 자료 묶음에서 겹말기 맞겨루기 생성기를 익히는 쓸모 있는 본이 된다.

## 1. 코드

```python
"""
MNIST에서 DCGAN 익히기

MNIST 자료 묶음에서 DCGAN을 익히는 온전한 대본.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

from dcgan import DCGANGenerator, DCGANDiscriminator
from gan_utils import (
    weights_init, save_samples, plot_training_progress,
    plot_discriminator_outputs, interpolate_latent, save_checkpoint
)


class DCGAN_MNIST:
    """MNIST에서 DCGAN을 익히는 감싸개 갈래."""
    
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64,
                 batch_size: int = 128, lr: float = 0.0002, beta1: float = 0.5,
                 device: str = None):
        """
        MNIST용 DCGAN을 첫자리매김한다.
        
        인수:
            latent_dim: 숨은 벡터의 차원
            feature_maps: 특징 지도의 바탕 수
            batch_size: 익히기 배치 크기
            lr: 학습률
            beta1: Adam 가장 좋게 하개의 beta1
            device: 학습에 쓸 장치
        """
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 신경망을 첫자리매김한다
        self.generator = DCGANGenerator(
            latent_dim=latent_dim,
            image_channels=1,
            feature_maps=feature_maps
        ).to(self.device)
        
        self.discriminator = DCGANDiscriminator(
            image_channels=1,
            feature_maps=feature_maps
        ).to(self.device)
        
        # 가중치 초기화
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # 가장 좋게 하개(DCGAN 논문을 따른다: lr=0.0002, beta1=0.5)
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr, betas=(beta1, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(beta1, 0.999)
        )
        
        # 손실 함수
        self.criterion = nn.BCELoss()
        
        # 한결같은 그림을 위한 붙박이 잡음
        self.fixed_noise = torch.randn(64, latent_dim, device=self.device)
        
        # 매개변수 개수 세기
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")
    
    def get_dataloader(self):
        """MNIST 자료 불러오개를 만든다."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]로 정규화
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    def train_discriminator(self, real_images: torch.Tensor) -> float:
        """
        판별기를 한 걸음 익힌다.
        
        인수:
            real_images: 실제 그림 배치
        
        반환값:
            판별기 손실
        """
        self.discriminator.zero_grad()
        
        batch_size = real_images.size(0)
        
        # 이름표
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # 실제 그림으로 익힌다
        d_real = self.discriminator(real_images)
        real_loss = self.criterion(d_real, real_labels)
        
        # 가짜 그림으로 익힌다
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        d_fake = self.discriminator(fake_images.detach())
        fake_loss = self.criterion(d_fake, fake_labels)
        
        # 결합된 손실
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self) -> float:
        """
        생성기를 한 걸음 익힌다.
        
        반환값:
            생성기 손실
        """
        self.generator.zero_grad()
        
        # 가짜 그림을 만든다
        noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        
        # 판별기를 속여 본다
        d_fake = self.discriminator(fake_images)
        real_labels = torch.ones(self.batch_size, 1, device=self.device)
        
        g_loss = self.criterion(d_fake, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train(self, n_epochs: int = 50, save_interval: int = 5,
             d_steps: int = 1):
        """
        DCGAN을 익힌다.
        
        인수:
            n_epochs: 익히기 바퀴 수
            save_interval: N바퀴마다 표본을 갈무리한다
            d_steps: 생성기 한 걸음마다 판별기 걸음 수
        """
        dataloader = self.get_dataloader()
        
        os.makedirs('samples', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        g_losses = []
        d_losses = []
        
        print(f"\nTraining DCGAN for {n_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc="Training")
            
            for i, (real_images, _) in enumerate(pbar):
                real_images = real_images.to(self.device)
                
                # 판별기를 익힌다
                for _ in range(d_steps):
                    d_loss = self.train_discriminator(real_images)
                
                # 생성기를 익힌다
                g_loss = self.train_generator()
                
                # 손실 기록
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1
                
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                
                pbar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}'
                })
            
            # 에포크 통계
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            
            print(f"Average G Loss: {avg_g_loss:.4f}")
            print(f"Average D Loss: {avg_d_loss:.4f}")
            
            # 표본을 갈무리한다
            if epoch % save_interval == 0 or epoch == 1:
                save_samples(
                    self.generator, epoch, self.device,
                    self.fixed_noise,
                    filename=f'samples/epoch_{epoch:04d}.png'
                )
            
            # 검사점 저장
            if epoch % 25 == 0:
                save_checkpoint(
                    self.generator, self.discriminator,
                    self.g_optimizer, self.d_optimizer,
                    epoch,
                    filename=f'checkpoints/dcgan_epoch_{epoch}.pth'
                )
        
        # 마지막 내놓기
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        
        # 마지막 표본을 갈무리한다
        save_samples(
            self.generator, n_epochs, self.device,
            self.fixed_noise,
            filename='final_samples.png'
        )
        
        # 익히기 나아감을 그린다
        plot_training_progress(g_losses, d_losses)
        
        # 마지막 되짚을 자리를 갈무리한다
        save_checkpoint(
            self.generator, self.discriminator,
            self.g_optimizer, self.d_optimizer,
            n_epochs,
            filename='dcgan_mnist_final.pth'
        )
        
        return g_losses, d_losses
    
    def generate_samples(self, n_samples: int = 64):
        """익힌 생성기에서 표본을 만든다."""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(noise)
            
        return samples
    
    def visualize_discriminator(self, real_data: torch.Tensor):
        """판별기 내놓기를 그려 본다."""
        noise = torch.randn(real_data.size(0), self.latent_dim, device=self.device)
        plot_discriminator_outputs(
            self.discriminator, real_data,
            self.generator, noise,
            self.device
        )
    
    def generate_interpolation(self):
        """아무 숨은 벡터 사이를 메워 만든다."""
        z1 = torch.randn(1, self.latent_dim, device=self.device)
        z2 = torch.randn(1, self.latent_dim, device=self.device)
        
        interpolate_latent(
            self.generator, z1, z2,
            steps=10, device=self.device
        )


def main():
    """으뜸 익히기 각본."""
    print("=" * 60)
    print("DCGAN Training on MNIST")
    print("=" * 60)
    
    # 설정
    config = {
        'latent_dim': 100,
        'feature_maps': 64,
        'batch_size': 128,
        'lr': 0.0002,
        'beta1': 0.5,
        'n_epochs': 50,
        'save_interval': 5,
        'd_steps': 1,  # G을 한 번 고칠 때마다의 D 고침 횟수
    }
    
    print("\nConfiguration:")
    print("-" * 60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("-" * 60)
    
    # DCGAN을 첫자리매김한다
    dcgan = DCGAN_MNIST(
        latent_dim=config['latent_dim'],
        feature_maps=config['feature_maps'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        beta1=config['beta1']
    )
    
    # 학습
    g_losses, d_losses = dcgan.train(
        n_epochs=config['n_epochs'],
        save_interval=config['save_interval'],
        d_steps=config['d_steps']
    )
    
    # 사이 메우기를 만든다
    print("\nGenerating interpolation...")
    dcgan.generate_interpolation()
    
    print("\n" + "=" * 60)
    print("All done! Check the following:")
    print("  - samples/ : Generated images during training")
    print("  - final_samples.png : Final generated samples")
    print("  - training_progress.png : Loss curves")
    print("  - interpolation.png : Latent space interpolation")
    print("  - checkpoints/ : Model checkpoints")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 2. 논의

DCGAN_MNIST 갈래는 온 익히기 흐름을 감싼다. DCGAN이 권하는 무게 첫자리매김으로 생성기와 판별기를 첫자리매김하고, 논문의 웃매개변수를 따라 $\beta_1 = 0.5$과 배움 빠르기 $2 \times 10^{-4}$인 Adam 가장 좋게 하개를 세우며, 두값 어긋 엔트로피 손실을 쓴다. 익히기의 나아감을 한결같이 그려 보려 붙박이 잡음 텐서를 지킨다.

익히기 절차는 판별기 고치기와 생성기 고치기를 번갈아 한다. 판별기는 실제라고 이름표 붙은 실제 MNIST 그림($[-1, 1]$으로 고르게 맞춘 것)과 가짜라고 이름표 붙은 만든 그림을 본다. 이어 판별기를 속이려 하며 생성기를 고친다. `d_steps` 매개변수는 생성기를 한 번 고칠 때마다 판별기를 몇 번 고칠지 다스린다. 붙박이 잡음 벡터로 이따금 표본을 만들어 익히기의 나아감을 지켜보며 바퀴마다 눈으로 견줄 수 있다.

이 물길에는 쓸모 있는 기능이 여럿 있다. 곧 오래 익힐 때를 위한 되짚을 자리 갈무리와 불러오기, 배운 나타냄을 살피기 위한 숨은 공간 사이 메우기, 익히기의 건강을 살피기 위한 판별기 내놓기 그려 보기이다. 사이 메우기 기능은 아무 숨은 벡터 둘 사이의 곧은 길을 따라 그림을 만들어 배운 숨은 공간의 매끄러움을 드러낸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
MNIST에서 DCGAN을 50바퀴 익히고 익히기 손실 곡선을 살펴라. 판별기 손실이 $\log 4 \approx 1.386$에 가까워지면 무슨 뜻인가?

</div>

??? success "연습문제 1 풀이"
    판별기 손실이 $\log 4$에 가까워지면 판별기가 실제 그림과 가짜 그림 모두에 확률 0.5을 매긴다는 뜻이며, 곧 둘을 가려내지 못한다는 것이다. 이는 최소최대 놀이의 내시 균형에 맞물린다. 판별기의 실제 손실은 $-\log(0.5)$, 가짜 손실은 $-\log(1 - 0.5)$이어서 온 손실은 $-2\log(0.5) = 2\log 2 = \log 4 \approx 1.386$이다. 이를 흔히 가장 좋은 익히기 움직임으로 여기지만 실제로는 정확히 모이기보다 이 값 둘레에서 흔들릴 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
생성기 무게의 지수 이동 평균(EMA)을 짜고 익힌 뒤 EMA 모델과 여느 모델의 표본을 견주어라. EMA이 표본 품질을 높이는가?

</div>

??? success "연습문제 2 풀이"
    ```python
    ema_decay = 0.999
    ema_generator = copy.deepcopy(generator)
    for ema_p, p in zip(ema_generator.parameters(), generator.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
    ```
    EMA은 매개변수의 자취를 부드럽게 하여 익히기 흔들림의 영향을 줄인다. EMA 모델은 잡음이 낄 수 있는 마지막 되풀이 대신 최근 매개변수 값을 평균 내므로 흔히 더 또렷하고 한결같은 표본을 낸다. 익히기가 불안정하거나 생성기와 판별기 손실이 크게 흔들릴 때 개선이 가장 두드러진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
숫자 이름표를 넣어 DCGAN에 갈래 조건 만들어 내기를 더하라. 숫자 갈래를 조건으로 삼도록 생성기(원핫 이름표를 숨은 벡터에 잇기)와 판별기(원핫 이름표를 그림 특징에 잇기)를 모두 고쳐라.

</div>

??? success "연습문제 3 풀이"
    생성기에서는 10차원 원핫 이름표 벡터를 숨은 벡터에 이어 들임을 100차원에서 110차원으로 넓힌다. 판별기에서는 그림 차원에 맞게 공간으로 되풀이하고 채널 축을 따라 이은 이름표 박아 넣기를 만들어 들임 채널을 1에서 11로 바꾼다. 익히는 동안 실제 그림에는 실제 이름표를, 가짜 그림에는 바라는 이름표를 준다. 만들 때는 바라는 숫자 갈래를 정한다. 그러면 바라는 숫자를 그때그때 만들 수 있고, 생성기에 배울 짜임을 더 주므로 흔히 전체 품질도 나아진다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 예의 결과를 어떤 잣대로 보고하겠는가?

</div>

??? success "연습문제 4 풀이"
    이 장에서 마련한 것들을 모두 쓴다. MNIST 분류기(시험 정확도 98.50%) 기준,
    표본 10,000개다.

    | | 인셉션 점수 | FID | 부류 엔트로피 |
    |---|---|---|---|
    | 참 시험 자료 | 9.467 | 2.52 | — |
    | 이 예 (표지 매끄럽게) | 6.964 | 40.7 | 2.206 |
    | 고른 열 부류 | — | — | 2.303 |

    기준점을 함께 적는 것이 중요하다. 참 자료의 FID 2.52가 이 설정의 바닥 근처이므로
    40.7이 어디쯤인지 읽을 수 있다.

    그리고 밝혀야 할 것들이 있다. 표본 수, 특징 그물과 그 층, 뽑기 씨앗, 인셉션 점수의
    조각 수와 섞었는지, 온도를 쓰지 않았다는 것
    ([값매김 연습문제 14](../gan_evaluation/complete_evaluation_example.md)).

    잣대와 함께 **고르지 않은 표본 격자**를 싣는다. 수가 못 보는 것이 있기 때문이다.

    여기에 정밀도와 재현율을 더하면 FID가 큰 까닭이 품질인지 다양성인지 갈라 볼 수
    있다([정밀도와 재현율](../gan_evaluation/precision_recall.md)).

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
학습이 안 될 때 확인할 순서를 정하라.

</div>

??? success "연습문제 5 풀이"
    값싼 것부터 본다.

    1. **자료의 범위.** 생성기 출력과 자료가 같은 범위인가. `tanh`면 자료도 $[-1,1]$
    2. **`detach()`가 제자리에 있는가.** 판별기 갱신의 거짓 그림에 있어야 한다
    3. **`zero_grad()`를 빠뜨리지 않았는가.** 두 최적화기 모두
    4. **$z$가 쓰이는가.** 다른 $z$에 다른 출력이 나오는지 확인한다
    5. **배치 정규화와 드롭아웃이 있는가.** 없으면 무너지기 쉽다 (FID 1792.8 대 49.85)
    6. **표지를 매끄럽게 해 본다.** 이 장에서 가장 잘 들었다 (40.68)
    7. **표본 격자를 에포크별로 본다.** 언제 틀어졌는지 보인다

    1번이 가장 흔하고 가장 조용하다. 범위가 어긋나면 판별기가 그것만으로 가려내므로
    생성기가 배울 것이 없는데, 오류가 나지 않는다.

    **손실을 보는 것이 목록에 없다**는 점을 거듭 짚어 둔다. 적대적 생성망의 손실은 진단에
    쓸모가 없다([GAN 기초 연습문제 2](../gan_architectures/45_gan.md)).

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
이 예에서 에포크를 두 배로 늘리면 나아지는가?

</div>

??? success "연습문제 6 풀이"
    **여기서는 크게 나아졌다.** 재어 보면 이렇다.

    | 에포크 | 30 | 60 |
    |---|---|---|
    | 인셉션 점수 | 6.964 | **8.026** |
    | FID | 40.68 | **10.25** |
    | 부류 엔트로피 | 2.206 | 2.271 |

    FID가 네 배 나아졌다. 곧 30 에포크의 모델은 아직 덜 익은 것이었다.

    이 수치를 먼저 내놓는 까닭이 있다. "적대적 생성망은 오래 익혀도 좋아지지 않는다"는
    말이 흔히 오가는데, 적어도 이 설정에서는 그렇지 않았다. **재어 보지 않고 말할
    일이 아니다.**

    그렇다고 무한히 좋아진다는 뜻은 아니다. 지도 학습과 다른 점이 남아 있다. 최소화할
    하나의 값이 없으므로 "수렴"이라는 말이 같은 뜻을 갖지 않는다. 두 그물이 균형 근처에서
    오르내릴 수 있고, 어느 시점에 무너질 수도 있다.

    그래서 실무의 방식이 이렇다.

    - **FID를 에포크마다 재어 기록한다**
    - 가장 좋은 시점의 가중치를 저장한다
    - 마지막 시점이 가장 좋다고 가정하지 않는다

    둘째가 중요하다. 지도 학습에서는 검증 손실로 이른 멈춤을 하는데, 여기서는 FID가
    그 노릇을 한다. 다만 FID로 고르면 FID를 겨냥하게 되므로, 마지막 보고는 다른 자료로
    다시 재는 것이 옳다
    ([값매김 연습문제 12](../gan_evaluation/complete_evaluation_example.md)).

    그리고 표본 수를 고정해 두어야 곡선을 읽을 수 있다. 2,000개로 재면 바닥이 3 근처라
    그만큼의 요동은 늘 있다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
표본이 그럴듯한데 FID가 높다면 무엇을 의심하겠는가?

</div>

??? success "연습문제 7 풀이"
    몇 가지가 있고, 순서대로 보면 된다.

    **범위나 전처리의 어긋남.** 특징 그물에 넣는 값이 익힐 때와 다른 범위면 값이 크게
    나빠진다. 참 자료로 기준점을 재어 보면 곧 드러난다. 참 자료의 FID가 2.52가 아니라
    크게 나오면 코드 문제다.

    **표본 수가 적다.** 100개로 재면 참 자료끼리도 81.4가 나온다
    ([FID 연습문제 12](../gan_evaluation/fid.md)).

    **다양성 부족.** 격자에 보이는 몇 장은 그럴듯한데 전체가 몇 가지뿐일 수 있다. 눈으로는
    잘 안 보인다. 부류 엔트로피와 재현율을 보면 갈린다.

    셋째가 가장 실질적이다. 서로 다른 그림 열 장을 되풀이한 배치가 인셉션 점수 9.164에
    FID 356.5였다([인셉션 점수 연습문제 2](../gan_evaluation/inception_score.md)). 격자로
    보면 완벽해 보이는데 FID가 이렇게 된다.

    곧 **FID가 높고 격자가 좋다는 조합은 다양성 문제의 전형적인 신호**다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
이 예를 조건부로 바꾸려면 무엇을 고치는가?

</div>

??? success "연습문제 8 풀이"
    부류 표지를 두 그물 모두에 넣는다. [26장의 조건부 변분 오토인코더](../../ch26/architecture/conditional_vae.md)와
    같은 생각이다.

    ```python
    # 생성기: z 에 표지를 이어 붙인다
    zc = torch.cat([z, F.one_hot(y, 10).float()], dim=1)
    fake = g(zc)

    # 판별기: 그림에 표지를 이어 붙인다
    out = d(torch.cat([x.flatten(1), F.one_hot(y, 10).float()], dim=1))
    ```

    판별기에 표지를 넣는 방식에 갈래가 있다.

    | 방식 | 어떻게 |
    |---|---|
    | 이어 붙이기 | 입력에 표지를 붙인다. 간단하다 |
    | 보조 분류기 (AC-GAN) | 판별기가 부류도 맞히게 한다 |
    | 사영 판별기 | 표지 묻기와 특징의 내적을 더한다 |

    셋째가 요즘 표준에 가깝고 잘 듣는다고 알려져 있다.

    값매김도 달라진다. "지정한 부류대로 나오는 비율"을 잴 수 있게 된다. 조건부 변분
    오토인코더에서 90.1%였던 그 잣대다. 조건부 적대적 생성망에서 같은 것을 재면 두
    모델을 그 축에서 견줄 수 있다.

    그리고 부류 쏠림 문제가 줄어든다. 부류를 우리가 지정하므로 생성기가 한 부류로 무너지는
    일이 어려워진다. 다만 **부류 안에서** 무너지는 것은 여전히 가능하다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
이 예의 결과를 26장의 변분 오토인코더와 나란히 놓을 때 조심할 점은 무엇인가?

</div>

??? success "연습문제 9 풀이"
    **같은 조건으로 재었는지**를 먼저 확인해야 한다.

    이 장의 두 수치는 맞추어 두었다. 같은 특징 그물, 같은 표본 수(10,000), 같은 참 자료
    기준(학습 자료 앞 10,000개), 같은 씨앗.

    맞추지 않으면 견줄 수 없는 것들이 이렇다.

    | 어긋나면 | 어떻게 되는가 |
    |---|---|
    | 표본 수 | FID가 적은 쪽에 불리하다 |
    | 특징 그물 | 아예 다른 잣대다 |
    | 익히기 예산 | 어느 쪽이 덜 익었는지 모른다 |
    | 모델 크기 | 공정한 견줌이 아니다 |

    셋째와 넷째는 이 장에서 완전히 맞추지 못했다. 변분 오토인코더는 20 에포크, 적대적
    생성망은 30 에포크로 익혔고 얼개도 다르다. 그러므로 **40.7 대 148.2라는 큰 차이는
    믿을 만하지만, 그 차이의 정확한 크기를 인용할 값은 아니다.**

    그리고 두 모델이 **다른 것을 잘하도록 익힌 것**임을 잊지 말아야 한다. 표본 잣대로
    재면 적대적 생성망이 이기고, 가능도로 재면 애초에 적대적 생성망을 못 재며, 되돌리기로
    재면 변분 오토인코더가 이긴다. 잣대를 고르는 것이 곧 답을 고르는 일이 될 수 있다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 예의 결과를 재현하려면 무엇을 적어 두어야 하는가?

</div>

??? success "연습문제 10 풀이"
    모델과 값매김 양쪽을 적어야 한다.

    ```python
    config = dict(
        # 모델
        latent_dim=64, g_hidden=(256,512,1024), d_hidden=(512,256),
        batchnorm_g=True, dropout_d=0.3,
        # 익히기
        epochs=30, batch_size=128, lr_g=2e-4, lr_d=2e-4, betas=(0.5,0.999),
        label_smoothing=0.1, seed=42,
        # 값매김
        feature_net='mnist_cnn_98.50', feature_layer='fc1', feature_dim=128,
        n_samples=10000, real_ref='train[:10000]', sample_seed=0,
        is_splits=10, is_shuffled=True, temperature=1.0,
    )
    ```

    적대적 생성망은 씨앗에 특히 민감하므로 `seed`가 중요하다. 그리고 씨앗을 **모델을
    만들기 직전에** 고정해야 한다. 순서가 어긋나면 같은 씨앗으로도 다른 모델이 나온다.

    `label_smoothing`을 빠뜨리면 안 된다. 이 장의 측정에서 FID 49.85와 40.68을 가른
    설정이다.

    값매김 쪽은 [값매김 연습문제 14](../gan_evaluation/complete_evaluation_example.md)의
    목록과 같다.

## 정리하며

**다룬 것** — MNIST에서 DCGAN 익히기

DCGAN_MNIST 갈래는 온 익히기 흐름을 감싼다.

고갱이 갈래는 `DCGAN_MNIST`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
