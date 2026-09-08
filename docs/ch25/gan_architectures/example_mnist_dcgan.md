# MNIST에서 DCGAN 익히기

이 단원은 MNIST 손글씨 숫자 자료 묶음에서 DCGAN을 익히는 온전한 물길을 보인다. 자료 불러오기와 미리 다듬기부터 알맞은 맞겨루기 만들개 재주(얼개에 은근히 담긴 이름표 부드럽게 하기, 특정 베타 값을 쓴 Adam)로 익히기를 거쳐 표본 만들기와 되짚을 자리 관리까지 온 흐름을 보인다. 이는 그림 자료 묶음에서 겹말기 맞겨루기 만들개를 익히는 쓸모 있는 본이 된다.

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
            batch_size: 익히기 묶음 크기
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
        가름개를 한 걸음 익힌다.
        
        인수:
            real_images: 실제 그림 묶음
        
        반환값:
            가름개 손실
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
        만들개를 한 걸음 익힌다.
        
        반환값:
            만들개 손실
        """
        self.generator.zero_grad()
        
        # 가짜 그림을 만든다
        noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        
        # 가름개를 속여 본다
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
            d_steps: 만들개 한 걸음마다 가름개 걸음 수
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
                
                # 가름개를 익힌다
                for _ in range(d_steps):
                    d_loss = self.train_discriminator(real_images)
                
                # 만들개를 익힌다
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
        """익힌 만들개에서 표본을 만든다."""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(noise)
            
        return samples
    
    def visualize_discriminator(self, real_data: torch.Tensor):
        """가름개 내놓기를 그려 본다."""
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
    main()```

## 2. 논의

DCGAN_MNIST 갈래는 온 익히기 흐름을 감싼다. DCGAN이 권하는 무게 첫자리매김으로 만들개와 가름개를 첫자리매김하고, 논문의 웃매개변수를 따라 $\beta_1 = 0.5$과 배움 빠르기 $2 \times 10^{-4}$인 Adam 가장 좋게 하개를 세우며, 두값 어긋 엔트로피 손실을 쓴다. 익히기의 나아감을 한결같이 그려 보려 붙박이 잡음 텐서를 지킨다.

익히기 절차는 가름개 고치기와 만들개 고치기를 번갈아 한다. 가름개는 실제라고 이름표 붙은 실제 MNIST 그림($[-1, 1]$으로 고르게 맞춘 것)과 가짜라고 이름표 붙은 만든 그림을 본다. 이어 가름개를 속이려 하며 만들개를 고친다. `d_steps` 매개변수는 만들개를 한 번 고칠 때마다 가름개를 몇 번 고칠지 다스린다. 붙박이 잡음 벡터로 이따금 표본을 만들어 익히기의 나아감을 지켜보며 바퀴마다 눈으로 견줄 수 있다.

이 물길에는 쓸모 있는 기능이 여럿 있다. 곧 오래 익힐 때를 위한 되짚을 자리 갈무리와 불러오기, 배운 나타냄을 살피기 위한 숨은 공간 사이 메우기, 익히기의 건강을 살피기 위한 가름개 내놓기 그려 보기이다. 사이 메우기 기능은 아무 숨은 벡터 둘 사이의 곧은 길을 따라 그림을 만들어 배운 숨은 공간의 매끄러움을 드러낸다.

## 연습문제

**연습문제 1.**
MNIST에서 DCGAN을 50바퀴 익히고 익히기 손실 곡선을 살펴라. 가름개 손실이 $\log 4 \approx 1.386$에 가까워지면 무슨 뜻인가?

??? success "연습문제 1 풀이"
    가름개 손실이 $\log 4$에 가까워지면 가름개가 실제 그림과 가짜 그림 모두에 확률 0.5을 매긴다는 뜻이며, 곧 둘을 가려내지 못한다는 것이다. 이는 최소최대 놀이의 내시 균형에 맞물린다. 가름개의 실제 손실은 $-\log(0.5)$, 가짜 손실은 $-\log(1 - 0.5)$이어서 온 손실은 $-2\log(0.5) = 2\log 2 = \log 4 \approx 1.386$이다. 이를 흔히 가장 좋은 익히기 움직임으로 여기지만 실제로는 정확히 모이기보다 이 값 둘레에서 흔들릴 수 있다.

---

**연습문제 2.**
만들개 무게의 지수 이동 평균(EMA)을 짜고 익힌 뒤 EMA 모델과 여느 모델의 표본을 견주어라. EMA이 표본 품질을 높이는가?

??? success "연습문제 2 풀이"
    ```python
    ema_decay = 0.999
    ema_generator = copy.deepcopy(generator)
    for ema_p, p in zip(ema_generator.parameters(), generator.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
    ```
    EMA은 매개변수의 자취를 부드럽게 하여 익히기 흔들림의 영향을 줄인다. EMA 모델은 잡음이 낄 수 있는 마지막 되풀이 대신 최근 매개변수 값을 평균 내므로 흔히 더 또렷하고 한결같은 표본을 낸다. 익히기가 불안정하거나 만들개와 가름개 손실이 크게 흔들릴 때 개선이 가장 두드러진다.

---

**연습문제 3.**
숫자 이름표를 넣어 DCGAN에 갈래 조건 만들어 내기를 더하라. 숫자 갈래를 조건으로 삼도록 만들개(원핫 이름표를 숨은 벡터에 잇기)와 가름개(원핫 이름표를 그림 특징에 잇기)를 모두 고쳐라.

??? success "연습문제 3 풀이"
    만들개에서는 10차원 원핫 이름표 벡터를 숨은 벡터에 이어 들임을 100차원에서 110차원으로 넓힌다. 가름개에서는 그림 차원에 맞게 공간으로 되풀이하고 채널 축을 따라 이은 이름표 박아 넣기를 만들어 들임 채널을 1에서 11로 바꾼다. 익히는 동안 실제 그림에는 실제 이름표를, 가짜 그림에는 바라는 이름표를 준다. 만들 때는 바라는 숫자 갈래를 정한다. 그러면 바라는 숫자를 그때그때 만들 수 있고, 만들개에 배울 짜임을 더 주므로 흔히 전체 품질도 나아진다.

## 정리하며

**다룬 것** — MNIST에서 DCGAN 익히기

DCGAN_MNIST 갈래는 온 익히기 흐름을 감싼다.

고갱이 갈래는 `DCGAN_MNIST`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
