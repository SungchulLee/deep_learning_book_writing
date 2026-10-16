# MNIST DCGAN 학습 파이프라인

이 예제는 데이터 로딩,교대 업데이트,고정 잠재 표본 표시와 체크포인트를 하나의 클래스에 묶는다. 다만 현재 import하는 로컬 모델과 스크립트의 해상도·출력·로스 계약이 서로 다르다. 코드를 실행해 얻은 완성 실험으로 설명하기보다 의도한 흐름과 실제 연결 상태를 구분해야 한다.

## 1. 학습 스크립트

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
    """MNIST에서 DCGAN을 익히는 감싸개 클래스."""
    
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64,
                 batch_size: int = 128, lr: float = 0.0002, beta1: float = 0.5,
                 device: str = None):
        """
        MNIST용 DCGAN을 첫자리매김한다.
        
        인수:
            latent_dim: 숨은 벡터의 차원
            feature_maps: 피처 지도의 바탕 수
            batch_size: 익히기 배치 크기
            lr: 러닝레이트
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
        
        # 뉴럴 넷트웍을 첫자리매김한다
        self.generator = DCGANGenerator(
            latent_dim=latent_dim,
            image_channels=1,
            feature_maps=feature_maps
        ).to(self.device)
        
        self.discriminator = DCGANDiscriminator(
            image_channels=1,
            feature_maps=feature_maps
        ).to(self.device)
        
        # 웨이트 초기화
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
        
        # 로스 함수
        self.criterion = nn.BCELoss()
        
        # 한결같은 그림을 위한 붙박이 잡음
        self.fixed_noise = torch.randn(64, latent_dim, device=self.device)
        
        # 파라메터 개수 세기
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
        판별기를 한 스텝 익힌다.
        
        인수:
            real_images: 실제 그림 배치
        
        반환값:
            판별기 로스
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
        
        # 결합된 로스
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self) -> float:
        """
        생성기를 한 스텝 익힌다.
        
        반환값:
            생성기 로스
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
            d_steps: 생성기 한 스텝마다 판별기 스텝 수
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
                
                # 로스 기록
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

main의 설정은 잠재 100,feature_maps=64,배치 128,Adam 러닝레이트 0.0002·betas=(0.5,0.999),50에포크와 D:G=1:1이다. MNIST 학습셋을 ToTensor와 Normalize(0.5,0.5)로 −1~1로 바꾸지만 Resize는 없다. 라벨은 버리는 무조건부 모델이며 라벨 스무딩·FID·IS 계산은 현재 학습 루프에 없다.

## 2. 실제 import 모델과의 불일치

이 디렉터리의 dcgan.py는 64×64 생성기와 판별기를 정의한다. 문서의 [28×28 모듈 예제](dcgan.md)와 같은 클래스 이름을 쓰지만 구조가 다르다. 실제 기본 단일 채널 로컬 G는 3,574,656개,D는 2,763,520개의 파라미터를 갖는다. G는 64×64를 만들고 D도 64×64 입력에서 마지막 4×4 커널을 적용하도록 되어 있다.

현재 28×28 실제 이미지는 D에서 28→14→7→3→1로 줄어 마지막 커널 4를 적용할 수 없다. 따라서 기본 train_discriminator의 실제 자료 경로가 먼저 실패한다. 로컬 파일 설명은 MNIST를 64로 키워 사용하라고 하지만 이 스크립트 전처리에는 그 처리가 없다.

또한 로컬 D의 출력은 Sigmoid 없는 **logits `(B,)`**다. 이 스크립트는 확률 BCELoss와 `(B,1)` 레이블을 사용해 해상도를 맞춰도 출력 모양·로스 계약이 맞지 않는다. 올바른 결합에는 목표 해상도와 logits용 로스·타깃 모양을 일관되게 정해야 한다. 이 편집에서는 모델 파일·표시된 코드를 보존하고 이 한계를 명시한다.

## 3. 의도한 교대 업데이트의 동작

계약을 맞춘다는 전제에서 D 단계는 같은 실제 배치와 새 가짜 배치를 분류하고 G 출력은 detach한다. G 단계는 새 잡음의 출력에 실제 레이블을 주는 비포화 목적이다. G 단계에서도 D grad는 계산되지만 D optimizer를 step하지 않으며 다음 D 초기화에서 지운다. BatchNorm 상태는 파라미터 업데이트와 별도로 바뀔 수 있다.

d_steps>1이면 같은 실제 배치를 반복 사용하고 생성 잡음은 매번 새로 뽑는다. 로그에는 마지막 D 로스만 기록하며 여러 D step의 평균이 아니다. d_steps는 양수여야 하고 0이면 d_loss가 정의되지 않는다. G는 마지막 실제 배치가 작아도 설정된 self.batch_size개를 생성한다.

두 모델은 생성 시 기본 train 모드지만 train 메서드 시작에서 모드를 명시적으로 재설정하지 않는다. 다른 생성 메서드가 eval 상태를 남긴 뒤 재학습하면 동작이 달라질 수 있어 호출 순서를 확인한다. 초기화 도구는 Conv·BatchNorm을 처리하며 실제 적용 범위는 모델 구조에 따른다.

## 4. 저장·표시와 평가의 범위

고정 잡음 64개를 첫 에포크와 save_interval 간격에 표시하고,25에포크마다 두 모델·두 optimizer 상태를 저장한다. 종료 때도 최종 표본·로스 곡선·체크포인트를 저장하는 구조다. save_samples·interpolate_latent는 G를 무조건 train으로 돌리는 보조 함수여서 원래 모드 복원이 아니다. generate_samples는 eval 상태를 남긴다.

체크포인트에는 구조·전처리·평가 설정·난수 상태 전체가 없고 이 클래스에 학습 재개 메서드도 없다. 같은 모델을 구성하고 필요한 상태를 복원하는 별도 절차가 필요하다. 학습셋만 사용하며 검증·독립 테스트·최적 모델 선택은 없다.

이 페이지에는 실제 FID·IS·클래스 엔트로피 측정이 없어 다른 완전 연결 실험의 수치를 이 DCGAN의 결과로 제시할 수 없다. 표시·격자·보간의 품질도 실험으로 확인해야 한다. 이 편집에서는 다운로드·학습·샘플링·그림 생성을 실행하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
D 로스가 log4에 가깝다는 사실은 분포 일치의 증명인가?

</div>

??? success "연습문제 1 풀이"

    D가 모든 실제·생성 표본에 0.5를 주면 두 평균 BCE의 합이 log4인 것은 맞다. 그러나 같은 합이 다른 예측 조합에서도 나올 수 있고 약한 D도 0.5를 출력할 수 있다. 유한 표본의 값 하나로 최적 판별기·내시 균형·좋은 G를 증명할 수 없다. 먼저 실제 logits·로스 계약을 맞추고 분류 진단과 품질·다양성을 함께 평가한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 EMA 골격을 학습에 연결할 때 무엇을 보완해야 하는가?

</div>

??? success "연습문제 2 풀이"

    copy import와 실제 generator 정의가 필요하다. EMA 모델은 학습 시작 시 한 번 만들고 매 G 업데이트 뒤 정한 일정으로 파라미터를 평균해야 한다. 아래 코드는 한 번의 평균만 보이며 BatchNorm running 통계 등 버퍼는 처리하지 않는다. EMA 버퍼를 복사·평균·재추정할 폴리시와 평가 모드·장치를 정해야 한다. 평균이 항상 더 좋은 생성 품질을 주는 것은 아니므로 같은 평가 조건에서 비교한다.

    ```python
    ema_decay = 0.999
    ema_generator = copy.deepcopy(generator)
    for ema_p, p in zip(ema_generator.parameters(), generator.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
    ```


---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
조건부 생성과 판별을 추가하려면 데이터·구조를 어떻게 연결하는가?

</div>

??? success "연습문제 3 풀이"

    실제 레이블을 보존하고 G의 조건 입력,D의 이미지·조건 쌍을 함께 정의한다. G에 원핫을 연결하면 투영 입력을 latent_dim+10으로 바꾼다. 컨볼루션 D에는 조건 지도를 채널로 연결하거나 피처에서 결합하는 등 구체적인 구조가 필요하다. 해상도·출력·로스 계약부터 맞추고 생성 시 조건 분포도 정한다. 레이블을 전달한다고 지정 숫자 생성·클래스 내 다양성·품질이 자동 보장되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 스크립트의 결과 보고에 어떤 평가가 추가로 필요한가?

</div>

??? success "연습문제 4 풀이"

    검증·최종 테스트 분할,실제 생성 품질·다양성 평가와 대표 표본이 필요하다. 피처 추출기·레이어·전처리·표본 수·참조 자료·시드·IS 분할·생성 폴리시를 명시하고 변동을 보고한다. MNIST 분류기 피처를 쓰면 표준 ImageNet Inception 평가와 구분해 이름을 붙인다. 실제끼리의 유한 표본 점수는 계약의 진단값이지 보편적 바닥 상수가 아니다. 본문에는 해당 결과가 없어 고정 FID·IS 우열을 제시할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
이 코드에서 가장 먼저 확인할 학습 실패 원인은 무엇인가?

</div>

??? success "연습문제 5 풀이"

    일반 하이퍼파라미터보다 로컬 D의 64×64 입력 요구,28×28 실제 전처리,logits `(B,)` 대 확률 BCE·레이블 `(B,1)` 불일치를 먼저 확인한다. 그 뒤 범위·그래프 분리·그래디언트 초기화·상태·유한값·잠재 의존성을 점검한다. BatchNorm·드롭아웃·라벨 스무딩은 원인과 목적에 맞춰 검증할 후보이며 필수 성공 조건이 아니다. 로스도 진단 자료로 유용하지만 생성 품질의 단독 증거는 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
훈련을 오래 하면 품질이 반드시 좋아지는가?

</div>

??? success "연습문제 6 풀이"

    보장할 수 없다. 초기 학습 부족이 개선될 수도 있고 붕괴·과적합·진동·다양성 저하가 나타날 수도 있다. 현재 스크립트는 최종 웨이트만 최종 결과로 쓰며 검증 기준의 최적 선택은 없다. 평가 빈도·표본 수·검증 목적을 미리 정해 체크포인트를 비교하고 독립 최종 자료를 분리한다. 실행 없는 30 대60에포크의 특정 수치를 이 예제 결과라고 주장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
표본 격자는 좋아 보이지만 FID가 높으면 무엇을 점검하는가?

</div>

??? success "연습문제 7 풀이"

    표시 변환·피처 전처리·참조 통계·표본 수·수치 구현과 평가 피처의 적합성을 확인한다. 유한 격자가 놓친 다양성·클래스 비중·피처 차이도 가능하다. 그러나 높은 FID가 유일하게 다양성 부족을 뜻하거나 실제 자료 기준이 특정 값과 다르면 반드시 버그라는 것은 아니다. 같은 피처 통계 FID의 수치 일관성 등 알려진 성질과 여러 독립 진단을 함께 사용한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
아래 조건 연결 골격을 현재 컨볼루션 D에 바로 쓸 수 있는가?

</div>

??? success "연습문제 8 풀이"

    아니다. 아래 D는 펼친 이미지와 원핫을 벡터로 연결하는 계약이지만 현재 D는 4차원 이미지의 컨볼루션 입력이다. G의 첫 레이어도 추가 조건 차원을 받도록 바꿔야 한다. 채널 연결·보조 분류·projection 등 조건화는 서로 다른 목적·구조라 실제 경로를 정의해야 한다. 지정 클래스 일치율뿐 아니라 클래스별 품질·다양성도 평가한다. 조건을 주어도 모델이 무시하거나 클래스 내 붕괴가 생길 수 있다.

    ```python
    # 생성기: z 에 표지를 이어 붙인다
    zc = torch.cat([z, F.one_hot(y, 10).float()], dim=1)
    fake = g(zc)

    # 판별기: 그림에 표지를 이어 붙인다
    out = d(torch.cat([x.flatten(1), F.one_hot(y, 10).float()], dim=1))
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
VAE와 이 GAN을 비교하려면 어떤 공통 조건이 필요한가?

</div>

??? success "연습문제 9 풀이"

    평가 피처·전처리·표본 수·참조·선택 예산·시드를 맞추고 구조·학습 시간·파라미터·관측 모형 차이를 보고한다. 같은 파라미터 비교와 같은 시간 비교도 다른 질문이다. 현재 코드의 실행 계약이 맞지 않으므로 다른 실험의 숫자를 빌려 우열을 단정할 수 없다. GAN에는 기본 재구성 인코더가 없고 통상 계산 가능한 밀도도 없어 비교 목적 자체를 명확히 정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
아래 config 기록이 실제 main 설정과 어떻게 다른가?

</div>

??? success "연습문제 10 풀이"

    골격은 잠재 64·완전 연결 hidden 목록·D 드롭아웃·30에포크·라벨 스무딩·MNIST 평가기를 가정한다. 실제 main은 잠재 100·컨볼루션 모델·50에포크이고 스무딩·FID·IS 계산은 없다. 따라서 아래 사전을 그대로 저장해도 현재 실험을 재현하는 설정이 아니다. 실제 import 모델 버전·해상도·출력·로스·전처리·optimizer·난수·데이터 순서·평가를 기록하고 체크포인트와 함께 연결해야 한다.

    ```python
    config = dict(
        # 모델
        latent_dim=64, g_hidden=(256,512,1024), d_hidden=(512,256),
        batchnorm_g=True, dropout_d=0.3,
        # 익히기
        epochs=30, batch_size=128, lr_g=2e-4, lr_d=2e-4, betas=(0.5,0.999),
        label_smoothing=0.1, seed=42,
        # 밸류
        feature_net='mnist_cnn_98.50', feature_layer='fc1', feature_dim=128,
        n_samples=10000, real_ref='train[:10000]', sample_seed=0,
        is_splits=10, is_shuffled=True, temperature=1.0,
    )
    ```


## 정리하며

학습 파이프라인은 모델·데이터·로스 계약이 맞아야 동작한다. 현재 로컬 64×64 logits 판별기와 28×28 확률 BCE 스크립트는 서로 맞지 않는다. 의도한 교대 학습·저장 흐름과 실제 실행 상태를 구분하고 검증·생성 평가를 별도로 설계해야 한다.
