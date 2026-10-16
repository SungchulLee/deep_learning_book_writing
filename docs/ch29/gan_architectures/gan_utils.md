# GAN 학습·평가 유틸리티

GAN 보조 함수는 초기화,로스,표본 표시와 체크포인트의 반복 작업을 묶는다. 그러나 모든 생성기 출력 범위·장치·잠재 입력 형태가 같지는 않다. 유틸리티 이름보다 실제 가정과 학습 상태에 미치는 영향을 먼저 확인해야 한다.

## 1. 도구 구현

```python
"""
맞겨루기 생성기 도구

이 단원은 맞겨루기 생성기를 익히고 그려 보고 따지는 도구 함수를 담는다.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Tuple, List

# ========================================================================
# 메인
# ========================================================================


def weights_init(m):
    """
    DCGAN 논문의 권고를 따라 뉴럴 넷트웍 무게를 첫자리매김한다.
    
    Conv와 ConvTranspose 레이어: 평균=0, 표준편차=0.02
    배치 정규화 레이어: 무게=1, 치우침=0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_samples(generator: nn.Module, epoch: int, device: str, 
                fixed_noise: torch.Tensor, filename: str = None):
    """
    생성기에서 보기 그림을 만들어 갈무리한다.
    
    인수:
        generator: 생성기 뉴럴 넷트웍
        epoch: 현재 에포크 번호
        device: 돌릴 장치
        fixed_noise: 한결같은 그림을 위한 붙박이 잡음
        filename: 골라 쓰는 파일 이름, 기본값은 'samples_epoch_{epoch}.png'
    """
    generator.eval()
    
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    
    # [-1, 1]에서 [0, 1]으로 되돌린다
    fake = (fake + 1) / 2.0
    
    # 격자 생성
    grid = make_grid(fake, nrow=8, padding=2, normalize=False)
    
    # 그림
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f'Generated Samples - Epoch {epoch}')
    
    if filename is None:
        filename = f'samples_epoch_{epoch:04d}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    generator.train()


def plot_training_progress(g_losses: List[float], d_losses: List[float],
                          filename: str = 'training_progress.png'):
    """
    생성기와 판별기의 로스 곡선을 그린다.
    
    인수:
        g_losses: 생성기 로스 목록
        d_losses: 판별기 로스 목록
        filename: 내놓을 파일 이름
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training progress to {filename}")


def plot_discriminator_outputs(discriminator: nn.Module, real_data: torch.Tensor,
                               generator: nn.Module, noise: torch.Tensor,
                               device: str, filename: str = 'discriminator_outputs.png'):
    """
    실제 자료와 가짜 자료에 대한 판별기 내놓기의 히스토그램을 그린다.
    
    인수:
        discriminator: 판별기 뉴럴 넷트웍
        real_data: 실제 자료 표본
        generator: 생성기 뉴럴 넷트웍
        noise: 가짜 표본을 만들 잡음
        device: 돌릴 장치
        filename: 내놓을 파일 이름
    """
    discriminator.eval()
    generator.eval()
    
    with torch.no_grad():
        # 실제 자료에 대한 판별기 내놓기를 얻는다
        d_real = discriminator(real_data).cpu().numpy()
        
        # 가짜 자료를 만들고 판별기 내놓기를 얻는다
        fake_data = generator(noise)
        d_fake = discriminator(fake_data).cpu().numpy()
    
    # 히스토그램을 그린다
    plt.figure(figsize=(10, 5))
    plt.hist(d_real, bins=50, alpha=0.5, label='Real', color='blue')
    plt.hist(d_fake, bins=50, alpha=0.5, label='Fake', color='red')
    plt.xlabel('Discriminator Output')
    plt.ylabel('Frequency')
    plt.title('Discriminator Output Distribution')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved discriminator outputs to {filename}")
    
    discriminator.train()
    generator.train()


def interpolate_latent(generator: nn.Module, z1: torch.Tensor, z2: torch.Tensor,
                      steps: int = 10, device: str = 'cpu',
                      filename: str = 'interpolation.png'):
    """
    숨은 벡터 둘 사이를 메워 만든다.
    
    인수:
        generator: 생성기 뉴럴 넷트웍
        z1: 첫째 숨은 벡터
        z2: 둘째 숨은 벡터
        steps: 사이 메우기 스텝 수
        device: 돌릴 장치
        filename: 내놓을 파일 이름
    """
    generator.eval()
    
    # 선형 사이 메우기
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_samples = []
    
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            sample = generator(z)
            interpolated_samples.append(sample)
    
    # 잇고 고르게 맞추기를 되돌린다
    samples = torch.cat(interpolated_samples, dim=0)
    samples = (samples + 1) / 2.0
    
    # 격자 생성
    grid = make_grid(samples, nrow=steps, padding=2)
    
    # 그림
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Latent Space Interpolation')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved interpolation to {filename}")
    
    generator.train()


def generate_latent_grid(generator: nn.Module, latent_dim: int = 100,
                        grid_size: int = 10, device: str = 'cpu',
                        filename: str = 'latent_grid.png'):
    """
    숨은 차원 둘을 바꾸어 표본 격자를 만든다.
    
    인수:
        generator: 생성기 뉴럴 넷트웍
        latent_dim: 숨은 공간의 차원
        grid_size: 격자의 크기(grid_size x grid_size)
        device: 돌릴 장치
        filename: 내놓을 파일 이름
    """
    generator.eval()
    
    # 두 차원의 값 격자를 만든다
    x = torch.linspace(-2, 2, grid_size)
    y = torch.linspace(-2, 2, grid_size)
    
    samples = []
    
    with torch.no_grad():
        for yi in y:
            for xi in x:
                # 숨은 벡터를 만든다(두 차원만 빼고 모두 0)
                z = torch.randn(1, latent_dim, device=device) * 0.5
                z[0, 0] = xi
                z[0, 1] = yi
                
                sample = generator(z)
                samples.append(sample)
    
    # 잇고 고르게 맞추기를 되돌린다
    samples = torch.cat(samples, dim=0)
    samples = (samples + 1) / 2.0
    
    # 격자 생성
    grid = make_grid(samples, nrow=grid_size, padding=2)
    
    # 그림
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Latent Space Grid (z[0] and z[1])')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent grid to {filename}")
    
    generator.train()


class GANLosses:
    """여러 맞겨루기 생성기 로스 함수 모음."""
    
    @staticmethod
    def vanilla_gan_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
                        mode: str = 'discriminator') -> torch.Tensor:
        """
        본디 GAN 잃음(두 클래스 엇갈린 엔트로피).
        
        인수:
            d_real: 실제 자료에 대한 판별기 내놓기
            d_fake: 가짜 자료에 대한 판별기 내놓기
            mode: 'discriminator'이나 'generator'
        
        반환값:
            로스 값
        """
        criterion = nn.BCELoss()
        
        if mode == 'discriminator':
            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)
            
            real_loss = criterion(d_real, real_labels)
            fake_loss = criterion(d_fake, fake_labels)
            
            return real_loss + fake_loss
        
        elif mode == 'generator':
            real_labels = torch.ones_like(d_fake)
            return criterion(d_fake, real_labels)
    
    @staticmethod
    def nonsaturating_loss(d_fake: torch.Tensor) -> torch.Tensor:
        """
        포화하지 않는 생성기 로스.
        
        인수:
            d_fake: 가짜 자료에 대한 판별기 내놓기
        
        반환값:
            생성기 로스
        """
        return -torch.mean(torch.log(d_fake + 1e-8))
    
    @staticmethod
    def wasserstein_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
                        mode: str = 'discriminator') -> torch.Tensor:
        """
        바서슈타인 맞겨루기 생성기 로스.
        
        인수:
            d_real: 참 자료에 대한 판별기(크리틱)의 내놓음
            d_fake: 가짜 자료에 대한 판별기(크리틱)의 내놓음
            mode: 'discriminator'이나 'generator'
        
        반환값:
            로스 값
        """
        if mode == 'discriminator':
            return -(torch.mean(d_real) - torch.mean(d_fake))
        elif mode == 'generator':
            return -torch.mean(d_fake)


def label_smoothing(labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    실제/가짜 이름표에 이름표 부드럽게 하기를 쓴다.
    
    인수:
        labels: 본디 이름표(0 또는 1)
        smoothing: 부드럽게 하는 정도
    
    반환값:
        부드럽게 한 이름표
    """
    return labels * (1 - smoothing) + smoothing * 0.5


def add_noise_to_inputs(data: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """
    판별기 들임에 잡음을 더한다(익힘이 든든해진다).
    
    인수:
        data: 들임 자료
        noise_std: 잡음의 표준 편차
    
    반환값:
        잡음 섞인 자료
    """
    noise = torch.randn_like(data) * noise_std
    return data + noise


def calculate_gradient_penalty(discriminator: nn.Module, real_data: torch.Tensor,
                               fake_data: torch.Tensor, device: str,
                               lambda_gp: float = 10.0) -> torch.Tensor:
    """
    WGAN-GP의 그래디언트 벌점을 셈한다.
    
    인수:
        discriminator: 판별기 뉴럴 넷트웍
        real_data: 실제 자료 표본
        fake_data: 만든 가짜 표본
        device: 돌릴 장치
        lambda_gp: 그래디언트 벌점 계수
    
    반환값:
        그래디언트 벌점 로스
    """
    batch_size = real_data.size(0)
    
    # 사이 메우기를 위한 아무 무게
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # 실제와 가짜 사이를 메운다
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    
    # 판별기 내놓기를 얻는다
    d_interpolates = discriminator(interpolates)
    
    # 그래디언트를 셈한다
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # 그래디언트를 펼친다
    gradients = gradients.view(batch_size, -1)
    
    # 벌점을 셈한다
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def save_checkpoint(generator: nn.Module, discriminator: nn.Module,
                   g_optimizer, d_optimizer, epoch: int,
                   filename: str = 'checkpoint.pth'):
    """
    모델 되짚기 지점을 갈무리한다.
    
    인수:
        generator: 생성기 뉴럴 넷트웍
        discriminator: 판별기 뉴럴 넷트웍
        g_optimizer: 생성기 가장 좋게 하개
        d_optimizer: 판별기 가장 좋게 하개
        epoch: 현재 에포크
        filename: 되짚을 자리 파일 이름
    """
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, filename)
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(generator: nn.Module, discriminator: nn.Module,
                   g_optimizer, d_optimizer, filename: str, device: str):
    """
    모델 되짚기 지점을 불러온다.
    
    인수:
        generator: 생성기 뉴럴 넷트웍
        discriminator: 판별기 뉴럴 넷트웍
        g_optimizer: 생성기 가장 좋게 하개
        d_optimizer: 판별기 가장 좋게 하개
        filename: 되짚을 자리 파일 이름
        device: 불러올 기기
    
    반환값:
        되짚을 자리의 바퀴 수
    """
    checkpoint = torch.load(filename, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {filename}, epoch {epoch}")
    
    return epoch


if __name__ == "__main__":
    pass
```

weights_init는 클래스 이름에 Conv가 있는 레이어의 웨이트를 N(0,0.02²),BatchNorm의 스케일을 N(1,0.02²),바이어스를 0으로 초기화한다. Linear나 Conv 바이어스는 여기서 초기화하지 않는다. 해당 속성을 갖춘 기본 affine 정규화 레이어를 가정하며 모든 모듈에 무조건 적용 가능한 함수는 아니다. 이 초기화가 붕괴·그래디언트 문제를 자동 방지하지도 않는다.

## 2. 표시 함수와 모델 상태

save_samples·interpolate_latent·generate_latent_grid는 출력이 −1~1이라는 가정으로 (x+1)/2를 적용한다. 이미 0~1을 반환하는 G에는 잘못된 표시 변환이다. 평가 피처의 입력 정규화와 표시 범위 변환은 별도 계약으로 둔다.

표시 함수는 eval·no_grad를 적용하지만 종료할 때 원래 모드를 복원하지 않고 무조건 train으로 바꾼다. 평가 중 호출한 모델의 상태가 달라질 수 있고 예외 발생 때 모드 복구도 보장하지 않는다. device 인수가 있어도 모델·fixed_noise·실제 자료를 자동으로 이동하지 않는 경로가 있으므로 호출자가 맞춰야 한다.

interpolate_latent는 두 끝점의 선형 보간이다. 단계 수가 양수이고 끝점 배치·잠재 모양·장치·자료형이 같아야 한다. 끝점이 배치 여러 행이면 그리드의 배치·단계 배치 순서도 따로 해석한다. generate_latent_grid는 첫 두 좌표를 바꾸면서 **나머지 좌표도 매 칸 새 잡음으로 뽑는다**. 따라서 두 축만 통제한 순수한 단면 그림이 아니다. latent_dim≥2와 G의 벡터 잠재 입력 계약이 필요하다.

## 3. 로스 이름과 실제 목적

GANLosses.vanilla_gan_loss의 D 모드는 실제·생성 평균 BCE의 합이다. G 모드는 실제 레이블 1의 BCE라서 이미 **비포화** 로스다. 별도 nonsaturating_loss도 같은 확률 목적에 작은 ε를 더한 로그 구현이다. 따라서 이 두 함수를 원래 minimax 대 비포화의 서로 다른 로스로 비교하면 안 된다. 지원하지 않는 mode 문자열에는 오류 대신 None을 반환하는 점도 호출부에서 주의한다.

확률 BCE에는 0~1의 D 출력이 필요하다. Wasserstein 모드는 실수 critic의 평균 차이와 생성 점수 목적이며 Sigmoid 확률 BCE와 다른 계약이다. Lipschitz 제어·critic 최적화가 함수 안에 자동 포함되지는 않는다. 수치적으로 안정적인 logits 목적과 원래·비포화 부호는 [로스 절](../gan_losses/original.md)을 따른다.

label_smoothing은 labels×(1−s)+s/2다. s=0.1이면 1→0.95,0→0.05로 **양쪽**을 가운데로 옮긴다. 실제 타깃만 0.9로 만드는 one-sided 방식과 다르다. s의 범위를 정하고 실제 적용한 타깃을 기록해야 한다. add_noise_to_inputs는 제한 없는 가우시안 잡음을 더하므로 원래 자료 범위를 넘을 수 있고 이것이 항상 학습에 유리한 것은 아니다.

## 4. 이미지용 gradient penalty

calculate_gradient_penalty는 alpha `(B,1,1,1)`로 이미지 `(B,C,H,W)`를 표본별 보간하고 비배치 축을 펼쳐 L2 노름을 계산한다. 이 alpha를 2차원 벡터 자료에 그대로 쓰면 원치 않는 broadcast·표본 혼합이 생길 수 있다. 실제·생성 자료의 모양·device·자료형을 맞춰야 한다.

반환값에는 **이미 lambda_gp가 곱해져 있다**. 호출부에서 다시 같은 계수를 곱하지 않도록 한다. 샘플 간 독립적인 critic 출력이라는 조건도 필요하며 BatchNorm처럼 표본을 섞는 연산에서는 출력 합의 입력 그래디언트와 표본별 노름을 구분해야 한다. 생성기 쪽 그래프를 분리할지도 호출부가 정한다. 보간점의 소프트 벌점은 전역 1-Lipschitz의 엄밀한 강제가 아니다.

## 5. 기록·복원과 평가 범위

save_checkpoint는 두 모델·두 optimizer·에포크를 저장한다. 모델 state_dict에는 등록된 BatchNorm 버퍼도 포함되지만 구조·전처리·잠재 분포·난수 상태·데이터 순서 전체는 없다. 같은 구조를 만든 뒤 로드해야 하고 정확한 학습 재현에는 추가 상태가 필요하다. 파일 저장이 매 에포크 기록이나 최적 모델 선택을 자동 구현하는 것은 아니다.

D·G 로스와 판별 출력 히스토그램은 현재 분류·게임 동작의 진단 자료다. D 평균 0.5가 내시 균형이나 좋은 생성의 충분조건은 아니다. 품질·다양성과 독립 평가를 함께 보며 이 페이지는 실제 학습·성능·그림 결과를 새로 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이 유틸리티의 vanilla G 로스와 nonsaturating_loss는 어떤 관계인가?

</div>

??? success "연습문제 1 풀이"

    둘 다 −log D(fake)를 최소화하는 비포화 확률 목적이다. 후자는 로그 안에 ε를 더해 수치·미분이 약간 달라진다. 원래 minimax G는 음의 BCE(D(fake),0)로 별도 정의해야 한다. Wasserstein 비교에는 실수 critic과 Lipschitz 처리도 포함한다. 각 로스 이름만으로 안정성 순위를 정하지 말고 동일 평가·튜닝 예산에서 품질·다양성·비용을 비교한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
Gradient penalty가 전역 1-Lipschitz를 강제하지 않는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    유한한 실제·생성 보간점에서 λ(||∇c||−1)²를 벌할 뿐 모든 입력의 그래디언트 노름≤1을 검사하지 않는다. 무작위 α는 선택한 표본 쌍 사이 위치를 정하며 이 쌍이 반드시 최적 수송 결합은 아니다. 이미지용 모양·표본별 그래디언트·λ 중복 적용도 맞춰야 한다. 이 벌점이 엄밀한 전역 보장이나 모든 critic의 최적 표현을 주지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 slerp 골격의 수치·기하 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"

    끝점 노름 0을 처리하고 내적을 −1~1로 제한해야 acos의 수치 오류를 피할 수 있다. 각도가 0·π에 가까우면 sin(omega) 분모와 경로 선택을 별도로 처리해야 한다. 원래 끝점 반경이 다르면 이 식이 일정 반경을 보장하지도 않는다. 원점은 가우시안 점별 밀도가 높고 전형적 확률 질량은 고차원 껍질에 모인다는 점을 구분한다. slerp가 항상 더 자연스러운 이미지를 만드는 것은 아니다.

    ```python
    def slerp(z1, z2, alpha):
        z1_norm = z1 / z1.norm(dim=-1, keepdim=True)
        z2_norm = z2 / z2.norm(dim=-1, keepdim=True)
        omega = torch.acos((z1_norm * z2_norm).sum(dim=-1, keepdim=True))
        return (torch.sin((1-alpha)*omega)/torch.sin(omega)) * z1 + \
               (torch.sin(alpha*omega)/torch.sin(omega)) * z2
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
표본 표시의 범위·스케일·보간을 어떻게 맞추는가?

</div>

??? success "연습문제 4 풀이"

    G의 실제 출력 단위에 맞게 0~1 표시 변환을 하고 같은 vmin·vmax·색상 폴리시를 쓴다. raw 픽셀을 보려면 nearest 같은 보간을 정하며 확대 표시와 원래 샘플 해상도를 밝힌다. 아래 make_grid는 생략 부호·미정 grid와 반환값이 없어 완성된 함수가 아니다. 동일 이름으로 정의하면 import한 torchvision make_grid를 가릴 수도 있다. 각 이미지 자동 대비 조절은 비교를 왜곡할 수 있어 명시적으로 관리한다.

    ```python
    def make_grid(imgs, nrow=8):
        imgs = imgs.detach().cpu().clamp(0, 1)         # 범위를 자른다
        ...
        plt.imshow(grid, cmap='gray', vmin=0, vmax=1,  # 눈금을 못 박는다
                   interpolation='nearest')            # 보간을 끈다
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
FixedNoise 예시가 본문 표시 함수보다 개선하는 상태 관리와 한계를 설명하라.

</div>

??? success "연습문제 5 풀이"

    별도 CPU torch.Generator로 잡음을 만들면 해당 표본 추출이 전역 난수 스트림을 바꾸지 않는다. 원래 루트 training 값을 저장해 복원하고 no_grad를 적용한다. 다만 예외 발생에도 복원하려면 try/finally가 필요하고 모델·잡음 자료형·장치를 맞춰야 한다. 특수하게 서로 다른 하위 모듈 모드를 사용한다면 루트 train(was_training)만으로 그 세부 상태가 모두 원래대로 보존되는지도 확인한다.

    ```python
    class FixedNoise:
        def __init__(self, n, dim, device, seed=0):
            g = torch.Generator().manual_seed(seed)
            self.z = torch.randn(n, dim, generator=g).to(device)
        def grid(self, generator):
            was_training = generator.training
            generator.eval()
            with torch.no_grad():
                imgs = generator(self.z)
            generator.train(was_training)
            return imgs
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
학습 로그의 판별 출력·품질 지표를 어떻게 해석하는가?

</div>

??? success "연습문제 6 풀이"

    로스·D(real)·D(fake)·러닝레이트·업데이트 횟수·평가 설정을 함께 기록한다. 아래는 사전 기록 골격이며 변수들은 호출부에서 정의해야 한다. D 평균이 0.5여도 약한 판별기·값 상쇄가 가능하고 1·0으로 분리되어도 모든 목적에서 G 그래디언트가 사라지는 것은 아니다. FID·IS·클래스 엔트로피도 각각 한계가 있어 표본·다양성·검증 목적을 함께 해석한다.

    ```python
    history = []
    history.append(dict(epoch=ep, d_loss=dl, g_loss=gl,
                        d_real=d_real_mean, d_fake=d_fake_mean,
                        fid=fid_value, is_=is_value, class_entropy=ent))
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
학습 재개 체크포인트에 필요한 정보를 설명하라.

</div>

??? success "연습문제 7 풀이"

    두 모델·두 optimizer·진행 위치와 구조·전처리·잠재 분포·클래스 계약을 저장한다. 완전 재현에는 난수·데이터 순서·스케줄러 등도 필요하다. 아래 config 골격은 본문보다 메타데이터를 더 담지만 자동으로 같은 구조를 생성하지는 않는다. weights_only 로드는 지원되는 기본 자료형·텐서 등의 객체 계약에 맞춰야 하며 map_location·모델 장치도 정한다. optimizer 상태가 없다고 반드시 발산하는 것은 아니지만 같은 업데이트를 이어 가는 것은 아니다.

    ```python
    def save_ckpt(path, g, d, og, od, ep, config):
        torch.save(dict(g=g.state_dict(), d=d.state_dict(),
                        og=og.state_dict(), od=od.state_dict(),
                        epoch=ep, config=config), path)

    def load_ckpt(path, g, d, og=None, od=None):
        ck = torch.load(path, weights_only=True)
        g.load_state_dict(ck['g']); d.load_state_dict(ck['d'])
        if og: og.load_state_dict(ck['og'])
        if od: od.load_state_dict(ck['od'])
        return ck['epoch'], ck['config']
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
아래 evaluate 골격이 완성된 평가 함수가 아닌 이유를 설명하라.

</div>

??? success "연습문제 8 풀이"

    반환값 fid·is_·entropy는 ...인 Ellipsis이고 ref·net도 실제 피처 계산에 사용되지 않는다. g.k도 본문 DCGANGenerator의 latent_dim과 다른 속성 계약이다. 난수 분리·eval·finally 복원 구조는 유용하지만 실제 샘플링·범위 변환·피처·통계·평가 지표를 구현해야 한다. 모델의 자료형·장치·잠재 모양도 맞춰야 하며 함수 이름만으로 숫자 평가가 완료되지는 않는다.

    ```python
    @torch.no_grad()
    def evaluate(g, ref, net, n=2000, seed=0, device='cpu'):
        was = g.training
        g.eval()
        try:
            gen = torch.Generator().manual_seed(seed)       # 전역을 안 건드린다
            z = torch.randn(n, g.k, generator=gen).to(device)
            imgs = g(z)
            return dict(fid=..., is_=..., entropy=...)
        finally:
            g.train(was)                                    # 반드시 되돌린다
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
도구의 정적·수치 검증에서 어떤 기준을 사용할 수 있는가?

</div>

??? success "연습문제 9 풀이"

    같은 피처 통계의 FID가 수치 허용오차 안에서 0에 가까운지,균형 원핫 예측의 전체 배치 IS가 클래스 수인지,격자 모양·범위와 고정 잡음 재현을 확인할 수 있다. 분할 IS에서는 각 분할의 클래스 균형도 영향을 준다. 아래 저장·복원 비교는 같은 구조·장치·모드·버퍼·잡음에서 해야 하며 train 모드 상태 변화·dropout이 있으면 출력 일치 검사가 달라진다. 참 자료의 FID·IS에 모든 파이프라인이 따라야 할 보편적인 고정 숫자는 없다.

    ```python
    z = torch.randn(4, k)
    before = g(z)
    save_ckpt('t.pt', g, d, og, od, 0, {})
    g2 = G(k); load_ckpt('t.pt', g2, D())
    assert torch.allclose(before, g2(z)), "되불러 오기가 어긋난다"
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
난수 시드 설정이 보장하는 범위를 설명하라.

</div>

??? success "연습문제 10 풀이"

    Python·NumPy·PyTorch 초기화 전에 필요한 시드를 설정하고 DataLoader·worker·평가용 생성기의 상태도 관리한다. 아래 set_seed에는 random import가 추가로 필요하다. 같은 시드만으로 서로 다른 버전·하드웨어·비결정 연산의 동일 결과를 보장하지는 않는다. 모델 생성 뒤 시드를 설정해도 이미 만들어진 웨이트가 바뀌지는 않는다. 평가의 난수 스트림을 분리하면 평가 빈도 변경이 학습 난수 소비를 바꾸는 일부 영향을 줄일 수 있다.

    ```python
    def set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    ```
    ```python
    set_seed(42)
    g, d = G(k), D()                       # 이 순서
    loader = DataLoader(..., generator=torch.Generator().manual_seed(42))
    ```


## 정리하며

GAN 유틸리티의 출력 범위·잠재 모양·모델 모드와 로스 이름의 실제 의미를 확인한다. 이 코드의 vanilla G는 비포화 목적,라벨 스무딩은 양쪽 0.95·0.05 변환이며 이미지 GP에는 계수가 이미 포함된다. 평가 골격·저장 상태와 완성된 검증 절차를 구분해야 한다.
