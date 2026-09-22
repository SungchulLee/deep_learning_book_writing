# 맞겨루기 만들개 도구

맞겨루기 만들개를 익히고 그려 보고 따지려면 여러 얼개에서 두루 쓰는 도구 함수 모음이 필요하다. 이 단원은 DCGAN 지침을 따른 무게 첫자리매김, 격자 배치의 표본 그려 보기, 익히기 나아감 그리기, 숨은 공간 사이 메우기, 여러 손실 함수 짜기(여느 것, 포화하지 않는 것, 바서슈타인) 같은 꼭 필요한 연장을 준다.

## 1. 코드

```python
"""
맞겨루기 만들개 도구

이 단원은 맞겨루기 만들개를 익히고 그려 보고 따지는 도구 함수를 담는다.
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
    DCGAN 논문의 권고를 따라 신경망 무게를 첫자리매김한다.
    
    Conv와 ConvTranspose 층: 평균=0, 표준편차=0.02
    배치 정규화 층: 무게=1, 치우침=0
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
    만들개에서 보기 그림을 만들어 갈무리한다.
    
    인수:
        generator: 만들개 신경망
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
    만들개와 가름개의 손실 곡선을 그린다.
    
    인수:
        g_losses: 만들개 손실 목록
        d_losses: 가름개 손실 목록
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
    실제 자료와 가짜 자료에 대한 가름개 내놓기의 히스토그램을 그린다.
    
    인수:
        discriminator: 가름개 신경망
        real_data: 실제 자료 표본
        generator: 만들개 신경망
        noise: 가짜 표본을 만들 잡음
        device: 돌릴 장치
        filename: 내놓을 파일 이름
    """
    discriminator.eval()
    generator.eval()
    
    with torch.no_grad():
        # 실제 자료에 대한 가름개 내놓기를 얻는다
        d_real = discriminator(real_data).cpu().numpy()
        
        # 가짜 자료를 만들고 가름개 내놓기를 얻는다
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
        generator: 만들개 신경망
        z1: 첫째 숨은 벡터
        z2: 둘째 숨은 벡터
        steps: 사이 메우기 걸음 수
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
        generator: 만들개 신경망
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
    """여러 맞겨루기 만들개 손실 함수 모음."""
    
    @staticmethod
    def vanilla_gan_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
                        mode: str = 'discriminator') -> torch.Tensor:
        """
        본디 GAN 잃음(두 갈래 엇갈린 엔트로피).
        
        인수:
            d_real: 실제 자료에 대한 가름개 내놓기
            d_fake: 가짜 자료에 대한 가름개 내놓기
            mode: 'discriminator'이나 'generator'
        
        반환값:
            손실 값
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
        포화하지 않는 만들개 손실.
        
        인수:
            d_fake: 가짜 자료에 대한 가름개 내놓기
        
        반환값:
            만들개 손실
        """
        return -torch.mean(torch.log(d_fake + 1e-8))
    
    @staticmethod
    def wasserstein_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
                        mode: str = 'discriminator') -> torch.Tensor:
        """
        바서슈타인 맞겨루기 만들개 손실.
        
        인수:
            d_real: 참 자료에 대한 가름개(비평가)의 내놓음
            d_fake: 가짜 자료에 대한 가름개(비평가)의 내놓음
            mode: 'discriminator'이나 'generator'
        
        반환값:
            손실 값
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
    가름개 들임에 잡음을 더한다(익힘이 든든해진다).
    
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
    WGAN-GP의 기울기 벌점을 셈한다.
    
    인수:
        discriminator: 가름개 신경망
        real_data: 실제 자료 표본
        fake_data: 만든 가짜 표본
        device: 돌릴 장치
        lambda_gp: 기울기 벌점 계수
    
    반환값:
        기울기 벌점 손실
    """
    batch_size = real_data.size(0)
    
    # 사이 메우기를 위한 아무 무게
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # 실제와 가짜 사이를 메운다
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    
    # 가름개 내놓기를 얻는다
    d_interpolates = discriminator(interpolates)
    
    # 기울기를 셈한다
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # 기울기를 펼친다
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
        generator: 만들개 신경망
        discriminator: 가름개 신경망
        g_optimizer: 만들개 가장 좋게 하개
        d_optimizer: 가름개 가장 좋게 하개
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
        generator: 만들개 신경망
        discriminator: 가름개 신경망
        g_optimizer: 만들개 가장 좋게 하개
        d_optimizer: 가름개 가장 좋게 하개
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

## 2. 논의

`weights_init` 함수는 DCGAN 논문의 첫자리매김 방식을 짠다. 곧 Conv과 ConvTranspose 층은 평균 0, 표준 편차 0.02인 정규 첫자리매김을 쓰고, BatchNorm 층은 무게에 평균 1, 표준 편차 0.02을, 치우침에 0을 쓴다. 이 꼼꼼한 첫자리매김은 익히기 앞머리의 무너짐을 막고 깊은 얼개에서 기울기가 제대로 흐르게 한다.

`GANLosses` 갈래는 흔한 맞겨루기 만들개 손실 세 가지를 짠다. 여느 손실은 두값 어긋 엔트로피를 쓰는데 가름개가 너무 자신 있으면 기울기가 사라질 수 있다. 포화하지 않는 손실 $-\mathbb{E}[\log D(G(z))]$은 익히기 앞머리에 더 센 기울기를 준다. 바서슈타인 손실은 가름개에서 시그모이드를 없애고 흙 나르기 거리를 가장 작게 하여 뜻있는 손실 값과 함께 더 안정된 익히기를 준다. `calculate_gradient_penalty` 함수는 립시츠 묶음을 지키게 하는 WGAN-GP 규칙 세우기 항을 짠다.

그 밖의 도구로는 이름표 부드럽게 하기(가름개가 지나치게 자신 있게 내놓지 않도록 목표 1.0을 0.9으로 바꾸기), 익히기의 안정을 위한 들임 잡음 더하기, 두루 갖춘 되짚을 자리 갈무리와 불러오기가 있다. 사이 메우기와 숨은 격자 함수는 배운 숨은 공간을 체계적으로 살필 수 있게 하며, 이는 만들개가 무엇을 배웠는지 아는 데 결정적이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
같은 자료 묶음에서 여느 맞겨루기 만들개 손실, 포화하지 않는 손실, 바서슈타인 손실을 견주는 익히기 되풀이를 짜라. 익히기 되풀이에 따라 손실 값을 좇아 그려라. 어느 손실이 가장 안정된 익히기 신호를 주는가?

</div>

??? success "연습문제 1 풀이"
    바서슈타인 손실은 손실 값이 뜻있고(흙 나르기 거리를 어림한다) 포화하지 않으므로 흔히 가장 안정된 익히기 신호를 준다. 여느 맞겨루기 만들개 손실은 심하게 흔들릴 수 있고 값을 풀이하기도 어렵다. 포화하지 않는 손실은 그 중간으로 여느 것보다 기울기가 낫지만 바서슈타인보다는 덜 안정되다. 바서슈타인 손실 곡선은 꾸준히 내려가며 표본 품질이 나아지는 것과 이어지지만, 여느 손실과 포화하지 않는 손실은 품질이 나아져도 또렷한 흐름을 보이지 않을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
WGAN-GP의 기울기 벌점은 실제 자료와 가짜 자료 사이를 메운다. 이 벌점이 왜 1-립시츠 묶음을 지키게 하는지 수학으로 이끌어 내고 사이 메우기 계수 $\alpha$의 몫을 설명하라.

</div>

??? success "연습문제 2 풀이"
    기울기 벌점 $\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$은 가름개의 기울기 잣대가 1에서 벗어날 때마다 벌을 준다. $\alpha \sim \text{Uniform}(0, 1)$일 때 메운 점 $\hat{x} = \alpha x_{\text{real}} + (1 - \alpha) x_{\text{fake}}$은 가장 좋은 평가개의 기울기 잣대가 1이어야 하는, 실제 자료와 만든 자료 사이의 자리를 뽑는다. 립시츠 이어짐의 뜻매김에 따라 모든 $x_1, x_2$에 대해 $|D(x_1) - D(x_2)| \leq K \|x_1 - x_2\|$이며, 기울기 벌점은 이 사이 메우기 길을 따라 기울기 잣대를 묶어 $K = 1$을 지키게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`interpolate_latent` 함수를 선형 사이 메우기에 더해 공 모양 선형 사이 메우기(slerp)도 받쳐 주도록 넓혀라. 두 방법으로 사이 메우기 격자를 만들고 중간 표본의 보기 품질을 견주어라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def slerp(z1, z2, alpha):
        z1_norm = z1 / z1.norm(dim=-1, keepdim=True)
        z2_norm = z2 / z2.norm(dim=-1, keepdim=True)
        omega = torch.acos((z1_norm * z2_norm).sum(dim=-1, keepdim=True))
        return (torch.sin((1-alpha)*omega)/torch.sin(omega)) * z1 + \
               (torch.sin(alpha*omega)/torch.sin(omega)) * z2
    ```
    slerp은 메우는 내내 잣대를 한결같이 지켜 정규 분포의 확률 높은 껍질 위에 머문다. 선형 사이 메우기는 가운데 점에서 원점 쪽으로 내려가는데 차원이 높으면 그곳은 밀도가 낮다. 숨은 차원이 클수록 보기의 차이가 뚜렷하다. 곧 slerp은 한결같이 또렷한 중간 그림을 내지만 선형 사이 메우기는 가운데가 더 흐릴 수 있다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
표본 격자를 만드는 함수에서 무엇을 챙겨야 하는가?

</div>

??? success "연습문제 4 풀이"
    범위와 눈금과 보간이다.

    ```python
    def make_grid(imgs, nrow=8):
        imgs = imgs.detach().cpu().clamp(0, 1)         # 범위를 자른다
        ...
        plt.imshow(grid, cmap='gray', vmin=0, vmax=1,  # 눈금을 못 박는다
                   interpolation='nearest')            # 보간을 끈다
    ```

    셋 다 빠뜨리기 쉽고 셋 다 그림을 실제와 다르게 보이게 한다.

    **눈금을 못 박지 않으면** 칸마다 다른 눈금이 쓰여 견줄 수 없다. 아주 밝은 화소 하나가
    있으면 그 칸 전체가 어두워 보인다.

    **보간을 끄지 않으면** 28×28을 키울 때 뭉개져 실제보다 매끄럽게 보인다. 표본의 품질을
    부풀리는 셈이다.

    `tanh` 출력이면 `clamp` 전에 $[0,1]$로 옮겨야 한다
    ([DCGAN 연습문제 2](46_dcgan.md)).

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
고정된 잡음을 다루는 도구 함수를 어떻게 짜겠는가?

</div>

??? success "연습문제 5 풀이"
    익히기 밖에서 한 번 뽑아 들고 다니게 한다.

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

    챙긴 것이 셋이다.

    - `Generator`에 씨앗을 주어 **전역 상태를 건드리지 않는다.** `torch.manual_seed`를
      부르면 익히기의 무작위성까지 바뀐다
    - `eval()`과 원래 모드 되돌리기. 배치 정규화가 있으면 필요하다
    - `no_grad()`

    첫째가 놓치기 쉽고 고약하다. 값매김 함수가 전역 씨앗을 건드리면 익히기의 재현성이
    깨지는데, 그 관계가 눈에 안 보인다.

    원래 모드로 되돌리는 것도 중요하다. `generator.train()`을 무조건 부르면 값매김 중에
    불렀을 때 모드가 바뀐다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
익히기 기록을 어떤 구조로 남기겠는가?

</div>

??? success "연습문제 6 풀이"
    에포크마다 사전 하나씩 모아 두는 것이 간단하고 충분하다.

    ```python
    history = []
    history.append(dict(epoch=ep, d_loss=dl, g_loss=gl,
                        d_real=d_real_mean, d_fake=d_fake_mean,
                        fid=fid_value, is_=is_value, class_entropy=ent))
    ```

    손실 말고 **$D(x)$와 $D(G(z))$의 평균**을 남기는 것이 특히 쓸모 있다. 균형이 어떤지
    직접 보여 준다.

    | 보이는 것 | 뜻 |
    |---|---|
    | $D(x) \approx D(G(z)) \approx 0.5$ | 균형. 바라는 모습 |
    | $D(x) \to 1$, $D(G(z)) \to 0$ | 판별기가 이긴다. 기울기가 사라진다 |
    | 둘 다 0.5 근처에서 요동 | 대개 정상 |

    손실보다 이 두 값이 읽기 쉽다. 손실은 두 항이 섞여 있어 어느 쪽이 움직였는지 안
    보인다.

    FID와 부류 엔트로피를 함께 남기면 나중에 곡선을 그릴 수 있다. 익히기를 다시 돌리지
    않고 분석할 수 있으므로 값이 크다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
가중치를 저장하고 되불러 오는 함수를 어떻게 짜겠는가?

</div>

??? success "연습문제 7 풀이"
    이어서 익힐 수 있도록 최적화기 상태까지 담는다.

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

    최적화기 상태를 담는 까닭은 Adam이 모멘텀을 들고 있기 때문이다. 빼고 이어 익히면
    그 상태가 초기화되어 흔들린다.

    **판별기도 저장해야** 한다. 표본을 만드는 데는 필요 없지만
    ([GAN 기초 연습문제 7](45_gan.md)) 이어 익히려면 있어야 한다.

    `config`를 함께 담으면 같은 얼개를 다시 만들 수 있다. 이것이 없으면 가중치의 모양만
    보고 얼개를 짐작해야 한다.

    `weights_only=True`를 쓰는 편이 안전한데, `config`에 파이썬 객체가 들어 있으면 못
    쓴다. 그래서 `config`를 사전과 기본 자료형만으로 두는 것이 좋다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
값매김 도구를 익히기 코드와 어떻게 떼어 놓겠는가?

</div>

??? success "연습문제 8 풀이"
    값매김이 **모델을 건드리지 않게** 하는 것이 핵심이다. 건드릴 수 있는 것이 셋이다.

    | 무엇 | 어떻게 막는가 |
    |---|---|
    | 전역 난수 상태 | `torch.Generator`를 따로 쓴다 |
    | 모델의 모드 | 원래 모드를 기억해 되돌린다 |
    | 기울기 | `torch.no_grad()` |

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

    `finally`를 쓰는 까닭은 값매김이 오류를 내도 모드가 되돌아가게 하는 것이다. 그러지
    않으면 오류 한 번 뒤로 익히기가 평가 모드로 돌아가는데, 배치 정규화가 있으면 결과가
    달라지고 알아채기 어렵다.

    참 자료 통계량은 **밖에서 받는다.** 값매김 함수가 스스로 셈하면 부를 때마다 비싸고,
    설정이 어긋날 여지가 생긴다.

    이렇게 떼어 두면 익히기 코드가 값매김을 몰라도 되고, 값매김을 따로 시험할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
이 도구들을 시험하는 방법을 적어라.

</div>

??? success "연습문제 9 풀이"
    값을 아는 입력으로 시험한다. 값매김 함수는 특히 시험해 두는 값이 크다. 틀려도 그럴듯한
    수가 나오기 때문이다.

    | 무엇을 시험하는가 | 어떻게 |
    |---|---|
    | FID | 같은 배치 두 번 → 0 |
    | 인셉션 점수 | 완벽한 원-핫 고른 배치 → 부류 수 |
    | 격자 | 모양과 값 범위 |
    | 고정 잡음 | 두 번 불러 같은 결과 |
    | 저장·되불러 오기 | 되불러 온 모델의 출력이 같은지 |

    마지막 것이 특히 값지다.

    ```python
    z = torch.randn(4, k)
    before = g(z)
    save_ckpt('t.pt', g, d, og, od, 0, {})
    g2 = G(k); load_ckpt('t.pt', g2, D())
    assert torch.allclose(before, g2(z)), "되불러 오기가 어긋난다"
    ```

    그리고 참 자료를 값매김 함수에 넣어 기준점을 얻는 것이 사실상 통합 시험이 된다.
    참 자료의 FID가 2.52 근처, 인셉션 점수가 9.467 근처로 나오면 물길 전체가 맞게
    돌아가고 있다는 뜻이다. 엉뚱한 수가 나오면 범위나 전처리를 의심한다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
난수 씨앗을 다루는 도구를 어떻게 두겠는가?

</div>

??? success "연습문제 10 풀이"
    한곳에서 모두 고정하는 함수를 두고, **모델을 만들기 직전에** 부른다.

    ```python
    def set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    ```

    부르는 자리가 중요하다. 모델을 만든 **뒤에** 부르면 가중치 초기화가 씨앗에 딸리지
    않아 재현되지 않는다. 이 책에서 실제로 겪은 사고이며, 같은 모델에 세 가지 정확도가
    나왔다([3.4절](../../ch03/mnist/04_cnn.md)).

    ```python
    set_seed(42)
    g, d = G(k), D()                       # 이 순서
    loader = DataLoader(..., generator=torch.Generator().manual_seed(42))
    ```

    자료 부르개의 생성기도 따로 고정해야 한다. `torch.manual_seed`만으로는 섞는 순서가
    고정되지 않는 경우가 있다.

    값매김 쪽은 전역 씨앗을 건드리지 않는 것이 좋다(연습문제 2). 두 곳의 무작위성을
    떼어 두면 값매김을 넣거나 빼도 익히기가 같게 돌아간다.

## 정리하며

**다룬 것** — 맞겨루기 만들개 도구

`weights_init` 함수는 DCGAN 논문의 첫자리매김 방식을 짠다.

고갱이 갈래는 `GANLosses`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
