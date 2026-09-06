# 뽑기 대본 보기

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 뽑기 대본 보기을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
뽑기 대본 보기

익힌 퍼짐 모델에서 표본을 만든다.
이 대본은 되짚을 자리를 불러와 새 MNIST 숫자를 만든다.
"""

import torch
import os
from mnist_diffusion import MNISTDiffusion
from diffusion_utils import visualize_samples

# ========================================================================
# 메인
# ========================================================================


def generate_grid(diffusion, n_samples=64, filename='generated_grid.png'):
    """표본 격자를 만든다."""
    print(f"Generating {n_samples} samples...")
    samples = diffusion.sample_images(n_samples=n_samples, use_ema=True)
    visualize_samples(samples, nrow=8, filename=filename)
    print(f"Saved to {filename}")


def generate_interpolation(diffusion, n_steps=10):
    """아무 출발점 사이를 메워 만든다."""
    print(f"Generating interpolation with {n_steps} steps...")
    
    # 아무 잡음 벡터 둘을 만든다
    noise1 = torch.randn(1, 1, 28, 28, device=diffusion.device)
    noise2 = torch.randn(1, 1, 28, 28, device=diffusion.device)
    
    # 보간
    alphas = torch.linspace(0, 1, n_steps)
    interpolated_samples = []
    
    for alpha in alphas:
        # 잡음 공간에서의 선형 사이 메우기
        noise = (1 - alpha) * noise1 + alpha * noise2
        
        # 이 출발점에서 잡음을 없앤다
        x_t = noise
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((1,), t, device=diffusion.device, dtype=torch.long)
            
            predicted_noise = diffusion.ema_model(x_t, t_tensor)
            
            beta_t = diffusion.diffusion_params['betas'][t]
            sqrt_recip_alpha_t = diffusion.diffusion_params['sqrt_recip_alphas'][t]
            sqrt_one_minus_alpha_cumprod_t = diffusion.diffusion_params['sqrt_one_minus_alphas_cumprod'][t]
            
            mean = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
            
            if t > 0:
                posterior_variance_t = diffusion.diffusion_params['posterior_variance'][t]
                noise_sample = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance_t) * noise_sample
            else:
                x_t = mean
        
        interpolated_samples.append(x_t)
    
    # 시각화한다
    samples = torch.cat(interpolated_samples, dim=0)
    visualize_samples(samples, nrow=n_steps, filename='interpolation.png')
    print("Saved to interpolation.png")


def main():
    # 되짚을 자리가 있는지 살핀다
    checkpoint_files = [
        'mnist_diffusion_model.pt',
        'mnist_diffusion_final.pt',
        'checkpoint_epoch_100.pt'
    ]
    
    checkpoint = None
    for ckpt_file in checkpoint_files:
        if os.path.exists(ckpt_file):
            checkpoint = ckpt_file
            break
    
    if checkpoint is None:
        print("Error: No checkpoint found!")
        print("Please train a model first using example_train.py")
        print("Looking for one of:", checkpoint_files)
        return
    
    print(f"Loading checkpoint: {checkpoint}")
    print("-" * 50)
    
    # 모형을 시작한다
    diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=64,
        learning_rate=2e-4
    )
    
    # 체크포인트를 불러온다
    diffusion.load_checkpoint(checkpoint)
    
    print("\nGenerating samples...")
    print("-" * 50)
    
    # 여러 표본을 만든다
    generate_grid(diffusion, n_samples=64, filename='samples_8x8.png')
    generate_grid(diffusion, n_samples=100, filename='samples_10x10.png')
    
    # 사이 메우기를 만든다
    generate_interpolation(diffusion, n_steps=10)
    
    # 다양함을 살피려 큰 묶음을 만든다
    print("\nGenerating large batch for visualization...")
    samples = diffusion.sample_images(n_samples=256)
    visualize_samples(samples, nrow=16, filename='samples_large.png')
    
    print("\n" + "=" * 50)
    print("Sampling complete!")
    print("Generated files:")
    print("  - samples_8x8.png: 64 samples in 8x8 grid")
    print("  - samples_10x10.png: 100 samples in 10x10 grid")
    print("  - samples_large.png: 256 samples")
    print("  - interpolation.png: Smooth interpolation between digits")
    print("=" * 50)


if __name__ == "__main__":
    main()```

## 논의

뽑기 대본 보기의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.
