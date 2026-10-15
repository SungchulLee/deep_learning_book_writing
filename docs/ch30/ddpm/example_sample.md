# 체크포인트를 이용한 MNIST 샘플링

학습된 확산 모델을 생성에 사용하려면 가중치뿐 아니라 모델 구조와 노이즈 스케줄도 학습 당시와 맞아야 한다. 이 예제는 저장 파일을 찾아 EMA 모델로 이미지 격자를 만들고 초기 노이즈 사이의 보간을 시도한다. 보간 루프와 파일 이름에는 실제 동작을 따로 확인해야 하는 부분이 있다.

## 1. 코드

```python
"""
뽑기 대본 보기

익힌 확산 모델에서 표본을 만든다.
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
    
    # 다양함을 살피려 큰 배치를 만든다
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
    main()
```

**보존된 출력:**

```
Error: No checkpoint found!
Please train a model first using example_train.py
Looking for one of: ['mnist_diffusion_model.pt', 'mnist_diffusion_final.pt', 'checkpoint_epoch_100.pt']
```

이 출력은 후보 체크포인트가 없어 생성 전에 종료한 경우다. 학습 완료나 샘플 생성 성공을 보여 주지 않는다.

## 2. 체크포인트 선택

작업 디렉터리에서 mnist_diffusion_model.pt, mnist_diffusion_final.pt, checkpoint_epoch_100.pt 순으로 존재 여부를 확인하고 첫 파일을 선택한다. 수정 시각이나 학습 성능을 비교해 최신·최상 모델을 고르는 방식은 아니다. 모두 없으면 안내 메시지와 함께 반환한다.

로드할 객체는 T=1000과 기본 코사인 스케줄, 같은 SimpleUNet 구조로 생성한다. load_checkpoint는 학습·EMA 가중치와 optimizer 상태를 복원하지만 저장된 diffusion_params는 사용하지 않는다. 따라서 다른 스케줄로 학습한 파일을 선택하면 생성 조건이 어긋날 수 있다.

## 3. 이미지 격자와 표시 범위

generate_grid는 EMA 모델로 지정 수량을 생성하지만 nrow는 항상 8로 전달한다. 64개는 8×8 격자지만 100개는 8열·13행의 격자에 배치되며 나머지 칸은 비어 있다. **samples_10x10.png라는 파일 이름이나 마지막 출력 문구와 달리 실제 10×10 배치는 아니다.** 256개 생성 부분은 nrow=16을 별도로 전달해 16×16 격자를 만든다.

생성 텐서는 B×1×28×28이며 픽셀 범위를 강제로 제한하지 않는다. visualize_samples는 배치 최솟값과 최댓값으로 표시 범위를 조정한다. 서로 다른 격자의 명암이 비슷하게 보이더라도 원래 값의 범위가 같다는 뜻은 아니다.

## 4. 초기 노이즈 보간의 의미

두 독립 표준정규 텐서 z₁, z₂에 대해 z(a)=(1−a)z₁+az₂를 만든다. 중간점의 분산은 (1−a)²+a²이므로 a=1/2에서 1/2이다. 따라서 선형 보간 경로 전체가 학습 시 가정한 단위 분산 초기 분포를 유지하지는 않는다.

또한 각 a에 대해 역과정의 모든 중간 단계에서 **새 노이즈를 독립적으로 뽑는다**. 이웃 결과의 차이는 초기점 변화와 역과정 난수 변화가 섞인 결과다. 출력 문구의 “Smooth interpolation”처럼 숫자의 의미가 매끄럽게 변한다고 보장할 수 없다. 조건을 분리하려면 초기점과 역과정 난수의 제어 방법을 함께 설계해야 한다.

## 5. 보간 루프의 계산 그래프

generate_grid가 호출하는 sample_images에는 no_grad가 있지만 generate_interpolation 자체에는 해당 문맥이 없다. EMA 모델도 파라미터의 requires_grad가 켜져 있으므로 1,000단계의 모델 호출을 따라 계산 그래프가 쌓일 수 있다. 단순 추론으로 기대한 것보다 메모리 사용이 커지거나 실패할 수 있다.

main에서는 격자 생성이 먼저 평가 모드를 설정하지만, eval은 자동 미분을 끄지 않는다. 이 두 설정의 역할을 구분해야 한다. 이 페이지의 코드는 보존하며, 저장된 성공 결과나 실제 메모리·시간 측정이 없는 상태에서 보간 완료를 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
100개를 generate_grid로 생성할 때 실제 격자 크기와 파일 이름을 비교하라.

</div>

??? success "연습문제 1 풀이"

    함수는 nrow=8을 고정하므로 13행·8열에 100개를 배치한다. 마지막 네 칸은 비어 있다. samples_10x10.png라는 이름이 10행·10열을 구현하지는 않는다. 코드의 인수와 표시 문구를 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
초기 노이즈 선형 보간의 중간점 분산을 구하고, 결과가 매끄럽지 않을 수 있는 다른 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    독립 z₁,z₂의 각 좌표 분산이 1이면 (z₁+z₂)/2의 분산은 1/2이다. 다른 보간 계수에서도 단위 분산이 유지되지 않는다. 역과정 노이즈도 각 보간점에서 새로 뽑으므로 관찰된 차이가 초기점 변화만의 효과가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
보존된 출력이 의미하는 실행 경로와 보간 함수의 추론 설정 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    출력은 세 후보 파일이 모두 없어 main이 조기 반환한 경우다. 생성 성공의 증거가 아니다. 보간 함수에는 no_grad가 없어 EMA 파라미터를 통한 계산 그래프가 누적될 수 있으며 eval만으로 이를 막을 수 없다. 체크포인트의 스케줄과 생성자의 스케줄도 별도로 맞아야 한다.



## 정리하며

샘플링에서는 체크포인트 선택 순서와 확산 설정, 실제 격자 열 수를 확인한다. 초기 노이즈의 선형 보간은 단위 분산을 유지하지 않으며, 독립적인 역과정 노이즈와 추론 그래프 누적도 결과 해석과 메모리에 영향을 준다.
