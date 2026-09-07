# 표본 만들기

익힌 변분 자기 부호기 모델에서 표본을 만들고 그려 보기

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 코드

```python
"""
익힌 변분 자기 부호기 모델에서 표본을 만들고 그려 보기
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========================================================================
# 메인
# ========================================================================

import sys
sys.path.append('..')
from models.vae import VAE
from models.conv_vae import ConvVAE
from models.conditional_vae import ConditionalVAE
from models.conv_cvae import ConvConditionalVAE
from models.beta_vae import BetaVAE, ConvBetaVAE
from utils.visualization import (
    visualize_reconstruction,
    visualize_samples,
    visualize_latent_traversal,
    visualize_interpolation,
    plot_latent_space
)


def load_model(model_type, checkpoint_path, device, **model_kwargs):
    """되짚기 지점에서 익힌 모델을 불러온다"""
    # 모델 생성
    if model_type == 'vae':
        model = VAE(**model_kwargs)
    elif model_type == 'conv_vae':
        model = ConvVAE(**model_kwargs)
    elif model_type == 'cvae':
        model = ConditionalVAE(**model_kwargs)
    elif model_type == 'conv_cvae':
        model = ConvConditionalVAE(**model_kwargs)
    elif model_type == 'beta_vae':
        model = BetaVAE(**model_kwargs)
    elif model_type == 'conv_beta_vae':
        model = ConvBetaVAE(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 무게를 불러온다
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} from {checkpoint_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model


def main(args):
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 시험 데이터셋 불러오기
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # 모델 인자
    model_kwargs = {
        'latent_dim': args.latent_dim,
    }
    
    if args.model_type in ['vae', 'beta_vae']:
        model_kwargs['input_dim'] = 784
        model_kwargs['hidden_dim'] = 256
    elif args.model_type in ['cvae']:
        model_kwargs['input_dim'] = 784
        model_kwargs['hidden_dim'] = 256
        model_kwargs['num_classes'] = 10
    elif args.model_type in ['conv_vae', 'conv_beta_vae']:
        model_kwargs['img_channels'] = 1
        model_kwargs['img_size'] = 28
    elif args.model_type in ['conv_cvae']:
        model_kwargs['img_channels'] = 1
        model_kwargs['img_size'] = 28
        model_kwargs['num_classes'] = 10
    
    if args.model_type in ['beta_vae', 'conv_beta_vae']:
        model_kwargs['beta'] = 4.0
    
    # 모델을 불러온다
    model = load_model(args.model_type, args.checkpoint_path, device, **model_kwargs)
    
    # 조건부인지 정한다
    is_conditional = 'cvae' in args.model_type
    
    # 그림을 만든다
    print("\n=== Generating Visualizations ===")
    
    if args.reconstruction:
        print("\n1. Reconstruction visualization...")
        visualize_reconstruction(
            model, test_loader,
            num_images=args.num_samples,
            device=device,
            conditional=is_conditional
        )
    
    if args.samples:
        print("\n2. Random sample generation...")
        if is_conditional:
            # 갈래마다 표본을 만든다
            for class_label in range(10):
                print(f"   Generating samples for class {class_label}...")
                visualize_samples(
                    model,
                    args.latent_dim,
                    num_samples=args.num_samples,
                    device=device,
                    class_label=class_label
                )
        else:
            visualize_samples(
                model,
                args.latent_dim,
                num_samples=args.num_samples,
                device=device
            )
    
    if args.interpolation and not is_conditional:
        print("\n3. Latent space interpolation...")
        visualize_interpolation(
            model,
            test_loader,
            device=device,
            num_steps=args.num_steps
        )
    
    if args.traversal and hasattr(model, 'traverse_latent_dimension'):
        print("\n4. Latent dimension traversals...")
        num_dims = min(args.num_traversals, args.latent_dim)
        for dim_idx in range(num_dims):
            print(f"   Traversing dimension {dim_idx}...")
            visualize_latent_traversal(
                model,
                dim_idx=dim_idx,
                num_steps=args.num_steps,
                range_limit=3.0,
                device=device
            )
    
    if args.latent_space and args.latent_dim == 2:
        print("\n5. Latent space visualization (2D only)...")
        plot_latent_space(
            model,
            test_loader,
            device=device,
            num_batches=20
        )
    
    print("\n=== All visualizations complete! ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['vae', 'conv_vae', 'cvae', 'conv_cvae', 'beta_vae', 'conv_beta_vae'],
                        help='Type of VAE model')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension (must match training)')
    
    # 만들어 내기 인자
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of steps for interpolation/traversal')
    parser.add_argument('--num-traversals', type=int, default=5,
                        help='Number of latent dimensions to traverse')
    
    # 무엇을 만들지
    parser.add_argument('--reconstruction', action='store_true',
                        help='Visualize reconstructions')
    parser.add_argument('--samples', action='store_true',
                        help='Generate random samples')
    parser.add_argument('--interpolation', action='store_true',
                        help='Visualize latent space interpolation')
    parser.add_argument('--traversal', action='store_true',
                        help='Visualize latent dimension traversals')
    parser.add_argument('--latent-space', action='store_true',
                        help='Visualize 2D latent space (only for 2D latent dim)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # --all을 주면 모든 그림을 켠다
    if args.all:
        args.reconstruction = True
        args.samples = True
        args.interpolation = True
        args.traversal = True
        args.latent_space = True
    
    # 그림을 정하지 않으면 다시 세우기와 표본을 붙박이로 한다
    if not any([args.reconstruction, args.samples, args.interpolation, 
                args.traversal, args.latent_space]):
        args.reconstruction = True
        args.samples = True
    
    main(args)```

## 논의

이 짜기는 말끔하고 읽기 쉬운 PyTorch 부호로 만들어 내는 모델 익히기의 핵심 개념을 보인다. 모듈 짜임 덕분에 조각마다 살펴보고 다른 일이나 자료 묶음에 맞춰 고치기 쉽다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 변분 자기 부호기 일에 대한 실전 직관이 선다.

## 연습문제

**연습문제 1.**
부호를 훑어 핵심 설계 결정을 가려내어라. 구체적인 짜기 고름 셋을 들고 저마다 왜 만들어 내는 모델 익히기에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
표본 만들기 짜기를 확인하는 두루 살핀 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 극단 값(0, 아주 큰 수)이 든 들임 같은 모서리 경우를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_generate samples():
        model = Generate Samples(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
