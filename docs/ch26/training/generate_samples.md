# 표본 만들기

익힌 변분 오토인코더 모델에서 표본을 만들고 그려 보기

오토인코더와 변분 오토인코더는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
익힌 변분 오토인코더 모델에서 표본을 만들고 그려 보기
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
    
    main(args)
```

## 2. 논의

이 짜기는 말끔하고 읽기 쉬운 PyTorch 부호로 만들어 내는 모델 익히기의 핵심 개념을 보인다. 모듈 짜임 덕분에 조각마다 살펴보고 다른 일이나 자료 묶음에 맞춰 고치기 쉽다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 변분 오토인코더 일에 대한 실전 직관이 선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
부호를 훑어 핵심 설계 결정을 가려내어라. 구체적인 짜기 고름 셋을 들고 저마다 왜 만들어 내는 모델 익히기에 알맞은지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

</div>

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
표본 만들기 짜기를 확인하는 두루 살핀 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 극단 값(0, 아주 큰 수)이 든 들임 같은 모서리 경우를 시험하라.

</div>

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


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
표본을 만드는 데 인코더가 필요한가?

</div>

??? success "연습문제 5 풀이"
    필요 없다. 사전 분포에서 뽑아 디코더에만 넣는다.

    ```python
    z = torch.randn(n, latent_dim)
    imgs = model.decode(z)
    ```

    인코더는 익히는 데만 쓰인다. 익히고 나면 **디코더가 곧 만들어 내는 모델**이다.

    이것이 `encode`와 `decode`를 나누어 두는 설계의 값어치를 보여 주는 자리다
    ([25장 모듈 연습문제 2](../../ch25/architecture/autoencoder.md)). 둘이 붙어 있으면
    이 일을 할 수 없다.

    오토인코더에도 같은 코드를 돌릴 수 있다. 문법적으로 막는 것이 없다. 다만 결과가
    얼룩이다([23.5절](../../ch25/limits/latent_sampling.md)). 그 차이가 이 장의
    존재 이유다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
뽑은 $z$에 온도를 곱해 키우면 표본이 좋아지는가?

</div>

??? success "연습문제 6 풀이"
    **판정 점수는 올라간다.** 그리고 그것이 이 문제의 함정이다.

    | 온도 | 1.0 | 1.15 | 1.3 | 1.5 | 1.75 | 2.0 | 2.5 |
    |---|---|---|---|---|---|---|---|
    | 확신도 0.9 넘는 비율 | 57.4% | 59.7% | 60.0% | 62.9% | 65.6% | 67.2% | **69.1%** |

    끝까지 단조롭게 오른다. 그런데 온도 2.5는 $\mathcal{N}(0, 6.25I)$에서 뽑는 것이라
    사전 분포와 거의 겹치지 않는다. $\mathbb{E}\|z\|$가 3.95에서 9.89로 커지는데 사전
    분포의 값은 4.00이다. **모델이 정의한 분포에서 멀어질수록 점수가 오른다.**

    그러므로 이 점수를 품질로 읽을 수 없다. 무엇이 오르고 있는지 보자.

    | 온도 | 평균 밝기 | 대비(표준편차) | 0.05/0.95 밖의 화소 |
    |---|---|---|---|
    | 1.0 | 0.1224 | 0.2655 | 78.5% |
    | 2.5 | 0.1477 | 0.3278 | 89.1% |
    | **실제 자료** | 0.1223 | 0.2984 | 89.8% |

    온도를 올리면 **더 날카로워진다.** 온도 1.0의 표본은 실제 자료보다 대비가 낮고
    흐릿한데(0.2655 대 0.2984), 온도를 키우면 그 값이 자료 쪽으로 올라간다. 분류기는
    날카로운 그림에 더 확신을 준다.

    곧 이 잣대는 **날카로움과 그럴듯함을 구별하지 못한다.** 그리고 흐릿함은 변분 자기
    인코더의 알려진 약점이므로([47_vae 연습문제 4](../architecture/47_vae.md)), 온도를
    올리는 것은 그 증상을 가리는 것이지 모델을 고치는 것이 아니다.

    온도를 쓸 수는 있다. 다만 **모델의 표본이 아니라 모델을 손본 표본**임을 밝혀야 한다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
GAN에서 쓰는 잘라 내기(truncation)를 여기 쓰면 어떻게 되는가?

</div>

??? success "연습문제 7 풀이"
    이 모델에서는 **도움이 되지 않는다.**

    | 설정 | 확신도 0.9 넘는 비율 |
    |---|---|
    | $|z| < 1$로 자름 | 37.6% |
    | $|z| < 2$로 자름 | 55.6% |
    | 자르지 않음 | **57.4%** |

    자르면 나빠진다. GAN에서는 잘라 내기가 품질을 올리는 알려진 요령인데 여기서는
    반대다.

    까닭은 사전 분포와 아우른 사후 분포의 관계에 있다. 차원마다 재어 보면 아우른 사후
    분포의 표준편차가 살아 있는 12차원에서 평균 **1.032**이고 사전 분포는 1.000이다.
    **거의 정확히 맞아 있다.**

    곧 코드가 사전 분포 전체에 고르게 퍼져 있으므로, 중심만 남기고 자르면 **실제로
    쓰이던 자리를 버리는** 것이 된다. KL 항이 바로 이 맞춤을 시킨 것이라 당연한 결과다.

    GAN은 그 맞춤을 강제하는 항이 없어 사정이 다르다. 잘라 내기가 듣는다는 것 자체가
    GAN의 숨은 공간에서는 중심부가 더 믿을 만하다는 뜻이다.

    교훈은 요령을 옮겨 쓸 때 **그 요령이 기대는 성질이 새 모델에도 있는지** 확인해야
    한다는 것이다. 여기서는 없었다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
코드 사이 끼움은 변분 오토인코더가 오토인코더보다 매끄러운가?

</div>

??? success "연습문제 8 풀이"
    **이 측정에서는 차이가 없다.** 흔히 그렇다고 이야기되는 것이라 재어 볼 값이 있다.

    시험 자료에서 두 그림을 200쌍 골라 코드를 선형으로 이어 아홉 단계를 풀고, 단계마다
    분류기의 확신도를 재면 이렇다.

    | 단계 | 1 | 3 | 5 (가운데) | 7 | 9 |
    |---|---|---|---|---|---|
    | 오토인코더 | 0.97 | 0.93 | **0.84** | 0.93 | 0.97 |
    | 변분 오토인코더 | 0.96 | 0.93 | **0.83** | 0.92 | 0.97 |

    양 끝에서 가운데로 가며 떨어지는 폭이 오토인코더 0.132, 변분 오토인코더 0.140이다.
    **변분 오토인코더가 오히려 조금 나쁘다**(차이가 의미 있을 만큼 크지는 않다).

    왜 기대와 다른가. 사이 끼움의 양 끝이 **자료에서 온 코드**이기 때문이다. 두 끝이
    모두 코드가 실제로 놓인 자리이므로 그 사이도 대체로 아는 영역을 지난다. 자기
    인코더의 약점은 코드가 놓인 자리에서 **멀리** 떨어진 곳이고, 사이 끼움은 거기까지
    가지 않는다.

    그래서 사이 끼움은 두 모델을 가르는 잣대가 못 된다. 가르는 것은
    [무작위로 뽑기](../../ch25/limits/latent_sampling.md)이고, 거기서는 0.2% 대 57.4%로
    크게 벌어진다.

    무엇을 재는 잣대인지 정하는 것이 그만큼 중요하다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
표본의 다양성을 어떻게 재겠는가? 확신도만으로 무엇을 놓치는가?

</div>

??? success "연습문제 9 풀이"
    확신도는 **모든 표본이 똑같아도** 높게 나온다. 그것이 가장 큰 구멍이다.

    싸게 쓸 수 있는 잣대가 판정된 부류의 쏠림을 보는 것이다. 표본 1,000개를 분류기에
    넣어 나온 부류의 히스토그램으로 엔트로피를 잰다.

    | | 엔트로피 |
    |---|---|
    | 완전 연결 VAE | 2.199 |
    | 누비기 VAE | 2.196 |
    | 열 부류가 완전히 고를 때 | **2.303** |

    둘 다 2.2 근처로 고른 쪽에 가깝다. 곧 열 숫자를 두루 만들고 있으며 몇 부류에만
    쏠려 있지 않다.

    이 잣대의 한계도 분명하다. **부류 사이의 다양성만 보고 부류 안의 다양성은 보지
    않는다.** 모든 3이 똑같은 3이어도 엔트로피는 높다.

    부류 안까지 보려면 표본끼리의 거리를 쓸 수 있다. 같은 부류로 판정된 표본들의 쌍별
    거리 평균을 내고, 실제 자료에서 같은 부류끼리 잰 값과 견준다. 모델 쪽이 훨씬 작으면
    원형 몇 개만 내고 있는 것이다.

    이것이 [29장](../../ch29/index.md)의 잣대들이 풀려는 문제다. 인셉션 점수는 이
    엔트로피 발상을 다듬은 것이고, 프레셰 인셉션 거리는 특징 공간에서 **분포 전체**를
    견주므로 다양성과 품질을 함께 본다.

    그리고 그 잣대들도 완전하지 않다. 이 장에서 본 온도 함정처럼, 잣대를 올리는 방법과
    모델을 좋게 하는 방법이 갈리는 일이 늘 있다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
표본을 격자로 그릴 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 10 풀이"
    **고른 표본을 고르지 않는 것**이 가장 중요하다.

    좋아 보이는 것만 골라 싣는 것은 결과를 부풀리는 일이다. 씨앗을 고정하고 처음 $n$개를
    그대로 싣는 것이 정직하다.

    ```python
    torch.manual_seed(0)                 # 고정하고
    z = torch.randn(64, latent_dim)      # 처음 64개를 그대로
    ```

    그리고 밝힐 것들이 있다.

    | 밝힐 것 | 왜 |
    |---|---|
    | 온도를 썼는가 | 썼다면 모델의 표본이 아니다 |
    | 고른 것인가 | 골랐다면 그 사실과 기준 |
    | 어떤 설정인가 | $\beta$, 숨은 차원, 에포크 |

    그림을 볼 때도 눈여겨볼 것이 있다. 표본들이 서로 얼마나 다른지 보라. 격자 안에
    비슷한 것이 여럿이면 다양성 문제의 신호다. 확신도 수치로는 안 보이는 것이 그림에서는
    보인다.

    실제 자료 몇 장을 같은 격자 옆에 나란히 놓는 것도 좋다. 흐릿함처럼 절대적으로
    판단하기 어려운 것이 견줄 대상이 있으면 금방 보인다. 이 장의 수치로는 대비 0.2655
    대 0.2984의 차이인데, 눈으로는 나란히 놓아야 알아챈다.

## 정리하며

**다룬 것** — 표본 만들기

이 짜기는 말끔하고 읽기 쉬운 PyTorch 부호로 만들어 내는 모델 익히기의 핵심 개념을 보인다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
