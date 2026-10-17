# 체크포인트 복원과 샘플 생성

VAE의 사전 생성은 잠재 분포에서 코드를 뽑아 디코더에 전달하는 과정이다. 재구성·코드 보간·잠재 축 탐색은 서로 다른 경로를 살펴보는 평가 도구다. 학습 설정과 체크포인트 구조를 정확히 복원한 뒤 각 그림의 의미를 해석해야 한다.

## 1. 생성 스크립트

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
            # 클래스마다 표본을 만든다
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

모델 종류와 체크포인트 경로를 지정하고 생성·재구성 등의 플래그를 선택하는 구조다. 아무 개별 플래그도 없으면 재구성과 사전 생성을 선택한다. all을 지정해도 조건부 보간은 건너뛰고, 축 탐색은 해당 메서드가 있을 때만, 잠재 공간 표시는 차원이 2일 때만 시도한다. 따라서 모든 모델에 모든 시각화를 제공하는 것은 아니다.

## 2. 제공된 구현의 연결 상태

현재 로컬 utils.visualization에는 visualize_latent_traversal, visualize_interpolation과 plot_latent_space가 없다. 표시된 import는 그대로라면 프로그램 시작 단계에서 실패한다. 이하 설명은 해당 보조 함수 연결을 전제로 한 설계 설명이며 이 코드를 새로 실행해 확인한 결과는 아니다.

conv_beta_vae 분기는 img_size=28을 생성자에 전달하지만 로컬 ConvBetaVAE 생성자는 이 인수를 받지 않는다. 완전 연결 cvae의 decode는 원핫 조건을 받는데 현재 visualize_samples는 정수 레이블을 전달하므로 이 경로도 그대로 호환되지 않는다. 보조 함수나 모델 이름이 비슷해도 인수 계약까지 같은 것은 아니다.

## 3. 체크포인트와 평가 설정

load_model은 먼저 model_kwargs로 구조를 만들고 model_state_dict를 불러온다. 잠재 차원뿐 아니라 은닉 폭·채널·클래스 매핑도 학습 때와 맞아야 한다. 이 스크립트는 은닉 256·클래스 10 등을 고정하고 β 모델에는 4를 지정한다. 체크포인트의 실제 설정을 자동으로 읽어 복원하지 않는다. β는 일반 속성이라 웨이트 로드만으로 덮어써지지 않는다.

map_location은 체크포인트 텐서를 읽을 장치를 정하고 이후 모델도 그 장치로 옮긴다. eval은 평가 모드이며 그래디언트 기록을 중단하는 기능은 아니다. 현재 시각화 보조 함수의 재구성·생성 경로는 no_grad를 별도로 사용한다. 저장된 loss가 없으면 기본 문자열 unknown을 소수 형식으로 출력하려 해 오류가 나므로 체크포인트 메타데이터의 계약도 확인해야 한다.

순수한 사전 생성에는 MNIST가 필요하지 않지만 이 main은 모든 경우에 테스트셋을 다운로드·생성한다. 현재 보조 함수 호출은 저장 경로나 명시적인 표시를 지정하지 않으므로 출력 메시지만으로 그림 파일 저장을 확인할 수는 없다.

## 4. 생성 결과를 읽는 기준

디코더의 Sigmoid 출력은 베르누이 확률 이미지다. 관측 모델 자체의 표본을 원하면 픽셀도 해당 분포에서 뽑아야 한다. 회색조 BCE 관례의 해석은 [VAE 모듈](../architecture/vae.md)을 따른다. 좋은 재구성·높은 분류기 확신도·부드러운 보간이 각각 전체 생성 분포의 정확성을 보장하지는 않는다.

잠재 온도나 truncation을 사용하면 잠재 샘플링 분포가 바뀐다. 해당 폴리시와 원래 사전에서의 결과를 구분하고 품질·다양성·조건 일치율을 함께 보고해야 한다. 이 페이지는 새 생성 실험 수치를 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 스크립트의 실제 설계 결정 세 가지를 설명하라.

</div>

??? success "풀이"

    모델 종류에 따라 구조를 먼저 생성하고 state_dict를 로드하는 점, 재구성·사전 생성·보간·축 탐색을 플래그로 선택하는 점, 조건부 생성에서 클래스별로 평가하는 점이다. 첫 결정은 구조 설정의 정확한 복원을 요구하고 둘째는 평가 목적을 구분하며 셋째는 클래스별 품질 차이를 드러낼 수 있다. 잔차 연결 등 이 스크립트에 없는 일반 구조 설명으로 대신하지 않는다. 실제 제공된 보조 함수·생성자 계약과 맞는지도 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
생성 전에 확인할 입력·설정 계약을 설명하라.

</div>

??? success "풀이"

    체크포인트 키와 구조·잠재 차원·전처리·클래스 매핑, 장치·자료형을 확인한다. 표본 수·보간 단계 수와 조건 범위도 유효해야 한다. 재구성 입력은 모델에 맞춰 펼치거나 이미지 형태로 유지한다. 생성 스크립트에는 모든 모델에 공통인 forward 입력 하나가 있는 것이 아니므로 임의의 expected_dims 검사로 충분하지 않다. float32만 강제하기보다 모델과 평가 폴리시의 호환성을 확인한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
코드의 연결 실패와 생성 품질 실패를 구분하라.

</div>

??? success "풀이"

    없는 보조 함수 import, 지원하지 않는 생성자 인수, 다른 잠재 차원의 state_dict, decode 조건 모양 불일치는 실행 계약 문제다. 반면 계약을 지켜도 흐릿함·모드 누락·조건 제어 실패가 나타날 수 있으며 학습·모형·평가 요인을 따로 진단해야 한다. 기존 출력 문구는 각 경로의 성공이나 품질을 증명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 테스트 골격의 문제점과 적절한 검증 대상을 설명하라.

</div>

??? success "풀이"

    아래의 test_generate samples와 Generate Samples는 공백 때문에 유효한 Python 이름이 아니며 실제 생성 스크립트에도 그런 클래스가 없다. 생략 기호·미정 변수도 남아 있어 실행 가능한 테스트가 아니다. 적절한 검증은 모델 복원 계약, 출력 모양·범위·유한값, 조건별 인수와 평가 모드 등 실제 경로를 대상으로 한다. 사전 생성은 추론 작업이므로 모든 파라미터에 그래디언트가 생겨야 한다는 검사는 필수 조건이 아니다. 빈 배치·극단 잠재 입력에 대한 폴리시도 명확히 정의해야 한다.

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


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
사전 생성에서 인코더의 역할은 무엇인가?

</div>

??? success "풀이"

    일반적인 사전 생성에는 인코더가 필요 없고 p(z)와 디코더의 p(x|z)가 필요하다. 아래는 무조건부 확률 이미지 생성의 골격이다. 조건부 모델에는 맞는 표현의 조건을 더 전달해야 한다. 인코더는 학습 후에도 재구성·후방 추론·피처 추출과 데이터 기반 보간에 사용된다. 메서드를 분리하면 이 경로를 명확히 호출할 수 있지만 메서드 이름을 분리하지 않았다고 생성 자체가 불가능한 것은 아니다. 결정적 AE도 잠재 분포를 별도로 정해 생성할 수 있으나 그 분포의 적합성은 따로 검증해야 한다.

    ```python
    z = torch.randn(n, latent_dim)
    imgs = model.decode(z)
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
잠재 표준정규 표본에 T를 곱하는 것의 의미를 설명하라.

</div>

??? success "풀이"

    T>0이면 생성 잠재 분포가 N(0,T²I)로 바뀐다. E||z||²=T²d이며 평균 노름은 일반적으로 T√d와 정확히 같지는 않다. 이 변경이 출력 대비나 분류기 확신도를 어떻게 바꾸는지는 디코더·학습 상태에 달려 있다. 확신도가 높아져도 품질·다양성·우도가 함께 개선되는 것은 아니다. 원래 사전과 같은 지지집합을 가져도 확률 질량 배치는 다를 수 있다. 기본 사전에서의 결과와 온도 변경 폴리시의 결과를 따로 보고한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
잠재 truncation을 평가할 때 먼저 정의할 것은 무엇인가?

</div>

??? success "풀이"

    좌표별 재추출로 절단정규를 쓰는지, 범위를 벗어난 좌표를 경계로 clipping하는지, 노름으로 거르는지 명시해야 한다. 이들은 서로 다른 분포다. truncation은 다양성·품질을 바꿀 수 있지만 GAN이나 VAE에서 보편적으로 유리하거나 불리하지 않다. 집계 사후의 좌표별 표준편차가 1에 가깝다고 표준정규와 일치하는 것도 아니다. 평균·고차 구조·좌표 의존성과 생성 결과를 함께 진단한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
선형 잠재 보간이 의미적으로 좋은 경로를 보장하는가?

</div>

??? success "풀이"

    연속 디코더에 연속 경로를 넣으면 출력도 연속이지만 이것이 의미적 자연스러움이나 높은 데이터 밀도를 보장하지는 않는다. 데이터에서 얻은 두 코드 사이도 집계 사후에서 드문 영역을 지날 수 있다. 독립 표준정규 끝점의 선형 보간은 중간점에서 분산이 줄어드는 성질도 있다. 분류기 확신도 변화는 보간의 한 측면이며 VAE와 AE의 전반적 우열을 결정하지 못한다. 사용한 끝점·평균 또는 표본·경로 정의를 함께 기록한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
생성 클래스 엔트로피와 FID가 각각 놓칠 수 있는 것은 무엇인가?

</div>

??? success "풀이"

    생성 이미지를 분류한 클래스 비율의 엔트로피는 클래스 간 분포만 요약한다. 10클래스 균등일 때 최대값은 ln 10≈2.303 nat이지만 참 자료가 불균형이면 균등이 정답은 아니다. 클래스마다 같은 이미지 하나만 생성해도 이 값은 높을 수 있다. 쌍별 거리 역시 노이즈·배경·표현 선택에 민감하다. FID는 선택한 피처의 평균·공분산을 비교하는 척도이며 분포 전체의 모든 차이를 식별하지 못한다. 평가 피처가 MNIST에 적합한지, 표본 수와 바이어스가 어떤지도 확인하고 여러 지표를 함께 사용한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
생성 격자를 보고할 때 어떤 조건을 밝혀야 하는가?

</div>

??? success "풀이"

    표본 선택·제외 기준, 시드·잠재 샘플링 폴리시·조건 분포, 모델 설정·체크포인트와 확률 이미지인지 관측 표본인지 밝힌다. 아래 코드는 CPU 잠재 64개를 고정 시드로 만드는 골격이며 모델 장치와 맞춰야 한다. 처음 표본을 그대로 표시하면 선택 바이어스를 줄일 수 있지만 단일 격자가 전체 분포를 대표하지는 않는다. 동일 표시 범위·전처리의 실제 자료와 비교하고 여러 배치·정량 진단을 함께 살핀다.

    ```python
    torch.manual_seed(0)                 # 고정하고
    z = torch.randn(64, latent_dim)      # 처음 64개를 그대로
    ```


## 정리하며

사전 생성은 잠재 분포와 디코더를 사용하며 복원·보간·축 탐색과 구분한다. 체크포인트 구조와 조건 인수 계약을 먼저 확인하고 현재 스크립트의 빠진 보조 함수·인수 불일치를 이해해야 한다. 온도·truncation·표본 선택을 밝히고 품질과 다양성을 따로 평가한다.
