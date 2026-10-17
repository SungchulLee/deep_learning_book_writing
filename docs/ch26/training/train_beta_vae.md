# β-VAE의 학습과 비교

β-VAE 실험에서는 β마다 다른 훈련 로스를 최소화하므로 전체 로스만으로 설정을 비교하면 안 된다. 재구성·표준 KL·표준 ELBO를 같은 단위로 기록하고 생성이나 표현 분리 등 목적별 지표를 추가한다. 이 절은 제공된 스크립트의 동작과 한계를 정리한다.

## 1. 학습 스크립트

```python
"""
β-변분 오토인코더(베타 VAE) 익히기 각본
β 파라메터로 얽힘 풀린 나타냄을 배운다
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================

import sys
sys.path.append('..')
from models.beta_vae import BetaVAE, ConvBetaVAE
from utils.losses import beta_vae_loss
from utils.visualization import (
    visualize_reconstruction,
    visualize_samples,
    visualize_latent_traversal
)


def train_epoch(model, train_loader, optimizer, device):
    """한 에포크 동안 학습한다"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        
        # 온전히 이어진 β-변분 오토인코더를 위해 자료를 펼친다
        if isinstance(model, BetaVAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # 순전파
        reconstruction, mu, logvar = model(data_input)
        
        # 로스를 계산한다
        if isinstance(model, BetaVAE):
            target = data.view(data.size(0), -1)
        else:
            target = data
        
        loss, recon_loss, kl_loss = model.loss_function(reconstruction, target, mu, logvar)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 지표를 추적한다
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
        
        # 진행 막대를 고친다
        pbar.set_postfix({
            'loss': loss.item() / data.size(0),
            'recon': recon_loss.item() / data.size(0),
            'kl': kl_loss.item() / data.size(0)
        })
    
    num_samples = len(train_loader.dataset)
    return train_loss / num_samples, train_recon / num_samples, train_kl / num_samples


def test_epoch(model, test_loader, device):
    """시험 배치로 밸류한다"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 온전히 이어진 β-변분 오토인코더를 위해 자료를 펼친다
            if isinstance(model, BetaVAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # 순전파
            reconstruction, mu, logvar = model(data_input)
            
            # 로스를 계산한다
            if isinstance(model, BetaVAE):
                target = data.view(data.size(0), -1)
            else:
                target = data
            
            loss, recon_loss, kl_loss = model.loss_function(reconstruction, target, mu, logvar)
            
            test_loss += loss.item()
            test_recon += recon_loss.item()
            test_kl += kl_loss.item()
    
    num_samples = len(test_loader.dataset)
    return test_loss / num_samples, test_recon / num_samples, test_kl / num_samples


def main(args):
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # MNIST 데이터셋 불러오기
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 모델 생성
    if args.model_type == 'fc':
        model = BetaVAE(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta=args.beta
        )
    elif args.model_type == 'conv':
        model = ConvBetaVAE(
            latent_dim=args.latent_dim,
            beta=args.beta,
            img_channels=1
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta (disentanglement weight): {args.beta}")
    print(f"  β=1: Standard VAE")
    print(f"  β>1: More disentangled (β=4-10 typical)")
    
    # 최적화기
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 루프
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 학습
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
        
        # 시험
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device)
        
        # 통계를 찍는다
        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
        
        # 최고 성능 모델 저장
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'beta': args.beta,
            }, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    
    # 가장 좋은 모델을 불러와 그려 본다
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating visualizations...")
    
    # 여느 그림
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=False)
    visualize_samples(model, args.latent_dim, num_samples=10, device=device)
    
    # β-변분 오토인코더 전용: 얽힘 풀림을 보이는 숨은 훑기
    print("\nGenerating latent dimension traversals (disentanglement visualization)...")
    num_dims_to_traverse = min(10, args.latent_dim)
    for dim_idx in range(num_dims_to_traverse):
        print(f"  Traversing dimension {dim_idx}...")
        visualize_latent_traversal(model, dim_idx=dim_idx, num_steps=10, range_limit=3.0, device=device)
    
    print("\nTraining complete!")
    print(f"\nInterpretation tip:")
    print(f"  Look at the latent traversal images to see what each dimension encodes.")
    print(f"  With β={args.beta}, dimensions should encode independent factors of variation")
    print(f"  (e.g., one dimension for rotation, another for thickness, etc.)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train β-VAE on MNIST for disentangled representations')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, default='fc', choices=['fc', 'conv'],
                        help='Type of β-VAE (fc or conv)')
    parser.add_argument('--latent-dim', type=int, default=10,
                        help='Latent dimension (smaller for better disentanglement visualization)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (for FC β-VAE)')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta parameter (1.0=standard VAE, 4-10=disentangled)')
    
    # 익히기 인자
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # 체크포인트
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/beta_vae_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # 체크포인트 디렉터리를 만든다
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
```

기본값은 완전 연결 모델, 잠재 10, 은닉 256, β=4, 배치 128, Adam 러닝레이트 0.001, 20에포크다. 입력에는 ToTensor를 적용하며 완전 연결 모델은 이미지를 펼친다. 로스는 모델의 loss_function을 호출하고 에포크 기록은 배치 합을 누적해 데이터셋 크기로 나눈 표본 평균이다. beta_vae_loss도 import하지만 실제 루프에서는 사용하지 않는다.

## 2. 현재 코드의 연결 상태

현재 로컬 utils.visualization에는 visualize_latent_traversal이 정의되어 있지 않다. 따라서 표시된 import는 그대로라면 학습 시작 전에 실패한다. 본문의 코드는 보존했으며 이후 설명은 필요한 보조 함수가 연결되었다는 전제의 학습 흐름이다. 구현되지 않은 함수를 호출한다고 잠재 축 그림이 자동으로 만들어지는 것은 아니다.

로컬 ConvBetaVAE는 ConvVAE를 상속하고 self.beta만 저장한다. 그런데 상속한 loss_function의 기본 인수는 beta=1이고 이 학습 루프는 β를 전달하지 않는다. 따라서 이 컨볼루션 경로에서는 명령행 β가 저장·출력되어도 실제 로스는 기본 β=1로 계산된다. 완전 연결 BetaVAE는 self.beta를 로스에 사용한다. 두 경로가 같은 방식으로 β를 적용한다고 해석하면 안 된다.

## 3. 평가·선택·재현 조건

완전 연결 경로에서 출력하는 Loss는 R+βK이며 β가 다른 실험의 값은 직접적인 우열 기준이 아니다. 공통 지표로 R+K를 기록하고 관측 모형과 합산 축을 맞춘다. eval·no_grad는 잠재 표본 추출을 중단하지 않아 평가 변동이 남는다. 재구성·생성 그림과 잠재 축 탐색은 서로 다른 정보를 준다.

코드는 테스트 로스로 최적 에포크를 고른다. 독립적인 최종 성능을 보고하려면 별도 검증셋으로 선택해야 한다. 체크포인트에는 β를 저장하지만 모델 구조·전처리 전체는 포함하지 않는다. state_dict 복원은 일반 속성 self.beta를 자동 복원하지 않으므로 실제 설정도 다시 적용해야 한다.

출력 안내문의 “β>1이면 더 분리된 표현”이나 “각 차원은 독립 요인을 담아야 한다”는 문장은 이론적 보장이 아니다. [β-VAE 모듈](../architecture/beta_vae.md)에서 설명한 것처럼 의미 요인 분리는 별도 가정과 검증을 요구한다. 이 페이지는 β 탐색 실험의 결과를 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
매 배치 step을 유지하면서 zero_grad를 없애면 왜 의도와 달라지는가?

</div>

??? success "풀이"

    이전 배치의 그래디언트가 다음 backward에 더해지고 이전 파라미터 위치의 그래디언트까지 누적 업데이트에 사용된다. Adam의 모멘트에도 이 누적값이 들어가므로 단순한 러닝레이트 배수로 설명할 수 없다. 올바른 배치 학습에서는 각 step의 그래디언트를 초기화한다. 의도한 누적은 여러 배치의 적절히 정규화한 그래디언트로 한 번 step하는 별도 설계다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문에서 Adam을 대안 옵티마이저와 비교하려면 무엇을 바꿔야 하는가?

</div>

??? success "풀이"

    기본이 이미 Adam 0.001이므로 Adam으로 교체하라는 기존 질문은 변화가 없다. 대안의 종류·러닝레이트·튜닝 예산을 정하고 같은 구조·β·자료 분할에서 비교한다. 동일 초기 상태와 여러 시드의 결과를 사용하고 반복 횟수뿐 아니라 시간·검증 목적도 확인한다. 구현 연결 오류를 먼저 해결해야 성능 비교에 의미가 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
로스가 예상과 다를 때 가장 먼저 확인할 설정은 무엇인가?

</div>

??? success "풀이"

    총로스가 실제 R+βK와 맞는지 확인해 β 전달 누락을 찾는다. 이 스크립트의 컨볼루션 경로가 구체적인 예다. 그다음 합산 기준·입력 전처리·출력 모양·비유한 값을 확인한다. 낮은 차원별 KL은 붕괴 가능성을 시사하지만 항상 오류는 아니며 코드 의존성과 검증 목적을 함께 본다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료 예시의 웨이트 복사와 종료 처리를 검토하라.

</div>

??? success "풀이"

    아래 state_dict().copy는 얕은 복사라 최적 시점 텐서가 고정되지 않을 수 있다. 독립 복제나 파일 저장이 필요하다. evaluate·val_loader·학습 단계는 생략되어 있으므로 별도로 연결해야 하고, 인내 횟수에 도달하지 않고 정상 종료할 때도 최적 상태를 복원하도록 한다. 선택 지표는 β별 목적과 공통 평가 목적을 구분하고 테스트셋 대신 검증셋을 사용한다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
β 탐색의 로그에 무엇을 기록해야 하는가?

</div>

??? success "풀이"

    실제 적용한 β, 재구성·수정 없는 KL·표준 ELBO, 차원별 KL과 사용 차원 임계, 생성·표현 평가 지표, 학습 곡선과 설정·시드를 기록한다. 아래 alive는 KL>0.01이라는 임의 임계의 개수이며 내재 차원이나 의미 요인의 수가 아니다. rec·kl·kl_dim은 동일한 배치 정규화 기준으로 계산해야 한다. model 설정과 명령행 출력이 다른 경우도 로그가 감지해야 한다.

    ```python
    log = {'beta': beta, 'rec': rec.item(), 'kl': kl.item(),
           'kl_per_dim': kl_dim.tolist(), 'alive': int((kl_dim > 0.01).sum())}
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
여러 β의 비교에서 어떤 조건을 통제할 수 있는가?

</div>

??? success "풀이"

    자료 분할·전처리·구조·잠재 차원·로스 축·평가 표본 수·평가기를 맞춘다. 초기 웨이트·자료 순서·생성용 잠재 표본을 공유하면 일부 변동을 줄일 수 있지만 같은 시드만으로 모든 환경의 결과가 같아지는 것은 아니다. 여러 초기화의 변동도 보고한다. 고정 학습 예산 비교와 동일 튜닝 예산 비교는 별도로 설계한다. 본문에는 재현 시드 설정이나 실제 β 탐색 결과가 없다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
재구성·KL 산점도에서 파레토 경계를 해석할 때 주의할 점은 무엇인가?

</div>

??? success "풀이"

    각 설정의 검증 재구성과 KL을 점으로 표시하고 β·목적별 품질을 색이나 표식으로 덧붙일 수 있다. 하지만 유한 학습 결과의 모든 점이 파레토 최적이라고 할 수 없다. 두 비용 모두 다른 점보다 나쁜 점은 지배된 점일 수 있다. 또한 사용 차원 감소가 품질 저하의 유일한 원인이라고 단정하지 않는다. 학습 변동과 지표의 정의·단위를 함께 제시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
β를 바꾸면 러닝레이트를 정확히 반비례 조절해야 하는가?

</div>

??? success "풀이"

    아니다. 그래디언트는 ∇R+β∇K이므로 β 변화는 전체 그래디언트의 상수 배율이 아니라 방향·항별 균형까지 바꾼다. SGD에서도 β를 32배 바꿨다고 전체 스텝이 32배 되는 것은 아니다. Adam 역시 이런 변화를 자동으로 상쇄하지 않는다. 같은 러닝레이트가 충분할 수도 있지만 안정성·수렴과 검증 성능으로 확인하며 필요하면 동일 예산으로 조절한다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
다른 데이터셋에 β를 옮길 때 어떤 정규화 차이를 확인해야 하는가?

</div>

??? success "풀이"

    픽셀 수·관측 모형·전처리·잠재 차원·두 항의 합산 축을 확인한다. 더 많은 픽셀이 재구성 합을 키울 수 있어도 데이터 내용과 단위가 다르므로 MNIST에서 CIFAR로 가면 실효 β가 정확히 4분의 1이라고 말할 수 없다. KL도 잠재 차원에 정확히 비례하지 않는다. R를 D개 픽셀로, K를 d개 잠재 차원으로 나눈 목표 R/D+β_norm K/d는 원래 단위로 R+β_norm(D/d)K와 대응한다. 이 정의에서는 β_norm=1이 일반적으로 표준 ELBO가 아니다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
매우 작은 β에서 예상 가능한 경향과 보장할 수 없는 결과를 구분하라.

</div>

??? success "풀이"

    KL 압력이 약해져 더 많은 정보와 작은 사후 분산을 사용하는 해를 선호할 수 있다. 그러나 유한 학습의 재구성이 언제나 개선되거나 분산이 반드시 0으로 가거나 특정 생성 실패율에 도달한다는 보장은 없다. 확률적 인코더·사전·디코더가 남으므로 자동으로 생성 모델이 아니게 되는 것도 아니다. 재구성만으로 β를 고르면 다른 목적을 놓칠 수 있어 생성·우도·표현 지표를 목적에 맞게 별도 평가한다.



## 정리하며

β 탐색은 실제 로스에 적용된 웨이트부터 확인해야 한다. 이 스크립트에는 빠진 시각화 import와 컨볼루션 경로의 β 전달 문제가 있다. 코드의 연결 상태를 이해한 뒤 공통 ELBO·생성·표현 평가와 검증셋으로 설정을 비교한다.
