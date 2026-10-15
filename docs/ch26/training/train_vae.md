# VAE 학습 절차

이 학습 스크립트는 MNIST에서 완전 연결 VAE 또는 합성곱 VAE를 선택하고 재구성·KL 손실을 함께 최적화한다. 핵심은 모델별 입력 모양, 손실의 합산 기준과 평가·체크포인트 선택 규칙을 일치시키는 것이다.

## 1. 학습과 평가 구현

```python
"""
여느 변분 오토인코더 익히기 각본
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
from models.vae import VAE
from models.conv_vae import ConvVAE
from utils.losses import vae_loss
from utils.visualization import visualize_reconstruction, visualize_samples


def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """한 에포크 동안 학습한다"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        
        # 온전히 이어진 변분 오토인코더를 위해 자료를 펼친다
        if isinstance(model, VAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # 순전파
        reconstruction, mu, logvar = model(data_input)
        
        # 손실을 계산한다
        if isinstance(model, VAE):
            target = data.view(data.size(0), -1)
        else:
            target = data
        
        loss, recon_loss, kl_loss = vae_loss(reconstruction, target, mu, logvar, beta)
        
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


def test_epoch(model, test_loader, device, beta=1.0):
    """시험 배치로 값매김한다"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 온전히 이어진 변분 오토인코더를 위해 자료를 펼친다
            if isinstance(model, VAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # 순전파
            reconstruction, mu, logvar = model(data_input)
            
            # 손실을 계산한다
            if isinstance(model, VAE):
                target = data.view(data.size(0), -1)
            else:
                target = data
            
            loss, recon_loss, kl_loss = vae_loss(reconstruction, target, mu, logvar, beta)
            
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
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 모델 생성
    if args.model_type == 'fc':
        model = VAE(input_dim=784, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'conv':
        model = ConvVAE(latent_dim=args.latent_dim, img_channels=1, img_size=28)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta: {args.beta}")
    
    # 최적화기
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 루프
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 학습
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args.beta)
        
        # 시험
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device, args.beta)
        
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
            }, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    
    # 가장 좋은 모델을 불러와 그려 본다
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating visualizations...")
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=False)
    visualize_samples(model, args.latent_dim, num_samples=10, device=device)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, default='fc', choices=['fc', 'conv'],
                        help='Type of VAE (fc or conv)')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (for FC VAE)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for KL weight')
    
    # 익히기 인자
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # 체크포인트
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/vae_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # 체크포인트 디렉터리를 만든다
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
```

모델은 이 디렉터리의 models 모듈에서, 손실과 시각화는 utils에서 가져온다. 모듈 검색 경로와 실행 작업 디렉터리에 따라 import·상대 데이터 경로가 달라질 수 있다. 기본 장치는 CUDA 사용 가능 시 CUDA, 아니면 CPU다. 학습 데이터는 다운로드를 요청하지만 테스트 데이터 생성에는 download=True를 지정하지 않으므로 필요한 파일이 준비되어 있어야 한다.

기본 설정은 잠재 32, 완전 연결 은닉 256, 배치 128, Adam 학습률 0.001, β=1, 20에포크다. ToTensor로 입력을 0~1 범위로 바꾸고 별도 표준화나 이진화는 하지 않는다. 완전 연결 VAE만 `(B,784)`로 펼치며 합성곱 모델은 `(B,1,28,28)`을 받는다. isinstance 분기는 이 두 구체적 클래스에 맞춘 방식이지 모든 VAE 구조에 대한 일반적인 입력 판별기는 아니다.

## 2. 손실 기록의 단위

가져온 vae_loss는 기본적으로 픽셀·잠재 차원·배치에 대해 합산한 BCE와 KL을 반환한다. 학습 루프는 배치 손실 합으로 역전파하고, 기록에서는 합을 누적한 뒤 데이터셋 표본 수로 나눈다. 따라서 출력은 표본당 픽셀 BCE 합과 표본당 잠재 KL 합이다. 픽셀별 평균 BCE와 같은 수치가 아니다.

현재 로더는 전체 자료를 한 번씩 사용하므로 len(dataset)이 분모와 맞는다. 샘플러·drop_last 등을 바꾸면 실제 처리한 표본 수를 세어야 한다. 배치 합으로 역전파하므로 배치 크기를 바꾸면 기울기 스케일도 달라진다. β가 1이 아닌 기록의 Loss는 표준 음의 ELBO가 아니며 비교용 표준 ELBO를 따로 기록할 수 있다.

## 3. 평가와 체크포인트의 해석

test_epoch는 eval과 no_grad를 적용한다. BatchNorm 통계 사용과 기울기 기록은 바뀌지만 두 모델의 재매개변수화는 계속 표본을 뽑으므로 평가 재구성에는 몬테카를로 변동이 남는다. train 지표는 학습 도중 서로 다른 가중치에서 누적한 값이므로 에포크 종료 모델의 train 평가와도 다르다.

코드는 테스트 손실로 최적 에포크를 선택한다. 독립적인 최종 테스트 성능을 보고하려면 학습 자료에서 검증셋을 따로 만들고 선택에 사용해야 한다. 현재 저장에는 가중치·옵티마이저 상태·에포크·손실만 포함된다. 다른 환경에서 복원하려면 구조 설정, 전처리, β와 합산 기준 등도 별도로 기록해야 한다.

저장된 최적 가중치를 불러온 뒤 시각화 보조 함수를 호출한다. 그림을 반환하는 코드이며 호출부에는 저장 경로나 명시적인 표시 호출이 없다. 생성 그림의 파일 저장까지 자동으로 완료된다고 설명할 수 없다. 이 절은 코드의 동작을 설명하며 새 학습 결과를 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
zero_grad를 제거하면 어떤 일이 일어나는가? 의도한 기울기 누적과 구분하라.

</div>

??? success "연습문제 1 풀이"

    backward는 기존 grad에 새 기울기를 더한다. 매 배치 step을 수행하면서 초기화하지 않으면 이전 파라미터 위치에서 계산한 기울기까지 재사용하므로 원래 최적화와 달라진다. 이것이 단순히 학습률에 단계 수를 곱한 것과 같거나 반드시 발산하는 것은 아니다. 의도한 누적은 여러 미니배치의 손실을 적절히 스케일해 backward한 뒤 한 번 step하고 초기화하는 방식이다. 누적 구간의 마지막 불완전 배치도 처리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문 옵티마이저를 확인하고 대안과 비교하는 절차를 설계하라.

</div>

??? success "연습문제 2 풀이"

    본문은 이미 Adam과 기본 학습률 0.001을 사용한다. 따라서 “Adam으로 변경”은 새로운 실험이 아니다. SGD 등과 비교하려면 같은 초기 가중치·데이터 분할·평가 지표를 사용하고 각 옵티마이저에 같은 튜닝 예산을 준다. 데이터 순서와 잠재 표본의 변동도 관리하며 여러 초기화의 검증 성능·비용을 비교한다. 같은 학습률 하나만으로 옵티마이저의 일반적인 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사후 붕괴와 비유한 손실을 구분해 진단하라.

</div>

??? success "연습문제 3 풀이"

    붕괴 가능성은 입력별·차원별 KL, 평균의 변동과 디코더의 코드 의존성으로 확인한다. 낮은 KL 차원이 있다는 사실만으로 학습 실패인 것은 아니다. NaN·Inf는 입력 범위, logvar.exp의 크기, 출력·기울기와 학습률 등을 점검한다. 클리핑·초기화·정밀도 조정이나 KL 일정은 원인에 맞춰 선택해야 하며 모든 문제에 공통으로 효과적인 처방은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 예시에서 최적 가중치 보존이 불완전한 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    state_dict().copy()는 사전의 얕은 복사라 텐서 저장소가 학습 중 값과 공유될 수 있다. 독립 복제나 파일 저장이 필요하다. 예시는 evaluate와 val_loader가 정의되어 있어야 하고 실제 학습 단계도 생략되어 있다. 조기 종료 조건이 발생할 때만 복원하므로 정상 종료 때도 최적 상태를 복원하도록 설계해야 한다. 체크포인트 선택에는 검증 자료를 사용하고 최종 테스트 자료는 분리한다.

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
재구성·KL·β를 별도로 기록해야 하는 이유는 무엇인가?

</div>

??? success "연습문제 5 풀이"

    전체 손실이 내려가도 어느 항이 개선되었는지 알 수 없고 β 일정이 바뀌면 같은 모델에서도 숫자가 달라진다. 아래 기록은 rec_sum·kl_sum이 합이고 n이 처리 표본 수라는 전제에서 표본당 항을 표시한다. KL 감소는 정보 비용 감소나 집계 사후 일치 등 여러 원인이 있고 증가도 반드시 더 유용한 정보 획득을 뜻하지 않는다. 차원별 KL과 코드 개입·재구성·생성 평가를 함께 사용한다.

    ```python
    print(f"epoch {ep}  rec {rec_sum/n:.2f}  kl {kl_sum/n:.2f}")
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
재구성 그림과 사전 생성 그림의 차이를 설명하라.

</div>

??? success "연습문제 6 풀이"

    재구성은 관측에서 추론한 q(z|x)를 거쳐 출력을 만들고 사전 생성은 p(z)에서 직접 뽑는다. 두 경로가 방문하는 잠재 영역과 평가하는 능력이 달라 좋은 재구성이 좋은 생성의 충분조건은 아니다. 차이는 사전 불일치뿐 아니라 디코더·관측 모형·학습 부족 등에도 영향을 받는다. 픽셀 확률 이미지를 보는 것과 관측 픽셀을 샘플링하는 것도 구분한다. 특정 확신도 문턱 비율을 실패 확률로 해석하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
언제 멈출지 결정하는 지표를 목적별로 설명하라.

</div>

??? success "연습문제 7 풀이"

    우도 목적에는 같은 관측 모형의 검증 표준 ELBO나 적절한 중요도 가중 평가를, 생성에는 검증용 품질·다양성 지표를, 특징 재사용에는 별도 검증 작업의 성능을 사용할 수 있다. 각 지표에는 변동과 편향이 있으므로 정한 평가 예산·인내 횟수로 선택한다. 병목이 작아도 과적합이 불가능하지 않고 낮은 KL 차원도 반드시 최적화 실패는 아니다. 최종 테스트셋을 선택 과정에 반복 사용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
KL warm-up을 본문 루프에 연결할 때 주의할 점은 무엇인가?

</div>

??? success "연습문제 8 풀이"

    에포크나 스텝별 현재 β를 계산해 train_epoch에 전달하고 일정·목표값을 기록한다. 아래는 개념적 골격이며 loader의 `(x,)`는 MNIST의 `(data,label)` 반환과 다르다. rec·kl과 optimizer 단계도 정의·연결해야 한다. warmup_epochs는 양수여야 하고 0에서 시작하는 ep의 종료 범위에 따라 마지막 값이 목표에 도달하는지 확인한다. β가 변하는 훈련 손실 대신 고정 β=1의 검증 지표를 별도로 기록하면 시점 간 비교가 쉽다. warm-up의 필요성과 효과는 진단해야 하며 일정한 권장 비율을 보장할 수 없다.

    ```python
    for ep in range(epochs):
        beta = min(1.0, ep / warmup_epochs)        # 0 -> 1
        for x, in loader:
            out, mu, logvar = model(x)
            loss = rec(out, x) + beta * kl(mu, logvar)
    ```


---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
배치 크기·학습률 선택에서 본문 구현의 어떤 특징을 고려해야 하는가?

</div>

??? success "연습문제 9 풀이"

    기본값은 배치 128과 Adam 학습률 0.001이며 보편적인 최적값은 아니다. 배치 합 손실, 마지막 배치 크기, 합성곱 모델의 BatchNorm과 잠재 샘플링 변동이 학습에 영향을 준다. 배치를 바꾸면 정규화와 스텝 수·시간 예산도 함께 고려한다. Adam이 임의의 손실 스케일 변화나 β 변경을 자동으로 상쇄하는 것은 아니다. 검증 기준과 계산비용으로 후보를 비교한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
재구성 곡선만으로 알 수 없는 실패와 추가 진단을 설명하라.

</div>

??? success "연습문제 10 풀이"

    사전 생성의 다양성 부족, 잠재 미사용, 집계 사후 불일치와 수치적으로 극단적인 분산은 재구성 곡선만으로 구분하기 어렵다. 차원별 KL·평균·로그분산 분포, 사전 표본과 코드 개입 결과를 함께 본다. σ→0이면 표준정규 사전에 대한 KL은 일반적으로 발산하지만 재구성이 반드시 단조 개선되는 것은 아니다. 생성 품질 저하만으로 prior hole의 크기를 직접 측정할 수도 없다. 지표를 진단 근거로 사용하되 단일 패턴에 유일한 원인을 붙이지 않는다.



## 정리하며

VAE 학습에서는 입력 모양과 배치 합 손실을 확인하고 재구성·KL·β를 함께 기록한다. 검증셋으로 모델을 선택하며 사전 생성도 별도로 평가한다. 본문의 테스트 기반 선택과 최소 체크포인트는 그 한계를 이해하고 사용해야 한다.
