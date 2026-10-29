# 조건부 VAE의 학습과 평가

조건부 VAE 학습은 관측 이미지와 대응 레이블을 함께 전달한다. 로스의 재구성·KL 계산은 무조건부와 같은 형태를 사용할 수 있지만 근사 사후와 디코더가 조건을 받는다는 해석이 달라진다. 레이블의 모양과 각 메서드의 입력 계약을 맞추는 것이 중요하다.

## 1. 학습 구현

```python
"""
조건부 변분 오토인코더 익히기 각본
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
from models.conditional_vae import ConditionalVAE
from models.conv_cvae import ConvConditionalVAE
from utils.losses import vae_loss
from utils.visualization import visualize_reconstruction, visualize_samples


def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """한 에포크 동안 학습한다"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, labels in pbar:
        data = data.to(device)
        labels = labels.to(device)
        
        # 온전히 이어진 조건부 변분 오토인코더를 위해 자료를 펼친다
        if isinstance(model, ConditionalVAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # 순전파
        reconstruction, mu, logvar = model(data_input, labels)
        
        # 로스를 계산한다
        if isinstance(model, ConditionalVAE):
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
    """시험 배치로 밸류한다"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            # 온전히 이어진 조건부 변분 오토인코더를 위해 자료를 펼친다
            if isinstance(model, ConditionalVAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # 순전파
            reconstruction, mu, logvar = model(data_input, labels)
            
            # 로스를 계산한다
            if isinstance(model, ConditionalVAE):
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
        model = ConditionalVAE(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_classes=args.num_classes
        )
    elif args.model_type == 'conv':
        model = ConvConditionalVAE(
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            img_channels=1,
            img_size=28
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Number of classes: {args.num_classes}")
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
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=True)
    
    # 클래스마다 표본을 만든다
    print("Generating conditional samples...")
    for class_label in range(min(10, args.num_classes)):
        visualize_samples(model, args.latent_dim, num_samples=10, device=device, class_label=class_label)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Conditional VAE on MNIST')
    
    # 모델 인자
    parser.add_argument('--model-type', type=str, default='fc', choices=['fc', 'conv'],
                        help='Type of cVAE (fc or conv)')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (for FC cVAE)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes')
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
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/cvae_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # 체크포인트 디렉터리를 만든다
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
```

기본값은 완전 연결 모델, 잠재 32, 은닉 256, 클래스 10, β=1, 배치 128, Adam 러닝레이트 0.001과 20에포크다. MNIST 이미지는 ToTensor로 0~1 범위로 변환하고, 완전 연결 모델만 `(B,784)`로 펼친다. 레이블은 정수 인덱스 그대로 모델의 forward에 전달한다. num_classes만 바꾸어 데이터의 실제 레이블 범위가 바뀌는 것은 아니다.

가져온 vae_loss는 기본적으로 배치 전체의 BCE와 KL 합을 계산한다. 매 배치 이 합으로 역전파하고 에포크 기록은 데이터셋 표본 수로 나눈다. 따라서 재구성 기록은 픽셀당 평균이 아닌 표본당 픽셀 BCE 합이다. 조건에 무관한 표준정규 사전을 사용하므로 별도 조건부 사전 네트워크는 없다.

## 2. 평가·선택과 생성 호출

test_epoch는 eval·no_grad를 적용하지만 잠재 표본을 계속 뽑으므로 재구성 평가에는 변동이 남는다. 테스트 로스로 에포크를 선택하는 현재 구조는 독립 최종 평가와 구분해야 하며 실제 성능 보고에는 검증 분할이 필요하다. 저장되는 웨이트·옵티마이저 상태·에포크·로스 외에 구조, 클래스 매핑, 전처리·β도 기록해야 복원이 명확하다.

완전 연결 ConditionalVAE의 forward는 정수 레이블을 원핫으로 바꾸지만 decode는 원핫을 직접 받는다. 반면 ConvConditionalVAE의 decode는 정수 레이블을 받는다. 현재 visualize_samples는 정수 레이블을 바로 decode에 전달하므로 완전 연결 모델과 호환되지 않는다. 따라서 기본 학습 경로가 완료되어도 조건부 생성 시각화가 그대로 성공한다고 할 수 없다. 모델별 변환이나 맞는 sample 호출이 필요하지만 이 절에서는 코드를 보존한다.

시각화 함수는 그림 객체를 반환하며 이 호출부에는 저장 경로나 명시적 표시가 없다. 클래스별 생성 그림의 파일 저장·품질 평가는 별도다. 이 스크립트에는 새 성능 측정 결과가 포함되어 있지 않다.

## 3. 조건 제어의 진단

조건을 제대로 전달하는지 모양·인덱스·원핫 변환을 확인한 뒤, 동일 z에서 클래스만 바꾸는 비교와 클래스별 생성 평가를 수행한다. 지정 클래스 일치율, 클래스 내 다양성, 재구성·KL은 서로 다른 지표다. KL이 무조건부보다 작아야만 올바른 학습이라는 기준은 없다.

클래스가 불균형해도 생성 시 조건 분포 p(y)는 사용 목적에 맞춰 별도로 선택할 수 있다. 조건별 모델의 품질과 전체 생성 클래스 비율을 구분하면 학습 샘플링·로스 웨이트·추론 조건 분포를 각각 설계할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
그래디언트 초기화를 생략한 학습과 의도한 누적 학습의 차이를 설명하라.

</div>

??? success "풀이"

    본문은 매 배치 step을 하므로 backward 전에 기존 그래디언트를 비워야 한다. 생략하면 이전 배치의 그래디언트가 반복해서 더해져 의도와 다른 업데이트를 한다. 이는 단계 수에 비례한 단순 러닝레이트 조정과 같지 않다. 누적 학습은 여러 배치의 정규화한 그래디언트를 모아 한 번 step하고 초기화하는 방식이며 레이블 정렬과 마지막 구간의 스케일도 유지해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 옵티마이저 설정을 확인하고 비교 실험의 조건을 설명하라.

</div>

??? success "풀이"

    기본은 이미 Adam 0.001이다. 대안을 비교하려면 구조·β·자료 분할·전처리를 맞추고 각 옵티마이저에 동일한 튜닝 예산을 준다. 여러 초기화의 검증 재구성·공통 ELBO·조건 일치율·다양성과 비용을 보고한다. 단일 곡선이나 같은 러닝레이트 하나만으로 보편적인 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
조건 전달 오류와 조건 생성 실패를 어떻게 구분하는가?

</div>

??? success "풀이"

    먼저 레이블 dtype·범위·배치 순서와 forward·decode의 서로 다른 계약을 확인한다. 계약을 지켜도 학습 부족·모형 한계·조건 무시로 지정 클래스가 잘 생성되지 않을 수 있다. 고정 z에서 조건 변경과 클래스별 평가가 진단에 도움이 되지만 단일 실패 패턴으로 원인을 확정하지 않는다. 입력 범위·비유한 로그분산 등 수치 문제도 별도로 점검한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 예시를 완전한 최적 모델 선택으로 쓰려면 무엇이 필요한가?

</div>

??? success "풀이"

    state_dict().copy는 텐서를 독립 복제하지 않는 얕은 복사이므로 최적 웨이트가 뒤 학습으로 변할 수 있다. 독립 복제나 파일 체크포인트가 필요하다. evaluate·val_loader와 실제 학습 단계도 연결해야 한다. 조기 종료뿐 아니라 정상 종료에서도 최적 상태를 복원하고 조건별 평가 지표·선택 규칙을 미리 정한다. 반복 선택에는 테스트셋 대신 검증셋을 쓴다.

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
무조건부 루프에서 조건부 루프로 바뀌는 입력과 로스의 의미를 설명하라.

</div>

??? success "풀이"

    레이블을 버리지 않고 이미지와 같은 장치로 옮겨 forward에 전달한다. 완전 연결 모델은 정수 또는 원핫 조건을 받을 수 있지만 컨볼루션 모델은 정수 레이블을 요구한다. 아래 골격의 원핫 c는 완전 연결 모델용이며 이미지를 펼치는 처리·장치 이동·최적화는 생략되어 있다. 표준정규 사전을 쓰면 KL 수식 형태는 같아도 대상은 q(z|x,y)다. 조건을 아예 인수로 주지 않으면 이 클래스들은 오류를 내므로 모든 누락이 조용히 무시되는 것은 아니다.

    ```python
    for x, y in loader:                       # 1. 표지를 함께 꺼낸다
        c = F.one_hot(y, 10).float()
        out, mu, logvar = model(x, c)         # 2. 조건을 넘긴다
        loss = recon(out, x) + kl(mu, logvar) # 로스는 그대로
    ```


---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
KL 크기나 클래스 일치율만으로 조건 누락을 판정할 수 있는가?

</div>

??? success "풀이"

    없다. KL에는 조건부 입력 정보와 집계 사후 불일치가 함께 들어가므로 특정 값에 머물렀다는 사실이 조건 누락의 증명은 아니다. 잘못 생성되는 것도 버그뿐 아니라 데이터·학습·평가기 문제 때문일 수 있다. 조건 벡터·중간값·장치·클래스 매핑을 직접 확인하고 고정 잠재 표현의 조건 개입 결과를 함께 본다. 10% 같은 기준도 균형 10클래스라는 전제가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
학습 조건과 생성 조건은 어떻게 정하는가?

</div>

??? success "풀이"

    학습에는 관측과 대응하는 레이블을 사용한다. 생성에는 지정 클래스나 선택한 p(y)에서 뽑은 클래스를 사용한다. 아래 원핫 예시들은 개념적 조건 구성이며 마지막 줄에는 float 변환이 없다. 실제로 forward와 decode 중 어디에 전달하는지에 따라 정수·원핫 계약을 맞춰야 한다. 균등 조건 생성은 데이터의 클래스 빈도와 다를 수 있다. 혼합 원핫도 허용하는 디코더라면 수학적으로 조건부 출력은 정의되지만 학습 분포 밖에서 의미 있는 결과가 나올지는 별도 문제다.

    ```python
    # 익히기: 자료에서 온다
    c = F.one_hot(y, 10).float()

    # 뽑기: 우리가 만든다
    c = F.one_hot(torch.full((n,), 7), 10).float()   # 7을 n개
    c = F.one_hot(torch.randint(0, 10, (n,)), 10)    # 고루 섞어서
    ```


---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
클래스 불균형에 따른 학습과 생성 분포를 구분하라.

</div>

??? success "풀이"

    희소 클래스는 학습 정보가 적을 수 있으나 품질이 항상 빈도 순서로 정해지는 것은 아니다. 클래스별 표본 수·자료 특성·재구성·생성 지표를 함께 보고, 빈도와 품질의 상관만으로 인과를 확정하지 않는다. 재표집이나 가중 로스는 학습 목적을 바꾸며 효과를 검증해야 한다. 조건부 모델은 학습 비중과 별도로 생성 시 p(y)를 선택할 수 있으므로 희소 클래스의 생성 빈도와 그 클래스의 조건부 품질은 다른 문제다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
분류기 기반 생성 평가의 한계를 설명하라.

</div>

??? success "풀이"

    높은 확신도나 지정 클래스 판정은 사람이 보는 품질, 전체 다양성 또는 학습 분포 적합성을 보장하지 않는다. 같은 원형을 반복해도 일치율이 높을 수 있고 분포 밖 이미지에 확신할 수도 있다. 학습·평가 자료와 평가기 바이어스도 확인한다. 잠재 z에 T를 곱하면 사전은 N(0,T²I)로 바뀐다. 모든 양의 T의 가우시안은 같은 지지집합을 가지므로 “거의 겹치지 않는다”는 표현에는 차원과 겹침 척도가 필요하다. 변경된 샘플링 폴리시와 평가 지표를 밝히고 여러 품질·다양성 진단을 함께 사용한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
조건부 모델을 클래스별로 어떤 기준에서 살펴야 하는가?

</div>

??? success "풀이"

    재구성·KL, 지정 클래스 일치율과 품질·다양성을 클래스별로 기록하고 표본 수와 변동도 제시한다. 클래스 평균만 보면 취약한 그룹을 놓칠 수 있다. 이미지 밝기·전처리 차이도 로스에 영향을 주지만 정규화가 항상 정답은 아니다. 표본 간 평균 거리는 노이즈만 늘려도 커질 수 있어 다양성의 충분한 지표가 아니다. 여러 진단을 결합하고 특정 패턴에 유일한 원인을 대응시키지 않는다.



## 정리하며

조건부 학습은 이미지·레이블의 대응과 메서드별 입력 계약을 유지해야 한다. 본문의 로스는 표본당 합산 기준이며 생성 보조 함수에는 완전 연결 decode와의 계약 차이가 있다. 검증셋에서 조건 제어·다양성·재구성을 따로 확인한다.
