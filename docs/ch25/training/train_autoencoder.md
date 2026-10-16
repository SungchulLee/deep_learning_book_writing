# 오토인코더 학습과 평가

이 스크립트는 MNIST를 읽고 `SimpleAutoencoder`를 학습하며 평가 로스가 가장 낮은 시점의 체크포인트를 저장한다. 마지막에는 저장한 모델을 다시 불러 복원 이미지를 표시한다. 일반 AE 학습이며 VAE의 분포 파라미터·KL·샘플링 과정은 포함하지 않는다.

## 1. 예제 코드

```python
"""
기본 오토인코더 익히기 각본
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
from models.autoencoder import SimpleAutoencoder
from utils.visualization import visualize_reconstruction


def train_epoch(model, train_loader, optimizer, device):
    """한 에포크 동안 학습한다"""
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        data_flat = data.view(data.size(0), -1)
        
        # 순전파
        reconstruction = model(data_flat)
        loss = model.loss_function(reconstruction, data_flat)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # 진행 막대를 고친다
        pbar.set_postfix({'loss': loss.item() / data.size(0)})
    
    return train_loss / len(train_loader.dataset)


def test_epoch(model, test_loader, device):
    """시험 배치로 밸류한다"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            data_flat = data.view(data.size(0), -1)
            
            reconstruction = model(data_flat)
            loss = model.loss_function(reconstruction, data_flat)
            test_loss += loss.item()
    
    return test_loss / len(test_loader.dataset)


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
    model = SimpleAutoencoder(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    )
    model = model.to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    
    # 최적화기
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 루프
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 시험
        test_loss = test_epoch(model, test_loader, device)
        
        # 통계를 찍는다
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss:  {test_loss:.4f}")
        
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
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Autoencoder on MNIST')
    
    # 모델 인자
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    
    # 익히기 인자
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # 체크포인트
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/autoencoder_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # 체크포인트 디렉터리를 만든다
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
```

## 2. 입력·모듈·실행 환경

입력은 `ToTensor`로 [0,1]에 맞춘 B×1×28×28 이미지이고 모델에는 B×784로 펼쳐 전달한다. 모델의 기본 은닉 너비는 256, 잠재 차원은 32다. 구조는 [오토인코더 모듈](../architecture/autoencoder.md)에서 설명하며 실제 import 대상은 이 디렉터리의 `models/autoencoder.py`다.

상대 경로 `sys.path.append('..')`는 현재 작업 디렉터리에 의존한다. 코드가 패키지와 도구를 찾을 수 있는 실행 환경을 준비해야 한다. 테스트 데이터셋에는 `download=True`가 없어 학습 파일만 존재하는 경우 테스트 파일이 자동으로 준비된다고 가정할 수 없다. 체크포인트 경로에 부모 디렉터리가 없는 파일명만 주면 `os.path.dirname`이 빈 문자열이 되는 것도 처리해야 한다.

## 3. 로스의 단위와 에포크 집계

가져온 모델은 복원 MSE를 `reduction='sum'`으로 반환한다. 한 배치의 모든 표본·픽셀 제곱 오차를 더하고 에포크에는 그 합을 데이터셋 표본 수로 나눈다. 따라서 출력 로스는 **표본당 784픽셀 제곱 오차 합의 평균**이다. 원소별 MSE로 읽으려면 784로 더 나눈다.

마지막 배치 크기가 작아도 배치 합을 합산하므로 기본 로더가 전체 데이터를 한 번씩 처리하는 조건에서는 표본 가중 에포크 평균이 맞는다. 학습에 사용하는 배치 로스 자체는 표본 수로 나누지 않아 배치 크기에 따라 그래디언트 척도가 달라진다. 로깅에서 나누는 것과 역전파 전에 나누는 것을 구분한다.

## 4. 학습·평가 모드

`train_epoch`는 `train()` 후 순전파·로스·그래디언트 초기화·역전파·Adam 갱신을 수행한다. `test_epoch`는 `eval()`과 `no_grad()`를 사용한다. 현재 모델의 Linear·ReLU·Sigmoid는 모드에 따라 계산이 바뀌지 않지만, 나중에 Dropout·BatchNorm을 추가하면 모드가 중요하다. `eval()`은 그래디언트 기록을 끄지 않으므로 두 기능을 구분한다.

그래디언트 초기화는 이전 갱신의 그래디언트가 새 그래디언트에 더해지는 것을 막는다. 의도한 미니배치 누적에서는 초기화·로스 평균·갱신 간격을 따로 설계한다. 누적이 언제나 단순히 러닝레이트를 단계 수만큼 늘리는 것과 같은 것은 아니다.

## 5. 체크포인트 선택과 데이터 분할

코드는 MNIST 테스트 로스로 매 에포크 최적 체크포인트를 고른다. 이 값을 모델 선택에 썼다면 같은 데이터가 독립 최종 테스트 역할을 유지하지는 않는다. 학습 데이터 안에서 검증 구간을 나누어 선택하고, 최종 테스트는 선택을 마친 뒤 평가하는 구성이 필요하다.

이 스크립트에는 조기 종료가 없고 지정 에포크 전체를 수행한다. 모델·옵티마이저 상태와 에포크·로스는 저장하지만 구조 인수와 전처리는 체크포인트에 포함하지 않는다. 재사용하려면 이를 별도로 기록한다. 학습 재개에는 옵티마이저 상태 복원도 필요하고, 재현성을 위해 시드·버전과 데이터 순서 상태를 관리할 수 있다.

## 6. 복원 그림과 실제 품질

최종 복원 그림은 원본과 모델 출력을 같은 표본에 대해 비교한다. 로스 하나로 가려지는 획 손상과 특정 클래스·희귀 표본의 오류를 살피는 데 유용하지만, 모든 출력이 비슷하다고 인코더만이 원인이라고 단정하지 않는다. 출력 범위·입력 전처리·최적화와 디코더도 함께 확인한다.

복원이 좋아도 코드의 분류 유용성이나 임의 잠재 샘플의 품질은 보장되지 않는다. 목적에 따라 검증 복원, 코드 기반 후속 태스크와 실제 저장·추론 비용을 따로 평가한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
zero_grad를 제거하면 무엇이 바뀌는지 설명하라.

</div>

??? success "연습문제 1 풀이"

    backward가 기존 grad에 더하므로 각 갱신에서 현재 배치뿐 아니라 이전 배치의 그래디언트도 사용한다. 서로 다른 파라미터 위치에서 계산한 그래디언트의 합이라 일정 러닝레이트 배수와 같지는 않다. 진동·발산이나 다른 경로가 가능하므로 의도하지 않은 누적을 피한다. 의도한 누적에는 초기화와 갱신 주기를 따로 정한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 옵티마이저와 다른 옵티마이저의 비교를 설계하라.

</div>

??? success "연습문제 2 풀이"

    본 코드는 이미 Adam(lr=0.001)을 사용하므로 같은 줄로 바꾸는 것은 비교가 아니다. SGD 등 실제 다른 후보를 정하고 같은 초기값·데이터 순서·평가 조건에서 각 설정의 러닝레이트를 검증한다. 로스 평균 방식과 갱신 수·시간도 통일해 비교한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
합제곱 로스 단위 혼동과 과적합을 각각 어떻게 진단하는가?

</div>

??? success "연습문제 3 풀이"

    출력 로스를 원소별 MSE로 해석하려면 784로 나누어야 한다. 값이 크다고 바로 학습 실패는 아니다. 학습·독립 검증의 같은 단위 로스를 비교해 과적합과 분포 차이를 확인하고 표본별 복원·출력 범위도 본다. 작은 병목도 암기를 원리적으로 막지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 조각의 state_dict 복사와 학습 흐름 문제를 설명하라.

</div>

??? success "연습문제 4 풀이"

    state_dict().copy()는 사전의 얕은 복사라 내부 텐서가 이후 학습과 저장 공간을 공유할 수 있다. 독립 복제를 만들거나 체크포인트로 저장해야 한다. 조각에는 실제 학습 호출과 evaluate 정의가 없고, 정상 종료 시 최적 상태 복원도 빠져 있다. 검증 로스가 유효하고 best_state가 존재하는 조건과 최소 개선폭도 정해야 한다.

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
모델과 학습 코드를 분리할 때 함께 관리해야 할 계약을 설명하라.

</div>

??? success "연습문제 5 풀이"

    입력 평탄화·피처 수, forward 반환형과 loss_function 인수·평균 축을 맞춘다. import 경로와 패키지 구조도 명시한다. 구조를 바꾸어 VAE 튜플을 반환하면 현재 단일 복원값 학습 루프는 그대로 호환되지 않는다. 한 곳의 구조 정의를 재사용해 중복 불일치를 줄일 수 있다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
train·eval과 no_grad가 현재 모델에서 각각 어떤 효과를 가지는가?

</div>

??? success "연습문제 6 풀이"

    Linear·ReLU·Sigmoid는 train/eval에 따라 수학적 계산이 달라지지 않는다. no_grad는 평가 연산의 그래프 기록을 끈다. Dropout·BatchNorm을 추가하면 모드 동작이 달라지고 VAE의 샘플링 여부는 forward 설계로 따로 정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
에포크별 복원 이미지를 비교하는 공정한 방법과 한계를 설명하라.

</div>

??? success "연습문제 7 풀이"

    같은 고정 보류 표본, 전처리·표시 범위와 평가 모드를 사용한다. 개별·클래스별 오차와 출력 다양성을 함께 본다. 모든 출력이 비슷하면 여러 원인이 가능하므로 인코더 붕괴로 단정하지 않는다. 이미지 일부의 개선이 전체 일반화나 좋은 잠재 샘플을 증명하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
현재 배치 크기를 바꾸면 학습과 보고 로스에 어떤 차이가 생기는가?

</div>

??? success "연습문제 8 풀이"

    역전파는 배치 전체 합을 사용하므로 같은 원소 오차에서 그래디언트 규모가 배치 수에 따라 달라진다. 에포크 보고는 표본 수로 나누어 다른 크기의 마지막 배치를 가중한다. 배치 평균으로 학습 규칙을 바꾸면 또 다른 설정이며 Adam의 변화가 러닝레이트 상수배만으로 같아지는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
언제 멈추고 어떤 체크포인트를 선택할지 목적에 따라 설명하라.

</div>

??? success "연습문제 9 풀이"

    복원 목적은 검증 오차와 비용, 표현 목적은 독립된 후속 태스크 지표도 사용한다. 개선폭·patience와 최대 예산을 정하고 선택한 상태를 복원한다. 작은 코드 차원만으로 과적합이 없다고 보지 않는다. 설정 선택과 최종 테스트를 분리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
아래 저장 조각을 재사용할 때 확인할 구조·전처리 불일치를 설명하라.

</div>

??? success "연습문제 10 풀이"

    조각은 latent_dim=16을 기록하지만 현재 기본 모델은 32이므로 실제 실행 설정과 같게 저장해야 한다. ToTensor 뒤 데이터에 다시 /255를 적용하면 전처리가 달라진다. 웨이트·구조·입력 범위를 함께 기록하고 필요하면 optimizer·스케줄러 상태도 저장한다. 불러온 state_dict는 같은 구조에 적용해야 한다.

    ```python
    torch.save({'state_dict': model.state_dict(),
                'config': {'input_dim': 784, 'hidden_dim': 256, 'latent_dim': 16},
                'preprocess': 'x / 255.0'}, path)
    ```


## 정리하며

이 학습 스크립트는 표본별 픽셀 합제곱 오차를 보고하고 테스트 로스로 체크포인트를 고른다. 로스 단위와 배치 갱신, 모드와 그래디언트 기록을 구분한다. 독립 평가를 위해 검증·테스트 역할을 나누고 구조·전처리·체크포인트의 실제 설정을 함께 관리해야 한다.
