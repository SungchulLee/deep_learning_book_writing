# 제 부호기 익히기

기본 자기 부호기 익히기 각본

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
기본 자기 부호기 익히기 각본
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
    """시험 배치로 값매김한다"""
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

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 변분 자기 부호기 일에 대한 실전 직관이 선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

</div>

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

</div>

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

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
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

</div>

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
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
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
익히기 스크립트가 `models.autoencoder`에서 모델을 들여온다. 모델의 뜻매김을 익히기 코드와 떼어 두는 것이 왜 나은가?

</div>

??? success "연습문제 5 풀이"
    같은 모델을 여러 스크립트가 나누어 쓸 수 있기 때문이다. 익히기, 값매김, 표본
    만들기가 저마다 모델을 다시 적으면 세 곳이 어긋나기 시작한다.

    그리고 **읽는 사람에게도 낫다.** 얼개가 궁금하면 모듈 한 곳만 보면 되고, 익히기
    절차가 궁금하면 스크립트만 보면 된다. 이 장이 얼개 절과 익히기 절을 나눈 것도
    같은 이유다.

    다만 책에서는 맞바꿈이 있다. 코드가 흩어지면 한 쪽만 읽고는 돌려 볼 수 없다.
    그래서 이 장은 모듈을 따로 두되 그 내용을
    [자기 부호기 모듈](../architecture/autoencoder.md) 쪽에 그대로 실어 둔다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
익히기와 값매김에서 `model.train()`과 `model.eval()`을 부르는데, 이 자기 부호기에서 실제로 달라지는 것이 있는가?

</div>

??? success "연습문제 6 풀이"
    이 모델에는 **없다.** 선형층과 ReLU뿐이라 두 모드가 같은 계산을 한다.

    그래도 적어 두는 까닭은 [3.2절 연습문제 8](../../ch03/linear_softmax/06_implementation.md)에서
    본 것과 같다. 모델을 바꾸면 곧 필요해지고, 빠뜨려도 오류가 나지 않아 알아채기
    어렵기 때문이다.

    자기 부호기 계열에서 특히 조심할 것이 둘이다.

    - **드롭아웃을 넣은 경우**: 평가 때 꺼야 다시 세우기 오차가 제대로 나온다
    - **배치 정규화를 넣은 경우**: 모드에 따라 쓰는 통계가 달라진다

    그리고 변분 자기 부호기에서는 한 가지가 더 붙는다. 평가할 때 $z$를 뽑을지
    $\mu$를 그대로 쓸지 정해야 하는데, 이는 `eval()`이 알아서 해 주지 않으므로
    코드로 나누어 주어야 한다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
다시 세우기 그림을 에포크마다 저장해 두면 무엇을 알 수 있는가?

</div>

??? success "연습문제 7 풀이"
    손실 숫자만으로는 보이지 않는 것이 드러난다.

    - **무엇을 먼저 배우는가.** 초기에는 대개 흐릿한 평균 모양이 나오고, 차츰 획의
      자리가 잡히고, 마지막에 굵기와 기울기 같은 세부가 온다.
    - **어떤 부류를 못 배우는가.** 8과 5가 오래 뭉개져 있다면
      [클래스별 오차](../ae/reconstruction_analysis.md)에서 본 것이 그림으로 확인되는
      셈이다.
    - **무너지고 있는가.** 모든 출력이 같은 그림으로 수렴하면 부호기가 죽은 것이다.
      손실은 그저 높은 값에서 평평해 보일 뿐이라 그림 없이는 구별하기 어렵다.

    마지막 경우가 특히 값지다. 손실 곡선이 평평한 것과 모델이 무너진 것은 숫자만으로
    구별되지 않지만, 그림 한 장이면 곧바로 안다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
배치 크기를 바꾸면 무엇이 달라지는가? 자기 부호기에서도 [3.2절](../../ch03/linear_softmax/05_gradient_descent.md)과 같은 이야기인가?

</div>

??? success "연습문제 8 풀이"
    같다. 배치가 작으면 갱신이 잦아 빨리 내려가지만 잡음이 많고, 크면 반대다.
    배치 크기를 바꾸면 실효 학습률이 함께 바뀌므로 학습률도 손봐야 한다.

    자기 부호기에 특별한 점이 하나 있다. 손실을 화소에 대해 `sum`으로 더하고 표본에
    대해서만 나누는 관례를 쓰면, **배치 크기를 바꿔도 기울기의 크기가 유지된다.**
    화소 축을 평균 내면 손실이 784배 작아져 학습률을 그만큼 키워야 한다
    ([손실 함수 연습문제 2](../ae/loss_functions.md)).

    익히기 코드를 읽을 때 `reduction`과 나누는 축을 먼저 확인하는 버릇을 들이면
    학습률이 왜 그 값인지 이해하기 쉬워진다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
자기 부호기를 익힐 때 검증 자료가 필요한가? 무엇을 보고 언제 멈출 것인가?

</div>

??? success "연습문제 9 풀이"
    필요하다. 다만 보는 것이 지도 학습과 다르다.

    지도 학습에서는 검증 손실이 올라가기 시작하면 멈춘다. 자기 부호기도 같은 신호를
    쓸 수 있지만, **병목이 좁으면 그 신호가 거의 오지 않는다.** MNIST에 병목 16이면
    외울 그릇이 없어 학습 오차와 검증 오차가 붙어 간다
    ([자기 부호기 연습문제 7](../ae/48_autoencoder.md)).

    그래서 멈출 자리를 정하는 기준이 둘로 갈린다.

    - **압축이 목적이면** 검증 오차가 평평해질 때 멈춘다.
    - **나타냄이 목적이면** 검증 오차가 아니라 **아래쪽 일의 성능**을 보아야 한다.
      코드로 선형 분류기를 익혀 정확도를 재는 식이다. 오차가 계속 줄어드는 동안에도
      그 성능은 이미 꺾였을 수 있다.

    두 번째가 중요한 까닭은 이 장이 되풀이해 말하는 것과 같다. **다시 세우기 오차는
    나타냄의 품질을 재는 자가 아니다.**

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
익힌 자기 부호기를 저장할 때 무엇을 함께 저장해야 다시 쓸 수 있는가?

</div>

??? success "연습문제 10 풀이"
    가중치만으로는 부족하다. 최소한 이것들이 필요하다.

    | 저장할 것 | 없으면 |
    |---|---|
    | `state_dict` | 가중치가 없다 |
    | `input_dim`, `hidden_dim`, `latent_dim` | 같은 모양의 모델을 만들 수 없다 |
    | 전처리 방법 | 화소를 어떻게 고르게 했는지 몰라 다른 자료를 넣을 수 없다 |
    | 손실과 정규화 설정 | 결과를 되살려 견줄 수 없다 |

    전처리를 함께 적어 두는 것이 특히 자주 빠뜨리는 부분이다. $[0,1]$로 나눈 모델에
    표준화한 자료를 넣으면 조용히 엉뚱한 결과가 나온다.

    ```python
    torch.save({'state_dict': model.state_dict(),
                'config': {'input_dim': 784, 'hidden_dim': 256, 'latent_dim': 16},
                'preprocess': 'x / 255.0'}, path)
    ```

    `torch.load`로 되불러 올 때는 `weights_only=True`를 쓰는 편이 안전하다. 이 사전에
    임의의 파이썬 객체가 들어 있으면 불러오는 것만으로 코드가 실행될 수 있기 때문이다.

## 정리하며

**다룬 것** — 제 부호기 익히기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
