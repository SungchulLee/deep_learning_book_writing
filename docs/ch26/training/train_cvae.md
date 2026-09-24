# 조건 VAE 익히기

조건부 변분 오토인코더 익히기 각본

오토인코더와 변분 오토인코더는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

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
        
        # 손실을 계산한다
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
    """시험 배치로 값매김한다"""
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
            
            # 손실을 계산한다
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
    
    # 갈래마다 표본을 만든다
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

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 변분 오토인코더 일에 대한 실전 직관이 선다.

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
익히기 반복문에서 조건부가 아닌 것과 달라지는 곳은 몇 군데인가?

</div>

??? success "연습문제 5 풀이"
    두 군데다.

    ```python
    for x, y in loader:                       # 1. 표지를 함께 꺼낸다
        c = F.one_hot(y, 10).float()
        out, mu, logvar = model(x, c)         # 2. 조건을 넘긴다
        loss = recon(out, x) + kl(mu, logvar) # 손실은 그대로
    ```

    손실 함수는 손대지 않는다. KL 항의 뜻이 달라지지만
    ([조건부 VAE 연습문제 3](../architecture/conditional_vae.md)) 식은 같다.

    자료 부르기에서 표지를 쓰게 되는 점이 실질적인 변화다. 오토인코더 계열을 쓰다가
    조건부로 옮기면 `for x, _ in loader`로 표지를 버리던 습관이 남아 있기 쉽다.
    그러면 오류 없이 조건이 무시되는 일이 생긴다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
표지를 넣는 것을 잊으면 어떤 증상이 나타나는가?

</div>

??? success "연습문제 6 풀이"
    조건을 0으로 채워 넣거나 모양이 맞으면 **오류가 나지 않는다.** 그래서 증상으로
    알아내야 한다.

    | 증상 | 무엇을 뜻하는가 |
    |---|---|
    | 부류를 지정해도 아무 숫자나 나온다 | 디코더가 조건을 받지 못한다 |
    | 다시 세우기가 조건 없는 것과 같다 | 조건이 실제로 쓰이지 않는다 |
    | KL이 조건 없는 것과 같다 | 코드가 여전히 부류를 나른다 |

    세 번째가 가장 이른 신호다. 조건이 제대로 들어가면 코드가 나를 정보가 줄어 KL이
    내려가야 한다. 실제로 16.54 대 20.32로 확실한 차이가 있다. KL이 20 근처에 머물면
    조건이 안 먹고 있다고 의심할 만하다.

    가장 확실한 확인은 **지정한 부류대로 나오는 비율을 재는 것**이다. 90% 근처가 아니라
    10% 근처라면 조건이 전혀 안 들어간 것이다. 이 한 줄 검사를 익히기 끝에 붙여 두면
    좋다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
익힐 때와 뽑을 때 조건을 만드는 방식이 어떻게 다른가?

</div>

??? success "연습문제 7 풀이"
    익힐 때는 자료의 표지를 그대로 쓴다. 뽑을 때는 **우리가 정한다.**

    ```python
    # 익히기: 자료에서 온다
    c = F.one_hot(y, 10).float()

    # 뽑기: 우리가 만든다
    c = F.one_hot(torch.full((n,), 7), 10).float()   # 7을 n개
    c = F.one_hot(torch.randint(0, 10, (n,)), 10)    # 고루 섞어서
    ```

    이 차이에서 실수가 나온다. 뽑을 때 부류를 고루 뽑으면 자료의 실제 비율과 어긋날
    수 있다. MNIST는 부류가 거의 고르니 문제가 적지만, 치우친 자료에서는 표본의 분포가
    자료의 분포와 달라진다.

    그리고 조건을 **익힐 때 본 적 없는 값**으로 주지 않도록 조심해야 한다. 원-핫이
    아닌 것(예컨대 두 부류를 반씩 섞은 벡터)을 넣으면 모델이 무엇을 낼지 알 수 없다.
    재미있는 실험거리이지만 결과를 "모델의 표본"이라 부를 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
부류마다 자료 수가 크게 다르면 무엇을 조심해야 하는가?

</div>

??? success "연습문제 8 풀이"
    드문 부류의 품질이 나빠진다. 본 것이 적으니 당연하다.

    MNIST는 부류가 고른 편인데도 부류마다 차이가 난다. 지정한 부류로 판정된 비율이
    3은 99%, 9는 83%다. 이 차이는 자료 수가 아니라 **모양의 어려움**에서 온다.

    치우친 자료에서는 두 원인이 겹치므로 나누어 보아야 한다.

    | 확인할 것 | 방법 |
    |---|---|
    | 자료 수 탓인가 | 부류별 품질과 부류별 자료 수의 상관 |
    | 모양 탓인가 | 자료 수를 맞추어 다시 익혀 보고 견줌 |

    다룰 방법으로는 드문 부류를 더 자주 뽑거나(가중 표집), 손실에 부류별 무게를 주는
    것이 있다. 다만 이것이 늘 옳지는 않다. 자료의 참된 비율을 따르는 것이 목적이라면
    드문 부류가 드물게 나오는 것이 **맞는** 행동이다.

    무엇을 원하는지 먼저 정할 일이다. 모든 부류를 고르게 잘 만들고 싶은 것인지, 자료의
    분포를 충실히 따르고 싶은 것인지는 다른 목표다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
조건부 모델의 품질을 잴 때 분류기를 쓰는 데에 어떤 함정이 있는가?

</div>

??? success "연습문제 9 풀이"
    함정이 여럿이다.

    **분류기와 모델이 같은 약점을 가질 수 있다.** 둘 다 MNIST로 익혔으므로 같은 것을
    놓칠 수 있다. 판정하는 쪽과 판정받는 쪽이 독립이 아니다.

    **분류기가 속는다.** 사람 눈에 얼룩인 그림에 높은 확신도를 주기도 한다. 판정
    정확도가 높다고 사람이 보기에 좋은 표본이라는 보장이 없다.

    **다양성을 보지 않는다.** 지정한 부류의 표본이 모두 똑같은 3이어도 100%가 나온다.
    이것이 가장 큰 구멍이다. 조건부 모델이 부류마다 **한 가지 원형만** 내는 쪽으로
    무너져도 이 잣대는 알아채지 못한다.

    **잣대가 품질의 눈금이 아니다.** 이 장에서 실제로 부딪힌 함정이다. 뽑을 때 $z$에
    온도를 곱해 키우면 판정 비율이 계속 올라간다.

    | 온도 | 1.0 | 1.5 | 2.0 | 2.5 |
    |---|---|---|---|---|
    | 확신도 0.9 넘는 비율 | 57.4% | 62.9% | 67.2% | **69.1%** |

    그런데 온도 2.5는 $\mathcal{N}(0, 6.25I)$에서 뽑는 것이라 사전 분포와 거의 겹치지
    않는다. 모델이 정의한 분포에서 **벗어날수록 점수가 오르는** 셈이니, 이 수를 품질로
    읽으면 안 된다([표본 만들기](../training/generate_samples.md)에서 까닭을 캔다).

    그래서 이 장의 수치는 "같은 조건에서 견주는 상대 비교"로만 쓴다. 절대적인 품질을
    말하려면 [30장](../../ch29/index.md)의 잣대가 필요하고, 그것들도 다양성 문제를
    완전히 풀지는 못한다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
조건부 모델을 익힌 뒤 부류별로 나누어 살펴야 할 것은 무엇인가?

</div>

??? success "연습문제 10 풀이"
    총합만 보면 놓치는 것이 있다. 부류마다 볼 것이 셋이다.

    **부류별 다시 세우기 오차.** [26장에서 본 대로](../../ch25/ae/reconstruction_analysis.md)
    부류마다 크게 다르고, 그 차이의 상당 부분이 밝기 같은 시시한 것에서 온다. 조건부에서도
    같은 확인이 필요하다.

    **부류별 표본 품질.** 잰 값이 83%에서 99%까지 벌어진다. 평균 90.1%만 보면 9와 8이
    뒤처지는 것을 모른다.

    **부류별 다양성.** 앞 문제에서 말한 구멍이다. 부류를 고정해 여러 개 뽑고 서로
    얼마나 다른지 보아야 한다. 눈으로 보는 것이 가장 빠르고, 수로 재려면 표본끼리의
    거리 평균을 쓸 수 있다.

    셋을 함께 보면 진단이 가능해진다. 어떤 부류의 품질이 낮으면서 다양성도 낮으면 그
    부류를 거의 못 배운 것이고, 품질은 낮은데 다양성이 높으면 너무 흩어진 것이다.
    손 볼 방향이 다르다.

## 정리하며

**다룬 것** — 조건 VAE 익히기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
