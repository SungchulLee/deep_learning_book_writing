# 베타 VAE 익히기

β-변분 자기 부호기(베타 VAE) 익히기 각본. β 매개변수로 얽힘 풀린 나타냄을 배운다

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
β-변분 자기 부호기(베타 VAE) 익히기 각본
β 매개변수로 얽힘 풀린 나타냄을 배운다
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
        
        # 온전히 이어진 β-변분 자기 부호기를 위해 자료를 펼친다
        if isinstance(model, BetaVAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # 순전파
        reconstruction, mu, logvar = model(data_input)
        
        # 손실을 계산한다
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
    """시험 묶음으로 값매김한다"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 온전히 이어진 β-변분 자기 부호기를 위해 자료를 펼친다
            if isinstance(model, BetaVAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # 순전파
            reconstruction, mu, logvar = model(data_input)
            
            # 손실을 계산한다
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
    
    # β-변분 자기 부호기 전용: 얽힘 풀림을 보이는 숨은 훑기
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
$\beta$를 쓸어 볼 때 무엇을 기록해 두어야 하는가?

</div>

??? success "연습문제 5 풀이"
    적어도 이 넷이다.

    | 기록할 것 | 왜 |
    |---|---|
    | 다시 세우기와 KL을 **따로** | 합만 보면 어느 쪽이 움직였는지 모른다 |
    | 차원마다의 KL | 죽은 차원을 보려면 총합으로는 안 된다 |
    | 표본 품질 | 손실 두 조각으로는 $\beta$를 고를 수 없다 |
    | 익히기 곡선 | 무너짐이 언제 일어났는지 알려 준다 |

    두 번째와 세 번째가 특히 자주 빠진다. 그런데 이 장의 결론이 모두 그 둘에서 나왔다.
    $\beta=4$가 표본에서 가장 좋다는 것도, 그 대가로 9개 차원이 죽는다는 것도 총
    손실만 보고서는 알 수 없다.

    ```python
    log = {'beta': beta, 'rec': rec.item(), 'kl': kl.item(),
           'kl_per_dim': kl_dim.tolist(), 'alive': int((kl_dim > 0.01).sum())}
    ```

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
여러 $\beta$를 견줄 때 무엇을 같게 맞추어야 하는가?

</div>

??? success "연습문제 6 풀이"
    $\beta$ 말고는 다 같아야 한다. 이 장의 쓸기에서 맞춘 것은 이렇다.

    - 씨앗 42, 모델을 만들기 **직전에** 고정
    - DataLoader의 생성기도 같은 씨앗
    - 숨은 차원 16, 묶음 256, 학습률 1e-3, 20 에포크
    - 같은 판정 분류기, 표본 뽑기의 씨앗도 고정

    씨앗을 모델 만들기 직전에 두는 것이 특히 중요하다. 순서가 어긋나면 같은 설정에서도
    값이 달라져, [3.4절에서 겪은 대로](../../ch03/mnist/04_cnn.md) 한 모델에 세 가지
    정확도가 나온다.

    표본 뽑기의 씨앗을 고정하는 것도 빠뜨리기 쉽다. 1,000개로 재는 비율이므로 씨앗이
    다르면 몇 %는 흔들린다. 57.4%와 60.0% 같은 차이를 말하려면 고정해야 한다.

    그리고 익히기 예산을 같게 두는 것이 $\beta$가 큰 쪽에 불리하지 않은지 물어볼 만하다.
    KL이 세면 수렴이 느릴 수 있다. 이 쓸기는 20 에포크로 모두 평평해졌지만, 확인하지
    않고 넘어갈 일은 아니다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$\beta$ 쓸기의 결과를 한 장의 그림으로 어떻게 보이겠는가?

</div>

??? success "연습문제 7 풀이"
    가로축을 다시 세우기, 세로축을 KL로 두고 $\beta$마다 점을 찍으면 **맞바꿈 앞머리**가
    그려진다. 점마다 $\beta$를 적어 둔다.

    이 그림이 좋은 까닭은 두 손실이 대등하게 보이고, 앞머리의 모양에서 어디가 값싼
    구간인지 읽히기 때문이다. $\beta$를 가로축에 두고 두 곡선을 겹쳐 그리면 눈금이
    달라 견주기 어렵다.

    그런데 이 그림만으로는 $\beta$를 고를 수 없다. 앞머리 위의 모든 점이 그 나름대로
    최적이기 때문이다. 고르려면 **제3의 축**이 필요하다.

    그래서 점의 크기나 색으로 표본 품질을 함께 나타내는 것이 좋다. 그러면 앞머리의
    가운데쯤($\beta=4$)에서 색이 가장 진해지는 것이 한눈에 보이고, 양 끝이 왜 나쁜지도
    설명된다.

    죽은 차원 수를 점마다 적어 두면 더 낫다. $\beta=8$의 품질이 떨어지는 까닭이 바로
    그것이기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$\beta$를 바꾸면 학습률도 바꾸어야 하는가?

</div>

??? success "연습문제 8 풀이"
    엄밀히는 그렇다. $\beta$가 손실의 크기를 바꾸므로 기울기의 크기도 바뀐다.

    다만 Adam을 쓰면 걱정이 많이 줄어든다. Adam은 기울기를 그 크기의 추정치로 나누므로
    전체적인 눈금 변화에 무디다. 이 장의 쓸기가 학습률 1e-3 하나로 $\beta$ 0.25에서
    8까지 다룰 수 있었던 까닭이다.

    SGD였다면 사정이 달랐을 것이다. $\beta$를 32배 바꾸면 실효 걸음도 그만큼 달라져
    한쪽은 너무 느리고 다른 쪽은 흔들린다.

    그래도 Adam에서 완전히 무관하지는 않다. 두 항의 **비**가 바뀌면 어느 방향으로 갈지가
    달라지므로, 수렴에 걸리는 에포크가 달라질 수 있다. 그래서 쓸기에서는 모든 설정이
    수렴했는지 곡선으로 확인해 두는 것이 맞다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$\beta$ 쓸기에서 얻은 최적값을 다른 자료에 그대로 쓸 수 있는가?

</div>

??? success "연습문제 9 풀이"
    쓸 수 없다. $\beta$는 자료와 얼개에 딸린 값이다.

    까닭을 보자. $\beta$는 다시 세우기 항과 KL 항의 균형을 정하는데, 두 항의 자연스러운
    크기가 자료마다 다르다.

    | 무엇이 바뀌면 | 왜 $\beta$가 달라지는가 |
    |---|---|
    | 화소 수 | 다시 세우기 항이 화소 수에 비례해 커진다 |
    | 숨은 차원 | KL 항이 차원 수에 비례해 커진다 |
    | 자료의 복잡도 | 다시 세우기의 도달 가능한 값이 달라진다 |
    | 손실 종류 | BCE와 MSE의 눈금이 다르다 |

    첫 칸이 가장 크다. MNIST의 784화소에서 고른 $\beta=4$를 CIFAR의 3,072화소에 쓰면
    다시 세우기 항이 네 배 커진 셈이라 실효 $\beta$가 4분의 1이 된다.

    그래서 새 자료에서는 다시 쓸어야 하고, 다행히 쓸기가 싸다. 이 장의 쓸기는 여섯 번
    익히면 끝이다.

    눈금에 덜 민감한 값을 쓰고 싶으면 화소당으로 정규화해 두는 방법이 있다. 다시
    세우기를 화소 수로, KL을 숨은 차원 수로 나누면 자료를 옮길 때 $\beta$가 덜 흔들린다.
    다만 그러면 관례적인 $\beta=1$이 더는 ELBO에 대응하지 않으므로, 무엇을 하고 있는지
    적어 두어야 한다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$\beta$를 아주 작게, 예컨대 0.01로 두면 어떻게 되는가?

</div>

??? success "연습문제 10 풀이"
    사실상 [자기 부호기](../../ch25/index.md)가 된다.

    쓸기의 방향이 그것을 가리킨다. $\beta$를 0.25까지 내렸을 때 다시 세우기는 70.60까지
    좋아졌고 표본 품질은 40.3%까지 떨어졌다. 더 내리면 이 추세가 이어진다.

    극한에서는 KL이 $\sigma \to 0$을 막지 못하므로 부호기가 결정적이 되고, 뽑기가
    [0.2%](../../ch25/limits/latent_sampling.md)로 무너진다.

    그런데 다시 세우기만 보면 $\beta$가 작은 쪽이 **언제나 이긴다.** 이 점이 함정이다.
    다시 세우기를 잣대로 $\beta$를 고르면 반드시 0으로 가라는 답이 나오고, 그러면
    만들어 내는 모델이 아니게 된다.

    25장이 되풀이한 교훈이 여기서도 같다. **다시 세우기 오차로 고를 수 없는 것들이
    있다.** $\beta$가 그중 하나다.

## 정리하며

**다룬 것** — 베타 VAE 익히기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
