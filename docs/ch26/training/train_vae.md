# VAE 익히기

여느 변분 자기 부호기 익히기 각본

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
여느 변분 자기 부호기 익히기 각본
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
        
        # 온전히 이어진 변분 자기 부호기를 위해 자료를 펼친다
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
            
            # 온전히 이어진 변분 자기 부호기를 위해 자료를 펼친다
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
익히기 반복문에서 두 손실 항을 따로 기록해야 하는 까닭은 무엇인가?

</div>

??? success "연습문제 5 풀이"
    합만 보면 무엇이 움직였는지 모르기 때문이다.

    특히 위험한 것이 초반이다. KL이 빠르게 떨어지면서 총손실이 내려가는 모습은
    순조로워 보이는데, 실제로는 차원이 죽고 있는 중일 수 있다
    ([47_vae 연습문제 6](../architecture/47_vae.md)).

    ```python
    print(f"epoch {ep}  rec {rec_sum/n:.2f}  kl {kl_sum/n:.2f}")
    ```

    두 항을 나란히 적으면 진단이 된다.

    | 보이는 것 | 뜻 |
    |---|---|
    | KL이 0으로 간다 | 무너짐. 코드가 쓰이지 않는다 |
    | KL이 계속 커진다 | 코드가 정보를 더 담는 중. 대개 괜찮다 |
    | 다시 세우기가 평평한데 KL이 준다 | 정보를 버리는 중 |

    차원마다의 KL을 몇 에포크에 한 번씩 찍어 두면 더 낫다. 총 KL이 20.32여도 안에
    0이 네 개 있을 수 있다([자유 비트](free_bits.md)).

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
익히기 중에 표본을 뽑아 보는 것이 다시 세우기 그림을 보는 것과 어떻게 다른가?

</div>

??? success "연습문제 6 풀이"
    재는 것이 다르다.

    **다시 세우기**는 부호기와 풀개가 함께 일하는 것을 본다. 자료에서 온 $x$를 넣으므로
    코드가 아는 자리에 있다.

    **표본**은 풀개만 본다. $z \sim \mathcal{N}(0,I)$을 넣으므로 코드가 아는 자리에 있다는
    보장이 없다.

    그래서 둘이 어긋날 수 있고, 그 어긋남이 진단이 된다.

    | 다시 세우기 | 표본 | 무엇을 뜻하는가 |
    |---|---|---|
    | 좋다 | 좋다 | 잘되고 있다 |
    | 좋다 | 나쁘다 | 구멍 문제. 사전 분포와 안 맞는다 |
    | 나쁘다 | 나쁘다 | 아직 덜 익었거나 무너졌다 |

    둘째 칸이 [자기 부호기의 상태](../../ch25/limits/latent_sampling.md)다. 다시 세우기는
    변분 자기 부호기보다 좋으면서 표본은 0.2%였다.

    그래서 만들어 내기가 목적이면 **표본을 반드시 함께 보아야 한다.** 다시 세우기 곡선만
    보고 있으면 정작 목적에서 실패하는 것을 알 수 없다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
검증 자료로 무엇을 보고 언제 멈출 것인가?

</div>

??? success "연습문제 7 풀이"
    검증 증거 하한을 보는 것이 기본이다. 다만 자기 부호기에서 말한 것과 같은 사정이 있다
    ([25장](../../ch25/training/train_autoencoder.md)). 병목이 좁으면 외울 그릇이 없어
    과적합 신호가 약하다.

    변분 자기 부호기에는 신호가 하나 더 있다. **차원이 죽는 것**이다. 이것은 과적합이
    아니라 최적화의 실패이며, 검증 손실이 아니라 차원별 KL을 보아야 잡힌다.

    그래서 볼 것이 목적에 따라 갈린다.

    | 목적 | 멈추는 기준 |
    |---|---|
    | 밀도 추정 | 검증 ELBO가 평평해질 때 |
    | 표본 만들기 | 표본 품질이 꺾일 때. ELBO와 어긋날 수 있다 |
    | 나타냄 | 아래쪽 일의 성능 |

    가운데 칸을 눈여겨볼 만하다. $\beta$ 쓸기에서 본 대로 ELBO가 좋은 설정과 표본이 좋은
    설정이 다르므로([베타 VAE](../architecture/beta_vae.md)), 익히기 도중에도 두 값이
    다른 시점에 최고가 될 수 있다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
KL 달구기를 이 반복문에 넣으려면 어떻게 하는가?

</div>

??? success "연습문제 8 풀이"
    에포크나 걸음 수에 따라 $\beta$를 올린다.

    ```python
    for ep in range(epochs):
        beta = min(1.0, ep / warmup_epochs)        # 0 -> 1
        for x, in loader:
            out, mu, logvar = model(x)
            loss = rec(out, x) + beta * kl(mu, logvar)
    ```

    고를 것이 둘이다. **얼마나 오래** 달구는가와 **어떤 모양**으로 올리는가.

    걸음 단위로 올리는 편이 에포크 단위보다 매끄럽고, 선형으로 올리는 것이 가장 흔하다.
    달구는 기간은 전체의 10~30% 정도를 흔히 쓴다.

    한 가지 조심할 것이 있다. **달구는 동안의 손실 값은 ELBO가 아니다.** 곡선을 그릴 때
    그 구간에서 값이 낮은 것이 모델이 좋아서가 아니라 KL을 덜 세어서다. 기록할 때
    $\beta$를 함께 적어 두어야 나중에 오해하지 않는다.

    달구기가 필요한지 먼저 확인하는 것도 좋다. 차원이 죽지 않는다면 넣을 까닭이 없다.
    이 장의 기본 설정($\beta=1$, 16차원)에서는 4개가 죽으니 시도해 볼 값이 있다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
배치 크기와 학습률을 어떻게 정했는가? 왜 그 값인가?

</div>

??? success "연습문제 9 풀이"
    이 장의 측정은 배치 256, 학습률 1e-3, Adam으로 했다. 23장과 같은 값이며, 그렇게
    맞춘 것이 의도다. **두 장의 수치를 견줄 수 있어야** 하기 때문이다.

    값 자체는 특별하지 않다. Adam에 1e-3은 흔한 출발점이고, 배치 256은 MNIST에서 빠르고
    안정된 범위다.

    변분 자기 부호기에 관련된 점이 하나 있다. 다시 뽑기 때문에 기울기에 잡음이 하나 더
    실리므로([부호기 연습문제 6](../architecture/encoder.md)), 배치가 아주 작으면
    자기 부호기보다 더 흔들린다. 배치를 32 아래로 내릴 때는 확인이 필요하다.

    Adam을 쓰는 것이 $\beta$ 쓸기에서 특히 편했다. 손실의 눈금이 32배 달라지는데도
    학습률 하나로 다룰 수 있었다([train_beta_vae 연습문제 4](train_beta_vae.md)).

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
익히기가 잘못되고 있다는 신호를 다시 세우기 곡선만 보고 알 수 있는가?

</div>

??? success "연습문제 10 풀이"
    알 수 없는 실패가 여럿 있다. 이것이 이 절이 두 항을 따로 기록하라고 하는 까닭이다.

    | 실패 | 다시 세우기 곡선에서 | 무엇을 보아야 하는가 |
    |---|---|---|
    | 차원이 죽는다 | 조금 나쁠 뿐, 평평해 보인다 | 차원별 KL |
    | 완전히 무너졌다 | 높은 값에서 평평 | 표본 또는 차원별 KL |
    | 구멍이 크다 | **전혀 안 보인다** | 표본 품질 |
    | $\sigma$가 0으로 간다 | 오히려 **좋아진다** | KL, 또는 $\sigma$ 값 |

    셋째와 넷째가 특히 고약하다.

    구멍 문제는 다시 세우기에 흔적을 남기지 않는다. 자기 부호기가 그 증거다. 다시
    세우기는 더 좋으면서 표본은 0.2%였다.

    $\sigma \to 0$은 다시 세우기를 **좋아지게** 한다. 잡음이 없으면 되돌리기가 쉬우니까.
    그래서 다시 세우기만 보면 "잘되고 있다"고 읽힌다. 실제로는 모델이 자기 부호기로
    되돌아가는 중이다.

    정리하면 익히기를 지켜볼 때 최소한 이 셋을 함께 보아야 한다. **다시 세우기, 차원별
    KL, 표본.** 셋 다 값이 싸고, 하나라도 빠지면 못 보는 실패가 있다.

## 정리하며

**다룬 것** — VAE 익히기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
