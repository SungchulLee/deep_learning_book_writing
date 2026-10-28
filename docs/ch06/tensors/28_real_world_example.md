# 실전 예제 — 파이프라인이 제 고장을 알아보는가

MNIST 숫자 분류기를 처음부터 끝까지 짠다. 데이터를 읽고, 모델을 세우고,
익히고, 검증하고, 추론한다. 이 절에서 하나씩 보아 온 조각들이 여기서 한 흐름으로
이어 붙는다.

그런데 끝까지 이어 붙인 보기가 값을 가지는 까닭은 98%라는 숫자를 받아 보는 데
있지 않다. 그 숫자는 다른 데서도 얻을 수 있다. 값은 **잘못 이어 붙였을 때
숫자가 그것을 알려 준다**는 데 있다. 그러니 알려 주기는 하는지를 재어 보아야 한다.

그래서 이 쪽은 바르게 돌린 다음 네 가지로 고장을 내고, 어느 고장이 숫자에
나타나는지 센다. 미리 말해 두면 **넷 가운데 하나만 큰 소리를 낸다.**

## 1. 코드

```python
"""튜토리얼 28: 파이프라인이 제 고장을 알아보는가.

MNIST 숫자 분류기를 처음부터 끝까지 짠다. 그러나 이 쪽의 일은 98%를
받아 보는 데서 끝나지 않는다. 끝까지 이어 붙인 보기가 값을 가지는
까닭은 **잘못 이어 붙였을 때 숫자가 그것을 알려 주기 때문**이고,
그러니 알려 주기는 하는지를 재어 보아야 한다.

그래서 바르게 돌린 다음, 네 가지로 고장을 내고 어느 고장이 숫자에
나타나는지 본다.

실행: python 28_real_world_example.py   (repo 뿌리에서, ./data 를 읽는다)
"""
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# ========================================================================
# 장치 — 있는 것을 쓴다
# ========================================================================

def pick_device():
    """cuda → mps → cpu. 이 책은 셋 다에서 돌아야 한다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEV = pick_device()

# ========================================================================
# 모델
# ========================================================================

class ConvNet(nn.Module):
    """28x28 한 채널 → 열 클래스. 파라미터 225,034개."""

    def __init__(self, p_drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28 → 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13 → 11
        self.pool = nn.MaxPool2d(2)                    # 26 → 13, 11 → 5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)                 # view(-1, ...) 보다 뜻이 분명하다
        x = torch.relu(self.fc1(x))
        return self.fc2(self.drop(x))

# ========================================================================
# 학습과 평가 — 고장을 낼 수 있게 손잡이를 둔다
# ========================================================================

def one_epoch(model, opt, crit, loader, *, zero_grad=True, scale=1.0):
    """한 에폭. zero_grad=False 와 scale=255 가 뒤에서 낼 고장이다."""
    model.train()
    tot = correct = seen = 0.0
    for x, y in loader:
        x, y = x.to(DEV) * scale, y.to(DEV)
        if zero_grad:
            opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tot += loss.item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.numel()
    return tot / seen, 100 * correct / seen


def accuracy(model, loader, *, eval_mode=True, scale=1.0):
    """eval_mode=False 가 'eval() 빼먹기' 고장이다."""
    model.eval() if eval_mode else model.train()
    correct = seen = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV) * scale, y.to(DEV)
            correct += (model(x).argmax(1) == y).sum().item()
            seen += y.numel()
    return 100 * correct / seen


def fresh(seed, p_drop=0.5, lr=1e-3, sgd=False):
    torch.manual_seed(seed)
    m = ConvNet(p_drop).to(DEV)
    opt = (torch.optim.SGD if sgd else torch.optim.Adam)(m.parameters(), lr=lr)
    return m, opt, nn.CrossEntropyLoss()


def train(loader, *, seed, epochs, **kw):
    m, opt, crit = fresh(seed)
    for _ in range(epochs):
        one_epoch(m, opt, crit, loader, **kw)
    return m

# ========================================================================
# 메인
# ========================================================================

TFM = transforms.ToTensor()
TR = datasets.MNIST("./data", train=True, download=True, transform=TFM)
TE = datasets.MNIST("./data", train=False, download=True, transform=TFM)
trl = DataLoader(TR, batch_size=128, shuffle=True)
tel = DataLoader(TE, batch_size=512)


def main():
    print(f"장치 {DEV}   학습 {len(TR)}장   검증 {len(TE)}장")

    # === 1. 바르게 이어 붙인 파이프라인 ===
    print("\n=== 1. 바른 파이프라인, 4 에폭 ===")
    m, opt, crit = fresh(0)
    print(f"파라미터 {sum(p.numel() for p in m.parameters()):,}개")
    for ep in range(1, 5):
        ls, ac = one_epoch(m, opt, crit, trl)
        print(f"에폭 {ep}  학습 로스 {ls:.4f}  학습 {ac:.2f}%  "
              f"검증 {accuracy(m, tel):.2f}%")

    # === 2. 라벨을 섞어 본다 ===
    # 라벨을 무작위로 섞으면 배울 규칙이 없어진다. 그런데도 검증 점수가
    # 높게 나오면 파이프라인이 어딘가에서 답을 흘리고 있다는 뜻이다.
    print("\n=== 2. 라벨을 섞으면 (학습 10,000장) ===")
    torch.manual_seed(0)
    idx = torch.randperm(len(TR))[:10000].tolist()
    sub_x = torch.stack([TR[i][0] for i in idx])
    sub_y = torch.tensor([TR[i][1] for i in idx])
    fake_y = sub_y[torch.randperm(10000)]
    print(f"섞고 나서 우연히 맞은 라벨 "
          f"{100 * (fake_y == sub_y).float().mean():.2f}%")
    fake = DataLoader(TensorDataset(sub_x, fake_y), batch_size=128, shuffle=True)
    for p_drop in (0.5, 0.0):
        m2, opt2, crit2 = fresh(0, p_drop=p_drop)
        print(f"  드롭아웃 p={p_drop}")
        for ep in range(1, 61):
            ls, ac = one_epoch(m2, opt2, crit2, fake)
            if ep in (10, 20, 30, 40, 60):
                print(f"    {ep:2d} 에폭  학습 로스 {ls:.4f}  학습 {ac:6.2f}%  "
                      f"검증 {accuracy(m2, tel):5.2f}%")

    # === 3. 조용한 고장 셋 ===
    print("\n=== 3. 조용한 고장 (2 에폭, 씨앗 셋) ===")
    leak = DataLoader(Subset(TR, list(range(10000))), batch_size=512)
    keys = ("정상", "eval 빼기", "검증 누수", "zero_grad 빼기")
    rows = {k: [] for k in keys}
    for seed in (0, 1, 2):
        base = train(trl, seed=seed, epochs=2)
        rows["정상"].append(accuracy(base, tel))
        rows["eval 빼기"].append(accuracy(base, tel, eval_mode=False))
        rows["검증 누수"].append(accuracy(base, leak))
        nz = train(trl, seed=seed, epochs=2, zero_grad=False)
        rows["zero_grad 빼기"].append(accuracy(nz, tel))
    for k in keys:
        v = rows[k]
        print(f"{k:14s} {sum(v) / 3:6.2f}%   퍼짐 {min(v):.2f}~{max(v):.2f}   "
              f"({', '.join(f'{x:.2f}' for x in v)})")

    # === 4. 같은 고장이 판에 따라 보이기도, 안 보이기도 한다 ===
    print("\n=== 4. 검증 누수를 외우는 판에서 다시 ===")
    for size, epochs in ((1000, 40), (60000, 2)):
        sub = Subset(TR, list(range(size)))
        mm = train(DataLoader(sub, batch_size=128, shuffle=True),
                   seed=0, epochs=epochs)
        s = accuracy(mm, DataLoader(sub, batch_size=512))
        h = accuracy(mm, tel)
        print(f"  학습 {size:>5}장 {epochs:2d}에폭   본 그림 {s:.2f}%   "
              f"못 본 그림 {h:.2f}%   부풀림 {s - h:+.2f}점")

    # === 5. 고장이라 불리지만 고장이 아닌 것 ===
    print("\n=== 5. 입력을 0~255로 넣으면 (2 에폭) ===")
    for name, kw in (("Adam lr=0.001", dict(lr=1e-3)),
                     ("SGD  lr=0.01 ", dict(lr=1e-2, sgd=True))):
        got = []
        for scale in (1.0, 255.0):
            m3, opt3, crit3 = fresh(0, **kw)
            ls = 0.0
            for _ in range(2):
                ls, _ = one_epoch(m3, opt3, crit3, trl, scale=scale)
            got.append((accuracy(m3, tel, scale=scale), ls))
        (a1, l1), (a2, l2) = got
        print(f"  {name}   0~1 {a1:6.2f}% (로스 {l1:.4f})   "
              f"0~255 {a2:6.2f}% (로스 {l2:.4f})")


if __name__ == "__main__":
    main()
```

**출력:**

```
장치 mps   학습 60000장   검증 10000장

=== 1. 바른 파이프라인, 4 에폭 ===
파라미터 225,034개
에폭 1  학습 로스 0.3273  학습 89.97%  검증 97.97%
에폭 2  학습 로스 0.1044  학습 96.83%  검증 98.57%
에폭 3  학습 로스 0.0770  학습 97.68%  검증 98.84%
에폭 4  학습 로스 0.0618  학습 98.08%  검증 98.84%

=== 2. 라벨을 섞으면 (학습 10,000장) ===
섞고 나서 우연히 맞은 라벨 10.19%
  드롭아웃 p=0.5
    10 에폭  학습 로스 2.2911  학습  12.73%  검증 10.46%
    20 에폭  학습 로스 2.2142  학습  18.73%  검증 11.93%
    30 에폭  학습 로스 2.0735  학습  24.59%  검증 11.67%
    40 에폭  학습 로스 1.9581  학습  29.64%  검증 10.86%
    60 에폭  학습 로스 1.7756  학습  35.20%  검증 10.59%
  드롭아웃 p=0.0
    10 에폭  학습 로스 2.2529  학습  16.17%  검증 12.49%
    20 에폭  학습 로스 2.0084  학습  28.10%  검증 10.46%
    30 에폭  학습 로스 1.7265  학습  40.01%  검증  9.57%
    40 에폭  학습 로스 1.4867  학습  48.97%  검증  9.98%
    60 에폭  학습 로스 1.1285  학습  62.02%  검증 10.94%

=== 3. 조용한 고장 (2 에폭, 씨앗 셋) ===
정상              98.49%   퍼짐 98.31~98.72   (98.72, 98.45, 98.31)
eval 빼기         97.47%   퍼짐 97.23~97.83   (97.83, 97.23, 97.34)
검증 누수           98.37%   퍼짐 98.21~98.54   (98.54, 98.21, 98.35)
zero_grad 빼기    87.74%   퍼짐 86.37~90.39   (86.37, 86.47, 90.39)

=== 4. 검증 누수를 외우는 판에서 다시 ===
  학습  1000장 40에폭   본 그림 99.80%   못 본 그림 94.43%   부풀림 +5.37점
  학습 60000장  2에폭   본 그림 98.67%   못 본 그림 98.72%   부풀림 -0.05점

=== 5. 입력을 0~255로 넣으면 (2 에폭) ===
  Adam lr=0.001   0~1  98.72% (로스 0.1026)   0~255  98.32% (로스 0.1648)
  SGD  lr=0.01    0~1  89.21% (로스 0.6721)   0~255  98.16% (로스 0.1611)
```

여섯 분 남짓 걸린다. 장치 줄에 `mps`라고 찍힌 것은 이 쪽을 돌린 기계가 애플
실리콘이기 때문이고, `pick_device`가 `cuda`와 `cpu`도 받으므로 다른 기계에서도
돌아간다. 다만 아래 숫자들은 이 기계에서 나온 값이다.

## 2. 첫 에폭에서 검증이 학습보다 높다

표를 처음 읽으면 어긋나 보이는 데가 있다. 1 에폭에서 학습 89.97%, 검증 97.97%다.
한 번도 보지 않은 그림을 익힌 그림보다 잘 맞힌다.

둘 다 맞다. 재는 조건이 다를 뿐이다.

- 학습 정확도는 **에폭 내내** 쌓은 평균이다. 에폭을 시작할 때의 설익은 모델이
  틀린 것까지 다 들어 있다.
- 검증 정확도는 **에폭이 끝난 뒤** 한 번 잰 값이다. 가장 많이 익은 모델로 잰다.
- 그 위에 학습 쪽은 드롭아웃이 켜진 채로 센다. 유닛 절반이 꺼진 모델로 맞힌
  셈이니 제 실력보다 낮게 나온다.

그러니 이 두 수를 나란히 놓고 과적합을 읽으려 하면 안 된다. 과적합은 같은 조건으로
재어야 보이고, 4 에폭에서 학습 98.08% · 검증 98.84%처럼 **검증이 계속 위에 있는
동안은** 아직 외우기 시작한 것이 아니다.

## 3. 라벨을 섞어 보는 진단

파이프라인이 제대로 이어졌는지 묻는 가장 매운 물음은 이것이다. **배울 것이 없는
문제를 주면 못 배운다고 말하는가.**

라벨을 무작위로 섞으면 그림과 라벨 사이의 규칙이 사라진다. 남는 것은 그림 만 장과
거기 아무렇게나 붙은 숫자뿐이다. 섞고 나서도 10.19%는 우연히 제자리에 남는데,
열 클래스니 그만큼이 제값이다.

이런 데이터로 익히면 **학습 정확도는 올라가고 검증 정확도는 10%에 머문다.**

| | 10 에폭 | 30 에폭 | 60 에폭 |
|---|---|---|---|
| 학습 (드롭아웃 0.5) | 12.73% | 24.59% | 35.20% |
| 학습 (드롭아웃 없음) | 16.17% | 40.01% | 62.02% |
| 검증 | 10.46% | 9.57% | 10.94% |

학습 쪽이 62%까지 가는 것은 그 만 장을 **외우는** 것이다. 규칙이 없으니 외우는
수밖에 없고, 외운 것은 다른 그림에 옮겨 가지 않으므로 검증은 끝까지 우연 수준에
붙어 있다. 드롭아웃을 끄면 외우기가 훨씬 빨라진다 — 드롭아웃이 막고 있던 것이
바로 이 암기다.

읽는 요령은 **높이가 아니라 벌어짐**이다. 60 에폭에서 62.02% 대 10.94%,
51점이 벌어져 있다. 이 벌어짐이 암기의 모양이다.

그리고 이 실험이 진단이 되는 까닭은 **반대로 나왔을 때**다. 섞은 라벨로 익혔는데
검증 점수가 50%, 80%로 올라간다면 배울 수 없는 것을 배운 것이고, 그럴 수 있는
길은 하나뿐이다. 검증 데이터가 학습 데이터와 섞여 있다. 새 파이프라인을 짰을 때
가장 먼저 해 볼 일이 이것이다.

## 4. 어느 고장이 숫자에 나타나는가

같은 코드에 세 가지 고장을 넣고, 바른 쪽과 견주었다. 2 에폭씩, 씨앗 셋으로 돌려
퍼짐까지 적는다. **퍼짐보다 작은 차이는 차이가 아니다.**

| | 검증 정확도 | 퍼짐 | 정상과의 차 |
|---|---|---|---|
| 정상 | 98.49% | 98.31~98.72 | — |
| `eval()` 빼기 | 97.47% | 97.23~97.83 | −1.02점 |
| 검증 누수 | 98.37% | 98.21~98.54 | −0.12점 |
| `zero_grad()` 빼기 | 87.74% | 86.37~90.39 | −10.75점 |

세 고장이 전혀 다른 소리를 낸다.

**`zero_grad()` 빼기는 큰 소리로 터진다.** 10점이 빠지고, 게다가 퍼짐이
86.37~90.39로 평소의 열 배 넘게 벌어진다. 경사가 배치마다 쌓여 걸음 크기가
들쭉날쭉해지니, 못하기도 하고 **못하는 정도가 흔들리기도** 한다. 숫자만 보고도
무언가 잘못되었음을 안다.

**`eval()` 빼기는 1점쯤 깎는다.** 작지만 퍼짐이 겹치지 않으므로(97.23~97.83 대
98.31~98.72) 재어서 알아낼 수 있는 차이다. 드롭아웃이 켜진 채로 평가하니 유닛
절반을 끄고 답을 내는 셈이다. 다만 한 번만 돌려 97.5%를 보았다면 그냥 좀 아쉬운
모델이라고 넘겼을 크기다.

**검증 누수는 보이지 않는다.** 검증 집합을 학습 데이터로 통째로 바꿔치기했는데도
−0.12점, 씨앗 퍼짐 안이다. 가장 큰 잘못이 가장 조용하다.

## 5. 같은 고장이 판에 따라 보이기도, 안 보이기도 한다

누수가 왜 안 보였는가. 누수가 점수를 부풀리려면 **익힌 그림이 새 그림보다 쉬워야**
한다. 그런데 6만 장을 2 에폭 돌린 모델은 아직 아무것도 외우지 않았다. 익힌
그림이라고 더 잘 맞히지 못한다.

외우는 판으로 옮기면 바로 나타난다.

| 학습 | 본 그림 | 못 본 그림 | 부풀림 |
|---|---|---|---|
| 1,000장 40 에폭 | 99.80% | 94.43% | **+5.37점** |
| 60,000장 2 에폭 | 98.67% | 98.72% | −0.05점 |

같은 잘못, 같은 코드인데 한 판에서는 5점을 부풀리고 다른 판에서는 아무 표도 내지
않는다. 그러니 이렇게 말해야 한다 — **고장이 숫자에 보이느냐는 고장만으로 정해지지
않는다. 어느 판에서 재는지가 함께 정한다.** 작은 데이터로 길게 돌리는 쪽에서 더
위험하다는 뜻이기도 하다.

## 6. 고장이라 불리지만 고장이 아닌 것

넷째로 넣은 것은 입력 정규화를 빼는 것이었다. `ToTensor`가 0\~255인 화소를
0\~1로 바꾸는데, 그걸 되돌려 0\~255를 그대로 먹인다. "입력을 정규화하라"는
말은 어디서나 들을 수 있다.

| | 0\~1 | 0\~255 |
|---|---|---|
| Adam lr=0.001 | 98.72% | 98.32% |
| SGD lr=0.01 | 89.21% | **98.16%** |

Adam에서는 0.40점 손해다. Adam이 파라미터마다 경사 크기로 걸음을 나누므로 입력
스케일이 대부분 상쇄된다. 그리고 SGD에서는 **거꾸로 9점 이득**이다. 입력을 255배
하면 경사도 그만큼 커지고, lr=0.01로 2 에폭이면 덜 익은 쪽이었으니 그 커짐이
도움이 되었다.

그러니 입력 스케일은 그 자체로 좋거나 나쁘지 않다. **러닝레이트와 맞물려서만 뜻을
가진다.** 「최적화 기초」에서 러닝레이트가 옵티마이저 선택보다 크게 작용한 것과
같은 이야기다.

이것도 적어 둘 만한 결과다. 고장 목록에 넣고 재어 보았더니 고장이 아니었다.
재어 보지 않고 목록에 넣어 둔 조언은 이런 식으로 틀린다.

## 7. 정리

| 고장 | 숫자에 나타나는가 | 무엇을 보아야 아는가 |
|---|---|---|
| `zero_grad()` 빼기 | 10점, 퍼짐도 커진다 | 검증 점수만 보아도 안다 |
| `eval()` 빼기 | 1점 | 씨앗 여럿으로 재야 확실하다 |
| 검증 누수 | 외우는 판에서만 | 섞은 라벨 진단, 그리고 집합 교집합 |
| 입력 0\~255 | 옵티마이저에 따라 방향이 바뀐다 | 고장이 아니다 |

끝까지 이어 붙인 파이프라인이 하는 일은 98.84%를 내놓는 것이 아니다. **고장을
냈을 때 어디까지 알려 주는지 알고서 쓰는 것**이다. 위 표의 셋째 줄이 그 알려 줌의
한계다 — 검증 누수는 점수를 보아서는 알 수 없고, 코드를 보거나 섞은 라벨로 물어야
알 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`ConvNet`의 파라미터가 225,034개라고 찍혔다. 층마다 손으로 세어 그 합이 맞는지
확인하라. 어느 층이 대부분을 차지하는가?

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    m = ConvNet().to(DEV)
    tot = 0
    for name, p in m.named_parameters():
        print(f"  {name:12s} {str(tuple(p.shape)):18s} {p.numel():>7,}")
        tot += p.numel()
    print(f"  {'합':12s} {'':18s} {tot:>7,}")
    ```

    ```
      conv1.weight (32, 1, 3, 3)          288
      conv1.bias   (32,)                   32
      conv2.weight (64, 32, 3, 3)      18,432
      conv2.bias   (64,)                   64
      fc1.weight   (128, 1600)        204,800
      fc1.bias     (128,)                 128
      fc2.weight   (10, 128)            1,280
      fc2.bias     (10,)                   10
      합                               225,034
    ```

    손으로 세면 이렇다. 합성곱은 출력 채널 × 입력 채널 × 커널 × 커널 + 편향,
    완전연결은 출력 × 입력 + 편향이다.

    $$
    \begin{aligned}
    \text{conv1} &: 32 \times 1 \times 3 \times 3 + 32 = 320 \\
    \text{conv2} &: 64 \times 32 \times 3 \times 3 + 64 = 18{,}496 \\
    \text{fc1} &: 128 \times 1600 + 128 = 204{,}928 \\
    \text{fc2} &: 10 \times 128 + 10 = 1{,}290 \\
    \text{합} &= 225{,}034
    \end{aligned}
    $$

    **`fc1` 하나가 204,928개로 전체의 91%다.** 합성곱 두 층을 합쳐도 18,816개,
    8%밖에 되지 않는다. 합성곱이 가중치를 자리마다 다시 쓰기 때문이다 — `conv2`의
    18,432개는 11×11 자리 전부에 같이 쓰이는데, `fc1`은 1600개 입력마다 128개
    가중치를 따로 가진다.

    그래서 이 모델을 줄이려면 합성곱 채널을 건드릴 일이 아니라 `fc1`로 들어가는
    1600을 줄여야 한다. 풀링을 한 번 더 넣거나, 전역 평균 풀링으로 바꾸는 길이
    여기서 나온다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`eval()`을 빼먹으면 점수가 1점쯤 낮아진다고 했다. 그런데 낮아지는 것만이
문제인가? **같은 모델을 같은 데이터로 다섯 번** 평가해 보고, `eval()`을 부른
쪽과 안 부른 쪽에서 각각 몇 점이 흔들리는지 재라.

</div>

??? success "풀이"
    ```python
    m = train(DataLoader(Subset(TR, list(range(10000))), batch_size=128,
                         shuffle=True), seed=0, epochs=3)
    ev = [accuracy(m, tel) for _ in range(5)]
    tm = [accuracy(m, tel, eval_mode=False) for _ in range(5)]
    print("  eval()  " + "  ".join(f"{v:.2f}" for v in ev)
          + f"   퍼짐 {max(ev) - min(ev):.2f}점")
    print("  train() " + "  ".join(f"{v:.2f}" for v in tm)
          + f"   퍼짐 {max(tm) - min(tm):.2f}점")
    ```

    ```
      eval()  96.70  96.70  96.70  96.70  96.70   퍼짐 0.00점
      train() 95.18  95.10  95.17  94.95  95.03   퍼짐 0.23점
    ```

    `eval()`을 부른 쪽은 **다섯 번이 소수점까지 같다.** 드롭아웃이 꺼져 있으니
    같은 입력에 같은 출력이 나오고, 평가란 본래 이래야 한다.

    안 부른 쪽은 95.18, 95.10, 95.17, 94.95, 95.03 — 0.23점 안에서 흔들린다.
    매번 다른 유닛이 꺼지므로 **매번 다른 모델을 평가한 것**이다.

    그러니 `eval()`을 빼먹은 값은 낮기만 한 것이 아니라 **되살릴 수 없다.** 어제
    기록해 둔 95.18%를 오늘 다시 내려 해도 나오지 않고, 모델을 고쳐서 0.1점이
    올랐다는 말도 할 수 없다. 흔들림이 그만하기 때문이다.

    여기서 흔들림(0.23점)이 깎임(1.5점)보다 작다는 점도 보아 둘 만하다. 한 번만
    재면 "좀 낮네"로 보이고, 두 번 재어 다른 값이 나와야 비로소 무언가 잘못되었음을
    눈치챈다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
본문에서는 검증 집합을 **통째로** 학습 데이터로 바꿨다. 실제 누수는 그렇게까지
가지 않고 일부만 겹친다. 1,000장을 40 에폭 돌려 외우게 한 뒤, 검증 집합의
0% · 25% · 50% · 100%를 학습 데이터로 채워 가며 점수가 얼마나 부풀려지는지
재라. 씨앗 셋으로 돌려, **씨앗 퍼짐보다 커지는 지점**을 찾아라.

</div>

??? success "풀이"
    ```python
    seen = Subset(TR, list(range(1000)))
    rows = {}
    for seed in (0, 1, 2):
        mm = train(DataLoader(seen, batch_size=128, shuffle=True),
                   seed=seed, epochs=40)
        for f in (0.0, 0.25, 0.5, 1.0):
            k = int(2000 * f)
            parts = []
            if k:
                parts.append(Subset(TR, [i % 1000 for i in range(k)]))
            if 2000 - k:
                parts.append(Subset(TE, list(range(2000 - k))))
            v = accuracy(mm, DataLoader(ConcatDataset(parts), batch_size=512))
            rows.setdefault(f, []).append(v)

    base = rows[0.0]
    print(f"  누수 없을 때 씨앗 퍼짐 {max(base) - min(base):.2f}점")
    for f in (0.0, 0.25, 0.5, 1.0):
        v = rows[f]
        print(f"  누수 {int(f * 100):3d}%   {sum(v) / 3:6.2f}%   "
              f"부풀림 {sum(v) / 3 - sum(base) / 3:+5.2f}점   "
              f"({', '.join(f'{x:.2f}' for x in v)})")
    ```

    ```
      누수 없을 때 씨앗 퍼짐 1.40점
      누수   0%    93.20%   부풀림 +0.00점   (92.55, 93.95, 93.10)
      누수  25%    95.32%   부풀림 +2.12점   (94.70, 95.85, 95.40)
      누수  50%    96.93%   부풀림 +3.73점   (96.75, 97.30, 96.75)
      누수 100%    99.77%   부풀림 +6.57점   (99.80, 99.90, 99.60)
    ```

    부풀림은 누수 비율에 거의 비례한다 — 100%에서 6.57점이니 25%에서 그 4분의 1인
    1.6점쯤을 기대할 수 있고, 실제로 2.12점이다.

    물음의 핵심은 그 옆 칸이다. **씨앗만 바꿔도 1.40점이 흔들린다.** 그래서
    25% 누수가 만드는 2.12점은 간신히 그 위에 있다. 모델을 두 가지 고쳐 보고
    "1점 올랐다"고 말하는 자리에서, 검증 집합 4분의 1이 새고 있어도 그 1점과
    구별되지 않는다는 뜻이다.

    여기서 두 가지가 따라온다.

    첫째, **누수는 비율로 줄여서 안전해지는 종류의 잘못이 아니다.** 조금 새면
    조금 부풀려지는데, 조금은 씨앗 잡음에 묻히고 묻힌 것은 고쳐지지 않는다.
    집합을 나눌 때 인덱스가 겹치지 않는지 한 번 확인하는 편이 훨씬 싸다.

    둘째, 재어서 확인할 길은 여전히 있다. 100% 누수에서 99.77%가 나오는 것처럼
    **외운 그림은 거의 다 맞힌다.** 검증 점수가 학습 정확도에 바싹 붙거나
    99%를 넘어서면 집합을 의심해 볼 자리다.
