# 조건부 변분 자기 부호기

조건부 변분 자기 부호기(cVAE)는 부호기와 풀개를 모두 갈래 이름표에 조건 지어 여느 변분 자기 부호기를 넓히고, 그래서 만들어 내기를 다스릴 수 있게 한다. 갈래 이름표가 주어지면 그 갈래의 다양한 표본을 만들 수 있고 하나만 뜨거운 이름표 벡터를 섞어 갈래 사이를 사이 끼움할 수도 있다. 특정 숫자 갈래 만들기, 모양새 옮기기, 자료 부풀리기 같은 쓰임새에 쓸모 있다.

## 1. 코드

```python
"""조건부 변분 자기 부호기(cVAE)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, num_classes=10):
        super().__init__()
        self.input_dim, self.latent_dim, self.num_classes = input_dim, latent_dim, num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x, c):
        return self.fc_mu(self.encoder(torch.cat([x, c], dim=1))), \
               self.fc_logvar(self.encoder(torch.cat([x, c], dim=1)))

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, labels):
        c = F.one_hot(labels, self.num_classes).float() if len(labels.shape) == 1 else labels
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, class_label, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        c = F.one_hot(labels, self.num_classes).float()
        return self.decode(z, c)

if __name__ == '__main__':
    model = ConditionalVAE()
    x = torch.randn(32, 784)
    labels = torch.randint(0, 10, (32,))
    reconstruction, mu, logvar = model(x, labels)
    print(f"Reconstruction shape: {reconstruction.shape}")
```

**출력:**

```
Reconstruction shape: torch.Size([32, 784])
```

## 2. 논의

조건 짓는 장치는 곧바르다. 곧 하나만 뜨거운 이름표를 부호기와 풀개의 들임에 잇는다. 부호기에서는 그물이 갈래마다 다른 부호화 분포를 배우게 돕는다. 풀개에서는 뽑은 숨은 부호로 옳은 숫자 갈래를 만드는 데 필요한 갈래 정체를 준다. 그래서 숨은 공간이 갈래 사이 흔들림(어느 숫자인가)이 아니라 갈래 안의 흔들림(모양새, 기울기, 굵기)을 잡는다.

`sample` 방법이 조건부 만들어 내기의 핵심 이점을 보인다. 곧 갈래 이름표를 정하고 $z \sim \mathcal{N}(0, I)$을 뽑으면 바라는 어떤 갈래의 다양한 것이든 만들 수 있다. `interpolate` 방법은 하나만 뜨거운 이름표를 선형으로 섞어 갈래 사이 매끄러운 옮아감(예컨대 "3"이 "8"로 바뀌는 것)을 낸다.

결정적인 설계 세부 하나는 부호기가 익히는 동안 이름표를 받는다는 것이다. 그래서 부호기가 이름표로 자료의 일부를 설명하며 "속임수"를 써서 앎이 덜 담긴 숨은 나타냄을 배울 수 있다. 어떤 얼개는 풀개만 조건 지어 숨은 공간이 모든 흔들림을 잡게 하지만 그러면 익히기가 더 어렵다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
MNIST에 조건부 변분 자기 부호기를 익히고 숫자마다(0~9) 표본 10개를 만들어라. 가로줄마다 한 숫자 갈래가 되도록 10x10 격자로 보여라. 만든 표본이 목표 갈래와 맞는가?

</div>

??? success "연습문제 1 풀이"
    ```python
    model.eval()
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        samples = model.sample(digit, 10).view(-1, 28, 28).detach()
        for j in range(10):
            axes[digit, j].imshow(samples[j].numpy(), cmap='gray')
            axes[digit, j].axis('off')
    plt.show()
    ```
    만든 표본이 목표 갈래와 또렷이 맞아야 하며, 이는 풀개가 조건 이름표로 내놓는 숫자를 다스리는 법을 배웠음을 보인다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`interpolate` 방법으로 갈래 이름표 "3"과 "8" 사이를 사이 끼움하라. 옮아가는 동안 어떤 중간 숫자 모양이 나타나는가?

</div>

??? success "연습문제 2 풀이"
    사이 끼움은 흔히 양 끝에서 "3"과 "8"을 닮은 모양을 지나며, 중간 걸음에서는 "8"의 위아래 고리가 차츰 드러나는 섞인 꼴이 보인다. 쓰는 숨은 부호에 따라 중간 걸음이 "5"나 "6"을 닮을 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
풀개만 조건 짓도록 조건부 변분 자기 부호기를 고쳐라(부호기에서 이름표 잇기를 없앤다). 숨은 공간의 배치와 표본 품질을 여느 조건부 변분 자기 부호기와 견주어라.

</div>

??? success "연습문제 3 풀이"
    부호기에 조건이 없으면 숨은 공간이 갈래 정체와 모양새 흔들림을 모두 잡아야 한다. 그러면 흔히 갈래마다 숨은 공간의 뚜렷한 자리를 차지하고(조건 없는 변분 자기 부호기와 비슷하다) 이름표에 조건 지어진 풀개가 또렷한 갈래 신호로 내놓기를 다듬는다. 풀개가 앎이 덜 담긴 숨은 부호를 받으므로 표본 품질이 조금 낮을 수 있지만 숨은 공간은 더 짜임새 있고 풀이하기 쉬워진다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
조건부 변분 자기 부호기는 부류 정보를 어디에 넣는가? 왜 두 곳인가?

</div>

??? success "연습문제 4 풀이"
    부호기와 풀개 **양쪽**에 넣는다.

    ```python
    mu, logvar = encoder(torch.cat([x, y], dim=1))   # 부호기
    xhat       = decoder(torch.cat([z, y], dim=1))   # 풀개
    ```

    풀개에 넣는 까닭은 분명하다. 뽑을 때 어떤 숫자를 만들지 지정하려면 풀개가 그 값을
    받아야 한다.

    부호기에도 넣는 까닭이 더 재미있다. 부호기가 $y$를 알면 코드에 부류를 담을 **까닭이
    없어진다.** 이미 아는 것을 코드에 실어 보낼 이유가 없으므로, 코드에는 부류를 뺀
    나머지(기울기, 굵기, 글씨체)만 남는다.

    그래서 같은 $z$에 다른 $y$를 넣어 "같은 글씨체의 다른 숫자"를 얻을 수 있다.
    이것이 조건부로 두는 큰 이득이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
조건부로 두면 표본 품질이 얼마나 나아지는가?

</div>

??? success "연습문제 5 풀이"
    부류를 지정해 500개씩 뽑아 분류기에게 판정시키면 이렇다.

    | 숫자 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 평균 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | 지정한 부류로 판정된 비율 | 90% | 94% | 86% | **99%** | 91% | 86% | 95% | 93% | **84%** | 83% | **90.1%** |

    평균 90.1%다. 조건 없는 변분 자기 부호기의 57.4%에 견주면 크게 나아진 것이지만,
    두 수가 **다른 것을 재고 있음**을 짚어 두어야 한다. 57.4%는 "무엇이든 숫자로
    보이는가"이고 90.1%는 "지정한 그 숫자로 보이는가"다. 뒤쪽이 더 어려운 물음인데도
    점수가 높다.

    다시 세우기도 조금 좋아진다(81.22 → 80.74). 부류를 공짜로 알려 주었으니 코드가
    나를 일이 줄어든 덕이다. KL이 20.32에서 16.54로 줄어든 것이 그 증거다. **코드가
    담아야 할 정보가 실제로 줄었다.**

    부류마다의 차이도 읽을 만하다. 3이 99%로 가장 쉽고 9가 83%, 8이 84%로 어렵다.
    9는 4·7과, 8은 3·5와 헷갈리기 쉬운 모양이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
조건부 변분 자기 부호기의 KL 항은 무엇이 달라지는가?

</div>

??? success "연습문제 6 풀이"
    식의 모양은 그대로이고 뜻이 달라진다.

    $$D_{\mathrm{KL}}\big(q(z \mid x, y) \,\|\, p(z)\big)$$

    조건이 $x$뿐에서 $(x,y)$로 바뀌었을 뿐 사전 분포는 여전히 $\mathcal{N}(0,I)$이다.
    그래서 코드는 **부류와 무관하게** 하나의 표준 정규 분포에 맞춰진다.

    이것이 뽑기를 쉽게 만든다. 부류마다 다른 데서 뽑을 필요 없이 $\mathcal{N}(0,I)$에서
    뽑고 원하는 $y$를 붙이면 된다.

    사전 분포를 $p(z \mid y)$로 조건부로 두는 선택도 있는데
    ([사전 분포 연습문제 9](prior.md)), 그러면 부류마다 코드가 다른 자리에 놓여
    숨은 공간이 갈라진다. 무엇을 원하는지에 따라 고를 일이다.

    실제로 잰 KL이 16.54로 조건 없는 쪽의 20.32보다 작다는 점이 이 설계를 뒷받침한다.
    부류를 코드에서 덜어 냈으므로 코드가 나르는 정보가 줄었고, 그만큼 KL도 줄었다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
부류 표지를 원-핫으로 넣는 것과 묻기(embedding)로 넣는 것은 어떻게 다른가?

</div>

??? success "연습문제 7 풀이"
    MNIST의 10부류라면 실질적인 차이가 거의 없다.

    원-핫 벡터에 선형층을 걸면 그것이 곧 묻기 표다. `nn.Embedding(10, d)`와
    `nn.Linear(10, d)`에 원-핫을 넣는 것은 같은 셈이다.

    차이가 생기는 것은 부류가 많을 때다.

    | | 원-핫 | 묻기 |
    |---|---|---|
    | 부류 10개 | 문제없다 | 문제없다 |
    | 부류 10,000개 | 입력이 10,000차원 | $d$차원으로 압축 |
    | 부류 사이 관계 | 모두 등거리 | 비슷한 부류가 가까워질 수 있다 |

    마지막 칸이 실질적인 이득이다. 원-핫은 모든 부류를 서로 무관하게 다루지만, 묻기는
    4와 9가 비슷하다는 것을 배울 수 있다. 부류 수가 많고 서로 관계가 있을 때 도움이 된다.

    MNIST에서는 원-핫이 간단하고 충분하다. 다만 조건을 **연속값**으로 주고 싶을 때
    (회전 각도, 굵기 같은 것) 원-핫이 아예 맞지 않으므로, 그때는 값을 그대로 넣거나
    묻기로 옮긴다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
조건부 변분 자기 부호기에서 코드 $z$와 부류 $y$가 정말 분리되었는지 어떻게 확인하겠는가?

</div>

??? success "연습문제 8 풀이"
    분리되었다는 것은 **코드에서 부류를 알아낼 수 없다**는 뜻이므로, 알아내려고 해 보면 된다.

    코드 $z$만으로 부류를 맞히는 분류기를 익힌다. 정확도가 10%(찍기)에 가까우면 분리된
    것이고, 높으면 코드가 여전히 부류를 나르는 것이다.

    이 검사가 좋은 까닭은 **실패 쪽으로 기울어 있다**는 점이다. 분류기가 부류를 못
    맞히는 것은 정보가 없다는 꽤 강한 증거다. 반대로 잘 맞히면 분리가 안 된 것이 분명하다.

    한 가지 조심할 것이 있다. 완전한 분리가 반드시 목표는 아니다. 부류와 글씨체가
    통계적으로 얽혀 있으면(예컨대 1은 가늘게 쓰는 경향이 있으면) 글씨체 정보가 부류를
    조금은 알려 주는 것이 **자연스럽다.** 그 경우 정확도가 10%보다 높은 것이 오류가 아니다.

    같은 생각의 다른 검사가 눈으로 확인하기에는 더 낫다. $z$를 고정하고 $y$만 0에서
    9까지 바꾸어 한 줄로 그려 보는 것이다. 열 개의 숫자가 **같은 글씨체로** 나오면
    분리가 잘된 것이 눈에 보인다. 굵기와 기울기가 유지되는지 보면 된다.

    이 검사는 수치가 안 나오지만 무엇이 얽혔는지 알려 준다. 두 검사를 함께 쓰는 것이 좋다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
익힐 때 쓰지 않은 부류 조합을 지정해 뽑을 수 있는가?

</div>

??? success "연습문제 9 풀이"
    MNIST처럼 조건이 하나(부류 하나)일 때는 그런 조합이 없다. 열 부류를 모두 익혔으니까.

    물음이 뜻을 갖는 것은 **조건이 여럿일 때**다. 예컨대 부류와 색을 조건으로 주고
    "빨간 3"을 본 적 없이 익혔다면, 뽑을 때 그 조합을 지정할 수 있는가?

    될 때도 있고 안 될 때도 있는데, 조건을 어떻게 넣었는지가 갈림길이다.

    - 조건을 **따로** 넣었으면(부류 벡터와 색 벡터를 각각) 조합을 밀어 넣을 수 있고,
      모델이 두 축을 독립으로 다루었다면 그럴듯한 것이 나온다
    - 조합마다 하나의 표지로 넣었으면(30가지 조합을 30차원 원-핫으로) 본 적 없는
      표지를 줄 수 없다

    그래서 **조건을 쪼개어 넣는 것이 일반화에 유리하다.** 다만 이렇게 만든 조합이
    그럴듯할 것이라는 보장은 없다. 모델이 색과 모양을 정말 독립으로 배웠어야 하고,
    그것은 자료와 얼개에 달렸다.

    이것이 조합적 일반화라 불리는 어려운 문제이며, 조건부 모델의 표현력을 가늠하는
    좋은 시험이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
조건부로 만들면 잃는 것이 있는가?

</div>

??? success "연습문제 10 풀이"
    몇 가지 있다.

    **표지가 필요하다.** 가장 큰 제약이다. 자기 부호기와 변분 자기 부호기의 매력 하나가
    표지 없이 익힌다는 것인데, 조건부는 그것을 버린다.

    **뽑을 때 부류를 정해야 한다.** 자료의 부류 비율대로 뽑고 싶으면 그 비율을 따로
    알아 그대로 뽑아 주어야 한다. 조건 없는 모델은 알아서 그 비율로 낸다.

    **코드가 부류를 안 나른다.** 이것은 이득이기도 하고 손실이기도 하다. 코드를 특징으로
    써서 부류를 맞히려 한다면, 조건부 모델의 코드는 그 정보를 일부러 뺀 것이라 쓸모가
    없다.

    마지막 것이 실수하기 쉬운 자리다. 나타냄 학습이 목적이면서 표지가 있다고 조건부로
    만들면, 코드에서 정작 원한 정보를 지워 버린다. **표지가 있다는 것이 조건부로 할
    이유가 되지는 않는다.** 무엇에 쓸 코드인지가 정한다.

## 정리하며

**다룬 것** — 조건부 변분 자기 부호기

조건 짓는 장치는 곧바르다.

고갱이 갈래는 `ConditionalVAE`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
