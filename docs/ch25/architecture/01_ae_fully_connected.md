# 온전히 이어진 자기 부호기

온전히 이어진 자기 부호기는 눌러 담은 자료 나타냄을 배우는 가장 단순한 신경망 얼개이다. 주성분 분석의 선형 쏘기와 달리 자기 부호기는 비선형 깨어남 함수로 더 풍부하고 차원이 낮은 부호를 찾아낸다. 이 보기는 부호기(784에서 32차원)와 풀개(32에서 784로)를 갖춘 온전한 자기 부호기를 세우고 두 값 엇갈린 엔트로피 손실로 MNIST에 익혀 이끌리지 않은 배움의 온전한 물길을 보인다.

## 1. 코드

```python
"""온전히 이어진 자기 부호기."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# === 자리매김 ===========================================================
input_dim, hidden_dim, latent_dim = 784, 128, 32
batch_size, learning_rate, num_epochs = 128, 1e-3, 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 모델 ===================================================================
class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.decode(self.encode(x))

# === 익히기 ================================================================
model = FullyConnectedAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for images, _ in train_loader:
        images_flat = images.to(device).view(images.size(0), -1)
        loss = criterion(model(images_flat), images_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    pass
```

## 2. 논의

병목 층(32차원)이 그물로 하여금 잡음과 남아도는 것은 버리고 숫자 그림의 요긴한 특징을 잡는 눌러 담은 나타냄을 배우게 한다. 성분 32개를 쓴 주성분 분석과 비슷하지만 비선형 부호기와 풀개는 선형 방법이 놓치는 굽은 다양체 짜임을 잡을 수 있다. 에스자 내놓기 깨어남이 다시 세운 것을 $[0, 1]$에 두어 고른 화소 범위에 맞춘다.

$[0, 1]$ 자료에서는 두 값 엇갈린 엔트로피 손실이 평균 제곱 어긋남보다 낫다. 화소마다 베르누이 확률 변수로 보아 0이나 1에 가까운 화소에 더 센 기울기를 주기 때문이다. 차원을 차츰 줄이는(784에서 128, 64, 32로) 대칭 부호기-풀개 얼개는 갑작스러운 병목보다 가장 좋게 하기 쉬운 매끄러운 눌러 담기 길을 만든다.

익힌 자기 부호기의 숨은 공간은 숫자 그림을 눈에 보이는 닮음으로 갈무리한다. 곧 비슷한 숫자가 함께 무리 지고 숨은 부호 사이를 사이 끼움하면 숫자 모양이 매끄럽게 옮아간다. 다만 변분 자기 부호기와 달리 여느 자기 부호기의 숨은 공간에는 벌주기가 없어 풀었을 때 뜻 없는 것이 나오는 "구멍"이 있을 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
숨은 차원 2, 8, 32, 128로 자기 부호기를 익혀라. 숨은 차원에 대한 시험 다시 세우기 평균 제곱 어긋남을 그리고 얻는 것이 줄어드는 지점을 찾아라.

</div>

??? success "연습문제 1 풀이"
    ```python
    for dim in [2, 8, 32, 128]:
        model = FullyConnectedAutoencoder(latent_dim=dim).to(device)
        # ... 20바퀴 익힌다 ...
        # ... 시험 평균 제곱 어긋남을 셈한다 ...
        print(f"latent_dim={dim}: test MSE = {test_mse:.6f}")
    ```
    평균 제곱 어긋남이 2에서 32차원까지 가파르게 떨어진 뒤 평평해진다. 64차원을 넘으면 나아짐이 미미하다. 풀개가 특징 32~64개로도 잘 다시 세울 담이를 갖췄기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
두 값 엇갈린 엔트로피 손실을 평균 제곱 어긋남 손실로 갈음해 다시 익혀라. 다시 세운 것의 눈에 보이는 품질을 견주어라. 어느 손실이 더 또렷한 숫자 그림을 내며 왜 그런가?

</div>

??? success "연습문제 2 풀이"
    `nn.BCELoss()` 대신 `nn.MSELoss()`을 쓰면 흔히 다시 세운 것이 조금 더 흐릿하다. 두 값 엇갈린 엔트로피는 0과 1 가까이에서 기울기를 더 세게 주어 내놓기를 더 두 값답게(또렷한 검정과 흰색으로) 이끌지만, 평균 제곱 어긋남은 모든 어긋남에 똑같이 벌을 주어 "평균 낸" 회색 화소 값이 많아진다. 두 값에 가까운 그림 자료에서는 대체로 두 값 엇갈린 엔트로피가 더 또렷한 결과를 낸다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
부호기 층에 떨구기(비율 0.3)를 더하고 떨구기가 있을 때와 없을 때의 시험 다시 세우기 어긋남을 견주어라. 이 이끌리지 않은 상황에서 벌주기가 지나치게 맞춰지는 것을 막는 데 도움이 되는가?

</div>

??? success "연습문제 3 풀이"
    ```python
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim // 2, latent_dim),
    )
    ```
    떨구기를 쓰면 익히기 손실은 흔히 더 높지만 익히기와 시험 손실의 벌어짐이 줄어든다. MNIST 자기 부호기에서는 자료 묶음이 모델에 견주어 커서 지나치게 맞춰짐이 크지 않으므로 떨구기의 덕이 크지 않다. 자료 묶음이 작으면 벌주기 효과가 더 뚜렷할 것이다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
완전 연결 자기 부호기와 누비기 자기 부호기를 **같은 16차원 병목**으로 맞추어 견주어라. 어느 쪽이 이기며 매개변수는 얼마나 드는가?

</div>

??? success "연습문제 4 풀이"
    | | 매개변수 | 코드 | 시험 MSE |
    |---|---|---|---|
    | 완전 연결 | 1,075,488 | 16 | 0.02433 |
    | 누비기 | **61,329** | 16 | **0.01133** |

    누비기가 **매개변수를 17분의 1만 쓰고도 오차를 절반으로** 줄인다.

    까닭은 3.4절에서 본 것과 같다. 완전 연결 부호기는 첫 줄에서 그림을 펼쳐 화소의
    이웃 관계를 버리므로, 어느 화소가 어느 화소 옆에 있었는지를 자료에서 다시
    배워야 한다. 누비기는 그것을 가정으로 들고 시작한다.

    이 비교는 [3.4절 연습문제 9](../../ch03/mnist/04_cnn.md)의 짝이다. 거기서는 분류로,
    여기서는 다시 세우기로 같은 결론에 이른다. **이미지에는 이미지를 아는 구조를
    쓰는 편이 낫다.**

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
앞의 비교에서 누비기 쪽에 일부러 16차원 병목을 끼웠다. 그 병목을 빼고 누비기 특징 맵을 그대로 코드로 쓰면 어떻게 되는가? 그 결과를 어떻게 읽어야 하는가?

</div>

??? success "연습문제 5 풀이"
    | | 매개변수 | 코드 차원 | 시험 MSE |
    |---|---|---|---|
    | 누비기, 16차원 병목 | 61,329 | 16 | 0.01133 |
    | 누비기, 병목 없음 | **9,569** | **1,568** | **0.00068** |

    오차가 **17배 낮아진다.** 매개변수도 가장 적다. 표만 보면 압도적인 승리다.

    그런데 코드가 $32 \times 7 \times 7 = 1568$차원으로 **입력 784보다 크다.** 눌러
    담기는커녕 두 배로 부풀렸다. 오차가 낮은 것이 당연하며, 극단적으로는 항등 함수에
    가까워지고 있을 뿐이다.

    그러므로 이 줄은 **다시 세우기 오차가 자기 부호기의 품질을 재지 못한다**는 증거로
    읽어야 한다. 가장 낮은 오차를 가장 쓸모없는 설정이 냈다.

    이 함정은 흔하다. 누비기 자기 부호기를 쓸 때 특징 맵을 그대로 코드라 부르는 코드를
    자주 보는데, 그때 실제 압축률이 얼마인지 세어 보면 1보다 작은 경우가 많다.
    **병목의 크기는 채널 수가 아니라 채널 × 높이 × 너비**다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
은닉층을 더 깊게 쌓으면 다시 세우기가 더 좋아지는가?

</div>

??? success "연습문제 6 풀이"
    어느 정도까지는 그렇고 곧 한계에 닿는다.

    깊이가 하는 일은 부호기가 표현할 수 있는 함수의 모둠을 넓히는 것이다.
    $784 \to 16$을 한 층으로 하면 선형 사영뿐이지만, 층을 쌓으면 굽은 다양체를 따라갈
    수 있다([주다양체](04_ae_principal_manifold.md)).

    그런데 병목이 16으로 고정되어 있는 한, 담을 수 있는 정보의 양은 깊이와 무관하게
    16개의 수로 묶여 있다. 깊이는 **그 16개를 얼마나 잘 고르느냐**를 도울 뿐 개수를
    늘리지 못한다. 그래서 이득이 빠르게 줄어든다.

    실무의 감각은 은닉층 두세 개면 MNIST 정도는 충분하다는 것이다. 그보다 깊게 가면
    익히기가 어려워지는 대가가 이득을 넘어선다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
부호기와 풀개의 가중치를 묶으면($W_{\text{dec}} = W_{\text{enc}}^{\top}$) 무엇이 좋고 무엇을 잃는가?

</div>

??? success "연습문제 7 풀이"
    **좋은 점**은 매개변수가 절반이 된다는 것이다. 자료가 적을 때 과적합을 줄이는
    효과도 있다.

    **잃는 것**은 표현력이다. 부호기와 풀개가 따로일 때는 $W_2 W_1$이 임의의 계수
    $k$ 행렬이 될 수 있지만, 묶으면 $W^{\top} W$ 꼴이라 **대칭이고 양의 준정부호**인
    행렬로 제한된다.

    다만 선형이고 제곱 손실이라면 이 제약이 손해가 아니다. 최적해가 어차피 직교
    사영이고 직교 사영은 $P = P^{\top} = P^2$를 만족하기 때문이다. 주성분 분석이
    바로 묶인 꼴이며, 그래서 사영과 복원에 같은 행렬을 쓴다.

    비선형이 되면 이야기가 달라져 묶는 것이 실제로 제약이 된다. 요즘 구조에서 가중치를
    묶는 일이 드문 까닭이며, 남아 있는 대표적인 자리가 말 모델의 들임/내놓기 묻기
    행렬을 공유하는 관례다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
병목에 ReLU를 걸면 어떻게 되는가? 코드가 음수를 가질 수 없다는 것이 문제가 되는가?

</div>

??? success "연습문제 8 풀이"
    문제가 된다. 실제로 재어 보면 같은 구조에서 병목에 ReLU를 건 쪽의 오차가 눈에
    띄게 나쁘다(0.02433 대 0.00977).

    까닭은 담이의 절반을 버리기 때문이다. 코드가 $\mathbb{R}^{16}$ 전체를 쓸 수 있으면
    $2^{16}$개의 부호 조합을 쓸 수 있지만, ReLU를 걸면 첫 사분면 하나에 갇힌다.
    게다가 0이 된 성분은 기울기도 0이라 되살아나기 어렵다.

    그래서 **병목에는 대개 활성화를 걸지 않는다.** 은닉층에는 걸되 코드를 내놓는
    마지막 층은 선형으로 둔다. 이 장의 코드도 그렇게 되어 있다.

    예외가 성긴 자기 부호기다. 거기서는 코드가 0이 되기를 **바라므로** ReLU가 오히려
    어울린다. 무엇을 원하느냐가 설계를 정한다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
자기 부호기의 코드에 아무 의미도 부여되지 않는데, 코드의 차원 하나를 움직이면 그림이 어떻게 달라지는가? 무엇을 기대할 수 있고 무엇은 기대할 수 없는가?

</div>

??? success "연습문제 9 풀이"
    **기대할 수 있는 것**은 그림이 달라진다는 것뿐이다. 코드가 바뀌면 풀개의 출력이
    바뀌고, 대개 매끄럽게 바뀐다.

    **기대할 수 없는 것**은 그 변화가 사람이 알아볼 뜻을 갖는 것이다. "이 차원은
    기울기, 저 차원은 굵기"처럼 갈리기를 바랄 근거가 전혀 없다. 손실에 그런 요구가
    들어 있지 않기 때문이다.

    실제로 [주다양체 연습문제 5](04_ae_principal_manifold.md)에서 본 것처럼, 임의의
    가역 $M$으로 코드를 섞어도 손실이 같다. 그러므로 좌표축에는 아무 특별함이 없으며
    우리가 보는 축은 초기값이 우연히 정한 것이다.

    이 성질을 **얽힘**(entanglement)이라 하고, 축마다 뜻이 갈리게 만드는 일을 얽힘
    풀기라 한다. 그러려면 손실에 따로 요구를 넣어야 한다. $\beta$-VAE가 KL 항의
    무게를 키워 그 일을 시도하며([26장](../../ch26/index.md)), 성김 벌점도 부분적으로
    같은 구실을 한다.

    다만 완전한 얽힘 풀기는 이름표 없이는 원리적으로 불가능하다는 것이 알려져 있다.
    무엇이 "뜻 있는 축"인지는 자료가 정해 주지 않기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 쪽의 자기 부호기를 MNIST가 아닌 자료에 그대로 쓸 수 있는가?

</div>

??? success "연습문제 10 풀이"
    입력 차원만 맞추면 돌아가기는 한다. 다만 두 가지를 손봐야 한다.

    **첫째, 출력 활성화.** 시그모이드는 값이 $[0,1]$인 자료를 전제한다. 표준화해서
    음수가 있는 자료라면 시그모이드를 떼고 손실도 BCE에서 MSE로 바꿔야 한다
    ([손실 함수](../ae/loss_functions.md)).

    **둘째, 병목 크기.** 16은 MNIST에 맞춘 값이다. 자료가 더 복잡하면 늘려야 하고,
    적당한 값은 오차가 꺾이는 자리를 보고 고른다([자기 부호기 연습문제 6](../ae/48_autoencoder.md)).

    그림 자료라면 애초에 [누비기 쪽](02_ae_cnn.md)을 쓰는 편이 낫다. 완전 연결
    자기 부호기는 구조가 없는 표 형태 자료에 더 어울린다.

## 정리하며

**다룬 것** — 온전히 이어진 자기 부호기

병목 층(32차원)이 그물로 하여금 잡음과 남아도는 것은 버리고 숫자 그림의 요긴한 특징을 잡는 눌러 담은 나타냄을 배우게 한다.

고갱이 갈래는 `FullyConnectedAutoencoder`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
