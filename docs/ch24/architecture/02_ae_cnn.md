# 누비기 자기 부호기

누비기 자기 부호기는 온전히 이어진 층을 누비기와 모으기 연산으로 갈음해 그림의 공간 짜임을 지키고 층층의 보기 특징을 배운다. 옮김 불변과 국소 이음을 써먹어 매개변수는 적으면서 다시 세우기 품질은 더 낫다. 부호기는 누비기와 최대 모으기로 내림 표집하고 풀개는 가장 가까운 이웃 사이 끼움과 누비기로 올림 표집한다.

## 1. 코드

```python
"""누비기 자기 부호기."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

batch_size, learning_rate, num_epochs = 128, 1e-3, 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

model = ConvolutionalAutoencoder().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for images, _ in train_loader:
        images = images.to(device)
        loss = criterion(model(images), images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    pass
```

## 2. 논의

누비기 얼개는 그림을 본디 2차원 꼴 그대로 처리해 거르개마다 모서리, 모퉁이, 결 같은 국소 결을 알아내게 한다. 부호기의 최대 모으기 층이 단계마다 공간 차원을 절반으로 줄여(28x28에서 14x14, 7x7로) 7x7 공간 해상도에 채널 128개짜리 병목(전체 특징 6,272개)을 만든다. 이 병목이 32차원 온전히 이어진 병목보다 크지만, 누비기 짜임 덕분에 공간에서 뜻 있는 특징만 부호로 담긴다.

누비기마다 뒤에 두는 묶음 고르게 맞추기가 묶음에 걸쳐 깨어남을 고르게 해 익히기를 안정시킨다. 없으면 층의 들임 분포가 익히는 동안 바뀌는 속 공변량 옮김이 생겨 모임이 느려질 수 있다. 묶음 고르게 맞추기와 정류 선형을 아우른 것은 요즘 누비기 얼개의 표준 벽돌이다.

풀개는 옮겨 놓은 누비기 대신 `nn.Upsample(mode='nearest')` 뒤에 누비기를 둔다. 이 "크기 바꾸고 누비기" 방식은 옮겨 놓은 알맹이가 고르지 않게 겹쳐 생기는 바둑판 무늬 헛것을 피한다. 마지막 에스자 깨어남이 내놓는 화소를 $[0, 1]$으로 옭아맨다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
첫 누비기 층에서 배운 거르개 32개를 그려 보아라. 어떤 갈래의 특징(모서리, 방울, 결)을 알아내는가?

</div>

??? success "연습문제 1 풀이"
    ```python
    filters = model.encoder[0].weight.data.cpu()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(32):
        ax = axes[i // 8, i % 8]
        ax.imshow(filters[i, 0].numpy(), cmap='viridis')
        ax.axis('off')
    plt.suptitle("First Layer Filters")
    plt.show()
    ```
    첫 층 거르개는 흔히 여러 각도의 방향 있는 모서리 알아내개, 방울 알아내개, 기울기 거르개를 배운다. 소벨이나 가보르 거르개처럼 손으로 만든 특징 뽑개에 있는 것과 같은 밑감이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
누비기 자기 부호기와 앞 절의 온전히 이어진 자기 부호기의 매개변수 수와 시험 다시 세우기 평균 제곱 어긋남을 견주어라. 어느 쪽이 매개변수를 더 아끼는가?

</div>

??? success "연습문제 2 풀이"
    누비기 자기 부호기는 흔히 온전히 이어진 자기 부호기(약 25만 개)보다 매개변수가 적으면서(약 20만 개) 다시 세우기 어긋남은 더 낮다. 누비기의 무게 나눠 쓰기(3x3 거르개마다 모든 공간 자리에 쓰임)가 화소와 숨은 단위의 이음마다 무게를 따로 배우는 것보다 매개변수를 아끼기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`nn.Upsample` 층을 `nn.ConvTranspose2d`(배울 수 있는 올림 표집)으로 갈음하라. 다시 세우기 품질을 견주고 온전한 해상도에서 다시 세운 것을 살펴 바둑판 무늬 헛것이 있는지 확인하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    # nn.Upsample(scale_factor=2)을 다음으로 갈음한다:
    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
    ```
    옮겨 놓은 누비기는 올림 표집 결을 배울 수 있어 다시 세우기 품질이 나아질 수 있다. 다만 특히 짝수 크기 알맹이에서 내놓는 것에 격자 무늬로 보이는 바둑판 헛것을 내기 쉽다. 홀수 알맹이(예컨대 `kernel_size=3, stride=2, padding=1, output_padding=1`)를 쓰면 누그러진다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
누비기 자기 부호기와 완전 연결 자기 부호기를 같은 16차원 병목으로 견주어라.

</div>

??? success "연습문제 4 풀이"
    | | 매개변수 | 시험 MSE |
    |---|---|---|
    | 완전 연결 | 1,075,488 | 0.02433 |
    | 누비기 | **61,329** | **0.01133** |

    매개변수를 17분의 1만 쓰고 오차가 절반이다. 가중치 공유 덕에 필터 하나를 모든
    자리에서 다시 쓰기 때문이며, 완전 연결층이 자리마다 따로 배워야 하는 것을
    한 번만 배운다.

    [3.4절](../../ch03/mnist/04_cnn.md)의 분류 실험과 같은 결론이다. 이미지에는
    이미지의 구조를 아는 얼개가 낫다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
풀개에서 크기를 되돌릴 때 `ConvTranspose2d`를 쓴다. 출력 크기가 어떻게 정해지는지 식으로 적고, `output_padding`이 왜 필요한지 말하라.

</div>

??? success "연습문제 5 풀이"
    전치 누비기의 출력 크기는 다음과 같다.

    $$H' = (H - 1)\,s - 2p + k + \text{output\_padding}$$

    보통 누비기의 $H' = \lfloor (H + 2p - k)/s \rfloor + 1$을 거꾸로 돌린 꼴이다.

    `output_padding`이 필요한 까닭은 **내림 때문에 정보가 하나 사라졌기** 때문이다.
    보통 누비기에서 $s = 2$이면 $H = 7$과 $H = 8$이 모두 $H' = 4$로 간다. 되돌릴 때
    4에서 출발해 7로 갈지 8로 갈지는 식만으로 정해지지 않는다.

    이 쪽의 코드에서 $k=3, s=2, p=1$로 $7 \to 14$를 만들려면

    $$(7-1)\cdot 2 - 2 + 3 + \text{op} = 13 + \text{op}$$

    이므로 `output_padding=1`이 있어야 14가 된다. 없으면 13이 되어 다음 층에서
    모양이 어긋난다.

    실무에서 이 값을 빠뜨려 생기는 모양 오류가 잦다. 그래서 전치 누비기 대신
    **크기 키우기 뒤에 보통 누비기**(`Upsample` + `Conv2d`)를 쓰는 구조도 많다.
    그쪽은 크기가 명시적이라 헷갈릴 일이 없고, 전치 누비기 특유의 바둑판 자국도
    생기지 않는다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
전치 누비기가 만드는 **바둑판 자국**(checkerboard artifact)은 왜 생기는가?

</div>

??? success "연습문제 6 풀이"
    커널 크기가 보폭으로 나누어떨어지지 않으면, 출력의 자리마다 기여하는 입력의
    개수가 달라진다. $k=3$, $s=2$이면 어떤 자리는 입력 두 개에서, 이웃한 자리는
    하나에서 값을 받는다. 그 불균형이 격자 무늬로 남는다.

    고치는 방법이 셋이다.

    - **$k$를 $s$의 배수로 둔다.** $k=4$, $s=2$이면 기여가 고르다.
    - **크기 키우기 + 보통 누비기로 바꾼다.** 가장 확실하다.
    - **키운 뒤 누비기를 한 번 더 건다.** 자국을 뭉갠다.

    생성 모델에서 이 자국은 눈에 잘 띈다. 초기 GAN 그림에 격자가 보이던 것이 대부분
    이 때문이며, 그래서 요즘 구조는 전치 누비기를 피하는 편이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
누비기 자기 부호기에서 코드의 차원은 무엇인가? 채널 수만 세면 되는가?

</div>

??? success "연습문제 7 풀이"
    **채널 × 높이 × 너비**를 모두 세어야 한다.

    이 쪽의 부호기가 $(1,28,28)$을 $(32,7,7)$로 만든다면 코드는 32차원이 아니라

    $$32 \times 7 \times 7 = 1568$$

    차원이다. 입력이 $28 \times 28 = 784$이므로 **줄어든 것이 아니라 두 배로 늘었다.**

    이 점을 놓치면 압축하고 있다고 착각하기 쉽다. 실제로 다시 세우기 오차를 재면
    0.00068로 아주 낮은데, 눌러 담지 않았으니 당연한 일이다
    ([완전 연결 자기 부호기 연습문제 5](01_ae_fully_connected.md)).

    그래서 누비기 자기 부호기로 정말 압축하려면 마지막에 `Flatten` 뒤 `Linear`를 넣어
    진짜 병목을 만들거나, 채널과 해상도를 충분히 줄여야 한다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
부호기에서 크기를 줄일 때 보폭 2 누비기와 풀링 가운데 무엇을 쓰는 편이 나은가?

</div>

??? success "연습문제 8 풀이"
    자기 부호기에서는 **보폭 2 누비기**가 대체로 낫다.

    풀링은 최댓값을 고르면서 어디서 왔는지를 버린다. 분류에서는 그 버림이 평행 이동에
    강해지는 이득으로 돌아오지만, 자기 부호기는 **그 위치를 도로 알아야** 그림을
    되세울 수 있다. 버린 정보를 다시 지어내야 하므로 손해다.

    보폭 누비기는 가중치를 익혀 가며 줄이므로 무엇을 남길지 스스로 고른다.

    풀링을 쓰되 위치를 기억해 두는 방법도 있다. `MaxPool2d(return_indices=True)`로
    최댓값의 자리를 받아 두었다가 `MaxUnpool2d`에 넘기면 그 자리에 되돌려 놓는다.
    분할(segmentation) 구조에서 쓰이던 방식이며, 요즘은 U-Net처럼 부호기의 특징 맵을
    풀개로 직접 건네주는 이음길(skip connection)로 대체되었다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
U-Net처럼 부호기와 풀개 사이에 이음길을 놓으면 자기 부호기로서 무엇이 문제인가?

</div>

??? success "연습문제 9 풀이"
    **병목이 무의미해진다.**

    이음길은 부호기의 특징 맵을 병목을 **건너뛰어** 풀개로 직접 넘긴다. 그러면 그림을
    되세우는 데 필요한 정보가 굳이 병목을 지날 이유가 없어진다. 극단적으로는 병목이
    아무 일도 하지 않고 이음길만으로 항등 함수를 배울 수 있다.

    자기 부호기의 목적이 "좁은 통로를 지나게 해서 무엇이 중요한지 고르게 하는 것"이므로,
    통로를 우회하는 길을 내주면 목적 자체가 사라진다.

    그러므로 이음길은 **나타냄을 배우려는 자기 부호기에는 쓰지 않는다.** 반대로
    출력의 품질만 중요한 일, 이를테면 분할이나 잡음 제거에서는 이음길이 큰 도움이
    된다. 거기서는 병목의 코드를 쓸 일이 없기 때문이다.

    [퍼짐 모델](../../ch29/index.md)이 U-Net을 쓰는 것도 같은 까닭이다. 그 그물이 할
    일은 잡음을 예측하는 것이지 압축된 코드를 만드는 것이 아니다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
누비기 자기 부호기의 매개변수가 입력 크기와 무관한가?

</div>

??? success "연습문제 10 풀이"
    부호기와 풀개의 **누비기 층은** 무관하다. 필터가 $k \times k \times C$로 정해지므로
    $28 \times 28$이든 $256 \times 256$이든 같다.

    그런데 병목을 만들려고 `Flatten` 뒤에 `Linear`를 넣으면 그 층은 입력 크기에
    비례해 커진다. 이 쪽의 예에서 $(32,7,7) \to 16$ 선형층만 해도
    $1568 \times 16 = 25{,}088$개로 전체 61,329개의 40%를 차지한다.

    그래서 큰 그림을 다룰 때는 선형 병목 대신 누비기로 충분히 줄이거나, 전역 평균
    풀링으로 공간 차원을 없앤 뒤 선형층을 둔다. [3.4절 연습문제 2](../../ch03/mnist/04_cnn.md)에서
    본 "매개변수의 95%가 완전 연결층에 있다"와 같은 이야기다.

## 정리하며

**다룬 것** — 누비기 자기 부호기

누비기 얼개는 그림을 본디 2차원 꼴 그대로 처리해 거르개마다 모서리, 모퉁이, 결 같은 국소 결을 알아내게 한다.

고갱이 갈래는 `ConvolutionalAutoencoder`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
