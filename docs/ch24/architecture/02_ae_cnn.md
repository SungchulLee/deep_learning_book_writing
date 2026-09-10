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

**연습문제 1.**
첫 누비기 층에서 배운 거르개 32개를 그려 보아라. 어떤 갈래의 특징(모서리, 방울, 결)을 알아내는가?

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

**연습문제 2.**
누비기 자기 부호기와 앞 절의 온전히 이어진 자기 부호기의 매개변수 수와 시험 다시 세우기 평균 제곱 어긋남을 견주어라. 어느 쪽이 매개변수를 더 아끼는가?

??? success "연습문제 2 풀이"
    누비기 자기 부호기는 흔히 온전히 이어진 자기 부호기(약 25만 개)보다 매개변수가 적으면서(약 20만 개) 다시 세우기 어긋남은 더 낮다. 누비기의 무게 나눠 쓰기(3x3 거르개마다 모든 공간 자리에 쓰임)가 화소와 숨은 단위의 이음마다 무게를 따로 배우는 것보다 매개변수를 아끼기 때문이다.

---

**연습문제 3.**
`nn.Upsample` 층을 `nn.ConvTranspose2d`(배울 수 있는 올림 표집)으로 갈음하라. 다시 세우기 품질을 견주고 온전한 해상도에서 다시 세운 것을 살펴 바둑판 무늬 헛것이 있는지 확인하라.

??? success "연습문제 3 풀이"
    ```python
    # nn.Upsample(scale_factor=2)을 다음으로 갈음한다:
    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
    ```
    옮겨 놓은 누비기는 올림 표집 결을 배울 수 있어 다시 세우기 품질이 나아질 수 있다. 다만 특히 짝수 크기 알맹이에서 내놓는 것에 격자 무늬로 보이는 바둑판 헛것을 내기 쉽다. 홀수 알맹이(예컨대 `kernel_size=3, stride=2, padding=1, output_padding=1`)를 쓰면 누그러진다.

## 정리하며

**다룬 것** — 누비기 자기 부호기

누비기 얼개는 그림을 본디 2차원 꼴 그대로 처리해 거르개마다 모서리, 모퉁이, 결 같은 국소 결을 알아내게 한다.

고갱이 갈래는 `ConvolutionalAutoencoder`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
