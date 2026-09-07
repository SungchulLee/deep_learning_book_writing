# LeNet

LeNet-5은 얀 르쿤과 그 동아리가 1998년 글 "글월 알아보기에 쓰인 기울기 바탕 배우기"에서 내놓았으며, 가장 이르고 가장 널리 미친 엮음 신경 그물 얼개의 하나다. 본디 MNIST 자료 묶음의 손글씨 숫자 알아보기를 위해 꾸며졌고, 엮음으로 배운 결을 쓰는 신경 그물이 손으로 짠 결 뽑개보다 나을 수 있음을 보였다. 그 꾸밈은 엮음 켜와 모으기 켜를 번갈아 두고 그 뒤에 온통 이은 켜를 두는 CNN의 밑바탕 무늬를 처음 선보였다.

## 코드

```python
#!/usr/bin/env python3
"""
LeNet-5 - Convolutional Neural Network
Paper: "Gradient-Based Learning Applied to Document Recognition" (1998)
Authors: Yann LeCun et al.
Key: Early CNN architecture using convolution, average pooling, and fully
connected layers; widely used for MNIST digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))      # -> (batch, 6, 24, 24)
        x = F.avg_pool2d(x, 2)         # -> (batch, 6, 12, 12)

        x = F.relu(self.conv2(x))      # -> (batch, 16, 8, 8)
        x = F.avg_pool2d(x, 2)         # -> (batch, 16, 4, 4)

        x = torch.flatten(x, 1)        # -> (batch, 16*4*4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = LeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)  # torch.Size([1, 10])
```

## 논의

LeNet-5은 요즘 CNN이 거의 다 따르는 본을 세웠다. 갈래 수가 늘어나는(1→6→16) 엮음 켜 둘을 두고, 켜마다 뒤에 자리 차수를 반으로 줄이는 고르게 모으기를 둔다. 그렇게 나온 결 그림을 펼쳐 차수를 차츰 줄이는(256→120→84→10) 온통 이은 켜 셋에 넣는다. 켜마다 곧지 않은 살림을 걸어 그물이 층을 이룬 결 드러냄을 배우게 한다.

엮음 켜는 가까운 자리의 결을 배운다. 첫 켜는 흔히 가장자리와 기울기를 알아내는 것을 배우고, 둘째 켜는 이를 아울러 굽이나 모서리 같은 더 위 켜의 무늬를 이룬다. 고르게 모으기는 옮겨도 안 바뀜과 자리 눌러 담기를 주지만, 요즘 얼개는 이를 가장 크게 모으기나 걸음 있는 엮음으로 거의 갈음했다. 온통 이은 켜는 배운 결 드러냄 위에서 움직이는 가름개 노릇을 한다.

요즘 잣대로는 단순하지만 LeNet에는 지금도 살아 있는 고갱이 원칙이 담겨 있다. 엮음으로 짐을 나누어 쓰기(온통 이은 켜보다 매개변수가 적다), 엮음과 모으기를 쌓아 이루는 자리의 층, 되돌려 퍼뜨리기로 끝에서 끝까지 익히기다. 매개변수가 6만 개쯤뿐인데도 LeNet은 MNIST에서 99%가 넘는 맞음을 이루어 그 일에 놀랍도록 잘 든다.

## 익힘 문제

**익힘 1.**
꼴이 $(32, 1, 28, 28)$인 들임에 대해 LeNet의 앞으로 걸음 내내 텐서 꼴이 어떻게 바뀌는지 좇아라.

??? success "익힘 1 풀이"
    들임: $(32, 1, 28, 28)$. conv1($5 \times 5$, 거르개 6개) 뒤: $28 - 5 + 1 = 24$이므로 $(32, 6, 24, 24)$. avg_pool2d(알 2) 뒤: $(32, 6, 12, 12)$. conv2($5 \times 5$, 거르개 16개) 뒤: $12 - 5 + 1 = 8$이므로 $(32, 16, 8, 8)$. avg_pool2d(알 2) 뒤: $(32, 16, 4, 4)$. 펼친 뒤: $16 \times 4 \times 4 = 256$이므로 $(32, 256)$. fc1 뒤: $(32, 120)$. fc2 뒤: $(32, 84)$. fc3 뒤: $(32, 10)$.

---

**익힘 2.**
LeNet-5에서 배울 수 있는 매개변수의 모든 수를 셈하여라(치우침도 넣는다).

??? success "익힘 2 풀이"
    conv1: $1 \times 6 \times 5 \times 5 + 6 = 156$개. conv2: $6 \times 16 \times 5 \times 5 + 16 = 2,416$개. fc1: $256 \times 120 + 120 = 30,840$개. fc2: $120 \times 84 + 84 = 10,164$개. fc3: $84 \times 10 + 10 = 850$개. 모두: $156 + 2,416 + 30,840 + 10,164 + 850 = 44,426$개. 매개변수의 거의 다(약 93%)가 온통 이은 켜에 있음을 눈여겨보아라.

---

**익힘 3.**
LeNet을 $32 \times 32$ RGB 그림(CIFAR-10 따위)을 받고 고르게 모으기 대신 가장 크게 모으기를 쓰도록 고쳐라. 차수 셈도 모두 손보아라.

??? success "익힘 3 풀이"
    ```python
    class LeNetCIFAR(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)       # (B,3,32,32) -> (B,6,28,28)
            self.conv2 = nn.Conv2d(6, 16, 5)      # (B,6,14,14) -> (B,16,10,10)
            self.fc1 = nn.Linear(16 * 5 * 5, 120) # after pool: (B,16,5,5) -> 400
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (B,6,14,14)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (B,16,5,5)
            x = torch.flatten(x, 1)                      # (B,400)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    ```
    들임 갈래가 1에서 3으로, 자리 차수가 28에서 32으로 바뀌고, 펼친 크기는 $16 \times 4 \times 4 = 256$이 아니라 $16 \times 5 \times 5 = 400$이 된다.
