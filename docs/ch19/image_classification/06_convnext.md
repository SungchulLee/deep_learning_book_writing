# ConvNeXt

Liu 외의 2022년 논문 "A ConvNet for the 2020s"에서 나온 ConvNeXt는 보기 변환기의 꾸밈 고름을 받아들여 ResNet 얼개를 요즘 것으로 바꾼다. 그 결과는 ImageNet 갈래 매기기에서 Swin 변환기와 맞먹거나 그를 넘어서는 순수 누비기 그물이며, 제대로 꾸미면 누비기가 여전히 경쟁력 있음을 보여 준다.

## 1. 코드

```python
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 941,896
```

## 2. 논의

ConvNeXt은 트랜스포머의 설계 선택을 합성곱 얼거리에 짜임새 있게 들여온다. $7 \times 7$ 깊이별 합성곱은 자기 눈길의 너른 수용 영역을 본뜬다. 학습을 더 든든하게 하려고 배치 정규화 대신 층 정규화를 쓴다. ReLU 대신 GELU을 쓰고, 뒤집힌 병목(좁게-넓게-좁게)은 트랜스포머의 MLP 블록 무늬를 따른다. 켜 스케일, 곧 작은 값으로 초기화한 채널별 학습 가능 스케일 매개변수가 깊은 그물의 학습을 든든하게 한다.

조각내기 줄기는 보폭 4인 $4 \times 4$ 합성곱을 쓰며, 이는 비전 트랜스포머의 조각 임베딩과 닮았다. 들임에서 이렇게 세게 내림 표본하는 것은 여느 합성곱 그물의 차츰차츰 줄이는 방식과 다르지만 잘 듣는다. 그 결과로 합성곱 그물과 트랜스포머 사이의 틈을 메우는 구조가 나온다.

## 연습문제

**연습문제 1.**
$7 \times 7$ 깊이별 합성곱의 수용 영역을 $3 \times 3$ 여느 합성곱의 수용 영역과 견주어라. 같은 수용 영역을 얻으려면 $3 \times 3$ 켜가 몇 개 필요한가?

??? success "연습문제 1 풀이"
    $7 \times 7$ 합성곱 하나의 수용 영역은 $7 \times 7$이다. $3 \times 3$ 합성곱은 켜마다 수용 영역을 2화소씩 넓힌다. 3에서 비롯하면 $n$켜 뒤 수용 영역은 $2n + 1$이다. $2n + 1 = 7$으로 두면 $n = 3$이다. 그러므로 $3 \times 3$ 합성곱 셋을 쌓으면 $7 \times 7$ 하나의 수용 영역과 같아진다.

---

**연습문제 2.**
켜 스케일을 작은 값(예: $10^{-6}$)으로 초기화하면 학습이 든든해지는 까닭을 밝혀라.

??? success "연습문제 2 풀이"
    층 잣수 매개변수를 작은 값으로 첫자리매김하면 익히기가 시작될 때 잔차 덩이가 내놓음에 거의 아무것도 보태지 않는다. 그물이 처음에는 항등 함수처럼 굴어, 익히기를 뒤흔들 수 있는 큰 깨어남과 기울기를 막는다. 익히기가 나아가면서 층 잣수 매개변수가 알맞은 값으로 자라 덩이마다 뜻있게 이바지하게 된다.

---

**연습문제 3.**
ConvNeXtBlock이 LayerNorm 대신 BatchNorm을 쓰도록 고쳐라. 무엇을 바꿔야 하는지 밝히고 있을 수 있는 맞바꿈을 논하여라.

??? success "연습문제 3 풀이"
    ```python
    class ConvNeXtBlockBN(nn.Module):
        def __init__(self, dim, layer_scale_init=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.norm = nn.BatchNorm2d(dim)  # LayerNorm에서 바꿈
            self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # 대신 Conv2d 쓰기
            self.act = nn.GELU()
            self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

        def forward(self, x):
            input = x
            x = self.norm(self.dwconv(x))
            x = self.act(self.pwconv1(x))
            x = self.gamma * self.pwconv2(x)
            return input + x
    ```

    맞바꿈: BatchNorm은 묶음 통계에 기대므로 묶음 크기에 민감하고 아주 작은 묶음에는 맞지 않는다. LayerNorm은 표본마다 고르게 맞춰 더 든든하지만 조금 느릴 수 있다. BatchNorm은 익히는 동안 벌주기 효과를 볼 수 있다.

## 정리하며

**다룬 것** — ConvNeXt

ConvNeXt은 트랜스포머의 설계 선택을 합성곱 얼거리에 짜임새 있게 들여온다.

고갱이 갈래는 `ConvNeXtBlock`, `ConvNeXt`, `ConvNeXtBlockBN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
