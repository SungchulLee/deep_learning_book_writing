# Inception V3

Inception v3은 2015년 글 "Inception 얼개 다시 보기"에서 나왔으며, 몇 가지 고갱이 새로움으로 본디 GoogLeNet을 다듬었다. 큰 거르개를 작고 어긋난 것으로 쪼개는 나눈 엮음, 이름표 매끄럽게 하기 다독임, 익힘을 든든하게 하는 도움 가름개다. 이 나아짐이 함께 더 잘 들고 더 맞는 얼개를 이루어 그림 가름 연구의 여느 밑금이 되었다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Inception v3 - 다듬은 인셉션 얼개
논문: "Rethinking the Inception Architecture" (2015)
고갱이: 인수로 나눈 엮음(nx1과 1xn), 레이블 스무딩, 곁들이 가름개
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    model = InceptionV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 2. 논의

Inception v3의 가장 큰 얼개 이바지는 엮음 나누기다. $5 \times 5$ 엮음을 $3 \times 3$ 엮음 둘을 쌓아 갈음하여 매개변수를 $25C^2$에서 $18C^2$으로 줄인다. 더 나아가 $n \times n$ 엮음을 $n \times 1$ 다음에 $1 \times n$ 엮음으로 나누면 $n=7$일 때 매개변수가 $49C^2$에서 $14C^2$으로 준다. 이렇게 나누어도 받는 밭은 그대로면서 셈은 크게 줄어든다.

이름표 매끄럽게 하기도 종요로운 이바지다. 딱딱한 원핫 과녁 대신 Inception v3은 과녁 분포를 섞음으로 갈음한다. $(1-\epsilon) \cdot \delta_{k,y} + \epsilon / K$이며, $\epsilon$은 흔히 0.1이고 $K$은 갈래 수다. 이러면 모형이 지나치게 자신하지 않게 되고 두루 미침이 좋아진다. 이 솜씨는 그 뒤로 깊은 그물 익힘의 여느 버릇이 되었다.

익히는 동안 가운데 켜에 붙이는 도움 가름개는 덧붙은 기울기 신호를 주어 아주 깊은 그물에서 기울기가 사라지는 탈에 맞선다. 작은 값(0.3)으로 짐을 주고 미루어 볼 때는 떼어 낸다. 기울기 사라짐을 막는 데 얼마나 이바지하는지는 말이 갈리지만, 가운데 드러냄에서 가려내는 결을 북돋우어 잘 듣는 다독임 노릇을 한다.

## 연습문제

**연습문제 1.**
들임 갈래 256, 날임 갈래 256인 $7 \times 7$ 엮음을 $7 \times 1$ 다음 $1 \times 7$ 엮음으로 나눌 때 아끼는 매개변수를 셈하여라.

??? success "연습문제 1 풀이"
    본디 $7 \times 7$ 엮음: $256 \times 256 \times 7 \times 7 = 3,211,264$개. 나눈 뒤에는 $7 \times 1$ 엮음이 $256 \times 256 \times 7 \times 1 = 458,752$개, $1 \times 7$ 엮음이 $256 \times 256 \times 1 \times 7 = 458,752$개다. 나눈 것 모두: $917,504$개. 아낌: $3,211,264 / 917,504 \approx 3.5\times$ 적으니 약 71% 줄었다.

---

**연습문제 2.**
이름표 매끄럽게 하기가 지나치게 맞춰짐을 막는 데 도움이 되는 까닭을 밝혀라. 엇결 엔트로피 잃음 함수와는 어떻게 맞물리는가?

??? success "연습문제 2 풀이"
    딱딱한 원핫 과녁을 쓰면 엇결 엔트로피 잃음이 맞는 갈래의 로짓을 다른 갈래보다 한없이 크게 내도록 몬다. $-\log(\text{softmax})$은 로짓 차이가 끝없이 커질 때만 0에 다가가기 때문이다. 그래서 미루어 봄이 지나치게 자신하고 눈금이 어그러진다. 이름표 매끄럽게 하기는 과녁 분포를 바꾸어 맞는 갈래의 낌새를 1이 아니라 $1-\epsilon+\epsilon/K$으로, 틀린 갈래를 0이 아니라 $\epsilon/K$으로 둔다. 이제 잃음은 모형이 맞는 갈래에 높되 마디 지어진 낌새를 주는 자리에서 마무리 값을 지닌다. 이는 모형이 아리송함을 지니도록 북돋우는 다독임 노릇을 하여 눈금을 맞추고 익힘과 다짐 사이의 됨됨이 틈을 줄인다.

---

**연습문제 3.**
나란한 가지 셋을 지닌 Inception 묶음을 짜라. $1 \times 1$ 엮음, 나눈 $3 \times 3$ 엮음($3 \times 1$과 $1 \times 3$을 씀), 가장 크게 모으기 가지다.

??? success "연습문제 3 풀이"
    ```python
    class InceptionModule(nn.Module):
        def __init__(self, in_ch, out_1x1, out_3x3, pool_proj):
            super().__init__()
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, out_1x1, 1, bias=False),
                nn.BatchNorm2d(out_1x1), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, out_3x3, (3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
                nn.Conv2d(out_3x3, out_3x3, (1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
            )
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                nn.Conv2d(in_ch, pool_proj, 1, bias=False),
                nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
    ```

## 정리하며

**다룬 것** — Inception V3

Inception v3의 가장 큰 얼개 이바지는 엮음 나누기다.

고갱이 갈래는 `InceptionV3`, `InceptionModule`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
