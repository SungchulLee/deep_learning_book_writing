# EfficientNet

EfficientNet은 2019년 글 "EfficientNet: 엮음 신경 그물의 모형 키우기 다시 보기"에서 나왔으며, 깊이, 너비, 결의 세 차수를 한꺼번에 키우는 이치에 닿는 길을 내놓았다. 겹 잣대 값을 써서, 앞선 모형보다 매개변수와 뜨는 셈 횟수가 훨씬 적으면서도 가장 앞선 맞음을 이룬다. 밑 얼개인 EfficientNet-B0은 신경 얼개 찾기로 찾아냈다.

## 코드

```python
#!/usr/bin/env python3
'''
EfficientNet - 모형 키우기 다시 보기
논문: "EfficientNet: Rethinking Model Scaling for CNNs" (2019)
고갱이: 깊이·너비·해상도를 함께 키우며 매우 잘 든다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MBConv(32, 16, expand_ratio=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 40, stride=2),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(40, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

if __name__ == "__main__":
    model = EfficientNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

MBConv(손전화 뒤집힌 목 엮음) 덩이가 EfficientNet의 밑바탕 벽돌이다. 뒤집힌 나머지 얼개를 따른다. 먼저 갈래 차수를 몇 곱절(흔히 6배)로 넓히고, 넓힌 차수에서 깊이별로 가른 엮음을 건 뒤, 더 작은 날임 차수로 되비춘다. 들임과 날임의 차수가 같으면 나머지 이음이 덩이를 통째로 건너뛰어 기울기가 잘 흐르게 한다. ReLU 대신 SiLU(Swish) 살림을 써서 기울기가 더 매끄럽다.

겹 잣대 방법이 EfficientNet의 고갱이 이론 이바지다. 깊이(켜 늘리기), 너비(갈래 늘리기), 결(그림 키우기)을 따로따로 키우는 대신, 겹 값 $\phi$으로 셋을 함께 키운다. 깊이 $d = \alpha^\phi$, 너비 $w = \beta^\phi$, 결 $r = \gamma^\phi$이고 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$을 지킨다. 이 옭아맴 덕에 $\phi$이 1 오를 때마다 뜨는 셈 횟수가 대략 곱절이 되고, 세 차수 사이의 사이가 B0에서 B7까지 그대로 지켜진다.

겹 잣대가 참으로 미치는 힘은 크다. EfficientNet-B7은 매개변수 6600만 개로 이미지넷에서 으뜸 맞음 84.3%을 이루는데, GPipe은 같은 84.3%에 5억 5700만 개가 든다. 차수를 고르게 키우는 편이 한 차수만 끝까지 키우는 것보다 훨씬 잘 든다는 뜻이다.

## 익힘 문제

**익힘 1.**
겹 잣대 매개변수 $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$과 $\phi = 2$이 주어졌을 때 키운 깊이, 너비, 결의 곱값을 셈하여라.

??? success "익힘 1 풀이"
    깊이 곱값: $d = \alpha^\phi = 1.2^2 = 1.44$. 너비 곱값: $w = \beta^\phi = 1.1^2 = 1.21$. 결 곱값: $r = \gamma^\phi = 1.15^2 = 1.3225$. 따져 보면 $\alpha \cdot \beta^2 \cdot \gamma^2 = 1.2 \times 1.21 \times 1.3225 \approx 1.919 \approx 2$이다. 그러므로 $\phi = 2$이면 밑 모형에 견주어 켜가 약 $1.44 \times$, 갈래가 $1.21 \times$ 너르고, 들임의 결이 $1.32 \times$ 크다.

---

**익힘 2.**
MBConv 덩이가 마지막 되비춤에 또 다른 깊이별 엮음이 아니라 $1 \times 1$ 엮음을 쓰는 까닭은 무엇인가? 선형 목이 하는 몫을 밝혀라.

??? success "익힘 2 풀이"
    $1 \times 1$ 되비춤은 넓힌 드러냄을 곧지 않은 살림 없이 더 낮은 차수로 눌러 담는 "선형 목" 노릇을 한다. 이는 일부러 그런 것이다. 넓힌 밭(갈래 6배)이 깊이별 엮음으로 결끼리의 넉넉한 주고받음을 담고, 되비춤이 그 결을 곧게 아우른다. 여기에 곧지 않은 살림을 더하면 MobileNetV2 글이 보인 대로 낮은 차수의 목에서 소식이 무너진다. 목 차수에서 깊이별 엮음을 쓰면 이미 눌러 담긴 차수에서 갈래마다 따로 자리를 거를 뿐이라, $1 \times 1$ 엮음이 주는 갈래끼리 섞기를 놓친다.

---

**익힘 3.**
`MBConv` 덩이의 깊이별 엮음과 마지막 $1 \times 1$ 되비춤 사이에 쥐어짜 북돋우기(SE) 묶음을 더하여라.

??? success "익힘 3 풀이"
    ```python
    class MBConvSE(nn.Module):
        def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1, se_ratio=0.25):
            super().__init__()
            hidden_dim = in_channels * expand_ratio
            self.use_res = stride == 1 and in_channels == out_channels

            layers = []
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True)
                ])
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            ])
            self.pre_se = nn.Sequential(*layers)

            se_ch = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_ch, 1), nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden_dim, 1), nn.Sigmoid(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.pre_se(x)
            out = out * self.se(out)
            out = self.project(out)
            if self.use_res:
                out = out + x
            return out
    ```
    SE 묶음은 깊이별 엮음 뒤, 되비춤 앞에 놓여 넓힌 갈래 차수에서 움직인다.
