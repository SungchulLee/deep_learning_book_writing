# CBAM

엮음 덩이 눈길 묶음(CBAM)은 2018년 같은 이름의 글에서 나왔으며, 갈래 눈길과 자리 눈길을 차례로 걸어 결 그림을 다듬는다. "무엇"에 눈길을 줄지(갈래 눈길)와 "어디"에 줄지(자리 눈길)를 배워, 어떤 CNN 등뼈에도 거의 덤 없이 끼워 넣을 수 있는 가볍고도 잘 듣는 얼개를 준다.

## 코드

```python
#!/usr/bin/env python3
'''
CBAM - Convolutional Block Attention Module
Paper: "CBAM: Convolutional Block Attention Module" (2018)
Key: Sequential channel and spatial attention mechanisms
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CBAM_ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.cbam = CBAMBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CBAM_ResNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

CBAM은 잇따른 두 작은 묶음으로 이루어진다. 갈래 눈길 묶음은 고르게 모으기와 가장 크게 모으기로 자리 소식을 모아 갈래끼리의 사이를 쓴다. 둘을 나누어 쓰는 여러 켜 퍼셉트론에 넣고 그 결과를 아우른다. 이렇게 두 가지로 모으면 고른 되받음(어떤 결이 두루 있는지)과 가장 도드라진 되받음(어떤 결이 가장 센지)을 함께 담아, 하나만 쓸 때보다 넉넉한 갈래 밝힘을 얻는다.

자리 눈길 묶음은 결 그림 안에서 알려 주는 바가 큰 자리를 짚어 갈래 눈길을 채워 준다. 갈래 축을 따라 고르게 모으기와 가장 크게 모으기로 갈래 소식을 눌러 담고, 두 밝힘을 이어 붙인 뒤 엮음을 걸어 자리 눈길 그림을 낸다. 갈래 눈길 다음에 자리 눈길을 거는 차례가 고갱이다. 갈래 눈길이 먼저 "무엇"이 걸리는지 고르고, 그다음 자리 눈길이 그 결이 "어디"에서 가장 걸리는지 정한다.

꾸밈은 일부러 가볍게 했다. 갈래 눈길의 줄임 견줌(기본값 16)이 목 MLP을 작게 하고, 자리 눈길은 엮음 하나만 쓴다. 그래서 ResNet 같은 이미 있는 얼개에 넣기 쉽고, 매개변수는 흔히 1%도 안 늘면서 가름, 알아내기, 나누기 일에서 한결같이 맞음을 올린다.

## 익힘 문제

**익힘 1.**
꼴이 $(B, 64, H, W)$인 들임 결 그림과 줄임 견줌 16이 주어졌을 때 `ChannelAttention` 묶음의 매개변수 수를 셈하여라.

??? success "익힘 1 풀이"
    나누어 쓰는 MLP은 치우침 없는 $1 \times 1$ 엮음 둘로 이루어진다. 첫째는 $64 \to 64/16 = 4$으로 매개변수가 $64 \times 4 = 256$개, 둘째는 $4 \to 64$으로 $4 \times 64 = 256$개다. 모두 $256 + 256 = 512$개다. 모으기 켜에는 배울 매개변수가 없고 시그모이드도 매개변수가 없다.

---

**익힘 2.**
CBAM이 자리 눈길보다 갈래 눈길을 먼저 거는 까닭은 무엇인가? 거꾸로 하거나 나란히 하지 않는 이치를 느낌으로 밝혀라.

??? success "익힘 2 풀이"
    갈래 눈길이 먼저 어느 결 갈래가 종요로운지 정하여 "무엇"에 눈길을 줄지 답한다. 갈래의 종요로움으로 결에 짐을 다시 준 뒤라야 자리 눈길이 그 결이 "어디"에 있는지를 더 잘 짚는다. 자리 눈길을 먼저 걸면 어느 갈래가 가장 걸리는 소식을 지녔는지 모른 채 모든 갈래를 똑같이 다루게 된다. 차례로 걸면 소식이 폭포처럼 이어진다. 갈래를 골라 결을 다듬어 주므로 자리 눈길이 더 맑은 신호로 자리를 짚는다. 본디 글은 갈래를 먼저 거는 차례가 자리를 먼저 걸거나 나란히 거는 것보다 한결같이 낫다고 겪어 보아 밝혔다.

---

**익힘 3.**
`SpatialAttention` 묶음을 모으기 셈의 수를 골라 잡을 수 있게 넓혀라(고르게 모으기와 가장 크게 모으기에 더해 $L^2$ 노름 모으기를 넣는 따위). 고친 갈래를 써라.

??? success "익힘 3 풀이"
    ```python
    class SpatialAttentionExtended(nn.Module):
        def __init__(self, kernel_size=7, use_l2=True):
            super().__init__()
            in_channels = 3 if use_l2 else 2
            self.use_l2 = use_l2
            self.conv = nn.Conv2d(in_channels, 1, kernel_size,
                                  padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            descriptors = [avg_out, max_out]
            if self.use_l2:
                l2_out = torch.norm(x, p=2, dim=1, keepdim=True)
                descriptors.append(l2_out)
            combined = torch.cat(descriptors, dim=1)
            return self.sigmoid(self.conv(combined))
    ```
    $L^2$ 노름 모으기는 자리마다의 온 힘을 재어, 고른 값(평균 신호)과 가장 큰 값(봉우리 신호) 밝힘을 채워 준다.
