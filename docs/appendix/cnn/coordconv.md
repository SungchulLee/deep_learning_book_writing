# CoordConv

CoordConv은 2018년 글 "엮음 신경 그물의 흥미로운 어그러짐과 CoordConv 풀이"에서 나왔으며, 여느 엮음의 밑바탕 한계를 다룬다. 여느 엮음은 꾸밈부터 옮겨도 함께 움직이므로 자리를 그 자체로 담지 못한다. 들임에 자리 값 갈래를 이어 붙이면 CoordConv은 그물이 자리에 매인 거르개를 배우게 하여, 자리 값 되돌이나 물체 알아내기처럼 자리를 알아야 하는 일의 됨됨이를 크게 올린다.

## 코드

```python
#!/usr/bin/env python3
'''
CoordConv - 엮음 신경 그물에 자리 좌표 더하기
논문: "An Intriguing Failing of Convolutional Neural Networks" (2018)
고갱이: 자리 헤아림을 돕도록 좌표 채널을 더한다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device).repeat(1, height, 1)
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device).repeat(1, width, 1).transpose(1, 2)
        
        xx_channel = xx_channel / (width - 1)
        yy_channel = yy_channel / (height - 1)
        
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        
        ret = torch.cat([x, xx_channel, yy_channel], dim=1)
        
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)
        
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class CoordConvNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = CoordConv(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.conv2 = CoordConv(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CoordConvNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

CoordConv의 고갱이 깨침은, 여느 엮음이 자리 차수 어디에나 같은 거르개를 걸기에 자리를 가릴 수 없다는 것이다. 옮겨도 함께 움직이는 결은 많은 일에 이롭지만(어디에 있든 물체를 알아보는 따위), 낱그림점 자리 값을 원핫 격자로 옮기거나 어떤 자리의 물체를 알아내는 것처럼 붙박이거나 견준 자리를 알아야 하는 일에는 걸림돌이 된다.

`AddCoords` 묶음은 두 자리 축을 따라 $-1$에서 $1$까지 잣대 맞춘 자리 값 격자를 만든다. 이를 엮음 앞에서 덧붙은 들임 갈래로 이어 붙여, 옮겨도 함께 움직이는 결을 다스린 채로 깬다. 골라 쓰는 살 자리 값 갈래 $r = \sqrt{x^2 + y^2}$은 가운데에서의 거리 소식을 주어 살 대칭이 있는 일에 쓸모 있다. 종요롭게도 자리 값 갈래에는 배울 매개변수가 없다. 그물은 여느 엮음 짐으로 자리 소식에 얼마나 기댈지 배운다.

CoordConv의 덤은 아주 적다. 엮음 켜마다 들임 갈래 2개(또는 3개)만 는다. 본디 글은 이 단순한 고침만으로 여느 CNN이 아예 못 풀던 "자리 값 바꾸기" 문제가 풀렸고, 만들개 모형과 물체 알아내기의 됨됨이도 나아졌음을 보였다. 그 뒤로 CoordConv은 자리를 아는 일이 종요로운 얼개에서 여느 연장이 되었다.

## 익힘 문제

**익힘 1.**
꼴이 $(2, 3, 8, 8)$인 들임에서 `with_r=True`인 `AddCoords`의 날임 꼴은 무엇인가? x 자리 값 갈래의 $(0, 0)$과 $(7, 7)$ 자리의 값은 얼마인가?

??? success "익힘 1 풀이"
    날임 꼴은 $(2, 6, 8, 8)$이다. 본디 갈래 3개에 x 자리 값, y 자리 값, r 자리 값 갈래가 더해진다. $(0, 0)$ 자리에서는 $x = 0/(8-1) \times 2 - 1 = -1$이고, $(7, 7)$ 자리에서는 $x = 7/7 \times 2 - 1 = 1$이다. 자리 값 갈래는 자리 차수를 따라 $-1$에서 $1$까지 곧게 뻗는다.

---

**익힘 2.**
CoordConv이 만들개 모형(맞겨루기 만들개 따위)에 더욱 이로운 까닭을 밝혀라. 어떤 어그러짐을 콕 집어 다루는가?

??? success "익힘 2 풀이"
    만들개 모형에서 만들개는 숨은 벡터를 자리로 짜인 날임 그림으로 옮겨야 한다. 여느 뒤집은 엮음에는 붙박인 자리에 대한 깨침이 없어, 만들개가 자료에서 자리 얼개를 넌지시 배워야 한다. 그래서 되풀이되는 무늬가 생기고 자리에 따라 짜임새 있게 달라지는 것(어떤 자리에 물체를 놓기 따위)을 만들기 어렵다. CoordConv은 만들개에 자리 값을 드러내 놓고 주어 자리에 매인 만들기 규칙을 곧바로 배우게 한다. 이로써 바둑판 무늬 같은 자국이 줄고 만든 것의 자리를 다스리는 힘이 는다.

---

**익힘 3.**
붙박인 자리 값 대신 주어진 기준 점 $(r_x, r_y)$에 견준 자리 값을 자리마다 주는 `RelativeCoordConv` 갈래를 꾸며라. 고갱이 점 알아내기 같은 일에 쓸모 있을 것이다.

??? success "익힘 3 풀이"
    ```python
    class RelativeAddCoords(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, ref_x, ref_y):
            # x: (B, C, H, W), ref_x/ref_y: (B,) in [-1, 1]
            B, _, H, W = x.size()
            xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            # 기준 점을 뺀다
            dx = xx - ref_x.view(B, 1, 1, 1)
            dy = yy - ref_y.view(B, 1, 1, 1)
            dr = torch.sqrt(dx ** 2 + dy ** 2)
            return torch.cat([x, dx, dy, dr], dim=1)
    ```
    이러면 자리마다 기준 점에서 얼마나 떨어졌는지를 주므로 그물이 견준 거리와 방향을 따져 볼 수 있다.
