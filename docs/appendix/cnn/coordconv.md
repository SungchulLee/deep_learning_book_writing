# CoordConv

CoordConv는 2018년 논문 "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"에서 나왔으며, 표준 컨볼루션의 근본적인 한계를 다룬다. 표준 컨볼루션은 설계상 이동 등변성(translation equivariance, 입력을 옮기면 출력도 같이 옮겨지는 성질)을 가지므로 절대 위치를 직접 표현하지 못한다. 입력에 좌표 채널을 이어 붙이면 CoordConv는 네트워크가 위치에 의존하는 필터를 트레인하게 하여, 좌표 회귀나 객체 탐지처럼 위치를 알아야 하는 과제의 성능을 크게 올린다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CoordConv - 컨볼루션 뉴럴 네트워크에 위치 좌표 더하기
논문: "An Intriguing Failing of Convolutional Neural Networks" (2018)
핵심: 위치 추론을 돕도록 좌표 채널을 더한다
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

**출력:**

```
Parameters: 221,288
```

## 2. 논의

CoordConv의 핵심 통찰은, 표준 컨볼루션이 공간 차원 어디에나 같은 필터를 적용하기에 위치를 구별할 수 없다는 것이다. 이동 등변성은 많은 과제에 이롭지만(어디에 있든 물체를 인식하는 경우 등), 픽셀 좌표를 원핫 격자로 옮기거나 특정 위치의 물체를 탐지하는 것처럼 절대 위치나 상대 위치를 알아야 하는 과제에는 걸림돌이 된다.

`AddCoords` 모듈은 두 공간 축을 따라 $-1$에서 $1$까지 정규화한 좌표 격자를 만든다. 이를 컨볼루션 앞에서 추가 입력 채널로 이어 붙여, 이동 등변성을 제어된 방식으로 깬다. 선택 사항인 반지름 좌표 채널 $r = \sqrt{x^2 + y^2}$은 중심에서의 거리 정보를 주어 방사 대칭이 있는 과제에 쓸모 있다. 중요하게도 좌표 채널에는 트레인할 파라메터가 없다. 네트워크는 일반적인 컨볼루션 웨이트로 공간 정보에 얼마나 의존할지 트레인한다.

CoordConv의 추가 비용은 아주 적다. 컨볼루션 레이어마다 입력 채널 2개(또는 3개)만 는다. 원 논문은 이 단순한 변경만으로 표준 CNN이 아예 못 풀던 "좌표 변환" 문제가 풀렸고, 생성 모델과 객체 탐지의 성능도 나아졌음을 보였다. 그 뒤로 CoordConv는 위치 인식이 중요한 구조에서 자주 쓰이는 도구가 되었다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
형태가 $(2, 3, 8, 8)$인 입력에서 `with_r=True`인 `AddCoords`의 출력 형태는 무엇인가? x 좌표 채널의 $(0, 0)$과 $(7, 7)$ 위치의 값은 얼마인가?

</div>

??? success "풀이"
    출력 형태는 $(2, 6, 8, 8)$이다. 원래 채널 3개에 x 좌표, y 좌표, r 좌표 채널이 더해진다. $(0, 0)$ 위치에서는 $x = 0/(8-1) \times 2 - 1 = -1$이고, $(7, 7)$ 위치에서는 $x = 7/7 \times 2 - 1 = 1$이다. 좌표 채널은 공간 차원을 따라 $-1$에서 $1$까지 선형으로 변한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CoordConv가 생성 모델(GAN 등)에 특히 이로운 까닭을 설명하여라. 어떤 실패 양상을 구체적으로 다루는가?

</div>

??? success "풀이"
    생성 모델에서 제너레이터는 잠재 벡터를 공간 구조를 가진 출력 이미지로 옮겨야 한다. 표준 전치 컨볼루션(transposed convolution)에는 절대 위치라는 개념이 없어, 제너레이터가 데이터에서 공간 구조를 암묵적으로 학습해야 한다. 그래서 반복되는 패턴이 생기고, 위치에 따라 체계적으로 달라지는 내용(특정 위치에 물체를 놓기 등)을 생성하기 어렵다. CoordConv는 제너레이터에 좌표를 명시적으로 주어 위치에 의존하는 생성 규칙을 곧바로 트레인하게 한다. 이로써 체커보드 패턴 같은 아티팩트가 줄고 생성 결과의 위치를 제어하는 능력이 는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
절대 좌표 대신 주어진 기준점 $(r_x, r_y)$에 대한 상대 좌표를 위치마다 주는 `RelativeCoordConv` 클래스를 설계하여라. 키포인트 탐지 같은 과제에 쓸모 있을 것이다.

</div>

??? success "풀이"
    ```python
    class RelativeAddCoords(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, ref_x, ref_y):
            # x: (B, C, H, W), ref_x/ref_y: (B,) in [-1, 1]
            B, _, H, W = x.size()
            xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            # 기준점을 뺀다
            dx = xx - ref_x.view(B, 1, 1, 1)
            dy = yy - ref_y.view(B, 1, 1, 1)
            dr = torch.sqrt(dx ** 2 + dy ** 2)
            return torch.cat([x, dx, dy, dr], dim=1)
    ```
    이러면 위치마다 기준점에서 얼마나 떨어졌는지를 주므로 네트워크가 상대적인 거리와 방향을 추론할 수 있다.

## 정리하며

**다룬 것** — CoordConv

CoordConv의 핵심 통찰은, 표준 컨볼루션이 공간 차원 어디에나 같은 필터를 적용하기에 위치를 구별할 수 없다는 것이다.

핵심 클래스는 `AddCoords`, `CoordConv`, `CoordConvNet`, `RelativeAddCoords`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
