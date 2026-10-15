# YOLOv3의 특징 추출과 단일 탐지 head

[YOLOv3](https://arxiv.org/abs/1804.02767)는 Darknet-53 기반 특징과 세 해상도의 예측을 사용하는 단일 단계 객체 탐지 모델이다. 이 절의 코드는 앞부분 특징 추출과 하나의 탐지 head를 보여 주는 축약 예제다. 원 논문의 전체 backbone이나 세 스케일 탐지를 구현하지 않는다.

## 1. 축약 모델 코드

`ConvBlock`은 합성곱·배치 정규화·Leaky ReLU를 묶는다. `ResidualBlock`은 1×1 합성곱으로 채널을 줄이고 3×3 합성곱으로 복원해 입력을 더한다. `YOLOv3`은 채널을 32→64→128→256으로 늘리며 세 번 다운샘플링하고, 마지막 1×1 합성곱으로 예측 텐서를 만든다.

```python
#!/usr/bin/env python3
'''
YOLOv3 — 한 번만 본다 v3
논문: "YOLOv3: An Incremental Improvement" (2018)
핵심: 한 방 물체 알아내기로 3가지 잣수에서 어림한다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1)
        self.conv2 = ConvBlock(channels // 2, channels, 3)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # Darknet-53 등뼈(간추림)
        self.conv1 = ConvBlock(3, 32, 3)
        self.conv2 = ConvBlock(32, 64, 3, stride=2)
        self.res1 = ResidualBlock(64)
        
        self.conv3 = ConvBlock(64, 128, 3, stride=2)
        self.res2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])
        
        self.conv4 = ConvBlock(128, 256, 3, stride=2)
        self.res3 = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])
        
        # 잣수마다의 알아내기 머리
        self.detect1 = nn.Conv2d(256, (5 + num_classes) * 3, 1)  # 잣수마다 닻 3개
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        
        detections = self.detect1(x)
        return detections

if __name__ == "__main__":
    model = YOLOv3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**기존 출력:**

```
Parameters: 3,267,295
```

출력된 매개변수 수에는 이 축약 구조만 포함된다. 전체 YOLOv3의 매개변수·속도·탐지 성능과 동일한 값으로 인용하지 않는다.

## 2. 출력 텐서의 의미

입력 4×3×64×64에서 첫 합성곱은 4×32×64×64다. 이후 stride2 합성곱 뒤에는 각각 4×64×32×32, 4×128×16×16, 4×256×8×8이 된다. 같은 단계의 잔차 블록은 크기를 유지한다. 기본 객체 클래스 수 80이면 마지막 출력은 4×255×8×8이다.

255는 위치마다 앵커 3개에 대해 4개 좌표 값, objectness 1개, 클래스 값 80개를 출력하기 때문이다. 이미지당 3×8×8=192개 후보의 **raw 출력**이며 최종 192개 객체를 탐지했다는 뜻은 아니다. 이 텐서는 아직 확률이나 원본 이미지 좌표의 상자가 아니다.

YOLOv3의 일반적인 디코딩에서 격자 좌표 c_x,c_y, stride s와 픽셀 단위 앵커 크기 p_w,p_h를 사용하면 중심은 (sigmoid(t_x)+c_x)s, (sigmoid(t_y)+c_y)s이고 크기는 p_w exp(t_w), p_h exp(t_h)다. 클래스는 클래스 간 softmax 대신 독립 sigmoid로 예측하며 objectness와 결합해 후보 점수를 구성한다. 앵커 단위와 좌표 스케일을 일관되게 정해야 한다.

## 3. 전체 탐지기로 확장할 때 필요한 것

현재 코드는 앵커 prior·디코딩·정답 할당·학습 손실·점수 필터·NMS를 포함하지 않는다. 원 논문의 세 스케일 구조에는 더 깊은 특징 단계와 업샘플링·연결 경로, 각 스케일의 탐지 head가 필요하다. 현재 단일 stride8 출력에 이름만 붙여 세 스케일 예측이라고 설명할 수 없다.

클래스 수를 바꾸면 마지막 채널은 3(5+K)로 바뀌지만 데이터의 클래스 ID, 목표 텐서와 손실·디코더도 그 K에 맞아야 한다. 입력 크기에 대한 세 번의 stride2 연산은 홀수 크기에서 올림 효과가 있으므로 임의 입력에서 항상 정확히 H/8, W/8이라고 단정하지 않는다. 속도와 정확도는 해상도·장치·후처리·평가 조건을 함께 명시한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 클래스 수에서 4×3×64×64 입력의 단계별 모양과 최종 후보 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    32×64×64, 64×32×32, 128×16×16, 256×8×8의 특징을 배치 차원 4와 함께 얻는다. 탐지 head는 4×255×8×8이며 이미지당 앵커 후보는 192개다. 잔차 블록은 해당 단계의 모양을 유지하고 선형 층이나 풀링은 사용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
클래스 수를 10으로 바꾸면 출력 채널이 몇 개인가? 입력 채널과 마지막 선형 층도 고쳐야 하는가?

</div>

??? success "연습문제 2 풀이"

    3(5+10)=45채널이다. 입력은 이미 RGB 3채널을 받으며 마지막 head는 선형 층이 아니라 1×1 합성곱이다. 출력 채널뿐 아니라 정답 인덱스·클래스 손실·디코딩도 10클래스에 맞춰야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
raw 출력의 좌표·objectness·클래스 값을 왜 그대로 최종 탐지로 사용할 수 없는지 설명하라.

</div>

??? success "연습문제 3 풀이"

    좌표는 격자·앵커에 대한 매개변수이므로 정해진 변환으로 디코딩해야 한다. objectness와 클래스 로짓에는 sigmoid 등 정의된 변환이 필요하다. 점수 필터와 중복 억제, 입력 resize의 역변환까지 거쳐 최종 결과를 구성한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList를 사용해 잔차 블록 수를 바꿀 때 필요한 조건과 세 스케일 탐지와의 차이를 설명하라.

</div>

??? success "연습문제 4 풀이"

    같은 단계의 잔차 블록은 입력·출력 채널과 공간 크기를 유지해야 하며 단계 전환에서 stride와 채널을 맞춘다. 아래 YourBlock은 실제 정의가 필요한 자리 표시자다. 잔차 블록 반복 수를 늘리는 것만으로 다른 해상도 특징의 연결과 세 개의 탐지 head가 생기지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

이 예제는 합성곱·잔차 블록으로 stride8 특징을 만들고 위치별 앵커 예측을 출력한다. 전체 YOLOv3의 세 스케일 구조와 구분하고, raw 텐서에서 최종 탐지까지 필요한 좌표 변환·학습·후처리를 별도로 이해한다.
