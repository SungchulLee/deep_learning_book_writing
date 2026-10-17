# DeepLabV3

DeepLabv3는 dilated(atrous) convolution과 여러 공간 범위의 문맥을 결합하는 ASPP를 사용하는 시맨틱 분할 모델이다. 아래 구현은 ASPP의 병렬 분기와 입력 크기로의 보간을 보여 준다. backbone은 stem 수준으로 축약되어 있어 원 DeepLabv3의 전체 구조·학습·성능을 재현하지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
DeepLabV3 — 뜻 나누기를 위한 구멍 뚫린 컨볼루션 다시 생각하기
논문: "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
핵심: 구멍 뚫린 자리 피라미드 모으기(ASPP), 여러 벌림 비율
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = torch.nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv_out(x)

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.aspp = ASPP(64, 256)
        self.classifier = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    model = DeepLabV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 820,821
```

## 2. 논의

### Dilated convolution의 간격과 범위

커널 크기 k, dilation=r이면 이론적인 유효 커널 크기는 k_eff=k+(k−1)(r−1)이다.3×3에서는2r+1이므로 r=6·12·18은13·25·37 크기의 영역에 걸친9개 표본을 참조한다. 같은9개 웨이트의 간격을 늘리는 것이지 영역의 모든 픽셀에 별도 웨이트를 두는 큰 dense 커널이 아니다. 파라미터 수는 dilation만 바꿔서는 늘지 않는다.

padding=r, stride1이면3×3 dilated 연산의 공간 크기를 유지한다. 큰 dilation은 유한한 피처 맵에서 많은 표본이 패딩 밖으로 나가거나 사이 간격의 정보가 빠질 수 있다. 범위를 넓힌다는 것과 모든 세부를 포괄한다는 것은 다르다.

### ASPP의 다섯 분기

코드는1×1, r=6·12·18인3×3 세 개, 전역 평균 풀링의 다섯 분기를 쓴다. 각 분기는 O채널이며 global 분기를 원 피처 크기로 보간한 뒤5O채널로 연결한다. 마지막1×1은5O→O로 조합한다. 순차적인 큰 dilation 세 레이어가 아니라 같은 입력을 병렬 처리하는 구조다.

전역 풀링 분기는 BN 직전 N×O×1×1이므로 학습 모드 배치1에서는 채널당 표본이 하나여서 문제가 된다. 큰 입력 공간이 이 위치의 통계 표본 수를 늘려 주지는 않는다. 평가 모드의 저장 통계 사용과 학습 모드의 조건을 구분한다.

### backbone과 입력 좌표

예제 backbone은 stride2 컨볼루션과 stride2 풀링뿐이어서 피처 stride는 대략4다. 원 모델의 더 깊은 dilated backbone과 output stride8·16 등의 설정과 다르다. 같은 dilation 숫자를 쓰더라도 입력 이미지 기준 간격은 backbone stride와 수용영역에 따라 달라진다. 원 모델의 rate를 어떤 해상도의 피처에나 그대로 적용하는 것이 항상 같은 설계는 아니다.

입력64×64에서 backbone은64채널16×16, ASPP는256채널16×16, classifier는K채널16×16을 만들고 마지막에64×64로 보간한다. 반환값은 로짓이며 K클래스 CE에는 N×H×W의 정수 레이블을 사용한다. 보간은 정렬과 크기를 맞추지만 그 자체로 경계의 정확도를 보장하지 않는다.

### 모듈 확장

ASPP 분기를 늘리거나 rate를 바꿀 때는 병렬 목록, 채널 연결과 마지막 projection의 입력 폭을 함께 바꿔야 한다. 연습의 일반 ModuleList 순차 골격은 이런 병렬 설계를 구현하지 않는다. 정확한 scope와 실험은 [DeepLabv3 원 논문](https://arxiv.org/abs/1706.05587)을 참고한다. 기존 코드는 파라미터 수만 출력하며 실행·학습 결과를 새로 확인하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,64,16,16), ASPP(64,256)의 형태를 추적하라.

</div>

??? success "풀이"
    네 공간 분기는 각각 (4,256,16,16)이다. 전역 풀링 분기는 (4,256,1,1)을 만들고16×16로 보간한다. 연결 뒤 (4,1280,16,16), 마지막1×1 뒤 (4,256,16,16)이다. 분기 결과를 더하는 구조가 아니라 연결한 뒤 투영한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB의 최종 출력과 배치1 학습의 주의점을 설명하라.

</div>

??? success "풀이"
    피처는64×16×16, ASPP 뒤256×16×16, 분류 뒤K×16×16이고 마지막 보간으로K×64×64로 돌아온다. 입력 채널은 이미3이며 fc는 없다. 전역 풀링 분기의 BN은1×O×1×1을 받아 학습 모드 배치1에서 유효 통계가 부족하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
3×3 dilation6·12·18의 범위와 파라메터 수를 비교하라.

</div>

??? success "풀이"
    이론적 유효 커널 범위는 각각13·25·37이지만 모두9개 공간 웨이트를 사용한다. 일반 채널 컨볼루션이라 웨이트는9C_inC_out이고 dilation만으로 바뀌지 않는다. 피처 맵의 실제 유효 범위와 패딩, backbone stride까지 고려해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
ASPP rate 분기를 추가할 때 아래 일반 목록 골격에 부족한 부분을 설명하라.

</div>

??? success "풀이"

    각 분기가 같은 입력을 받는 병렬 forward, 공간 정렬, 채널 연결과 projection 폭을 정해야 한다. g개 공간 분기와 전역 분기를 모두O채널로 두면 projection 입력은(g+1)O다. YourBlock·dim의 순차 골격만으로 ASPP가 구현되지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

ASPP는 여러 dilation과 전역 문맥을 병렬 결합한다. dilation의 간격·범위와 파라메터 수, backbone stride를 구분하며 전역 풀링 뒤 BN의 학습 조건과 축약 모델의 범위를 확인한다.
