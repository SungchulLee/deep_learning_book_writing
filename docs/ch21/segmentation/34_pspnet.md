# PSPNet

PSPNet은 여러 격자 크기의 평균 풀링으로 장면 문맥을 모으고 원 피처와 결합하는 시맨틱 분할 모델이다. Zhao 외의 CVPR 2017 논문은 pyramid pooling을 깊은 backbone에 결합했다. 아래 예제는 풀링 모듈과 로짓 보간을 보여 주는 축약형이다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
PSPNet — 피라미드 장면 뜯어 읽기 넷트웍
논문: "Pyramid Scene Parsing Network" (2017)
핵심: 여러 잣수 맥락을 모으는 피라미드 모으기 단원
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
    
    def forward(self, x):
        h, w = x.shape[2:]
        out = [x]
        for stage in self.stages:
            out.append(torch.nn.functional.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(out, dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.ppm = PyramidPooling(64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64 + 64, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1(x)
        x = self.ppm(x)
        x = self.final(x)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    model = PSPNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 607,701
```

## 2. 논의

### 피라미드 풀링의 격자

입력 N×C×H×W를1×1,2×2,3×3,6×6의 격자로 각각 적응형 평균 풀링한다. 이 값은 커널 크기나 업샘플 배수가 아니라 **풀링 출력 격자 수**다.1×1은 전체 문맥을 요약하고 더 큰 격자는 여러 영역의 통계를 남긴다.

각 분기는1×1로 채널을 줄이고 BN·ReLU를 적용한 뒤 원 H×W로 보간한다. 원 피처도 유지해 연결하므로 정밀한 원 피처와 거친 문맥을 함께 제공한다. 풀링된 각 bin이 완전히 분리된 영역이거나 정보가 로스 없이 복원된다고 가정하지 않는다. 적응형 풀링의 bin 범위와 보간은 입력 크기에 따라 달라진다.

### 채널 수와 정규화 조건

분기 수 g와 m=⌊C/g⌋에서 연결 출력은 C+gm채널이다. C가 g로 나누어지면2C지만 일반 C에서는 같지 않다. 기본 C=64, g=4는 m=16, 출력128채널이어서 본체의 final 입력과 맞는다. 분기 목록·출력 폭을 바꾸면 final 입력도 함께 조정해야 한다.

g는 양수이고 m도 유효한 양수여야 한다. 입력 폭보다 분기를 과도하게 늘리면0채널이 될 수 있다.1×1 풀링 분기의 BN은 N×m×1×1을 받아 학습 모드 N=1에서는 문제가 된다. 원 영상이 커도 이 조건은 변하지 않는다.

### 본체의 형태와 한계

64×64 RGB에서 stride2 stem은64×32×32를 만든다. 풀링 각 분기는16채널의 지정 격자를 만든 뒤32×32로 보간한다. 원 피처와 연결하면128×32×32, final은512채널을 거쳐K채널 로짓을 만들고 마지막에64×64로 보간한다.

이 본체에는 깊은 dilated ResNet backbone이나 원 논문의 보조 학습 head 등이 없다. 입력 채널은 이미3이며 분할 출력은 공간을 펼친 fc가 아니라1×1 분류 컨볼루션이다. 임의 공간 크기에 맞춘 마지막 보간이 전체 PSPNet의 정밀한 경계·정확도와 같은 의미는 아니다.

### 다중 스케일 방법의 비교

PSP의 여러 풀링 격자는 전역·지역 영역의 평균 문맥을 제공한다. ASPP는 여러 dilation의 희소 공간 필터를 학습하는 다른 연산이다. FPN의 계층 피처 융합이나 위치 쌍 셀프 어텐션과도 구분한다. 서로 다른 문맥 구조가 같은 계산·표현력을 갖는다고 단정하지 않는다.

연습의 ModuleList는 정의되지 않은 일반 골격이다. 피라미드의 분기를 늘리는 것은 순차 깊이를 늘리는 것과 다르며 풀링 격자·채널 배분·보간·연결·후속 폭을 함께 정해야 한다. [원 논문](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)의 결과와 이 stem 기반 예제를 구분한다. 코드·출력은 실행하지 않고 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,64,32,32)와 격자 [1,2,3,6]에서 피라미드 출력 형태를 추적하라.

</div>

??? success "풀이"
    각 분기는64채널을16채널로 줄이고 지정 격자 (4,16,s,s)를 만든다.32×32로 보간한 네 분기와 원64채널을 연결해 (4,128,32,32)가 된다. 격자 크기는 풀링 출력 크기이며 컨볼루션 커널 크기가 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력의 최종 로짓 형태와 학습 모드 배치1의 조건을 설명하라.

</div>

??? success "풀이"
    stem 뒤64×32×32, PPM 뒤128×32×32, final 뒤K×32×32, 마지막 보간 뒤K×64×64다.1×1 풀링 분기의 BN은 배치1에서 채널당 통계값이 하나라 문제가 된다. fc 입력이나 RGB 채널을 고치는 문제와는 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
C=65, 분기4개일 때 PPM의 출력 채널이2C인가?

</div>

??? success "풀이"
    아니다. 각 축소 폭은 floor(65/4)=16이므로 출력은65+4×16=129채널로2C=130과 다르다. 후속 final의 입력을 실제 연결 폭에 맞춰야 한다. 일반식은 C+g floor(C/g)다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 골격으로 피라미드 분기 수를 바꿀 때 필요한 구성을 설명하라.

</div>

??? success "풀이"

    pool_sizes 목록, 분기당 채널, 병렬 실행과 목표 공간 보간, 원 피처와의 연결 및 후속 컨볼루션 입력 폭을 정해야 한다. YourBlock·dim의 순차 목록만으로 이 병렬 피라미드가 구현되지 않는다. 분기 수와 sequential 깊이를 같은 인수로 혼용하지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

피라미드 풀링은 여러 출력 격자의 평균 문맥을 원 피처와 결합한다. 실제 연결 채널 C+g⌊C/g⌋와 전역 분기 BN의 조건, 깊은 원 모델과 축약 stem 예제의 차이를 구분한다.
