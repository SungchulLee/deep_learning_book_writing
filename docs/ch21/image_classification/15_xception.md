# Xception

Xception은 채널별 공간 컨볼루션과 1×1 채널 혼합을 결합한 depthwise separable convolution을 중심으로 구성한 CNN이다. Chollet의 CVPR 2017 논문은 Inception 설계를 이런 분해의 관점에서 해석한다. 아래 예제는 분리 컨볼루션과 잔차의 원리를 보여 주지만 전체 Xception 본체는 완성되어 있지 않다.

## 1. 코드

```python
import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.sep_conv = SeparableConv2d(32, 64)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.sep_conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = Xception()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 2,052,392
```

## 2. 논의

### 공간 연산과 채널 혼합

`SeparableConv2d`는 각 입력 채널에 공간 필터를 적용한 뒤 pointwise 1×1로 출력 채널을 만든다. 두 컨볼루션 사이에는 활성화가 없고, 뒤에 BN과 ReLU를 적용한다. 같은 폭의 표준 3×3에 비해 웨이트 수를 줄일 수 있지만, 공간·채널 상관이 실제 데이터에서 완전히 독립이라는 통계적 결론은 아니다. 모델의 연산을 분해해 표현하는 가정이다.

Inception은 1×1 병목 뒤 여러 공간 분기를 결합하는 구조이고, depthwise separable 연산은 채널별 필터링 뒤 채널을 혼합한다. 커널 순서뿐 아니라 분기·폭·비선형 배치가 다르므로 순서 하나만으로 성능의 우열을 설명하지 않는다.

### 전체 구조와 예제의 누락

원 Xception은 entry flow, 반복되는 middle flow와 exit flow를 사용하며 여러 분리 컨볼루션을 잔차 블록으로 연결한다. 아래 `Xception`은 stem과 32→64의 분리 컨볼루션 하나만 포함한다. 전역 풀링 뒤 피처는 64차원인데 선형 레이어는 2048입력을 요구하므로 순전파가 실패한다. 기존 출력의 파라미터 수는 이런 누락을 확인하는 실행 검증이 아니다.

### 잔차의 위치와 형태

연습의 지름길은 stride 또는 채널 수가 바뀌면 1×1과 BN으로 투영한다. 이 코드는 분리 컨볼루션 내부에서 ReLU를 적용한 뒤 지름길을 더하며, 덧셈 뒤에는 별도 활성화가 없다. 여러 분리 컨볼루션을 묶는 원 Xception의 블록과 동일한 구조라고 가정하지 않는다. padding·stride가 두 경로의 공간 크기를 실제로 맞추는지도 확인해야 한다.

분리 연산의 파라미터 절감과 실제 장치 지연시간은 다른 지표다. [원 논문](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html)의 ImageNet 비교는 전체 모델과 해당 학습·평가 설정의 결과이며 이 골격의 성능이 아니다. 코드·출력은 실행하지 않고 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
256→256의 3×3 표준·분리 컨볼루션을 바이어스·BN 없이 비교하라.

</div>

??? success "연습문제 1 풀이"
    표준은 9×256²=589,824개, 분리는 9×256+256²=67,840개로 약 8.69배 차이다. 이 파일의 SeparableConv2d에는 출력 BN도 있으므로 전체 학습 파라메터에는 스케일·바이어스 2×256개를 추가한다. 구조나 활성화까지 같은 연산이라는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Inception과 depthwise separable 연산이 다른 점을 설명하라.

</div>

??? success "연습문제 2 풀이"
    Inception은 채널 축소와 병렬 공간 분기의 결합을 사용하고, 분리 컨볼루션은 입력 채널별 공간 필터 뒤 pointwise로 채널을 섞는다. 분기 수·중간 폭·활성화 위치도 다르므로 단순히 순서만 바꾼 같은 모델이 아니다. 어떤 순서가 항상 더 좋다는 일반적 결론도 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 잔차 모듈이 형태를 맞추는 조건과 활성화 위치를 설명하라.

</div>

??? success "연습문제 3 풀이"

    stride≠1이거나 입력·출력 채널이 다르면 1×1 투영과 BN을 사용한다. 기본 3×3 padding=1에서는 투영과 공간 크기를 맞출 수 있지만 다른 커널 설정에서는 다시 확인해야 한다. 주 분기는 SeparableConv2d 안에서 ReLU를 거친 뒤 지름길과 더해지며, 덧셈 뒤 ReLU는 없다. 원 Xception 블록 전체와 동일하다고 부르면 안 된다.

    ```python
class SepConvWithResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.sep = SeparableConv2d(in_ch, out_ch, stride=stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        return self.sep(x) + self.shortcut(x)
```


## 정리하며

Xception의 분리 컨볼루션은 공간 필터링과 채널 혼합을 분리하는 구조다. 이 예제의 64차원 피처와 2048입력 분류기는 맞지 않으며, 잔차 모듈의 활성화 위치도 실제 코드 기준으로 해석한다.
