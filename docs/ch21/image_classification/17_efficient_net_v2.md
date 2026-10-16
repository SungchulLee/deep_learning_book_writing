# EfficientNet V2

EfficientNetV2는 학습 속도와 모델 규모를 함께 고려한 탐색·스케일링 및 점진적 학습을 사용하는 CNN 계열이다. Tan과 Le의 ICML 2021 논문은 일부 구간에서 Fused-MBConv를 활용한다. 아래 코드는 블록과 이미지 크기 일정의 골격이며 전체 V2 학습을 구현하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, 2, 1, bias=False)
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = EfficientNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,281,648
```

## 2. 논의

### Fused-MBConv의 연산

일반적인 확장 MBConv는 1×1 확장→depthwise 공간 연산→1×1 투영을 사용한다. Fused-MBConv는 앞의 확장과 공간 연산을 하나의 표준 공간 컨볼루션으로 결합할 수 있다. 아래 블록은 3×3으로 tC 채널을 만든 뒤 1×1로 출력한다.

입력 C, 출력 D, 확장 비율 t, 같은 공간 크기에서 바이어스·BN·SE를 제외한 웨이트는 일반 MBConv가 tC²+9tC+tCD, 해당 fused 구조가 9tC²+tCD다. fused의 웨이트·MAC이 더 많아도 장치의 효율적인 밀집 커널과 메모리 접근 특성 덕분에 빠를 수 있다. 반대로 모든 장치·폭·배치에서 더 빠르다는 보장은 없다.

코드에는 stride 인수, 잔차 덧셈, 확장 비율 1에서의 별도 분기가 없다. 이를 모든 공개 Fused-MBConv 설정과 동일한 블록이라고 가정하지 않는다. 원 방법은 구간별 비용을 고려해 fused와 일반 MBConv를 배치한다.

### 점진적 학습

작은 이미지로 시작해 목표 해상도로 키우면 초기 연산비용을 줄일 수 있다. 이미지 크기만 키웠을 때의 정확도 변화를 보완하려고 데이터 증강·드롭아웃 등의 정규화도 함께 조절한다. 작은 이미지는 언제나 과적합 위험이 작고 큰 이미지는 항상 더 강한 정규화가 필요하다는 보편 법칙은 아니다. 데이터·모델·학습 기간에 맞춘 설계이며 원 논문의 결과는 그 실험 조건에 해당한다.

해상도 변경은 데이터 전처리에 적용해야 하고 학습 일정과 평가 해상도를 구분해야 한다. 단순한 크기 함수만 정의했다고 점진적 학습이 구현된 것은 아니다.

### 본체와 일정 함수의 한계

`EfficientNetV2` 본체는 24채널 stem만 계산한 뒤 1280입력 분류기에 전달하므로 순전파에서 차원이 맞지 않는다. 정의한 `FusedMBConv`도 본체에서 사용하지 않는다. 기존 출력은 파라미터 합산이며 순전파·훈련 검증이 아니다.

연습의 `get_image_size`는 epoch/max_epochs를 선형 보간하고 32의 배수로 **내림**한다. 입력 범위를 제한하지 않아 음수 epoch나 max_epochs보다 큰 값에는 범위 밖 크기가 나올 수 있고, max_epochs=0은 정의되지 않는다. epoch를 0,…,max_epochs−1로 쓰면 목표 끝점에 정확히 도달하지 않을 수 있다. 예제의 range(0,100,20)는 0·20·40·60·80까지만 확인하며 목표 384의 도달을 보여 주지 않는다.

전체 구조와 학습법은 [원 논문](https://proceedings.mlr.press/v139/tan21a.html)을 참고한다. 코드와 출력은 실행하지 않고 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Fused-MBConv의 연산 수가 더 커도 빠를 수 있는 이유와 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    밀집 컨볼루션의 커널 처리량, 메모리 접근과 연산 호출 비용 때문에 적은 depthwise MAC이 곧 짧은 지연시간을 뜻하지 않는다. 그러나 장치·배치·폭과 구현에 따라 반대일 수도 있다. 학습 속도와 추론 속도, MAC과 파라미터 수를 분리해서 비교한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
점진적 학습에서 이미지 크기와 정규화 일정을 함께 정하는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    해상도가 바뀌면 계산량과 모델이 보는 정보, 적절한 증강·드롭아웃의 균형도 달라질 수 있다. 원 방법은 학습 속도와 최종 정확도의 균형을 위해 함께 조절한다. 단순한 보편적 비례 규칙이나 매 단계의 성능 개선 보장은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 크기 일정이 마지막 학습 epoch에서 목표 크기를 보장하는지 검토하라.

</div>

??? success "연습문제 3 풀이"

    epoch/max_epochs를 사용하므로 0부터 max_epochs−1까지의 학습에서는 비율이 1에 도달하지 않는다. 32배수 내림까지 있어 끝점 차이가 더 생길 수 있다. 유효한 범위·분모·마지막 시점의 정의와 실제 데이터 변환에 적용하는 절차가 필요하다. 함수는 정규화 강도를 조절하거나 모델을 학습하지 않는다.

    ```python
def get_image_size(epoch, max_epochs, min_size=128, max_size=384):
    progress = epoch / max_epochs
    size = int(min_size + (max_size - min_size) * progress)
    return (size // 32) * 32  # 32의 배수로 반올림

# 쓰는 보기:
for epoch in range(0, 100, 20):
    size = get_image_size(epoch, 100)
    print(f"Epoch {epoch}: image size = {size}x{size}")
# 내놓음: 128, 176, 224, 288, 336
```


## 정리하며

EfficientNetV2의 효율은 학습을 고려한 구조와 점진적 해상도·정규화의 결합에서 평가한다. 예제의 fused 블록·크기 함수와 미완성 본체를 구분하며 연산 수만으로 장치 속도를 보장하지 않는다.
