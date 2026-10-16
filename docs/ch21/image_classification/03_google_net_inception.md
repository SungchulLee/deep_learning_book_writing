# GoogLeNet 인셉션

GoogLeNet은 서로 다른 크기의 컨볼루션과 풀링을 병렬로 적용하는 Inception v1 구조다. Szegedy 외가 2014년에 공개하고 CVPR 2015에 발표한 모델로, 1×1 병목과 전역 평균 풀링을 활용한다. 논문의 대회 성능은 특정 학습·다중 크롭·앙상블 설정의 결과이며 아래 교육용 구현의 성능을 뜻하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    인셉션 단원(차원 줄이기를 곁들인 소박한 판)
    
    1x1, 3x3, 5x5 컨볼루션과 3x3 최대 모으기를 나란히 하고,
    그다음 내놓음을 이어 붙인다.
    
    차원을 줄이려 3x3과 5x5 앞에 1x1 컨볼루션을 쓴다
    셈 값을 줄인다.
    """
    
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1번 가지: 1x1 컨볼루션
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 2번 가지: 1x1 컨볼루션 -> 3x3 컨볼루션
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3번 가지: 1x1 컨볼루션 -> 5x5 컨볼루션
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 4번 가지: 3x3 최대 모으기 -> 1x1 컨볼루션
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionAux(nn.Module):
    """깊은 넷트웍의 그래디언트 사라짐을 이겨내는 곁딸린 클래스 매개."""
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = GoogLeNet(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
```

**출력:**

```
Total parameters: 13,378,280
Input shape: torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 1000])
```

## 2. 논의

### 병렬 분기와 채널 병합

Inception 모듈은 1×1 컨볼루션, 1×1→3×3, 1×1→5×5, 풀링→1×1의 네 분기를 계산하고 채널 축으로 이어 붙인다. 분기별 공간 크기가 같아야 하며 출력 채널 수는 각 분기의 최종 채널 수를 더한 값이다. 입력 192채널인 첫 모듈에서는 64+128+32+32=256채널이 된다.

1×1 컨볼루션은 공간 수용영역을 늘리지 않으면서 채널을 혼합하고 줄일 수 있다. 큰 커널 앞에 병목을 두면 그 커널의 입력 채널 수를 줄여 비용을 낮춘다. 모든 분기는 고정된 구조로 계산된다. 입력마다 한 커널만 자동으로 선택해 나머지 연산을 생략하는 동적 선택 모델은 아니다.

### 보조 분류기

코드는 중간 두 지점에 보조 분류기를 둔다. `training`과 `aux_logits`가 모두 참일 때 `(주 로짓, 보조 로짓1, 보조 로짓2)`를 반환하고 평가 모드에서는 주 로짓만 반환한다. 모델이 보조 출력을 만든다고 로스까지 자동으로 합산되는 것은 아니다. 학습 루프가 각 로스와 웨이트를 정의해야 한다. 원 논문은 보조 로스에 0.3의 웨이트를 사용했다.

보조 분류기는 중간 피처에 학습 신호를 더 주는 목적이다. 주 분류기보다 항상 부정확하다거나 모든 데이터에서 보조 예측의 결합이 무익하다는 결론은 아니다. 표준 추론 경로에서는 제거해 계산을 줄이고 주 출력만 사용한다.

### 논문 구조와 구현 규모

원 논문의 대회 top-5 오류율 6.67%는 앙상블·평가 설정과 함께 읽어야 한다. 단일 네트워크의 파라미터 효율을 설명할 때도 주 경로만 셌는지 보조 분류기를 포함했는지 구분한다. 아래 `total_params`는 `aux_logits=True`로 생성한 보조 분류기까지 포함하므로 흔히 인용하는 추론 경로의 약 700만 규모와 직접 같지 않다.

이 구현은 적응형 풀링과 별도 초기화를 사용하는 교육용 형태다. 사전 학습 웨이트를 읽지 않으며 출력 형태가 맞는다는 사실은 분류 정확도의 검증이 아니다. Inception v3의 컨볼루션 분해·레이블 스무딩과 Inception-ResNet의 잔차 연결은 서로 다른 후속 설계이므로 모든 v2·v3·v4에 잔차 연결이 있다고 설명하지 않는다.

### 큰 커널을 작은 커널로 바꾸기

stride=1인 3×3 두 개의 이론적 수용영역은 5×5 한 개와 같다. 중간 채널 폭과 바이어스를 포함해 파라메터를 세어야 하며 항상 더 적은 것은 아니다. 아래 연습의 16→32→32 설정에서는 오히려 파라메터가 늘어난다. 사이의 ReLU와 합성 구조가 달라 같은 연산은 아니다.

설계와 평가 배경은 [원 논문](https://arxiv.org/abs/1409.4842)을 참고한다. 코드·출력은 실행하거나 재생성하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
첫 Inception 모듈의 주어진 채널 구성에서 바이어스를 포함한 파라메터를 계산하라.

</div>

??? success "연습문제 1 풀이"

    네 분기는 각각 192×64+64=12,352개, (192×96+96)+(96×128×9+128)=129,248개, (192×16+16)+(16×32×25+32)=15,920개, 192×32+32=6,176개다. 합계는 163,696개다. 풀링과 ReLU는 학습 파라메터가 없으며, 출력은 256채널이다. 아래 확인 코드는 보존했지만 실행하지 않았다.

    ```python
    module = InceptionModule(192, 64, 96, 128, 16, 32, 32)
    print(sum(p.numel() for p in module.parameters()))
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
보조 분류기의 반환값과 학습 로스를 구분하라.

</div>

??? success "연습문제 2 풀이"
    학습 모드이며 aux_logits가 참이면 주 출력과 두 보조 출력이 반환된다. 로스는 학습 루프에서 따로 정의해야 하며 출력 반환만으로 보조 로스가 최적화되지는 않는다. 평가 모드에서는 주 출력만 반환한다. 보조 출력의 정확도나 결합 이득은 별도 평가 문제다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
16→32의 5×5 컨볼루션을 16→32→32의 3×3 두 개로 바꿀 때 비용과 수용영역을 비교하라.

</div>

??? success "연습문제 3 풀이"

    공통 1×1 병목을 제외하면 기존은 16×32×25+32=12,832개, 대체 구조는 (16×32×9+32)+(32×32×9+32)=13,888개다. 이 채널 설정에서는 1,056개 증가한다. 두 구조의 이론적 수용영역은 모두 5×5이고 대체 구조에는 중간 비선형이 추가된다. 수용영역이 더 커진다고 설명하면 안 된다.

    ```python
    # 원래 3번 가지: 1x1 -> 5x5
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        nn.ReLU(inplace=True)
    )

    # 고친 3번 가지: 1x1 -> 3x3 -> 3x3
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    ```


## 정리하며

Inception v1은 여러 공간 크기의 분기를 채널로 결합하고 1×1 병목으로 비용을 조절한다. 분기의 형태·출력 채널·보조 로스를 명시하고, 작은 커널로의 교체 비용은 실제 채널 수로 계산한다. 논문 성능과 보조 분류기를 포함한 구현 규모를 구분한다.
