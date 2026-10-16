# Faster R-CNN의 영역 제안

Faster R-CNN은 backbone 피처를 공유하는 RPN으로 후보 영역을 만들고, 영역별 분류와 상자 회귀를 수행하는 두 단계 탐지기다. [원 논문](https://arxiv.org/abs/1506.01497)의 핵심은 외부 영역 제안 대신 학습 가능한 RPN을 사용하는 데 있다. ‘두 단계’는 이미지마다 서로 다른 뉴럴 넷트웍을 처음부터 별도로 실행한다는 뜻이 아니다.

## 1. 축약 코드의 범위

```python
#!/usr/bin/env python3
'''
더 빠른 R-CNN — 자리 제안 넷트웍으로 실시간 물체 알아내기에 다가가기
논문: "Faster R-CNN: Towards Real-Time Object Detection" (2015)
핵심: 자리 제안에 RPN을 쓰며 끝에서 끝까지 익힐 수 있다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # 피처 뽑개
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 자리 제안 넷트웍
        self.rpn = nn.Sequential(
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 클래스 2개 * 닻 9개
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 자리표 4개 * 닻 9개
        
        # 관심 자리 모으기와 클래스 매기기
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
    
    def forward(self, x):
        feat = self.features(x)
        
        # 자리 제안 넷트웍
        rpn_feat = self.rpn(feat)
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        
        return {'features': feat, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = FasterRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**기존 출력:**

```
Parameters: 30,385,567
```

위 `forward`는 backbone·RPN만 호출해 피처·objectness 로짓·앵커 오프셋을 반환한다. `roi_pool`, `fc`, `cls_score`, `bbox_pred`는 선언되어 있지만 사용하지 않는다. 파라메터 출력에는 이 미사용 head도 포함되며 전체 탐지 순전파가 검증된 것은 아니다.

## 2. 실제 호출 경로와 빠진 연결

4×3×64×64 입력은 패딩된 컨볼루션 뒤에 4×64×64×64, 최대 풀링 뒤에 4×64×32×32가 된다. RPN 중간 피처는 4×512×32×32이고 분류·회귀는 각각 4×18×32×32, 4×36×32×32다. 18은 앵커 9개의 전경·배경 두 로짓씩, 36은 앵커 9개의 네 오프셋씩이다. 아직 객체 클래스나 이미지 좌표의 최종 상자가 아니다.

전체 탐지에는 앵커 생성, 오프셋 디코딩, 영역 필터·NMS와 선택, 영역별 피처 추출, 분류·회귀와 최종 후처리가 필요하다. 학습에서는 앵커·RoI와 정답의 매칭 및 각 로스도 정의한다. RPN과 영역별 head가 backbone을 공유한다는 구조와 실제 학습 절차를 구분한다.

## 3. 전역 풀링과 RoI 풀링의 차이

`AdaptiveMaxPool2d((7,7))` 자체는 주어진 피처 전체를 7×7로 만든다. 제안 상자를 받아 그 상자별 영역을 선택하는 연산이 아니다. 전체 피처에 한 번 적용하면 이미지당 한 피처만 생기므로 객체별 RoI pooling을 대체하지 못한다.

영역별로 64×7×7 피처를 얻었다고 가정하면 펼친 차원은 3136이고 `fc`는 4096차원으로 변환한다. 기본 `cls_score`의 21은 배경과 객체 클래스 20개, `bbox_pred`의 84는 클래스별 네 오프셋에 해당한다. 그러나 이 경로에 영역 피처를 공급하는 코드가 없다. RoI pooling은 영역별 bin을 최대 풀링하는 방식이고, 뒤의 Mask R-CNN에서 사용하는 RoIAlign의 연속 좌표 보간과는 다르다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
4×3×64×64 입력에서 실제 반환값의 텐서 모양을 계산하라.

</div>

??? success "연습문제 1 풀이"

    features는 4×64×32×32, rpn_cls는 4×18×32×32, rpn_reg는 4×36×32×32다. 사전으로 반환되며 fc·cls_score·bbox_pred를 통과하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력을 위해 첫 컨볼루션과 fc를 반드시 수정해야 하는가?

</div>

??? success "연습문제 2 풀이"

    첫 컨볼루션은 이미 3채널을 받으므로 수정할 필요가 없다. 실제 forward에서 fc는 호출되지 않는다. 영역별 7×7 피처를 만드는 전체 경로를 추가할 때 fc 입력 64×7×7을 확인해야 하며, 이미지 전체 크기를 곧바로 fc의 펼친 차원으로 사용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
AdaptiveMaxPool2d만으로 RoI pooling을 구현했다고 할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    영역 좌표를 받아 이미지별 피처에서 해당 영역을 선택하는 과정이 빠져 있다. 전역 피처를 7×7로 줄이는 것과 제안마다 서로 다른 7×7 피처를 만드는 것은 다르다. 영역 좌표와 피처 stride의 대응도 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 조각으로 backbone 깊이를 바꿀 때 탐지 경로에 어떤 연결이 필요한가?

</div>

??? success "연습문제 4 풀이"

    YourBlock을 정의하고 RPN 입력 채널·피처 stride와 앵커 좌표를 일치시켜야 한다. 영역별 피처 추출과 head 입력도 맞춰야 한다. 모듈을 등록하는 것만으로 현재 사용하지 않는 head가 실행되거나 제안 영역 생성이 완성되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

Faster R-CNN의 두 단계는 영역 제안과 영역별 예측이다. 이 예제는 RPN 출력까지만 반환한다. 전역 적응형 풀링을 RoI pooling과 혼동하지 않고, 후보 영역과 head 사이의 좌표·피처 연결을 확인한다.
