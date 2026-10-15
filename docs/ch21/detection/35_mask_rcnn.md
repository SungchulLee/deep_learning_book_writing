# Mask R-CNN의 구성 요소

Mask R-CNN은 영역 제안 기반 객체 탐지에 객체별 마스크 예측을 추가한 모델이다. backbone에서 특징을 추출하고 RPN으로 후보 영역을 만든 뒤, RoIAlign으로 추출한 영역별 특징에서 클래스·상자·마스크를 예측한다. 원리와 실제 마스크 처리는 [인스턴스 분할 절](../segmentation/mask_rcnn.md)에서 자세히 다룬다.

## 1. 축약 코드의 범위

아래 `MaskRCNN`은 전체 탐지기가 아니라 backbone과 RPN 출력의 형태를 살펴보는 축약 예제다. 영역 분류·회귀·마스크 모듈도 선언하지만 `forward`에서는 호출하지 않는다. 앵커 생성, 제안 상자 디코딩·선택, RoIAlign, 정답 매칭과 손실, 최종 탐지·마스크 후처리가 빠져 있다.

```python
#!/usr/bin/env python3
'''
마스크 R-CNN — 낱 물체 나누기 얼거리
논문: "Mask R-CNN" (2017)
핵심: 마스크 어림 가지를 더해 더 빠른 R-CNN을 넓힌다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        # 간추린 등뼈
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # RPN(자리 제안 그물)
        self.rpn_conv = nn.Conv2d(64, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 닻 9개, 갈래 2개(물체/아님)
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 닻 9개, 자리표 4개
        
        # 관심 자리 머리
        self.roi_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        
        # 갈래 매기기와 두름 상자 되돌리기
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # 마스크 어림
        self.mask_conv = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # 자리 제안 그물
        rpn_feat = torch.nn.functional.relu(self.rpn_conv(features))
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        
        # 간추린 내놓음
        return {'features': features, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = MaskRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**기존 출력:**

```
Parameters: 5,178,140
```

매개변수 개수에는 선언된 뒤 사용하지 않는 모듈도 포함된다. 이 출력만으로 마스크 추론이나 탐지 성능이 검증되었다고 볼 수 없다.

## 2. RPN 출력과 영역별 head

RGB 입력 4×3×64×64를 가정하면 stride2 합성곱은 4×64×32×32, stride2 최대 풀링은 4×64×16×16을 만든다. RPN 합성곱은 4×512×16×16이고 반환 사전에는 backbone 특징과 4×18×16×16의 `rpn_cls`, 4×36×16×16의 `rpn_reg`가 들어간다.

18채널은 위치마다 앵커 9개의 전경·배경 로짓 두 개씩이다. 의미 클래스 18개가 아니다. 36채널은 앵커마다 상자 오프셋 네 개이며 아직 이미지 좌표의 경계 상자가 아니다. 앵커의 크기·비율과 오프셋 정의를 정해 디코딩해야 한다.

`roi_head`의 첫 선형 층은 영역마다 64×7×7 특징을 펼친 입력을 가정한다. backbone 전체를 임의로 펼쳐 넣는 층이 아니며 이 코드에는 그 입력을 만드는 RoIAlign이 없다. `cls_score`는 영역별 클래스 로짓, `bbox_pred`는 클래스별 네 좌표 보정값을 만들도록 선언했다. 기본 81클래스는 배경 1개와 연속 인덱스로 재매핑한 객체 클래스 80개를 가정할 수 있지만, 데이터셋의 원래 클래스 ID와 자동으로 일치하지는 않는다.

마스크 모듈의 합성곱은 입력 영역의 공간 크기를 유지하고 클래스별 채널을 만든다. 업샘플링이나 원본 이미지로의 배치도 없으며 실제 순전파에서 사용하지 않는다. 마스크 학습에는 양성 RoI와 대응하는 정답 객체, 같은 좌표계로 정렬한 목표 마스크와 손실이 필요하다. [Mask R-CNN 원 논문](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)의 전체 구조와 이 축약 코드를 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
4×3×64×64 입력에서 실제 forward가 반환하는 텐서 모양을 추적하라.

</div>

??? success "연습문제 1 풀이"

    backbone은 4×64×16×16이다. RPN 중간 특징은 4×512×16×16, 분류 출력은 4×18×16×16, 회귀 출력은 4×36×16×16이다. 반환값은 이 중 backbone 특징·RPN 분류·회귀를 담은 사전이다. 선형 층과 mask_conv는 호출하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
현재 코드가 이미 64×64 RGB 입력을 받을 수 있는가? 이 입력에서 곧바로 객체 마스크도 얻을 수 있는가?

</div>

??? success "연습문제 2 풀이"

    첫 합성곱의 입력 채널이 이미 3이고 backbone·RPN은 해당 공간 크기를 처리한다. 그러나 반환값은 최종 상자나 마스크가 아니다. 이미지 크기 변경만으로 빠진 앵커·제안·RoIAlign·head 호출·후처리가 추가되지는 않는다. 사전 반환값에 .shape를 직접 접근하는 설명도 맞지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
rpn_cls의 18채널과 bbox_pred의 4×num_classes 출력의 의미를 비교하라.

</div>

??? success "연습문제 3 풀이"

    RPN은 앵커 9개마다 전경·배경을 분류하므로 위치별 18채널이다. 영역 head의 bbox_pred는 선택된 RoI별로 의미 클래스마다 네 개의 상자 보정값을 출력하도록 선언했다. 입력 단위와 클래스의 의미가 다르며 어느 쪽도 디코딩 없이 원본 이미지 상자로 해석할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 조각만으로 완전한 Mask R-CNN을 만들 수 있는지 검토하고 추가 모듈의 입출력을 설명하라.

</div>

??? success "연습문제 4 풀이"

    ModuleList는 모듈을 등록하지만 아래 YourBlock은 정의되지 않은 자리 표시자다. backbone 깊이를 바꾸면 RPN·RoI head가 기대하는 채널과 특징 stride를 맞춰야 한다. 여기에 앵커·영역 제안, RoIAlign, 영역별 분류·상자·마스크 예측과 학습·후처리 연결이 필요하다. 단순 반복만으로 빠진 탐지 절차가 완성되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

Mask R-CNN은 제안 영역을 공유해 탐지와 객체별 분할을 수행한다. 이 코드는 backbone·RPN 텐서만 반환하므로 선언된 모듈과 실제 호출 경로를 구분하고, 앵커 오프셋·RoI 특징·최종 상자·마스크의 좌표와 의미를 각각 확인한다.
