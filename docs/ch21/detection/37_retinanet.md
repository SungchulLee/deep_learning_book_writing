# RetinaNet과 다중 해상도 탐지

RetinaNet은 피처 피라미드(FPN)와 focal loss를 결합한 단일 단계 탐지기다. 촘촘한 앵커 예측에서 쉬운 배경 예제가 매우 많아지는 문제를 다룬다. 아래 코드는 단일 피처 맵과 두 예측 가지를 보여 주는 축약 구조이며 전체 FPN이나 focal loss를 구현하지 않는다.

## 1. 축약 모델 코드

```python
#!/usr/bin/env python3
'''
레티나넷 — 촘촘한 물체 알아내기를 위한 초점 로스
논문: "Focal Loss for Dense Object Detection" (2017)
핵심: 클래스 치우침을 다루는 초점 로스, 피처 피라미드 넷트웍(FPN)
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        # 아래에서 위로 가는 길
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 옆 이음
        self.lateral3 = nn.Conv2d(64, 256, 1)
        
        # 위에서 아래로 가는 길
        self.smooth = nn.Conv2d(256, 256, 3, 1, 1)
    
    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        
        # 옆 이음
        p3 = self.lateral3(c1)
        
        # 매끄럽게 하기
        p3 = self.smooth(p3)
        
        return [p3]

class RetinaNet(nn.Module):
    def __init__(self, num_classes=80, num_anchors=9):
        super().__init__()
        self.fpn = FPN()
        
        # 클래스 매기기 아래넷트웍
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * num_classes, 3, 1, 1)
        )
        
        # 상자 되돌리기 아래넷트웍
        self.box_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 4, 3, 1, 1)
        )
    
    def forward(self, x):
        features = self.fpn(x)
        
        cls_outputs = []
        box_outputs = []
        
        for feat in features:
            cls_outputs.append(self.cls_subnet(feat))
            box_outputs.append(self.box_subnet(feat))
        
        return {'classifications': cls_outputs, 'regressions': box_outputs}

if __name__ == "__main__":
    model = RetinaNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**기존 출력:**

```
Parameters: 4,719,156
```

`FPN`이라는 클래스 이름과 달리 이 구현은 하나의 backbone 피처에 1×1·3×3 컨볼루션을 적용해 원소 하나짜리 목록을 반환한다. 서로 다른 깊이의 피처를 연결하는 top-down·lateral 경로나 P3~P7 피라미드가 없다. 따라서 전체 RetinaNet과 같은 다중 스케일 탐지 능력이 구현되었다고 볼 수 없다.

## 2. 분류와 상자 회귀의 분리

입력이 4×3×64×64이면 stride2 컨볼루션과 최대 풀링 뒤의 피처는 4×64×16×16이다. lateral·smooth 컨볼루션은 4×256×16×16을 만든다. 기본 앵커 수 9, 객체 클래스 수 80에서 분류 가지는 4×720×16×16, 회귀 가지는 4×36×16×16이다. 반환 사전의 각 값은 이 텐서를 하나 담은 목록이다.

분류는 앵커마다 클래스별 로짓 80개를 출력하고, 배경 전용 softmax 채널을 추가하지 않는 sigmoid 방식으로 해석한다. 회귀는 앵커마다 네 오프셋이다. 앵커 정의·정답 매칭·로스·디코딩·NMS는 코드에 없으므로 raw 출력만으로 최종 객체를 얻지 못한다. 원 모델의 공유 subnet 아이디어와 달리 현재는 반복할 피처 맵 자체가 하나뿐이다.

## 3. Focal loss의 역할

정답 클래스에 부여한 확률 p_t에 대해 이진 focal loss는 다음과 같다.

$$\operatorname{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.$$

γ>0이면 잘 맞힌 예제의 로스를 상대적으로 줄이고 α_t로 클래스별 웨이트를 조절할 수 있다. γ=0이면 α 가중 교차 엔트로피가 된다. 쉬운 예제를 완전히 버리는 hard mining과는 다른 가중 방식이다. 전체 검출 로스에는 양성 앵커의 회귀 로스와 유효·무시 앵커의 규칙, 정규화 기준도 필요하다. 이 코드에는 로스 함수가 정의되어 있지 않으므로 focal loss로 학습했다고 설명하지 않는다. 세부 사항은 [원 논문](https://arxiv.org/abs/1708.02002)과 [focal loss 절](focal_loss.md)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 설정에서 4×3×64×64 입력의 피처·분류·회귀 모양과 반환 자료형을 설명하라.

</div>

??? success "연습문제 1 풀이"

    피처는 4×256×16×16, 분류는 4×720×16×16, 회귀는 4×36×16×16이다. 반환값은 사전이고 classifications와 regressions는 각각 텐서 하나를 담은 목록이다. 공간 위치마다 9개 앵커를 가정한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 FPN이 실제 피처 피라미드와 다른 점을 설명하라.

</div>

??? success "연습문제 2 풀이"

    서로 다른 해상도의 backbone 단계도, 상위 피처를 업샘플링해 하위 피처와 결합하는 경로도 없다. 단일 RGB 텐서를 받는 현재 FPN의 이름만으로 다중 스케일 출력이라고 판단하면 안 된다. 실제 피라미드는 각 단계의 채널·stride·정렬을 정의해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
γ=0과 γ>0에서 focal loss가 쉬운 배경 예제를 다루는 방식을 비교하라.

</div>

??? success "연습문제 3 풀이"

    γ=0은 α 가중 교차 엔트로피다. γ>0이면 p_t가 1에 가까운 쉬운 예제에서 (1−p_t)^γ가 작아져 상대 기여가 줄어든다. α·γ와 앵커 정규화의 영향은 구분하며 어떤 데이터에서도 항상 더 정확한 로스라는 보장은 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList만으로 FPN을 확장할 수 있는지 검토하라.

</div>

??? success "연습문제 4 풀이"

    YourBlock은 실제 정의가 필요한 자리 표시자다. 순차 블록 등록만으로 top-down·lateral 연결이 생기지 않는다. 여러 backbone 피처, 채널 정렬, 업샘플링과 결합을 명시하고 각 출력에 공유 예측 subnet을 적용해야 한다. 단계별 앵커와 목표 매칭도 함께 설계한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

RetinaNet은 다중 해상도 피처와 focal loss로 촘촘한 탐지를 구성한다. 현재 예제는 단일 피처 맵의 분류·회귀 가지를 보여 주므로 전체 FPN·학습 로스·탐지 후처리와 구분해 읽는다.
