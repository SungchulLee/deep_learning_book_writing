# Mask R-CNN

Mask R-CNN은 2017년 논문 "Mask R-CNN"에서 나왔다. Faster R-CNN에 마스크 예측 브랜치를 더해 확장했다. 그래서 물체마다 경계 상자와 클래스에 더해 픽셀 단위 마스크까지 내는 인스턴스 분할(instance segmentation, 같은 클래스라도 물체 하나하나를 따로 가르는 분할)을 한다.

여기 구현은 Mask R-CNN을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Mask R-CNN - 인스턴스 분할 구조
논문: "Mask R-CNN" (2017)
핵심: Faster R-CNN에 마스크 예측 브랜치를 더해 확장했다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        # 단순화한 백본
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # RPN(영역 제안 네트워크)
        self.rpn_conv = nn.Conv2d(64, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 9 anchors, 2 classes (obj/not)
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 9 anchors, 4 coords
        
        # ROI 헤드
        self.roi_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        
        # 분류와 경계 상자 회귀
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # 마스크 예측
        self.mask_conv = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # RPN
        rpn_feat = torch.nn.functional.relu(self.rpn_conv(features))
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        
        # 단순화한 출력
        return {'features': features, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = MaskRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 5,178,140
```

## 2. 논의

`MaskRCNN` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다. 다만 `forward`는 백본과 RPN(영역 제안 네트워크, 앵커 9개마다 물체일 점수와 상자 보정값을 내는 부분)까지만 계산하고 그 결과를 딕셔너리로 돌려준다. ROI 헤드, 분류·경계 상자 회귀 레이어, 마스크 브랜치는 정의만 되어 있고 쓰이지 않는다. 제안 상자를 골라 피처를 $7 \times 7$로 잘라 내는 RoIAlign 단계가 없기 때문이다. 출력의 파라메터 5,178,140개 가운데 약 485만 개(94%)가 이 쓰이지 않는 레이어에 있다. `forward` 메서드가 계산 그래프를 정의하므로 학습하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`MaskRCNN`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화(flatten) 뒤에는 $(B, F)$로 중간 형태를 적어라. `forward`가 실제로 지나는 것은 백본과 RPN뿐이다. $224 \times 224$ 입력이면 $(4, 3, 224, 224) \to (4, 64, 112, 112) \to (4, 64, 56, 56)$이 `features`이고, `rpn_conv` 뒤 $(4, 512, 56, 56)$, `rpn_cls`는 $(4, 18, 56, 56)$, `rpn_reg`는 $(4, 36, 56, 56)$이다. 선형 레이어는 순전파에 들어가지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    백본의 첫 컨볼루션 레이어는 이미 입력 채널이 3이다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 공간 차원을 다시 계산하면 $64 \to 32 \to 16$이고, RPN은 크기를 유지한다. 선형 레이어인 ROI 헤드는 $7 \times 7$로 잘라 낸 피처를 받도록 되어 있어 입력 크기와 무관하므로 고칠 차원은 없다. `model = MaskRCNN(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`처럼 확인하되, `forward`가 딕셔너리를 돌려주므로 `.shape`는 값마다 따로 본다. `features`는 $(1, 64, 16, 16)$, `rpn_cls`는 $(1, 18, 16, 16)$, `rpn_reg`는 $(1, 36, 16, 16)$이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 부동소수점 연산 횟수(FLOPs)를 비교하여라. 계산이 가장 크게 절약되는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크게 절약된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MaskRCNN`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = MaskRCNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Mask R-CNN

`MaskRCNN` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다.

핵심 클래스는 `MaskRCNN`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
