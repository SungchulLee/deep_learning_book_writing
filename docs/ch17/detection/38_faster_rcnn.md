# 더 빠른 R-CNN

더 빠른 R-CNN은 2015년 논문 "Faster R-CNN: Towards Real-Time Object Detection"에서 나왔다. 자리 제안에 RPN을 쓰며 끝에서 끝까지 익힐 수 있다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 물체 알아내기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
#!/usr/bin/env python3
'''
더 빠른 R-CNN — 자리 제안 그물로 실시간 물체 알아내기에 다가가기
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
        # 특징 뽑개
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 자리 제안 그물
        self.rpn = nn.Sequential(
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 갈래 2개 * 닻 9개
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 자리표 4개 * 닻 9개
        
        # 관심 자리 모으기와 갈래 매기기
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
        
        # 자리 제안 그물
        rpn_feat = self.rpn(feat)
        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        
        return {'features': feat, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = FasterRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## 논의

`FasterRCNN` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`FasterRCNN`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = FasterRCNN(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `FasterRCNN`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = FasterRCNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
