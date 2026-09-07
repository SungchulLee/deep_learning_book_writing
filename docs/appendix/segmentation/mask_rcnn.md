# Mask R-CNN

Mask R-CNN was introduced in the 2017 paper "Mask R-CNN." Extends Faster R-CNN by adding mask prediction branch.

This implementation provides a concise, educational reference for Mask R-CNN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## 코드

```python
#!/usr/bin/env python3
'''
Mask R-CNN - Instance Segmentation Framework
Paper: "Mask R-CNN" (2017)
Key: Extends Faster R-CNN by adding mask prediction branch
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        # Simplified backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # RPN (Region Proposal Network)
        self.rpn_conv = nn.Conv2d(64, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, 2 * 9, 1)  # 9 anchors, 2 classes (obj/not)
        self.rpn_reg = nn.Conv2d(512, 4 * 9, 1)  # 9 anchors, 4 coords
        
        # ROI Head
        self.roi_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        
        # Classification and bounding box regression
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # Mask prediction
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
        
        # Simplified output
        return {'features': features, 'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg}

if __name__ == "__main__":
    model = MaskRCNN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## 논의

`MaskRCNN` 갈래는 PyTorch의 `nn.Module` 낯을 써서 모형 얼개를 담는다. `forward` 방법이 셈 그림을 매기므로 익히는 동안 PyTorch의 autograd이 기울기 셈을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
`MaskRCNN`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "익힘 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**익힘 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "익힘 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = MaskRCNN(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**익힘 3.**
들임과 날임의 차수가 같을 때 여느 엮음과 깊이별로 가른 엮음의 매개변수 수와 뜨는 셈 횟수를 견주어라. 셈을 가장 크게 아끼는 때는 언제인가?

??? success "익힘 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 매개변수는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 엮음은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(들임 갈래마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 엮음)로 나눈다. 매개변수의 견줌은 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

**익힘 4.**
`MaskRCNN`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MaskRCNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
