# FCN

FCN은 이미지 수준 분류기의 피처와 분류 연산을 공간적으로 적용하고 업샘플링해 픽셀별 예측을 학습하는 접근이다. Long 외의 CVPR 2015 논문은 여러 해상도의 score map을 결합한 모델을 제시했다. 아래 `FCN8s`는 세 피처 단계와 단일 업샘플만 사용하는 축약형이며 원 FCN-8s의 융합 구조를 재현하지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
FCN — 뜻 나누기를 위한 온통 컨볼루션 넷트웍
논문: "Fully Convolutional Networks for Semantic Segmentation" (2015)
핵심: 뜻 나누기를 끝에서 끝까지 하는 첫 컨볼루션 뉴럴 넷트웍으로 온전히 이은 레이어를 컨볼루션 레이어로 바꾼다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # VGG 방식 인코더
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 건너뛰는 이음을 위한 점수 레이어
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        
        # 상향 표본화
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        score = self.score_pool3(conv3)
        upscore = self.upscore(score)
        
        return upscore

if __name__ == "__main__":
    model = FCN8s()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,853,802
```

## 2. 논의

### 완전 컨볼루션과 score map

완전연결 레이어도 입력 피처의 공간 크기에 대응하는 커널로 재해석하면 공간 여러 위치에 같은 분류 연산을 적용할 수 있다. 원 FCN은 이런 변환과 end-to-end 학습, 얕고 깊은 score의 결합을 활용한다. 모든 dense layer가 언제나1×1 하나로 대체된다는 뜻은 아니다. 원래 펼친 공간 범위까지 연결한 레이어는 그 범위의 커널이 필요하다.

코드는 세 VGG식 블록의 풀링으로 H·W를8분의1로 줄인다. 마지막256채널 피처를1×1로 K채널 score map으로 바꾼다. 출력은 확률이 아닌 로짓이며 클래스별 점수를 공간적으로 공유한다.

### 원 FCN-8s와 다른 점

원 FCN-8s는 깊은 예측을 업샘플하고 pool4·pool3의 score를 더해 상대적으로 세밀한 예측을 만든다. 이 예제는 pool3에서 바로 score를 만들고 뒤의 conv4·conv5 및 skip 융합을 생략한다. 최종 피처 stride가8이라는 사실만으로 원 FCN-8s와 같은 모델이 되는 것은 아니다.

### 전치 컨볼루션의 출력 크기

`upscore`는 kernel16, stride8이며 padding=0이다. 입력 공간 q에 대해 출력은 (q−1)×8+16=8q+8이다. 따라서 원 높이 H의 출력은8⌊H/8⌋+8이고, H가8의 배수여도 원 크기보다8 크다.64입력은 pool3가8이고 출력72,256입력은 pool3가32이고 출력264다.

이 코드에는 원 입력 크기에 맞추는 crop·padding·보간이 없으므로 정답 마스크와 바로 로스를 계산할 때 형태가 맞지 않는다. 원 FCN의 정렬 과정을 생략한 결과다. 같은 크기라는 가정 아래 분할 로스를 쓰면 안 된다.

### 입력·평가와 확장

RGB 입력은 이미3채널이며 분할 모델에는 마지막 공간 전체를 펼친 fc가 없다.64×64를 지원하려고 fc 입력을 바꾸는 일반 분류기 설명은 이 예제에 맞지 않는다. 출력 score의 위치 대응·목표 크기와 학습·평가 마스크를 일관되게 정렬해야 한다.

연습의 ModuleList는 `YourBlock` 등이 정의되지 않은 골격이다. 블록을 늘리면 stride와 수용영역, 사용할 skip score와 업샘플 배율도 함께 달라질 수 있다. 기존 실행부는 파라미터 수만 세며 출력 크기·학습 성능 검증이 아니다. 설계는 [원 논문](https://arxiv.org/abs/1411.4038)을 참고하고 코드·출력은 실행하지 않고 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 K=21, 입력 (4,3,64,64)의 score와 출력 형태를 계산하라.

</div>

??? success "연습문제 1 풀이"
    세 풀링 뒤 (4,64,32,32), (4,128,16,16), (4,256,8,8)이다.1×1 score는 (4,21,8,8), kernel16 stride8 padding0의 전치 컨볼루션 출력은 (4,21,72,72)다. 원64×64와 같지 않다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB를 위한 수정에서 fc보다 먼저 검토할 것은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    입력은 이미RGB이고 fc는 없다. 출력72×72와 정답64×64의 위치·크기를 맞추는 crop·패딩 또는 목표 크기 보간을 명시해야 한다. 공간 정렬의 규약 없이 단순 reshape로 고칠 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
왜 이 예제를 원 FCN-8s라고 부르기에는 불충분한가?

</div>

??? success "연습문제 3 풀이"
    출력 전 피처 stride는8이지만 pool3 score 하나만 쓰고 원 모델의 깊은 score와 pool4·pool3 융합을 생략했다. 원 FCN-8s의 이름은 그 전체 skip·업샘플 구성을 가리킨다. stride 숫자가 같다고 구조·성능이 같지는 않다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 목록 골격으로 FCN의 단계를 늘릴 때 확인할 항목을 설명하라.

</div>

??? success "연습문제 4 풀이"

    정의되지 않은 블록과 채널을 구체화하고 각 단계 stride, score projection 폭K, skip 덧셈의 형태, 전치 컨볼루션 배율과 최종 정렬을 함께 정해야 한다. ModuleList만으로 원 FCN-8s의 여러 score 경로가 생성되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

FCN은 공간 score와 업샘플링을 end-to-end로 학습한다. 이 예제는 원 FCN-8s의 융합을 생략하며 현재 전치 컨볼루션 출력도 원 크기보다 커진다. 클래스 로짓과 정답의 위치·크기를 맞추는 과정을 명시해야 한다.
