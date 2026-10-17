# U-Net

이 U-Net 예제는 네 단계 인코더와 전치 컨볼루션 디코더를 채널 연결로 결합한다. 컨볼루션에 padding=1과 BN을 사용하므로 원 논문의 valid convolution·crop 구조와 다르다. 기본 출력은 한 채널 로짓이며 입력 크기에 따른 스킵 정렬과 학습 목표의 형태를 확인해야 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
U-넷 — 생의학 그림 나누기를 위한 컨볼루션 넷트웍
논문: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
핵심: 인코더와 디코더 사이의 건너뛰는 이음, 나누기에 널리 쓰인다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 인코더
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # 병목
        self.bottleneck = DoubleConv(512, 1024)
        
        # 디코더
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # 인코더
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # 병목
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 건너뛰는 이음을 갖춘 디코더
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)

if __name__ == "__main__":
    model = UNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 3, 256, 256)
    print(f"Input: {x.shape}, Output: {model(x).shape}")
```

**출력:**

```
Parameters: 31,043,521
Input: torch.Size([2, 3, 256, 256]), Output: torch.Size([2, 1, 256, 256])
```

## 2. 논의

### 채널과 공간 크기

DoubleConv는 3×3–BN–ReLU를 두 번 적용해 공간 크기를 유지한다. 인코더 폭은64·128·256·512, 병목은1024다. 네 번의2×2 풀링으로 해상도를16분의1로 줄이고, 디코더는 전치 컨볼루션으로 채널과 공간을 바꾼 뒤 같은 해상도의 인코더 피처를 연결한다.

입력 (N,3,64,64)에서 인코더 피처는64×64×64,128×32×32,256×16×16,512×8×8이고 병목은1024×4×4다. 첫 업샘플은512×8×8, 스킵과 연결하면1024×8×8, dec4 뒤512×8×8이다. 같은 방식으로 마지막64×64×64를 만들고1×1로 out_channels를 출력한다.

### 입력 크기와 정렬

이 코드에는 crop·pad·목표 크기 보간이 없다. 양쪽 공간이16의 배수이면 네 번의 풀링과 업샘플이 같은 크기의 스킵과 맞는다. 홀수나 임의 크기에서는 바닥 나눗셈으로 스킵과 decoder가 달라질 수 있으므로 전처리나 명시적 정렬이 필요하다. 작은 입력에서 병목 공간·배치가 BN 통계에 충분한지도 확인한다.

전치 컨볼루션은 학습 가능한 업샘플링이며 인코더의 역연산이 아니다. 연결은 높은 해상도의 피처를 제공하지만 모든 세부를 보존하거나 정확도를 보장하지 않는다. 활성 저장과 연결 채널이 학습 메모리에 영향을 준다.

### 로짓과 마스크

기본 out_channels=1은 전경/배경의 한 로짓이다. BCEWithLogitsLoss에는 같은 형태의 float0/1 마스크를 주고, 평가에서는 sigmoid 뒤 정한 임계값으로 영역을 만든다. 상호 배타적인 K클래스로 바꾸면 출력 K채널과 정수 마스크 N×H×W의 CE를 사용할 수 있다. 출력에 활성화를 붙이지 않았다는 점은 logits 로스와 일관된다.

기존 실행부에는 배치2의256×256 예시가 있지만 이번 편집에서 실행하지 않았다. 그 형태와 파라미터 출력이 학습된 분할 정확도를 뜻하지는 않는다. 연습의 ModuleList 골격은 정의되지 않은 블록을 사용하므로 깊이·채널·스킵 구성을 실제로 제공하지 않는다. 원 설계는 [U-Net 논문](https://arxiv.org/abs/1505.04597)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
(4,3,64,64) 입력의 인코더·병목·첫 decoder 형태를 추적하라.

</div>

??? success "풀이"
    인코더는 (4,64,64,64), (4,128,32,32), (4,256,16,16), (4,512,8,8), 병목은 (4,1024,4,4)다. 첫 전치 컨볼루션 뒤 (4,512,8,8), 연결 뒤 (4,1024,8,8), dec4 뒤 (4,512,8,8)이다. 기본 최종 출력은 (4,1,64,64)다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
RGB64×64 입력을 위해 fc를 추가하거나 입력 채널을 바꿔야 하는가?

</div>

??? success "풀이"
    이미 in_channels=3이고 마지막은1×1 컨볼루션이므로 fc를 바꿀 필요가 없다. 이 크기는16의 배수라 스킵 정렬이 맞는다. 임의 홀수 크기에서도 같은 보장이 생기지는 않으며 선택한 BCE/CE와 마스크 형태를 맞춰야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
DoubleConv(3,64)의 학습 파라메터를 바이어스와 BN까지 포함해 계산하라.

</div>

??? success "풀이"
    두 컨볼루션 웨이트는9×3×64+9×64²=38,592개, 바이어스는128개, 두 BN의 스케일·바이어스는256개다. 총38,976개이며 BN 이동평균·분산 버퍼는 학습 파라메터로 세지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
U-Net 깊이를 바꿀 때 단순 블록 목록 외에 정해야 할 것은 무엇인가?

</div>

??? success "풀이"

    단계별 폭·풀링 수·병목 폭과 대응하는 업샘플·스킵·decoder 입력 채널을 함께 정해야 한다. 연결 뒤 채널은 두 경로의 합이다. YourBlock과 dim 등이 정의되지 않은 목록만으로 가변 깊이 UNet이 완성되지 않으며 입력 크기의 정렬 조건도 바뀐다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

이 예제는 padding을 유지하는 네 단계 U-Net이다. 입력 크기·채널 연결·로짓과 마스크의 형태를 추적하고, 임의 크기를 자동으로 정렬하거나 학습 성능을 검증한 코드로 해석하지 않는다.
