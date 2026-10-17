# ConvNeXt V2

ConvNeXt V2는 컨볼루션 기반 마스크드 오토인코더 사전 학습과 전역 응답 정규화(GRN)를 함께 설계한 모델이다. Woo 외의 CVPR 2023 논문은 채널 반응의 다양성과 자기지도 학습을 다룬다. 아래 구현은 GRN 블록을 살피는 축약형이며 실제 마스킹·재구성 학습은 포함하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x


class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(96) for _ in range(3)])
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXtV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 341,608
```

## 2. 논의

### GRN의 축과 수식

NHWC 입력 X에 대해 표본별·채널별 공간 L2 노름 G_nc=(∑ₕw X_nhwc²)¹ᐟ²를 계산한다. 채널 평균으로 나눈 N_nc=G_nc/(mean_c G_nc+ε)를 사용해 다음처럼 출력한다.

$$Y=X+\gamma\odot(X\odot N)+\beta.$$

γ와 β는 채널별 학습 파라메터다. 공간 평균을 빼거나 분산을 1로 만드는 LayerNorm·BatchNorm과 다른 연산이며, 표본 간 통계를 섞지 않는다. 코드의 ε=10⁻⁶은 영 노름에서 나눗셈을 안정화한다.

γ=β=0으로 초기화하므로 처음 GRN 자체는 항등이다. 채널 노름이 크면 N도 크지만 γ는 학습 중 양수·음수 모두 가능하고 잔차도 있으므로 ‘큰 채널은 항상 증폭되고 작은 채널은 억제된다’고 단정할 수 없다. 채널 경쟁을 도입해 특정 학습에서 피처 붕괴를 완화한 설계이지 모든 입력에서 붕괴를 불가능하게 하는 수학적 보장은 아니다.

### ConvNeXt V2 블록과 축약 본체

블록은 7×7 depthwise 뒤 NHWC의 채널 LayerNorm, 4배 확장 Linear–GELU–GRN–축소 Linear를 거쳐 입력과 더한다. GRN이 확장된 4C채널에 적용되는 점을 확인한다. 본체는 96채널 블록 3개만 포함하며 원 모델의 여러 해상도 단계·다운샘플링·학습 레시피를 재현하지 않는다.

이 본체의 순전파 형태는 일관되지만 `__main__`은 파라미터 수만 출력한다. 구조의 형태가 맞는 것과 학습된 모델의 분류 성능·표현 품질이 검증된 것은 다르다. 원 논문의 개선은 해당 전체 모델과 사전 학습 조건에서 평가한 결과다.

### 마스크드 오토인코더의 목적

마스크 M이 가린 위치를 표시할 때 재구성 로스의 예는 ∑ᵢMᵢ∥x̂ᵢ−xᵢ∥²/∑ᵢMᵢ다. 모델은 가린 관측을 직접 보지 않은 상태에서 그 위치의 목표를 복원해야 한다. 로스 단위·패치 정규화·마스크 비율과 데이터 유효 범위를 명시하고, 컨볼루션의 이웃 연산을 통해 가린 원본 정보가 누출되지 않도록 설계해야 한다.

아래 `MAEPretraining`은 `mask_ratio`를 저장하지만 사용하지 않는다. stem과 blocks가 **가리지 않은 전체 입력**을 처리하고 decoder만 적용하며 마스크 생성·재구성 목표·로스·훈련 루프가 없다. 따라서 실제 마스크드 사전 학습의 구현으로 볼 수 없다. 출력 48채널은 stride4 stem의 4×4 RGB 패치 3×16값과 연결해 해석할 수 있지만, 패치를 원래 공간으로 재배열하는 과정도 구현되지 않았다.

GRN을 넣었다는 사실만으로 이 골격이 FCMAE의 정보 누출 제어와 희소 연산을 재현하지는 않는다. 코드·출력은 그대로 보존했으며 실행하지 않았다. [원 논문](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)에서 모델과 사전 학습을 함께 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GRN의 축과 초기값을 쓰고 항상 채널 증폭을 보장하는지 설명하라.

</div>

??? success "풀이"
    각 표본의 채널별 공간 L2 노름을 구해 채널 평균 노름으로 나눈 뒤 X+γXN+β를 계산한다. γ=β=0에서는 항등이다. 학습 γ의 부호가 제한되지 않으므로 큰 노름의 채널이 항상 증폭된다고 할 수 없다. 이 정규화는 공간 평균·분산을 맞추는 LN/BN과 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
컨볼루션 기반 마스크드 사전 학습에서 입력 가림 외에 무엇을 설계해야 하는가?

</div>

??? success "풀이"
    가린 위치의 목표와 로스, decoder의 출력 형태, 패치 재배열, 이웃 컨볼루션을 통한 정보 누출과 가려진 피처의 처리를 함께 설계해야 한다. GRN은 채널 표현의 설계 요소이며 마스크 생성이나 누출 제어를 자동으로 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 MAEPretraining이 실제 마스크드 학습을 수행하는지 검토하라.

</div>

??? success "풀이"

    mask_ratio는 저장만 하고 입력을 가리지 않는다. 출력 재구성만 반환하며 마스크별 목표·로스·학습 루프도 없다. 48채널을 4×4 RGB 패치로 해석할 수 있어도 재배열과 가린 위치의 평가가 추가로 필요하다. 따라서 완성된 FCMAE 사전 학습 코드가 아니라 decoder 골격이다.

    ```python
class MAEPretraining(nn.Module):
    def __init__(self, encoder, decoder_dim=256, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Conv2d(96, decoder_dim, 1),
            nn.GELU(),
            nn.Conv2d(decoder_dim, 3 * 16, 1)  # 4x4 조각 어림
        )
    
    def forward(self, x):
        # 조각에 마구잡이 가리기를 하게 된다
        features = self.encoder.stem(x)
        features = self.encoder.blocks(features)
        reconstruction = self.decoder(features)
        return reconstruction
```


## 정리하며

GRN은 공간 노름의 채널 상대값을 활용하는 잔차 정규화다. 피처 붕괴 완화의 실험적 결과와 보편적 보장을 구분하고, 마스킹하지 않는 decoder 골격을 실제 마스크드 오토인코더 학습으로 해석하지 않는다.
