# 인셉션 V3

Inception v3은 컨볼루션 분해와 효율적인 해상도 축소, 레이블 스무딩 등을 결합한 CNN이다. Szegedy 외의 논문은 2015년에 공개되고 CVPR 2016에 발표되었다. 아래 클래스는 일부 연산의 골격만 담고 있으며 Inception v3 전체 네트워크가 아니다.

## 1. 코드

```python
import torch
import torch.nn as nn


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = InceptionV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 2,077,512
```

## 2. 논의

### 컨볼루션 분해의 비용

동일한 C채널과 stride=1에서 5×5 한 레이어는 25C²개, 3×3 두 레이어는 18C²개의 웨이트를 갖는다. 이론적 수용영역은 같지만 중간 활성화와 합성의 제약으로 같은 연산은 아니다. 채널 폭이 바뀌면 절감 비율도 달라진다.

n×n을 1×n과 n×1로 나누고 입출력·중간 폭을 모두 C로 두면 n²C² 대 2nC²다. 코드처럼 중간 폭을 C_out으로 두면 분해 웨이트는 nC_inC_out+nC_out²이므로 항상 n/2배 절감되는 것은 아니다. 정규화의 파라메터와 추가 활성화도 별도로 계산한다.

### 레이블 스무딩의 두 정의

균일분포 u(k)=1/K를 섞는 원 논문의 정의는 q(k)=(1−ε)1[k=y]+ε/K다. K=1000, ε=0.1이면 정답은 0.9001, 나머지는 각각 0.0001이다.

정답을 0.9로 두고 나머지에 0.1/999를 배분하는 정의도 정규화된 목표지만 **다른 규칙**이다. 두 방식을 같은 ε의 동일한 목표로 혼용하지 않는다. 스무딩은 날카로운 목표를 완화하는 정규화이며 모든 데이터에서 일반화나 캘리브레이션을 반드시 개선하는 것은 아니다. 분류 로스에서 적용하는 것이지 모델 클래스를 정의하는 것만으로 활성화되는 기능은 아니다.

### 코드의 미완성 경로

`forward`는 conv1만 사용하고 conv2·conv3는 정의되어 있어도 사용하지 않는다. 그 뒤 전역 평균 피처는 32차원인데 `fc`는 2048차원을 요구하므로 순전파가 실패한다. 병렬 Inception 모듈·축소 단계·보조 분류기도 포함하지 않는다. 기존 출력은 파라미터 수만 계산하므로 모델의 실행이나 정확도 검증이 아니다.

연습의 `AsymmetricConv`는 두 방향 커널과 BN·ReLU를 실제로 연결한다. 홀수 n에서 padding=n//2가 각 축의 공간 크기를 유지하며, 짝수 n에서는 이 패딩 규칙으로 크기가 달라질 수 있다. 공간 형태와 비용은 해당 n과 폭으로 계산해야 한다.

원 논문의 단일 평가·다중 크롭·앙상블 결과는 서로 다르다. 특정 설정의 top-5 오류율을 모든 Inception v3이나 이 미완성 코드의 성능으로 인용하지 않는다. [원 논문](https://openaccess.thecvf.com/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html)의 평가 조건을 함께 확인한다. 코드와 출력은 실행하지 않고 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입출력·중간 폭 256에서 5×5를 3×3 두 개로 바꿀 때 컨볼루션 웨이트 수를 비교하라.

</div>

??? success "풀이"
    바이어스·BN을 제외하면 각각 25×256²=1,638,400개, 18×256²=1,179,648개다. 차이는 458,752개로 28% 절감이다. 중간 폭이 다르거나 추가 정규화를 포함하면 비율을 다시 계산한다. 이론적 수용영역이 같다는 것이 연산의 동등성을 뜻하지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
균일분포 혼합과 오답에만 배분하는 스무딩을 K=1000, ε=0.1에서 비교하라.

</div>

??? success "풀이"
    균일 혼합은 정답 1−ε+ε/K=0.9001, 모든 오답 ε/K=0.0001이다. 오답 배분 방식은 정답 0.9, 오답 0.1/999다. 둘 다 합은 1이지만 서로 다른 목표이며 어느 정의를 쓰는지 명시해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 1×7–7×1 분해가 항상 3.5배 적은 파라메터를 쓰는가?

</div>

??? success "풀이"

    입력·중간·출력 폭이 모두 C이면 49C² 대 14C²로 3.5배다. 코드의 일반 폭에서는 7C_inC_out+7C_out²이며 같은 비가 아니다. BN의 스케일·바이어스는 별도이고 중간 비선형이 추가된다. 홀수 n에서 공간 크기를 유지하는 패딩 조건도 확인한다.

    ```python
class AsymmetricConv(nn.Module):
    def __init__(self, in_ch, out_ch, n=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, n), padding=(0, n//2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (n, 1), padding=(n//2, 0), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
```


## 정리하며

Inception v3의 분해 비용은 실제 채널 폭으로 계산하고 레이블 스무딩의 정의를 명시한다. 예제의 32차원 피처와 2048입력 분류기는 맞지 않으며 전체 Inception 구조와 논문 성능을 재현하지 않는다.
