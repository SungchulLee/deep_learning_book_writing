# 자기 부호기

자기 부호기 - 차원 줄이기와 특징 배우기

자기 부호기는 자료를 좁은 병목으로 눌러 담았다가 되돌리면서 눌러 담은 나타냄을 배우는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

여기서 배우는 것은 **압축**이지 만들어 내기가 아니다. 숨은 공간에서 코드를 지어내 새 자료를 뽑는 일은 자기 부호기가 약속하지 않으며, 왜 그런지는 [23.5 숨은 공간에서 뽑을 수 있는가](../limits/latent_sampling.md)에서 재어 본다. 그 물음에 답하는 것이 [26장 변분 자기 부호기](../../ch26/index.md)다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
자기 부호기 - 차원 줄이기와 특징 배우기
핵심: 이끌리지 않은 배움을 위한 부호기-풀개 짜임
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=32):
        super().__init__()
        
        # 부호기
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    model = Autoencoder()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,141,296
```

## 2. 논의

`Autoencoder` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다. `forward` 방법이 셈 그래프를 정하며, 그래서 PyTorch의 저절로 미분하기가 익히는 동안 기울기 셈하기를 알아서 다룬다. 이 모듈 설계 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 넣기가 쉽다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 나타냄 배우기 일에 대한 실전 직관이 선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
붙박이 첫자리매김에서 `Autoencoder`의 배울 수 있는 매개변수의 총수를 셈하라. 무게와 치우침을 모두 넣어 층마다 나누어 세어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

</div>

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
층이나 덩이의 수를 자리매김할 수 있도록 `Autoencoder`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 층 2, 4, 8개로 시험하라.

</div>

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (수수한 파이썬 목록이 아니라) `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 다음으로 시험하라: `for n in [2, 4, 8]: model = Autoencoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
부호기와 풀개의 층 너비가 대칭이어야 하는가? 대칭이 아니어도 되는 까닭과, 그런데도 대칭으로 두는 까닭을 말하라.

</div>

??? success "연습문제 5 풀이"
    대칭이어야 할 이유는 없다. 부호기가 $784 \to 512 \to 256 \to 16$이고 풀개가
    $16 \to 64 \to 784$여도 아무 문제 없이 돌아간다.

    그런데도 대칭으로 두는 까닭이 둘이다.

    **첫째, 담이를 가늠하기 쉽다.** 풀개가 부호기보다 약하면 코드에 정보가 남아
    있어도 되살리지 못하고, 반대면 풀개가 코드에 없는 것까지 지어내려 한다. 대칭이면
    둘의 힘이 엇비슷하리라 기대할 수 있다.

    **둘째, 매개변수가 절반으로 줄 수 있다.** 대칭이면 풀개의 가중치를 부호기의
    전치로 묶을 수 있다($W_{\text{dec}} = W_{\text{enc}}^{\top}$). 이를 **묶인
    가중치**(tied weights)라 하며, 매개변수가 반으로 주는 대신 표현력이 조금 준다.
    주성분 분석이 바로 이 묶인 꼴이다. 사영과 복원이 같은 행렬을 쓴다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
병목의 크기를 정하는 기준이 있는가? MNIST에서 병목을 2에서 64까지 바꾸며 다시 세우기 오차를 재고, 어디서 꺾이는지 보아라.

</div>

??? success "연습문제 6 풀이"
    | 병목 | 2 | 4 | 8 | 16 | 32 | 64 |
    |---|---|---|---|---|---|---|
    | 시험 MSE | 0.03689 | 0.02661 | 0.01581 | 0.00926 | 0.00654 | 0.00606 |
    | 두 배로 늘려 얻은 양 | — | 0.01028 | 0.01080 | 0.00655 | 0.00272 | 0.00048 |

    16에서 32로 갈 때까지는 두 배가 제값을 하다가, 32에서 64로 가면 얻는 양이
    0.00048로 뚝 떨어진다. 곧 **32 언저리에서 꺾인다.**

    이 꺾이는 자리가 자료의 **참된 차원**을 어림한 값이라고 볼 수 있다. 그보다 좁으면
    아직 버릴 것이 남아 있지 않아 억지로 버리는 것이고, 그보다 넓으면 더 담을 것이
    없어 통로만 헐렁해진다.

    다만 이 값을 곧이곧대로 믿으면 안 된다. 오차가 낮다고 나타냄이 좋은 것이 아니기
    때문이다([주다양체 연습문제 9](../architecture/04_ae_principal_manifold.md)). 병목을
    고를 때는 오차만이 아니라 그 코드로 무엇을 할지도 함께 보아야 한다. 분류에 쓸
    것이라면 코드로 분류기를 익혀 정확도를 재는 편이 낫다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
자기 부호기를 익힐 때 학습 자료와 시험 자료의 오차를 함께 재면 무엇을 알 수 있는가? 이 모델도 과적합하는가?

</div>

??? success "연습문제 7 풀이"
    과적합할 수 있다. 다만 모양이 지도 학습과 다르다.

    자기 부호기의 과적합은 **학습 자료를 외워 되돌리는** 것이다. 병목이 넓고 자료가
    적으면, 부호기가 사실상 "몇 번째 자료인가"를 코드에 적어 넣고 풀개가 그것을
    되짚는 일이 벌어진다. 그러면 학습 오차는 0에 가까워지지만 처음 보는 자료는
    되세우지 못한다.

    MNIST에 병목 16이면 그럴 일이 없다. 표본이 6만 개인데 코드가 16개뿐이라 외울
    그릇이 없다. 실제로 학습 오차와 시험 오차가 거의 붙어 있다.

    과적합이 실제로 문제가 되는 자리는 **자료가 적을 때**다. 표본 수백 개에 병목
    128이면 외우기가 쉬워진다. 그때 쓰는 처방이 병목을 좁히거나, 잡음을 더하거나,
    오그림 벌점을 주는 것이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
자기 부호기의 손실이 **비볼록**임을 보이고, 그 결과 무엇이 달라지는지 말하라.

</div>

??? success "연습문제 8 풀이"
    가장 단순한 꼴로 줄여도 곱셈 구조가 남는다. 스칼라 하나짜리 선형 자기 부호기의
    복원은 $\hat{x} = w_2 w_1 x$이고, 표본 하나 $x = 1$에 대한 손실은

    $$f(w_1, w_2) = (w_1 w_2 - 1)^2$$

    이다. $A = (1,1)$과 $B = (-1,-1)$에서 $f = 0$인데 중점 $(0,0)$에서 $f = 1$이므로

    $$f\!\left(\tfrac{A+B}{2}\right) = 1 > 0 = \tfrac{f(A)+f(B)}{2}$$

    로 볼록의 정의를 어긴다. $\square$

    [3.3절 연습문제 13](../../ch03/mnist/03_mlp.md)과 똑같은 반례이며, 원인도 같다.
    **층을 곱으로 쌓는다는 사실 자체**이지 활성화 함수가 아니다.

    달라지는 것이 둘이다.

    **첫째, 해가 하나로 정해지지 않는다.** 앞의 예에서 $w_1 w_2 = 1$이기만 하면
    되므로 답이 쌍곡선 전체다. 일반적으로도 $W_1 \to MW_1$, $W_2 \to W_2M^{-1}$이
    모두 같은 손실을 주므로, 어떤 코드를 얻을지는 초기값과 경로가 정한다.

    **둘째, 그런데도 크게 걱정하지 않아도 된다.** 선형 자기 부호기의 경우 비볼록임에도
    **모든 극소점이 전역 최소점**임이 알려져 있다(안장점은 있지만 나쁜 극소점은 없다).
    [주다양체 연습문제 5](../architecture/04_ae_principal_manifold.md)에서 경사 하강법이 주성분 분석의
    아래 공간을 실제로 찾아간 것이 그 증거다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff easy" title="쉬움"></span>
자기 부호기는 이름표를 쓰지 않는다. 그렇다면 무엇이 정답 노릇을 하는가?

</div>

??? success "연습문제 9 풀이"
    **입력 자신**이 정답이다. 손실이 $\lVert x - \hat{x} \rVert^2$이므로 맞혀야 할
    표적이 입력과 같다.

    그래서 이끌림 없는 배움(unsupervised learning)이라 부르지만, 요즘은 **스스로
    이끄는 배움**(self-supervised learning)이라 부르는 편이 더 정확하다고 본다.
    사람이 붙인 이름표는 없지만 표적이 없는 것은 아니고, 자료 자신에서 표적을
    만들어 낸 것이기 때문이다.

    같은 생각을 다르게 쓰면 다른 방법이 된다. 입력의 일부를 가리고 그 부분을
    맞히게 하면 마스크 모델링이 되고, 앞을 보고 다음을 맞히게 하면 자기 되돌이
    모델이 된다. 자기 부호기는 그 가운데 "전부를 가리지 않고 통로만 좁히는" 방식이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
익히고 난 자기 부호기의 부호기만 떼어 분류기의 앞단으로 쓸 수 있는가? 그렇게 하면 무엇이 좋은가?

</div>

??? success "연습문제 10 풀이"
    쓸 수 있고, 실제로 오랫동안 그렇게 써 왔다. 부호기를 얼려 두고 코드 위에 작은
    분류기만 얹어 익히는 방식이다.

    좋은 점은 **이름표가 적을 때** 드러난다. 부호기는 이름표 없는 자료로 익혔으므로
    자료가 아무리 많아도 쓸 수 있고, 이름표는 마지막 분류기를 익히는 데에만 필요하다.
    이름표 붙이기가 비싼 분야에서 값어치가 크다.

    다만 주의할 점이 있다. **다시 세우기에 좋은 코드가 분류에 좋은 코드는 아니다.**
    자기 부호기는 화소를 되돌리는 데 필요한 것을 담으므로, 획의 굵기나 밝기처럼
    분류에는 쓸모없는 정보에도 자리를 내준다. 반대로 분류에 결정적이지만 화소로는
    작은 차이(고리가 닫혔는가)는 소홀히 다룰 수 있다.

    그래서 요즘은 다시 세우기 대신 **대조 학습**처럼 분류에 쓸모 있는 성질을 직접
    겨냥하는 방법이 더 좋은 앞단을 만든다고 본다.

## 정리하며

**다룬 것** — 자기 부호기

`Autoencoder` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다.

고갱이 갈래는 `Autoencoder`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
