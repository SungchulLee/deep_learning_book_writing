# CNN에서 비전 트랜스포머로

CNN, 채널 어텐션, 비국소 연산과 비전 트랜스포머(ViT)를 연결해 이미지 표현의 설계 선택을 살펴본다. 이들은 한 방향으로 모든 가정을 제거하는 단계가 아니라 지역 연결·웨이트 공유·위치 표현과 입력에 따른 가중 결합을 다르게 조합한 구조이다.

패치 임베딩의 선형 투영과 컨볼루션의 대응, 토큰·파라메터 수, 주요 연산량의 계산 예제를 통해 공통점과 비용 차이를 확인한다.

## 1. 구조 패러다임의 변천

### 1단계: 순수한 CNN

전통적인 CNN은 지역 컨볼루션을 잇달아 적용하여 이미지를 처리한다. 출력 자리 $(i,j)$의 값은 입력의 같은 자리 둘레에 있는 작은 창 하나만 보고 정해진다.

$$\mathbf{y}_{i,j}^{(l)} = f\left(\sum_{c}\sum_{p,q} W_{c,p,q}^{(l)} \mathbf{x}_{c,\,i+p,\, j+q}^{(l-1)} + b^{(l)}\right)$$

여기서 $c$은 입력 채널을 훑고, $(p,q)$은 커널 안의 자리를 훑으며, $f$은 엑티베이션 함수이다. 채널에 걸친 합이 있어야 색 세 장이나 앞 레이어의 피처 맵 여러 장을 하나로 묶을 수 있다.

국소 연결은 연산량과 표현의 제약을 정하고, 웨이트 공유는 같은 검출기를 여러 위치에 적용한다. 이미지에 유용한 귀납 바이어스가 될 수 있지만 모든 과제의 효율이나 성능을 보장하지 않는다.

- 계산 효율
- 이미지 구조와 들어맞는 귀납 바이어스
- 공간에 걸친 파라메터 공유

### 2단계: 어텐션으로 보강한 CNN

CNN에 어텐션 메커니즘([Squeeze-and-Excitation](squeeze_excitation.md), CBAM)를 더하면 지역성을 지키면서도 채널과 공간의 웨이트를 적응적으로 다시 매길 수 있다.

$$\mathbf{y}_{i,j}^{(l)} = \alpha_{i,j}^{(l)} \odot f\left(\sum_{c}\sum_{p,q} W_{c,p,q}^{(l)} \mathbf{x}_{c,\,i+p, \,j+q}^{(l-1)} + b^{(l)}\right)$$

배율은 채널 또는 공간 위치에 따라 입력에서 계산한다. SE처럼 전역 평균을 이용하면 배율 경로는 전체 피처 맵에 의존한다. 따라서 최종 출력까지 순수하게 국소적인 연산이라고 할 수 없으며, 국소 피처 계산과 전역 배율 계산을 구분해야 한다.

### 3단계: 비국소 뉴럴 넷트웍

[비국소 연산](non_local.md)은 공간 전체에 걸친 의존 관계를 붙잡는다.

$$\mathbf{y}_{i} = \frac{1}{C(\mathbf{x})} \sum_{j} f(\mathbf{x}_i, \mathbf{x}_j) \cdot g(\mathbf{x}_j)$$

f는 위치 쌍의 관계를, g는 값 피처의 투영을 정의한다. 정규화 인수는 함수 종류에 따라 입력·질의 위치에 의존하거나 N 같은 상수일 수 있다. 지수화한 스케일 내적을 행 합으로 정규화하면 단일 헤드의 소프트맥스 어텐션 형태를 얻는다.

### 4단계: 혼합 구조

혼합 모델은 컨볼루션 스템에 트랜스포머 몸통을 잇는다.

$$\text{Tokens} = \text{PatchEmbed}(\text{ConvStem}(\mathbf{x}))$$

$$\mathbf{z}^{(l)} = \text{Transformer}(\mathbf{z}^{(l-1)})$$

앞쪽 레이어는 효율을 위해 CNN의 귀납 바이어스를 쓰고, 뒤쪽 레이어는 트랜스포머의 유연함을 쓴다. 줄기가 이미 공간을 줄여 두므로 트랜스포머가 받는 토큰 수가 작아지는데, 6절에서 보듯 이 토큰 수가 어텐션 비용을 이차로 좌우한다.

### 5단계: 순수한 비전 트랜스포머

기본 ViT는 이미지를 패치별 선형 투영으로 토큰화하고 전역 셀프 어텐션과 위치별 MLP를 반복한다. 패치 투영은 같은 크기·스트라이드의 Conv2d로도 구현할 수 있어, 컨볼루션 연산을 전혀 사용하지 않는다는 표현과는 구분한다. 사전 레이어 정규화 블록은 다음 두 단계로 쓴다.

$$\mathbf{z}'^{(l)} = \text{MSA}(\text{LN}(\mathbf{z}^{(l-1)})) + \mathbf{z}^{(l-1)}$$

$$\mathbf{z}^{(l)} = \text{MLP}(\text{LN}(\mathbf{z}'^{(l)})) + \mathbf{z}'^{(l)}$$

두 식 모두 왼쪽에 $\mathbf{z}^{(l)}$을 쓰면 두 번째 식이 갱신이 아니라 $\mathbf{z}^{(l)} = \text{MLP}(\text{LN}(\mathbf{z}^{(l)})) + \mathbf{z}^{(l)}$이라는 고정점 방정식이 되어 버린다.

---

## 2. 핵심적인 구조의 다리

!!! tip "패치 임베딩은 컨볼루션을 닮은 것이 아니라 컨볼루션이다"
    ViT의 패치 임베딩 레이어는 커널 크기와 스트라이드가 똑같은 컨볼루션 레이어 하나와 **같은 연산**이다. 비유가 아니라 등식이며, 6절에서 두 길의 출력이 float32 반올림 자리까지만 어긋남을 보인다. torchvision의 `vit_b_16`은 이 레이어를 아예 `nn.Conv2d(3, 768, kernel_size=16, stride=16)`으로 적어 두었다.

**컨볼루션 스템**: 여러 국소 컨볼루션으로 토큰화 전 피처를 계산하는 변형이다. 작은 데이터셋에서의 성능 개선은 가능한 효과이며 구조·학습 설정에 따라 검증해야 한다.

**피라미드 구조**: 계층적 시각 모델(Swin, PVT)은 어텐션을 쓰면서도 CNN과 비슷하게 여러 규모의 처리를 유지한다.

**혼합 셀프 어텐션**: 특정 자리의 트랜스포머 블록을 컨볼루션 레이어로 바꾸어 계산 비용과 모델링 능력을 저울질한다.

---

## 3. 비교 분석

아래 표의 수는 모두 6절에서 재어 나온 것이거나 그 자리에서 셈할 수 있는 것이다.

| 항목 | CNN | 혼합 구조 | 기본 ViT |
|---|---|---|---|
| 국소·전역 처리 | 컨볼루션은 국소, 깊이·풀링 등으로 확장 | 두 방식을 조합 | 패치 투영 뒤 전역 어텐션 |
| 위치 표현 | 격자·커널·패딩·스트라이드의 구조 | 구성에 따라 다름 | 패치 격자와 위치 임베딩 등 |
| 입력 크기 증가 | 고정 폭·깊이의 컨볼루션은 위치 수에 대체로 선형 | 토큰화·어텐션 방식에 의존 | 밀집 어텐션의 쌍 연산은 토큰 수에 이차 |
| 데이터 효율 | 구조·학습·사전 학습에 의존 | 구조·학습·사전 학습에 의존 | 구조·학습·사전 학습에 의존 |
| 이동 등변성 | 경계·스트라이드·후처리에 따라 제한 | 구성에 따라 제한 | 패치 격자와 위치 표현에 따라 제한 |

!!! warning "'전역'과 '이차'는 서로 다른 것에 붙는 말이다"
    표의 둘째 줄과 셋째 줄을 한 줄로 합쳐 "ViT는 리셉티브 필드가 이차로 자란다"고 적으면 두 번 틀린다. 첫째, 어텐션의 리셉티브 필드는 자라지 않는다 — 첫 레이어에서 이미 전부다. 둘째, 이차인 것은 리셉티브 필드가 아니라 **토큰 수에 대한 어텐션 행렬의 비용**이다.

    한편 CNN 쪽의 "선형"도 조건이 붙는다. [리셉티브 필드](../cnn/receptive_field.md) 쪽의 점화식

    $$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

    에서 커널 크기와 dilation이 일정하고 모든 스트라이드가 1이면 리셉티브 필드의 폭이 깊이에 선형으로 늘어난다. 스트라이드 2를 반복해서 넣으면 누적 스트라이드 $\prod s_i$가 더 빠르게 커진다. 이 예제의 폭 기준 계산에서는 필요한 레이어 수가 112레이어에서 12레이어로 줄어든다. 어텐션의 "첫 레이어부터 전역"은 이 점화식에서 나오는 결과가 **아니라** 연결 방식 자체가 다르다는 뜻이다.

---

## 4. 실무자를 위한 고려 사항

학습 데이터와 사전 학습, 어그멘테이션·정규화·증류 설정을 함께 고려해야 한다. 초기 ViT 논문의 대규모 사전 학습 결과를 모든 ViT에 필요한 최소 데이터량으로 읽으면 안 된다. 예를 들어 [DeiT 연구](https://proceedings.mlr.press/v139/touvron21a.html)는 외부 데이터 없이 ImageNet으로 학습한 트랜스포머를 보고했다.

연산 예산은 모델 이름보다 토큰 수 N, 폭 d와 깊이로 비교한다. 아래의 밀집 블록 근사에서 224×224·패치 16·폭 768은 주요 MAC 대부분이 투영·MLP 항에 있다. 하지만 이차 어텐션 행렬은 연산 비중이 작아도 메모리 병목이 될 수 있다. CNN이나 ViT 중 어느 것이 실제로 빠른지는 장치·배치·자료형과 구현에서 측정해야 한다.

---

## 5. 앞으로의 방향

새로 나오는 연구는 다음을 살핀다.

- **뉴로모픽 혼합**: 스파이킹 뉴런과 어텐션의 결합
- **효율적인 트랜스포머**: 고해상도 이미지를 위한 선형 복잡도 어텐션
- **학습된 라우팅**: 입력에 따라 컨볼루션과 어텐션 중에서 그때그때 고르기
- **멀티모달 융합**: 혼합 구조로 시각과 언어를 잇기

---

## 6. 다리를 수로 재어 보기

다음 코드는 패치 임베딩의 두 표현, 특정 torchvision ViT-Base/16의 구성과 파라메터 수, 컨볼루션의 리셉티브 필드 경계 너비와 주요 블록 MAC을 비교한다. 학습 후 정확도나 실제 지연을 재는 실험은 아니다.

```python
"""CNN과 ViT를 잇는 다리를 수로 재어 본다."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

torch.manual_seed(0)

IMG, PATCH, DIM = 224, 16, 768


# === 1. 조각 임베딩 = 겹치지 않는 컨볼루션 ===

def patch_embed_is_conv():
    """조각을 펼쳐 선형 레이어에 넣는 것과 stride=kernel 컨볼루션이 같음을 보인다."""
    x = torch.randn(1, 3, IMG, IMG)
    conv = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)

    # 컨볼루션 길
    z_conv = conv(x)                                 # (1, 768, 14, 14)
    tokens_conv = z_conv.flatten(2).transpose(1, 2)  # (1, 196, 768)

    # 펼치기 + 선형 레이어 길 (같은 웨이트를 모양만 바꿔 쓴다)
    patches = F.unfold(x, kernel_size=PATCH, stride=PATCH)  # (1, 3*16*16, 196)
    patches = patches.transpose(1, 2)                       # (1, 196, 768)
    W = conv.weight.reshape(DIM, -1)                        # (768, 768)
    tokens_lin = patches @ W.t() + conv.bias

    cls = torch.zeros(1, 1, DIM)
    with_cls = torch.cat([cls, tokens_conv], dim=1)

    print("[1] 조각 임베딩은 컨볼루션이다")
    print(f"  입력                          : {tuple(x.shape)}")
    print(f"  Conv2d(3, 768, k=16, s=16)    : {tuple(z_conv.shape)}")
    print(f"  펼친 토큰                     : {tuple(tokens_conv.shape)}")
    print(f"  클래스 토큰을 붙인 뒤         : {tuple(with_cls.shape)}")
    print(f"  조각 수 (224//16)**2          : {(IMG // PATCH) ** 2}")
    print(f"  토큰 수 = 조각 + 클래스       : {(IMG // PATCH) ** 2 + 1}")
    print(f"  두 길의 최대 어긋남           : "
          f"{(tokens_conv - tokens_lin).abs().max().item():.2e}")
    print(f"  torch.allclose(atol=1e-5)     : "
          f"{torch.allclose(tokens_conv, tokens_lin, atol=1e-5)}")
    n_w = DIM * 3 * PATCH * PATCH
    print(f"  조각 임베딩 파라메터          : {n_w + DIM:,} = {n_w:,} + {DIM}")


# === 2. 실제 ViT-Base/16을 뜯어 보기 ===

def real_vit_base():
    """torchvision의 ViT-Base/16에서 구조 수를 직접 읽는다."""
    m = vit_b_16(weights=None)
    enc = m.encoder.layers
    blk = enc[0]
    total = sum(p.numel() for p in m.parameters())
    head = sum(p.numel() for p in m.heads.parameters())

    print("\n[2] torchvision ViT-Base/16 (웨이트 없이 구조만)")
    print(f"  조각 임베딩 레이어                : {m.conv_proj}")
    print(f"  은닉 차수                     : {blk.ln_1.normalized_shape[0]}")
    print(f"  블록 수                       : {len(enc)}")
    print(f"  머리 수                       : {blk.self_attention.num_heads}")
    print(f"  머리 하나의 차수              : "
          f"{blk.ln_1.normalized_shape[0] // blk.self_attention.num_heads}")
    print(f"  위치 임베딩                   : {tuple(m.encoder.pos_embedding.shape)}")
    print(f"  파라메터 (분류 머리 포함)     : {total:,}")
    print(f"  파라메터 (분류 머리 뺀 몸통)  : {total - head:,}")

    out = m(torch.randn(1, 3, IMG, IMG))
    print(f"  출력 모양                     : {tuple(out.shape)}")


# === 3. 리셉티브 필드: 컨볼루션의 점화식 대 어텐션의 연결 ===

def receptive_field(layers):
    """r_l = r_{l-1} + (K_l - 1) * d_l * prod(s_i), 리셉티브 필드 쪽과 같은 점화식."""
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += (k - 1) * d * jump
        jump *= s
    return rf, jump


def depth_to_cover():
    """3x3 컨볼루션을 몇 레이어 쌓아야 224 화소를 덮는가."""
    print("\n[3] 224 화소를 덮는 데 드는 깊이")

    n = 0
    while receptive_field([(3, 1, 1)] * n)[0] < IMG:
        n += 1
    print(f"  3x3 스트라이드 1만 쌓을 때          : {n}레이어 "
          f"(리셉티브 필드 {receptive_field([(3, 1, 1)] * n)[0]})")

    n = 0
    while True:
        stack = [(3, 1, 1), (3, 2, 1)] * n
        if stack and receptive_field(stack)[0] >= IMG:
            break
        n += 1
    rf, jump = receptive_field(stack)
    print(f"  3x3 스트라이드 1,2를 번갈아 쌓을 때 : {len(stack)}레이어 "
          f"(리셉티브 필드 {rf}, 누적 스트라이드 {jump})")

    print(f"  어텐션 한 레이어                  : 1레이어 (토큰 197개가 서로를 모두 본다)")
    print(f"  토큰 하나가 덮는 화소         : {PATCH}x{PATCH}, 조각 196개면 {IMG}x{IMG} 전체")


# === 4. 계산 비용: 무엇이 이차이고 무엇이 일차인가 ===

def block_cost(n_tokens, dim=DIM, mlp_mult=4):
    """트랜스포머 블록 한 개의 곱셈-덧셈 수를 항별로 센다."""
    qkv_proj = 4 * n_tokens * dim * dim          # Q, K, V와 출력 사영 — 토큰에 일차
    attn_mat = 2 * n_tokens * n_tokens * dim     # QK^T와 가중합 — 토큰에 이차
    mlp = 2 * mlp_mult * n_tokens * dim * dim    # 다층 퍼셉트론 — 토큰에 일차
    return qkv_proj, attn_mat, mlp


def cost_exponents():
    print("\n[4] 토큰 수에 대한 비용의 차수")
    for img in (224, 384):
        n = (img // PATCH) ** 2 + 1
        lin1, quad, lin2 = block_cost(n)
        tot = lin1 + quad + lin2
        print(f"  {img}x{img}: 토큰 {n:3d} | 블록 곱셈-덧셈 {tot:,}")
        print(f"            일차 항 {lin1 + lin2:>15,} | "
              f"이차 항 {quad:>13,} ({100 * quad / tot:.1f}%)")

    n224 = (224 // PATCH) ** 2 + 1
    n384 = (384 // PATCH) ** 2 + 1
    print(f"  224에서 384로 갈 때 토큰이 {n384 / n224:.2f}곱")
    print(f"    이차 항은 {block_cost(n384)[1] / block_cost(n224)[1]:.2f}곱")
    print(f"    일차 항은 {(block_cost(n384)[0] + block_cost(n384)[2]) / (block_cost(n224)[0] + block_cost(n224)[2]):.2f}곱")
    print(f"    화소 수는 {(384 / 224) ** 2:.2f}곱 — 컨볼루션은 여기에 붙어 자란다")


if __name__ == "__main__":
    patch_embed_is_conv()
    real_vit_base()
    depth_to_cover()
    cost_exponents()
```

**출력:**

```text
[1] 조각 임베딩은 컨볼루션이다
  입력                          : (1, 3, 224, 224)
  Conv2d(3, 768, k=16, s=16)    : (1, 768, 14, 14)
  펼친 토큰                     : (1, 196, 768)
  클래스 토큰을 붙인 뒤         : (1, 197, 768)
  조각 수 (224//16)**2          : 196
  토큰 수 = 조각 + 클래스       : 197
  두 길의 최대 어긋남           : 1.25e-06
  torch.allclose(atol=1e-5)     : True
  조각 임베딩 파라메터          : 590,592 = 589,824 + 768

[2] torchvision ViT-Base/16 (웨이트 없이 구조만)
  조각 임베딩 레이어                : Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  은닉 차수                     : 768
  블록 수                       : 12
  머리 수                       : 12
  머리 하나의 차수              : 64
  위치 임베딩                   : (1, 197, 768)
  파라메터 (분류 머리 포함)     : 86,567,656
  파라메터 (분류 머리 뺀 몸통)  : 85,798,656
  출력 모양                     : (1, 1000)

[3] 224 화소를 덮는 데 드는 깊이
  3x3 스트라이드 1만 쌓을 때          : 112레이어 (리셉티브 필드 225)
  3x3 스트라이드 1,2를 번갈아 쌓을 때 : 12레이어 (리셉티브 필드 253, 누적 스트라이드 64)
  어텐션 한 레이어                  : 1레이어 (토큰 197개가 서로를 모두 본다)
  토큰 하나가 덮는 화소         : 16x16, 조각 196개면 224x224 전체

[4] 토큰 수에 대한 비용의 차수
  224x224: 토큰 197 | 블록 곱셈-덧셈 1,453,954,560
            일차 항   1,394,343,936 | 이차 항    59,610,624 (4.1%)
  384x384: 토큰 577 | 블록 곱셈-덧셈 4,595,320,320
            일차 항   4,083,941,376 | 이차 항   511,378,944 (11.1%)
  224에서 384로 갈 때 토큰이 2.93곱
    이차 항은 8.58곱
    일차 항은 2.93곱
    화소 수는 2.94곱 — 컨볼루션은 여기에 붙어 자란다
```

읽을 것은 네 가지다.

**패치 투영의 대응**: 같은 비중첩 패치에 같은 선형 웨이트를 적용하는 연산은 커널=스트라이드인 Conv2d로 표현할 수 있다. 웨이트 배열과 바이어스를 맞춘 두 경로의 기록된 차이는 약 1.25×10⁻⁶이다. 대수적으로 같은 연산이어도 부동소수점 계산 순서와 장치에 따라 차이가 있을 수 있다. 패치 순서와 채널 배열을 맞춰야 한다.

**특정 모델의 구성**: 224×224 입력을 16×16 패치로 나누면 196개이고 클래스 토큰 하나를 더하면 197개이다. 폭 768인 패치 투영은 590592개의 계수를 사용한다. 코드의 1000클래스 모델 전체는 86567656개, 분류 헤드를 제외하면 85798656개이다. 전체 계수는 블록 수·폭·헤드·MLP와 위치 임베딩 등 전체 구성에 의해 정해지며 패치 투영 하나만으로 결정되지 않는다.

**전역 연결과 경계 너비**: 3×3 스트라이드 1 컨볼루션의 리셉티브 필드 너비가 224 이상이 되는 데는 112레이어가 필요하다. 이는 출력의 위치와 관계없이 반대편 모서리까지 연결된다는 뜻은 아니다. 같은 해상도에서 모서리 출력과 반대 모서리 입력의 연결은 223레이어가 필요하다. 코드의 스트라이드 혼합 12레이어는 지정한 배열의 경계 너비 계산이다. 전역 어텐션은 한 집계에서 모든 토큰을 포함하지만 동일한 피처가나 추론 능력을 보장하지 않는다.

**주요 MAC의 분포**: 폭 768·MLP 확장 4인 기본 블록 근사에서 토큰 197개의 쌍 연산 비중은 약 4.1%, 577개에서는 약 11.1%이다. 투영·MLP·쌍 연산만 세고 레이어 정규화·소프트맥스·메모리 이동 등을 제외한 계산이다. 어느 항이 연산량을 차지하는지와 실제 메모리·시간 병목을 구분해야 한다.

---

## 7. 관련 주제

- [리셉티브 필드](../cnn/receptive_field.md) — 이 절 3절과 6절이 쓰는 점화식의 출처
- [Squeeze-and-Excitation](squeeze_excitation.md) — 2단계의 채널 어텐션
- [비국소 뉴럴 넷트웍](non_local.md) — 3단계의 공간 어텐션
- [항등 사상](../residual/identity_mapping.md) — 5단계 ViT 블록의 잔차 구조와 같은 꼴
- [ViT](../../appendix/vit/vision_transformer_vit.md) — 5단계의 온전한 구현
- [Swin Transformer](../../appendix/vit/swin_transformer.md) — 피라미드 구조

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
CNN과 기본 ViT의 지역 연결·위치 표현·계산 차이를 설명하라.

</div>

??? success "풀이"
    CNN은 국소 격자 연결과 웨이트 공유를 사용하며 깊이·스트라이드·팽창률에 따라 연결 범위를 넓힌다. 기본 ViT는 패치 투영과 위치 표현을 사용하고 전역 어텐션으로 모든 토큰을 결합한다. ViT에도 패치 격자 등의 귀납 바이어스가 있고 CNN의 엄격한 이동 등변성도 경계·스트라이드에 의해 제한된다. 어느 구조가 데이터 효율과 속도에서 유리한지는 학습 조건과 구현에 달려 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SE와 비국소 블록을 ViT 어텐션과 비교하라.

</div>

??? success "풀이"
    SE는 전역 통계에서 채널별 배율을 정해 곱하고, 비국소 embedded Gaussian은 위치별 점수로 다른 위치의 값 피처를 결합한다. 후자는 스케일·정규화·투영을 맞추면 단일 헤드 소프트맥스 어텐션과 대응한다. 기본 ViT는 다중 헤드, 잔차·레이어 정규화·MLP·위치 표현과 패치 토큰화를 함께 사용하므로 전체 구조의 차이가 토큰 종류 하나뿐인 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ViT의 데이터 요구량을 고정된 최소 이미지 수로 표현할 수 없는 이유를 설명하라.

</div>

??? success "풀이"
    초기 ViT 실험의 데이터 규모와 정확도 관계는 해당 구조·학습 절차의 결과이다. 어그멘테이션·정규화·사전 학습·증류와 모델 크기가 바뀌면 데이터 효율도 달라진다. DeiT처럼 ImageNet만으로 경쟁력 있게 학습한 사례도 있으므로 1억 장이나 3억 장이 모든 ViT의 필수 조건은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
CNN의 피처 추출과 트랜스포머의 처리를 결합한 혼합 구조를 설계하라. 줄기의 총 스트라이드를 정할 때 무엇을 보아야 하는지 수로 밝혀라.

</div>

??? success "풀이"
    컨볼루션 스템의 깊이·채널·커널·스트라이드와 토큰 투영, 어텐션 범위 등을 선택할 수 있다. 총 스트라이드는 입력이 나누어떨어지고 패딩 규약이 맞을 때 격자 크기를 가늠하는 한 기준이다. 총 스트라이드 4라면 224→56으로 패치 토큰 3136개, 스트라이드 16이라면 224→14로 196개가 된다. 클래스 토큰을 포함하면 3137과 197로 계산하며, 실제 혼합 모델이 모두 스트라이드 16을 사용해야 하는 것은 아니다.

    ```python
    class HybridViT(nn.Module):
        """컨볼루션 줄기로 총 스트라이드 16까지 줄인 뒤 트랜스포머에 넘긴다."""

        def __init__(self, dim=768, depth=12, heads=12, num_classes=1000):
            super().__init__()
            # 스트라이드 2짜리 컨볼루션 넷 = 총 스트라이드 16. 224 -> 14
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, dim, 3, stride=2, padding=1),
            )  # 출력: (B, 768, 14, 14) -> 토큰 196개, ViT-Base와 같다
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, 197, dim))
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=4 * dim,
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_classes)

        def forward(self, x):
            z = self.stem(x).flatten(2).transpose(1, 2)      # (B, 196, 768)
            z = torch.cat([self.cls.expand(z.size(0), -1, -1), z], dim=1)
            z = self.transformer(z + self.pos)               # (B, 197, 768)
            return self.head(self.norm(z)[:, 0])
    ```

    아래 HybridViT는 224×224 입력에서 총 스트라이드 16과 위치 임베딩 197개를 가정한다. 다른 해상도에서는 토큰 수와 위치 표현의 호환 처리가 필요하다. norm_first=True로 사전 정규화를 쓰지만 TransformerEncoderLayer의 기본 활성화·드롭아웃 등까지 원래 ViT와 같아지는 것은 아니다. 임포트와 주변 정의를 준비한 코드 패치으로 읽는다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
6절의 [4]는 224×224에서 어텐션 행렬이 블록 비용의 4.1%만 차지한다고 말한다. 이 몫이 50%가 되는 토큰 수를 은닉 차수 $d = 768$, 다층 퍼셉트론 배수 4에 대해 구하고, 패치 크기 16을 그대로 쓸 때 그것이 어느 해상도인지 말하라.

</div>

??? success "풀이"
    6절 `block_cost`의 세 항은 일차 항 $12 N d^2$(투영 $4Nd^2$ + 다층 퍼셉트론 $8Nd^2$)과 이차 항 $2N^2 d$이다. 몫이 50%라는 것은 두 항이 같다는 뜻이므로

    $$2N^2 d = 12 N d^2 \quad \Longleftrightarrow \quad N = 6d$$

    이다. $d = 768$이면 $N = 4608$개다. 이 값은 $d$에만 딸려 있고 패치 크기나 해상도와는 무관하다는 점이 요점이다 — 은닉 차수를 키우면 이차 항이 주인이 되는 자리도 함께 뒤로 밀린다.

    같은 계산에서 이차 항의 몫이 곧바로 나온다.

    $$\frac{2N^2 d}{12Nd^2 + 2N^2 d} = \frac{N}{6d + N}$$

    $d = 768$이므로 $6d = 4608$이고, $N = 197$이면 $197 / 4805 = 4.1\%$, $N = 577$이면 $577 / 5185 = 11.1\%$ — 6절이 찍은 두 값과 정확히 같다.

    패치 16과 클래스 토큰 하나를 가정하면 정사각 입력에서 $N=(H/16)^2+1$이다. N=4608에 대응하는 연속 높이는 약 1086이고, 정수 패치 격자에서 처음 넘는 것은 68×68, 높이 1088이다. 이것은 폭 768·MLP 확장 4의 주요 MAC 근사에서 두 항이 같아지는 기준일 뿐, 그 해상도 아래에서는 이차 비용이나 어텐션 메모리가 중요하지 않다는 뜻은 아니다.

---

## 정리하며

CNN과 ViT는 국소 연결, 웨이트 공유와 입력별 가중 결합을 다르게 구성한다. 비중첩 패치 투영은 웨이트를 맞춘 컨볼루션으로 표현할 수 있다. 토큰 수와 폭을 기준으로 주요 MAC과 어텐션 행렬 메모리를 따로 계산하고, 데이터 효율·성능·지연은 특정 실험 조건에서 비교해야 한다.
