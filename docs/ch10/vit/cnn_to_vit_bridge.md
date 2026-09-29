# 다리: CNN에서 비전 트랜스포머로

합성곱 신경망에서 비전 트랜스포머로 넘어간 것은 컴퓨터 비전 구조의 근본적인 전환이다. 이 흐름은 갑작스러운 단절이라기보다 CNN의 귀납 편향을 차츰 느슨하게 푸는 과정이며, 그 사이의 구조들이 지역성 기반 처리와 어텐션 기반 처리를 잇는 개념적 다리 노릇을 한다.

다만 "다리"라는 말이 비유에 그치면 아무것도 확인할 수 없다. 이 마당은 다리를 두 번 놓는다. 먼저 1절부터 5절까지 다섯 단계의 구조 변화를 말로 좇고, 그다음 6절 "다리를 수로 재어 보기"에서 그 말 가운데 수로 잴 수 있는 것을 모두 돌려서 확인한다. 조각 임베딩이 정말 합성곱 한 층인지, ViT-Base/16의 토큰이 정말 197개인지, 어텐션의 무엇이 이차이고 무엇이 일차인지는 외울 것이 아니라 재어 볼 것이다.

---

## 1. 구조 패러다임의 변천

### 1단계: 순수한 CNN

전통적인 CNN은 지역 합성곱을 잇달아 적용하여 이미지를 처리한다. 출력 자리 $(i,j)$의 값은 입력의 같은 자리 둘레에 있는 작은 창 하나만 보고 정해진다.

$$\mathbf{y}_{i,j}^{(l)} = f\left(\sum_{c}\sum_{p,q} W_{c,p,q}^{(l)} \mathbf{x}_{c,\,i+p,\, j+q}^{(l-1)} + b^{(l)}\right)$$

여기서 $c$은 입력 채널을 훑고, $(p,q)$은 핵 안의 자리를 훑으며, $f$은 활성 함수이다. 채널에 걸친 합이 있어야 색 세 장이나 앞 층의 특징 맵 여러 장을 하나로 묶을 수 있다.

지역성에 기댄 이 방식은 다음을 보장한다.

- 계산 효율
- 이미지 구조와 들어맞는 귀납 편향
- 공간에 걸친 매개변수 공유

### 2단계: 어텐션으로 보강한 CNN

CNN에 어텐션 장치([압축-여기](squeeze_excitation.md), CBAM)를 더하면 지역성을 지키면서도 채널과 공간의 가중치를 적응적으로 다시 매길 수 있다.

$$\mathbf{y}_{i,j}^{(l)} = \alpha_{i,j}^{(l)} \odot f\left(\sum_{c}\sum_{p,q} W_{c,p,q}^{(l)} \mathbf{x}_{c,\,i+p, \,j+q}^{(l-1)} + b^{(l)}\right)$$

여기서 $\odot$은 성분별 곱을 뜻하고 $\alpha_{i,j}^{(l)}$은 학습된 어텐션 가중치를 나타낸다. 가중치를 만들 때 전역 평균 풀링이 쓰이므로, 곱해지는 값에는 이미 특징 맵 전체의 정보가 들어 있다. 연산 자체는 여전히 지역적이되 그 크기를 정하는 잣대가 전역이 된 것이다.

### 3단계: 비국소 신경망

[비국소 연산](non_local.md)은 공간 전체에 걸친 의존 관계를 붙잡는다.

$$\mathbf{y}_{i} = \frac{1}{C(\mathbf{x})} \sum_{j} f(\mathbf{x}_i, \mathbf{x}_j) \cdot g(\mathbf{x}_j)$$

여기서 $f$은 자리 $i$와 $j$ 사이의 관계를 재고, $g$은 특징을 사영하며, $C(\mathbf{x})$은 입력에 따라 달라지는 정규화 인수이다. 정규화 인수가 상수가 아니라 입력의 함수라는 점이 중요하다. 이는 엄격한 지역성에서 벗어난 것이며, $f$을 배율 조정한 내적으로 두면 식 자체가 자기 어텐션과 같아진다.

### 4단계: 혼합 구조

혼합 모델은 합성곱 줄기에 트랜스포머 몸통을 잇는다.

$$\text{Tokens} = \text{PatchEmbed}(\text{ConvStem}(\mathbf{x}))$$

$$\mathbf{z}^{(l)} = \text{Transformer}(\mathbf{z}^{(l-1)})$$

앞쪽 층은 효율을 위해 CNN의 귀납 편향을 쓰고, 뒤쪽 층은 트랜스포머의 유연함을 쓴다. 줄기가 이미 공간을 줄여 두므로 트랜스포머가 받는 토큰 수가 작아지는데, 6절에서 보듯 이 토큰 수가 어텐션 비용을 이차로 좌우한다.

### 5단계: 순수한 비전 트랜스포머

ViT는 합성곱 없이 이미지를 조각의 순차열로 처리한다. 블록 하나는 두 걸음으로 이루어지며, 중간값을 $\mathbf{z}'^{(l)}$으로 따로 이름 붙여야 두 걸음이 섞이지 않는다.

$$\mathbf{z}'^{(l)} = \text{MSA}(\text{LN}(\mathbf{z}^{(l-1)})) + \mathbf{z}^{(l-1)}$$

$$\mathbf{z}^{(l)} = \text{MLP}(\text{LN}(\mathbf{z}'^{(l)})) + \mathbf{z}'^{(l)}$$

두 식 모두 왼쪽에 $\mathbf{z}^{(l)}$을 쓰면 두 번째 식이 갱신이 아니라 $\mathbf{z}^{(l)} = \text{MLP}(\text{LN}(\mathbf{z}^{(l)})) + \mathbf{z}^{(l)}$이라는 고정점 방정식이 되어 버린다.

---

## 2. 핵심적인 구조의 다리

!!! tip "조각 임베딩은 합성곱을 닮은 것이 아니라 합성곱이다"
    ViT의 조각 임베딩 층은 핵 크기와 보폭이 똑같은 합성곱 층 하나와 **같은 연산**이다. 비유가 아니라 등식이며, 6절에서 두 길의 출력이 float32 반올림 자리까지만 어긋남을 보인다. torchvision의 `vit_b_16`은 이 층을 아예 `nn.Conv2d(3, 768, kernel_size=16, stride=16)`으로 적어 두었다.

**합성곱 줄기**: ViT의 첫 조각 임베딩을 합성곱 층 여러 개로 바꾸어 저수준 특징을 잡으면 작은 데이터셋에서 성능이 좋아진다.

**피라미드 구조**: 계층적 시각 모델(Swin, PVT)은 어텐션을 쓰면서도 CNN과 비슷하게 여러 규모의 처리를 유지한다.

**혼합 자기 어텐션**: 특정 자리의 트랜스포머 블록을 합성곱 층으로 바꾸어 계산 비용과 모형화 능력을 저울질한다.

---

## 3. 비교 분석

아래 표의 수는 모두 6절에서 재어 나온 것이거나 그 자리에서 셈할 수 있는 것이다.

| 항목 | CNN | 혼합 | ViT |
|-----------|-----|--------|-----|
| **한 층이 닿는 범위** | 핵 크기만큼 (지역) | 줄기는 지역, 몸통은 전역 | 첫 층부터 전역 |
| **224 화소를 덮는 데 드는 깊이** | 3×3 보폭 1이면 112층, 보폭 2를 섞으면 12층 | 줄기가 덮은 뒤 한 층 | 1층 |
| **입력이 커질 때 드는 비용** | 화소 수에 일차 | 사이 | 어텐션 행렬만 토큰 수에 이차, 나머지는 일차 |
| **필요한 데이터양** | ImageNet-1k(128만 장)로 충분 | 보통 | JFT-300M급에서야 CNN을 앞지름 |
| **공간에 대한 사전 지식** | 지역성과 가중치 공유가 구조에 박혀 있음 | 줄기에만 | 위치 임베딩으로 배워야 함 |
| **평행 이동 동변성** | 보폭 1에서 엄격 | 느슨 | 조각 간격(16화소)의 정수배에만, 그마저 절대 위치 임베딩이 깨뜨림 |

!!! warning "'전역'과 '이차'는 서로 다른 것에 붙는 말이다"
    표의 둘째 줄과 셋째 줄을 한 줄로 합쳐 "ViT는 수용 영역이 이차로 자란다"고 적으면 두 번 틀린다. 첫째, 어텐션의 수용 영역은 자라지 않는다 — 첫 층에서 이미 전부다. 둘째, 이차인 것은 수용 영역이 아니라 **토큰 수에 대한 어텐션 행렬의 비용**이다.

    한편 CNN 쪽의 "선형"도 조건이 붙는다. [수용 영역](../cnn/receptive_field.md) 쪽의 점화식

    $$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

    에서 깊이에 선형인 것은 모든 보폭이 1일 때뿐이다. 보폭 2를 섞으면 누적 보폭 $\prod s_i$이 기하급수로 커져 112층이 12층으로 줄어든다. 어텐션의 "첫 층부터 전역"은 이 점화식에서 나오는 결과가 **아니라** 연결 방식 자체가 다르다는 뜻이다.

---

## 4. 실무자를 위한 고려 사항

!!! warning "데이터의 규모가 중요하다"
    순수한 ViT는 ImageNet-1k(128만 장) 정도만으로 학습하면 같은 크기의 CNN보다 대체로 못하다. ViT 논문이 JFT의 부분집합으로 그린 곡선에서 두 곡선이 갈리는 자리는 이미지 1억 장 언저리이며, JFT-300M까지 가야 ViT가 확실히 앞선다. 데이터가 적은 상황에서는 혼합 구조나 전이 학습이 꼭 필요해진다.

!!! note "계산 예산"
    계산 자원이 크게 모자랄 때는 여전히 CNN이 유리하다. 다만 6절의 셈이 보여 주듯 224×224에서 ViT 블록 비용의 96%는 토큰 수에 **일차**인 항이므로, ViT가 비싼 까닭은 어텐션의 이차 항보다 은닉 차수 768의 제곱에 붙은 사영·다층 퍼셉트론 쪽이 크다. 이차 항이 주인이 되는 것은 해상도를 크게 올려 토큰이 수천 개가 될 때다.

---

## 5. 앞으로의 방향

새로 나오는 연구는 다음을 살핀다.

- **뉴로모픽 혼합**: 스파이킹 뉴런과 어텐션의 결합
- **효율적인 트랜스포머**: 고해상도 이미지를 위한 선형 복잡도 어텐션
- **학습된 경로 배정**: 입력에 따라 합성곱과 어텐션 중에서 그때그때 고르기
- **다중 양식 융합**: 혼합 구조로 시각과 언어를 잇기

---

## 6. 다리를 수로 재어 보기

앞의 다섯 절은 모두 주장이다. 여기서는 그 가운데 잴 수 있는 넷을 실제로 돌려서 확인한다. (1) 조각 임베딩과 합성곱이 같은 연산인지, (2) ViT-Base/16의 토큰·층·머리·매개변수가 몇인지, (3) 224 화소를 덮는 데 합성곱과 어텐션이 각각 몇 층을 쓰는지, (4) 토큰 수가 늘 때 비용의 어느 항이 이차인지이다. 셋째는 [수용 영역](../cnn/receptive_field.md) 쪽의 점화식을 그대로 다시 쓴 것이다.

```python
"""CNN과 ViT를 잇는 다리를 수로 재어 본다."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

torch.manual_seed(0)

IMG, PATCH, DIM = 224, 16, 768


# === 1. 조각 임베딩 = 겹치지 않는 합성곱 ===

def patch_embed_is_conv():
    """조각을 펼쳐 선형층에 넣는 것과 stride=kernel 합성곱이 같음을 보인다."""
    x = torch.randn(1, 3, IMG, IMG)
    conv = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)

    # 합성곱 길
    z_conv = conv(x)                                 # (1, 768, 14, 14)
    tokens_conv = z_conv.flatten(2).transpose(1, 2)  # (1, 196, 768)

    # 펼치기 + 선형층 길 (같은 가중치를 모양만 바꿔 쓴다)
    patches = F.unfold(x, kernel_size=PATCH, stride=PATCH)  # (1, 3*16*16, 196)
    patches = patches.transpose(1, 2)                       # (1, 196, 768)
    W = conv.weight.reshape(DIM, -1)                        # (768, 768)
    tokens_lin = patches @ W.t() + conv.bias

    cls = torch.zeros(1, 1, DIM)
    with_cls = torch.cat([cls, tokens_conv], dim=1)

    print("[1] 조각 임베딩은 합성곱이다")
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
    print(f"  조각 임베딩 매개변수          : {n_w + DIM:,} = {n_w:,} + {DIM}")


# === 2. 실제 ViT-Base/16을 뜯어 보기 ===

def real_vit_base():
    """torchvision의 ViT-Base/16에서 구조 수를 직접 읽는다."""
    m = vit_b_16(weights=None)
    enc = m.encoder.layers
    blk = enc[0]
    total = sum(p.numel() for p in m.parameters())
    head = sum(p.numel() for p in m.heads.parameters())

    print("\n[2] torchvision ViT-Base/16 (가중치 없이 구조만)")
    print(f"  조각 임베딩 층                : {m.conv_proj}")
    print(f"  은닉 차수                     : {blk.ln_1.normalized_shape[0]}")
    print(f"  블록 수                       : {len(enc)}")
    print(f"  머리 수                       : {blk.self_attention.num_heads}")
    print(f"  머리 하나의 차수              : "
          f"{blk.ln_1.normalized_shape[0] // blk.self_attention.num_heads}")
    print(f"  위치 임베딩                   : {tuple(m.encoder.pos_embedding.shape)}")
    print(f"  매개변수 (분류 머리 포함)     : {total:,}")
    print(f"  매개변수 (분류 머리 뺀 몸통)  : {total - head:,}")

    out = m(torch.randn(1, 3, IMG, IMG))
    print(f"  출력 모양                     : {tuple(out.shape)}")


# === 3. 수용 영역: 합성곱의 점화식 대 어텐션의 연결 ===

def receptive_field(layers):
    """r_l = r_{l-1} + (K_l - 1) * d_l * prod(s_i), 수용 영역 쪽과 같은 점화식."""
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += (k - 1) * d * jump
        jump *= s
    return rf, jump


def depth_to_cover():
    """3x3 합성곱을 몇 층 쌓아야 224 화소를 덮는가."""
    print("\n[3] 224 화소를 덮는 데 드는 깊이")

    n = 0
    while receptive_field([(3, 1, 1)] * n)[0] < IMG:
        n += 1
    print(f"  3x3 보폭 1만 쌓을 때          : {n}층 "
          f"(수용 영역 {receptive_field([(3, 1, 1)] * n)[0]})")

    n = 0
    while True:
        stack = [(3, 1, 1), (3, 2, 1)] * n
        if stack and receptive_field(stack)[0] >= IMG:
            break
        n += 1
    rf, jump = receptive_field(stack)
    print(f"  3x3 보폭 1,2를 번갈아 쌓을 때 : {len(stack)}층 "
          f"(수용 영역 {rf}, 누적 보폭 {jump})")

    print(f"  어텐션 한 층                  : 1층 (토큰 197개가 서로를 모두 본다)")
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
    print(f"    화소 수는 {(384 / 224) ** 2:.2f}곱 — 합성곱은 여기에 붙어 자란다")


if __name__ == "__main__":
    patch_embed_is_conv()
    real_vit_base()
    depth_to_cover()
    cost_exponents()
```

**출력:**

```text
[1] 조각 임베딩은 합성곱이다
  입력                          : (1, 3, 224, 224)
  Conv2d(3, 768, k=16, s=16)    : (1, 768, 14, 14)
  펼친 토큰                     : (1, 196, 768)
  클래스 토큰을 붙인 뒤         : (1, 197, 768)
  조각 수 (224//16)**2          : 196
  토큰 수 = 조각 + 클래스       : 197
  두 길의 최대 어긋남           : 1.25e-06
  torch.allclose(atol=1e-5)     : True
  조각 임베딩 매개변수          : 590,592 = 589,824 + 768

[2] torchvision ViT-Base/16 (가중치 없이 구조만)
  조각 임베딩 층                : Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  은닉 차수                     : 768
  블록 수                       : 12
  머리 수                       : 12
  머리 하나의 차수              : 64
  위치 임베딩                   : (1, 197, 768)
  매개변수 (분류 머리 포함)     : 86,567,656
  매개변수 (분류 머리 뺀 몸통)  : 85,798,656
  출력 모양                     : (1, 1000)

[3] 224 화소를 덮는 데 드는 깊이
  3x3 보폭 1만 쌓을 때          : 112층 (수용 영역 225)
  3x3 보폭 1,2를 번갈아 쌓을 때 : 12층 (수용 영역 253, 누적 보폭 64)
  어텐션 한 층                  : 1층 (토큰 197개가 서로를 모두 본다)
  토큰 하나가 덮는 화소         : 16x16, 조각 196개면 224x224 전체

[4] 토큰 수에 대한 비용의 차수
  224x224: 토큰 197 | 블록 곱셈-덧셈 1,453,954,560
            일차 항   1,394,343,936 | 이차 항    59,610,624 (4.1%)
  384x384: 토큰 577 | 블록 곱셈-덧셈 4,595,320,320
            일차 항   4,083,941,376 | 이차 항   511,378,944 (11.1%)
  224에서 384로 갈 때 토큰이 2.93곱
    이차 항은 8.58곱
    일차 항은 2.93곱
    화소 수는 2.94곱 — 합성곱은 여기에 붙어 자란다
```

읽을 것은 네 가지다.

**조각 임베딩은 정확히 합성곱이다.** 같은 가중치를 `(768, 3, 16, 16)`으로 보면 합성곱, `(768, 768)`로 보면 선형층인데 두 길의 출력이 $1.25 \times 10^{-6}$까지만 어긋난다. 이는 두 길이 부동소수점 덧셈을 다른 차례로 하기 때문에 생기는 float32 반올림이지 서로 다른 연산이어서가 아니다. 이 등식이 있어 CNN을 알던 독자가 ViT의 첫 층에서 새로 배울 것은 아무것도 없다.

**ViT-Base/16의 수는 모두 이 등식에서 따라 나온다.** 조각 $(224/16)^2 = 196$개에 클래스 토큰 하나를 더해 197개이고, 위치 임베딩 모양 `(1, 197, 768)`이 이를 되비친다. 조각 임베딩 매개변수 590,592개는 $768 \times 3 \times 16 \times 16 + 768$이며, 전체 86,567,656개 가운데 0.7%에 지나지 않는다. 흔히 말하는 "8600만"은 어느 쪽으로 세든 나오는 값이다 — 1000갈래 분류 머리($768 \times 1000 + 1000 = 769{,}000$)를 넣으면 86,567,656개, 빼면 85,798,656개다.

**어텐션의 전역성은 점화식의 결과가 아니다.** 3×3 합성곱을 보폭 1로만 쌓으면 $r_l = 1 + 2l$이므로 224를 덮는 데 112층이 든다. 보폭 2를 섞으면 누적 보폭이 64까지 불어 12층으로 끝난다. 어텐션은 1층이다. 셋을 나란히 두면 어텐션이 빠른 것이 아니라 아예 다른 방식으로 연결되어 있음이 드러난다 — 합성곱의 수용 영역은 깊이의 함수이고, 어텐션의 그것은 깊이와 무관한 상수다.

**이차 항은 224에서는 아직 주인이 아니다.** 토큰 197개일 때 어텐션 행렬이 차지하는 몫은 블록 비용의 4.1%뿐이고, 나머지 96%는 토큰 수에 일차이면서 은닉 차수에 이차인 사영과 다층 퍼셉트론이다. 384×384로 올리면 토큰이 2.93곱 늘 때 이차 항은 8.58곱($2.93^2 = 8.58$) 늘어 몫이 11.1%로 커진다. 합성곱 쪽 비용은 화소 수를 따라가므로 같은 변화에서 2.94곱에 그친다. "ViT는 이차라 비싸다"는 말이 참이 되는 지점은 해상도가 훨씬 높아져 토큰이 수천 개가 되는 곳이지, 표준 224×224가 아니다.

---

## 7. 관련 주제

- [수용 영역](../cnn/receptive_field.md) — 이 마당 3절과 6절이 쓰는 점화식의 출처
- [압축-여기](squeeze_excitation.md) — 2단계의 채널 어텐션
- [비국소 신경망](non_local.md) — 3단계의 공간 어텐션
- [항등 사상](../residual/identity_mapping.md) — 5단계 ViT 블록의 잔차 구조와 같은 꼴
- [ViT](../../appendix/vit/vision_transformer_vit.md) — 5단계의 온전한 구현
- [Swin Transformer](../../appendix/vit/swin_transformer.md) — 피라미드 구조

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
CNN과 비전 트랜스포머의 구조적 차이 가운데 핵심은 무엇인가?

</div>

??? success "연습문제 1 풀이"
    CNN은 지역 수용 영역, 가중치 공유로 얻는 평행 이동 동변성, 계층적 하향 표본화, 공간 지역성에 대한 귀납 편향을 갖는다. ViT는 첫 층부터 전역 어텐션을 쓰고, 내장된 공간 편향이 없으며, 조각 단위로 토큰을 만들고, 공간 정보를 위해 위치 임베딩을 쓴다. 6절의 [3]이 이 차이를 수로 보여 준다 — 224 화소를 덮는 데 3×3 합성곱은 112층(보폭 1) 또는 12층(보폭 2를 섞을 때)이 들지만 어텐션은 1층이다. ViT는 데이터가 더 많이 필요하지만 더 유연한 표현을 배운다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
압축-여기 신경망과 비국소 신경망의 어텐션 장치가 어떻게 ViT를 앞서 보여 주었는지 설명하라.

</div>

??? success "연습문제 2 풀이"
    SE 신경망은 채널 어텐션(전역 평균 풀링과 완전 연결층)을 더하여 '무엇'에 주목할지 배운다. 비국소 신경망은 공간 어텐션(특징 맵 위의 자기 어텐션)을 더하여 '어디'에 주목할지 배운다. 둘 다 지역적인 CNN 특징에 전역 맥락을 더하여 ViT의 온전한 자기 어텐션으로 가는 다리를 놓는다. 특히 비국소 연산의 식

    $$\mathbf{y}_i = \frac{1}{C(\mathbf{x})} \sum_j f(\mathbf{x}_i, \mathbf{x}_j) \cdot g(\mathbf{x}_j)$$

    에서 $f$을 $\exp(\theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j) / \sqrt{d})$으로, $C(\mathbf{x})$을 그 합으로 두면 소프트맥스 자기 어텐션과 글자 그대로 같아진다. 남는 차이는 토큰을 무엇으로 삼느냐 — 특징 맵의 화소냐 이미지 조각이냐 — 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ViT가 비슷한 성능을 내는 데 CNN보다 학습 데이터가 더 많이 드는 까닭은 무엇인가?

</div>

??? success "연습문제 3 풀이"
    CNN은 이미지에 대한 사전 지식을 담은 강한 귀납 편향(지역성, 평행 이동 동변성)을 갖는다. ViT는 그런 성질을 데이터에서 배워야 한다. 데이터가 적으면 CNN의 사전 지식이 일반화를 돕는다. ViT 논문이 JFT의 부분집합(1000만·3000만·1억·3억 장)으로 그린 곡선에서 두 곡선이 갈리는 자리는 1억 장 언저리이고, JFT-300M까지 가야 ViT의 유연함이 CNN을 확실히 앞지른다. 강한 증강과 증류를 쓰면(DeiT) ImageNet-1k만으로도 좁힐 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
CNN의 특징 추출과 트랜스포머의 처리를 결합한 혼합 구조를 설계하라. 줄기의 총 보폭을 정할 때 무엇을 보아야 하는지 수로 밝혀라.

</div>

??? success "연습문제 4 풀이"
    설계에서 자유롭게 고를 수 있는 값은 줄기의 **총 보폭** 하나이고, 이것이 토큰 수를 정하며, 토큰 수가 어텐션 비용을 이차로 정한다. 총 보폭 4로 두면 $224/4 = 56$, 토큰이 $56^2 = 3136$개로 ViT-Base의 196개보다 16곱 많아지고 어텐션 행렬 비용은 $16^2 = 256$곱이 된다. 그래서 실제 혼합 모델은 줄기에서 총 보폭 16까지 줄인 뒤 트랜스포머에 넘긴다.

    ```python
    class HybridViT(nn.Module):
        """합성곱 줄기로 총 보폭 16까지 줄인 뒤 트랜스포머에 넘긴다."""

        def __init__(self, dim=768, depth=12, heads=12, num_classes=1000):
            super().__init__()
            # 보폭 2짜리 합성곱 넷 = 총 보폭 16. 224 -> 14
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

    합성곱 줄기는 (지역성 편향이 도움이 되는) 저수준 특징 추출을 맡고, 트랜스포머는 전역적인 추론을 맡는다. 줄기가 조각 임베딩 한 층을 대신하는 것이므로 `norm_first=True`로 두어 5절의 사전 정규화 블록과 같은 꼴을 지킨다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
6절의 [4]는 224×224에서 어텐션 행렬이 블록 비용의 4.1%만 차지한다고 말한다. 이 몫이 50%가 되는 토큰 수를 은닉 차수 $d = 768$, 다층 퍼셉트론 배수 4에 대해 구하고, 조각 크기 16을 그대로 쓸 때 그것이 어느 해상도인지 말하라.

</div>

??? success "연습문제 5 풀이"
    6절 `block_cost`의 세 항은 일차 항 $12 N d^2$(사영 $4Nd^2$ + 다층 퍼셉트론 $8Nd^2$)과 이차 항 $2N^2 d$이다. 몫이 50%라는 것은 두 항이 같다는 뜻이므로

    $$2N^2 d = 12 N d^2 \quad \Longleftrightarrow \quad N = 6d$$

    이다. $d = 768$이면 $N = 4608$개다. 이 값은 $d$에만 딸려 있고 조각 크기나 해상도와는 무관하다는 점이 요점이다 — 은닉 차수를 키우면 이차 항이 주인이 되는 자리도 함께 뒤로 밀린다.

    같은 셈에서 이차 항의 몫이 곧바로 나온다.

    $$\frac{2N^2 d}{12Nd^2 + 2N^2 d} = \frac{N}{6d + N}$$

    $d = 768$이므로 $6d = 4608$이고, $N = 197$이면 $197 / 4805 = 4.1\%$, $N = 577$이면 $577 / 5185 = 11.1\%$ — 6절이 찍은 두 값과 정확히 같다.

    조각 크기 16을 그대로 쓰면 토큰 수가 $(\text{해상도}/16)^2 + 1$이므로 $N = 4608$은 해상도 $16\sqrt{4607} \approx 1086$화소에 해당한다. 조각 격자가 정수여야 하니 실제로 절반을 처음 넘기는 것은 68×68 격자, 곧 1088화소이고 이때 $N = 4625$, 몫은 $4625 / 9233 = 50.1\%$다. 즉 흔히 쓰는 해상도에서 ViT가 비싼 까닭은 어텐션의 이차성이 아니라 $d^2$에 붙은 일차 항이고, 이차성이 실제로 아픈 곳은 1000화소를 넘는 밀집 예측 과제다.

---

## 정리하며

이 마당은 CNN에서 ViT로 가는 길을 다섯 단계로 좇은 뒤, 그 가운데 잴 수 있는 것을 6절에서 모두 돌려 보았다. 남는 것은 세 문장이다. 조각 임베딩은 합성곱을 닮은 것이 아니라 `Conv2d(3, 768, 16, stride=16)` 그 자체이고(어긋남 $1.25 \times 10^{-6}$), ViT-Base/16의 토큰 197개·매개변수 86,567,656개는 모두 이 한 층에서 따라 나온다. 어텐션이 첫 층에서 전역인 것은 수용 영역 점화식이 빨리 자라서가 아니라 연결 방식이 다르기 때문이며, 어텐션의 이차성은 224×224에서 블록 비용의 4.1%일 뿐이어서 토큰이 $6d = 4608$개가 되는 1086화소 언저리에 가서야 주인이 된다.
