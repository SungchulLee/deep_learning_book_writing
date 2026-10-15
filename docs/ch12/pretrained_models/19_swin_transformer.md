# 스윈 트랜스포머

Swin은 국소 윈도 어텐션, 시프트 윈도와 패치 병합을 결합한 계층적 비전 트랜스포머다. 아래 코드는 그 이름의 구성 요소 일부만 보여 주며 실제 Swin의 전체 계산을 구현하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.head = nn.Linear(embed_dim * 8, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = nn.functional.adaptive_avg_pool1d(
            x.transpose(1, 2), 1
        ).squeeze(-1)
        return self.head(x)


if __name__ == "__main__":
    model = SwinTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 773,704
```

## 2. 논의

**모델의 미완성 부분:** SwinTransformer는 패치 합성곱 뒤 평균 풀링을 수행할 뿐 WindowAttention, 시프트나 단계별 패치 병합을 호출하지 않는다. embed_dim 채널을 풀링한 결과에 embed_dim×8 입력의 분류기를 적용해 차원이 맞지 않는다. 기본 설정에서는 96차원 벡터에 768차원 입력층을 적용하므로 forward가 실패한다. 출력의 매개변수 773,704개는 생성만 가능한 구조의 수이며 분류 실행 결과가 아니다.

WindowAttention도 전달된 N개 토큰 전체에 어텐션한다. window_size는 저장만 되고 윈도 분할·병합이나 마스크에 쓰이지 않는다. 외부에서 윈도별 텐서를 준비하지 않으면 국소 윈도 연산이 아니다. 스케일도 일반적인 head_dim의 제곱근이 아니라 전체 C의 제곱근을 사용한다. C가 num_heads로 나누어떨어지는지 검증이 필요하다.

원래 Swin은 고정 크기 윈도를 나누고 시프트 시 경계 마스크를 적용해 이웃 윈도 사이에 정보를 전달한다. 패치 병합으로 해상도를 줄이며 채널을 늘린다. 고정 윈도 크기에서 어텐션 비용은 토큰 수에 선형이지만, 전역 정보가 모든 층 수에서 즉시 전달되는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
224×224 이미지, 패치 4, 한 변 7인 윈도에서 어텐션 점수 수를 비교하라.

</div>

??? success "연습문제 1 풀이"
    첫 격자는 56×56=3136토큰이다. 전역 점수는 헤드당 9,834,496개다. 64개 윈도에 각각 49토큰이 있으므로 점수는 64×49²=153,664개로 64분의 1이다. 이는 윈도 분할이 구현되었다는 가정의 점수 수이며, 현재 SwinTransformer 코드가 이 계산을 수행하는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
시프트 윈도의 목적과 경계 마스크가 필요한 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    분할을 달리해 이전 층의 윈도 경계를 넘는 정보 교환을 만든다. 순환 이동으로 반대편 가장자리가 붙는 경우 실제 이웃이 아닌 토큰 간 어텐션을 마스크해야 한다. 팽창 합성곱은 고정된 성긴 커널 표본을 사용하며 연결 방식이 다르다. 전역 수용 영역은 격자·깊이·병합 구성에 따라 달라진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 PatchMerging의 입출력 조건과 두 단계 모델을 완성하기 위해 남은 작업을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    class PatchMerging(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(4 * dim)

        def forward(self, x, H, W):
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1)
            x = x.view(B, -1, 4 * C)
            x = self.norm(x)
            return self.reduction(x)
    ```
    입력은 (B,H×W,C)이고 H·W가 짝수여야 네 슬라이스를 이어 붙일 수 있다. 4C 특징을 정규화하고 2C로 투영해 (B,H×W/4,2C)를 반환한다. 조각만 정의해도 기존 SwinTransformer에 연결되지는 않는다. 윈도 블록·단계별 해상도와 마지막 채널 수에 맞는 분류기를 따로 구성해야 한다.

## 정리하며

Swin의 국소 어텐션·시프트·계층 병합과 이 페이지의 미완성 코드를 구분한다. 모델의 미사용 어텐션과 분류기 차원 불일치 때문에 매개변수 출력만으로 실행·학습 가능성을 판단해서는 안 된다.
