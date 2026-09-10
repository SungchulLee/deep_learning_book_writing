# 비전 트랜스포머 (ViT)

비전 트랜스포머(ViT)는 컴퓨터 비전의 큰 전환으로, 본디 자연어 처리를 위해 설계된 순수 트랜스포머 구조가 이미지 분류 과제에서 최고 수준의 결과를 낼 수 있음을 보였다. Dosovitskiy 외가 "An Image is Worth 16x16 Words"(2021)에서 내놓은 비전 트랜스포머는 그림을 조각의 수열로 다루어 합성곱 신경망과 트랜스포머 사이의 틈을 잇는다.

---

## 1. 동기: 왜 트랜스포머를 비전에 쓰는가

전통적인 합성곱 신경망은 국소 수용 영역을 가진 위계적 합성곱으로 그림을 처리한다. 잘 통하지만 이 방식에는 한계가 있다.

**합성곱 신경망의 제약:**

- 깊어져도 받는 영역이 천천히 넓어진다
- 먼 거리 의존에는 층이 많이 필요하다
- 귀납 편향이 큰 데이터셋에서 자유로움을 제한할 수 있다

**트랜스포머의 이점:**

- 첫 층부터 받는 영역이 전역이다
- 데이터에서 배운 자유로운 주의 무늬
- 자연어 처리 분야에서 검증된, 키울 수 있음

비전 트랜스포머의 핵심 통찰은 그림을 조각으로 "토큰화"할 수 있고, 그러면 자연어 처리를 뒤바꾼 트랜스포머 구조에 그대로 맞아 들어간다는 것이다.

---

## 2. 구조 개관

비전 트랜스포머 구조는 주요 부품 넷으로 이루어진다.

```
Input Image → Patch Embedding → Transformer Encoder → Classification Head → Output
     ↓              ↓                    ↓                    ↓
  (H,W,C)    (N patches, D)      (N+1 tokens, D)         (classes)
```

### 수식으로 나타내기

입력 그림 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$이 주어지면 비전 트랜스포머의 파이프라인은 다음과 같이 나아간다.

**1단계: 조각 뽑기**
그림을 겹치지 않는 $P \times P$ 크기의 조각 $N = \frac{HW}{P^2}$개로 나눈다.

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, \ldots, N$$

$P = 16$인 표준 $224 \times 224$ 그림에서는 조각이 $N = \frac{224 \times 224}{16 \times 16} = 196$개이고 조각마다 차원이 $P^2 \cdot C = 16^2 \cdot 3 = 768$이다.

**2단계: 선형 사영**
편 조각마다 차원 $D$으로 사영한다.

$$\mathbf{z}_0^i = \mathbf{x}_p^i \mathbf{E} + \mathbf{e}_{pos}^i, \quad \mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$

실제로는 이 사영을 핵 크기 $P$, 걸음 $P$의 2차원 합성곱으로 구현한다.

```python
# 조각을 펴고 선형 사영을 하는 것과 같다
self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
```

**3단계: 분류 토큰 앞에 붙이기**
학습되는 분류 토큰을 앞에 붙인다.

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \mathbf{z}_0^2; \ldots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

`[CLS]` 토큰은 자기 주의로 모든 조각의 정보를 모으며 마지막 분류 판단에 쓰인다.

**4단계: 트랜스포머 인코딩**
트랜스포머 인코더 층 $L$개를 적용한다.

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

요즘 트랜스포머 방식과 같은 **앞 정규화** 설계(아래 층마다 그 앞에 층 정규화)임에 유의하라.

**5단계: 분류**
분류 토큰으로 예측한다.

$$\mathbf{y} = \text{MLP}_{head}(\text{LN}(\mathbf{z}_L^0))$$

### 위치 임베딩

비전 트랜스포머는 자리 색인을 벡터로 잇대는 간단한 찾기 표인 **학습된 1차원 위치 임베딩**을 쓴다.

$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

그림이 2차원인데도 학습된 1차원 임베딩이 잘 통하는 것은 모형이 데이터에서 2차원 짜임을 찾아내기 때문이다. 학습된 위치 임베딩을 그려 보면 (2차원 그림 공간에서) 가까운 조각들이 비슷한 위치 인코딩을 갖게 되어 사실상 격자 배치를 되찾음을 알 수 있다.

(본디 논문에서 살펴본) **2차원을 고려한 대안**으로는 행과 열 자리를 나눈 학습형 2차원 임베딩이 있다. 아주 조금(약 0.5%) 나아질 뿐이며 표준은 아니다.

---

## 3. PyTorch 구현

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    """
    이미지 분류를 위한 비전 트랜스포머.
    
    핵심 혁신:
    1. 그림을 조각의 수열로 다룬다
    2. 이미지 분류에 트랜스포머 인코더를 쓴다
    3. 합성곱 신경망 방식의 입력 처리와 트랜스포머 구조를 잇는다
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # 조각 임베딩 층
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches
        
        # 학습되는 분류 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 학습되는 위치 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 트랜스포머 인코더 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 분류 머리
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # 가중치 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # 그림을 조각 임베딩으로 바꾼다
        x = self.patch_embed(x)  # (B, N, D)
        
        # 분류 토큰을 앞에 붙인다
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # 위치 임베딩을 더한다
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 트랜스포머 블록을 적용한다
        for block in self.blocks:
            x = block(x)
        
        # 분류 토큰으로 분류한다
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS 토큰을 쓴다
```

---

## 4. 모형의 변형

비전 트랜스포머에는 표준 설정이 여럿 있다.

| 모형 | 매개변수 | 임베딩 차원 | 깊이 | 머리 | 조각 크기 |
|-------|-----------|-----------|-------|-------|------------|
| ViT-Tiny | 5M | 192 | 12 | 3 | 16 |
| ViT-Small | 22M | 384 | 12 | 6 | 16 |
| ViT-Base | 86M | 768 | 12 | 12 | 16 |
| ViT-Large | 307M | 1024 | 24 | 16 | 16 |
| ViT-Huge | 632M | 1280 | 32 | 16 | 14 |

### 공장 함수

```python
def create_vit_base(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Base: 매개변수 8600만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768,
        depth=12, n_heads=12, n_classes=n_classes
    )

def create_vit_large(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Large: 매개변수 3억 700만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024,
        depth=24, n_heads=16, n_classes=n_classes
    )
```

---

## 5. 핵심 통찰

### 1. 전역 수용 영역
받는 영역이 차츰 넓어지는 합성곱 신경망과 달리 비전 트랜스포머는 자기 주의로 첫 층부터 전역 수용 영역을 가진다. 그래서 합성곱 신경망이라면 층을 많이 쌓아야 얻을 먼 거리의 공간 의존을 앞쪽 층에서 잡아낼 수 있다.

### 2. 데이터가 얼마나 드는가
비전 트랜스포머가 합성곱 신경망을 앞서려면 큰 규모의 사전 학습(이를테면 ImageNet-21k나 JFT-300M)이 필요하다. 데이터가 적으면 합성곱 신경망의 귀납 편향이 이롭다. 갈림 문턱은 대략 그림 1000만~1억 장이다. 그 아래면 합성곱 신경망이 이기고 그 위면 비전 트랜스포머의 자유로움이 값을 한다.

### 3. 계산 복잡도
자기 주의는 수열 길이에 대해 이차 복잡도 $O(N^2)$을 가지는데 합성곱 신경망은 그림 크기에 대해 일차이다. 조각이 $16 \times 16$인 $224 \times 224$ 그림에서는 $N = 196$이어서 주의 행렬($196 \times 196$)을 감당할 만하다. 그림이 커지거나 조각이 작아지면 $N$과 그에 딸린 이차 비용이 는다.

### 4. 전이 학습
사전 학습된 비전 트랜스포머 모형은 아래쪽 과제로 매우 잘 옮겨 가며 합성곱 신경망의 성능을 넘어서는 일이 많다. 다른 그림 크기로 옮길 때는 (학습된 자리 격자를 2차원 쌍삼차로) 위치 임베딩의 사이를 채울 수 있다.

### 5. 섞은 구조
실용적인 절충은 합성곱 신경망 등뼈(이를테면 ResNet)로 특징 지도를 뽑은 뒤 그 특징을 "조각"으로 트랜스포머에 넣는 것이다.

$$\text{Image} \xrightarrow{\text{CNN}} \text{Feature Map} \xrightarrow{\text{Flatten}} \text{Patch Tokens} \xrightarrow{\text{Transformer}} \text{Output}$$

섞은 모형은 합성곱 신경망이 쓸모 있는 귀납 편향(옮김 동변성, 국소성)을 주고 트랜스포머가 전역 주고받음을 잡아내므로 작은 규모에서 순수 비전 트랜스포머를 앞서는 일이 많다.

### 6. 자기 지도 사전 학습

**MAE(가린 자동 부호기)**: 조각의 많은 몫(75%)을 가리고 빠진 화소를 되살린다. BERT의 가린 언어 모형화에 해당하는 비전 판이며 시각 표현을 배우는 데 매우 잘 통함이 밝혀졌다.

**DINO**: 이름표 없는 자기 증류로, 학생 신경망이 관성으로 갱신되는 스승 신경망의 출력에 맞추는 법을 배운다. DINO로 학습한 비전 트랜스포머의 특징은 분할을 전혀 가르치지 않았는데도 물체의 분할 경계를 찾아내는 등 놀라운 창발 성질을 보인다.

---

## 6. 비전 트랜스포머의 후예

| 모형 | 핵심 혁신 | 나아진 점 |
|-------|---------------|-------------|
| **DeiT** | 합성곱 신경망 스승에게서의 지식 증류 | ImageNet-1K만으로 잘 학습된다 |
| **스윈 트랜스포머** | 어긋난 창 주의, 위계를 이루는 특징 | 그림 크기에 대해 일차 복잡도 |
| **BEiT** | 시각 토큰을 쓰는 BERT 방식 사전 학습 | 더 나은 자기 지도 표현 |
| **MAE** | 높은 비율의 가리기와 되살리기 | 표본을 아끼는 사전 학습 |
| **EVA** | 비전 트랜스포머를 매개변수 10억 개 이상으로 키우기 | 최고 수준의 시각 표현 |

---

## 7. 학습할 때 살필 점

**잘 통하는 학습 방법:**

```python
# 더 나은 일반화를 위한 이름표 매끄럽게 하기
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 가중치 감쇠를 곁들인 AdamW 최적화기
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1
)

# 예열을 곁들인 코사인 담금질
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

**데이터 불리기:**

- RandAugment 또는 AutoAugment
- Mixup과 CutMix
- 무작위 지우기

---

## 8. 계량 금융에서의 쓰임

비전 트랜스포머 구조는 금융 분석에서도 쓰임을 찾았다.

1. **도표 무늬 알아보기**: 봉 도표와 기술적 무늬 분석
2. **문서 분석**: 재무 보고서와 명세서 처리
3. **위성 영상**: 하늘에서 찍은 그림으로 경제 활동 어림하기
4. **여러 양식의 금융**: 시각 금융 데이터와 글 금융 데이터 아우르기

---

## 연습문제

**연습문제 1.**
비전 트랜스포머가 그림을 조각으로 토큰화하고 트랜스포머로 처리하는 방식을 설명하라.

??? success "연습문제 1 풀이"
    $H \times W$ 그림을 $P \times P$ 조각으로 나누면 조각이 $N = HW/P^2$개이다. 조각마다 $P^2 \cdot C$ 차원으로 편다. 선형 임베딩으로 $d$ 차원에 사영한다. [CLS] 토큰을 앞에 붙인다. 학습된 자리 임베딩을 더한다. 표준 트랜스포머 인코더로 처리한다. [CLS] 출력으로 분류한다.

---

**연습문제 2.**
그림 크기가 224이고 조각 크기가 16인 비전 트랜스포머의 수열 길이를 셈하라.

??? success "연습문제 2 풀이"
    조각이 $N = (224/16)^2 = 14^2 = 196$개에 [CLS] 토큰 1개를 더해 토큰 197개이다. 주의 복잡도는 $O(197^2 \times d)$이다. 자연어 처리와 견주면 토큰 197개짜리 수열은 보통의 글보다 훨씬 짧아 비전 트랜스포머를 셈할 만하다.

---

**연습문제 3.**
비전 트랜스포머가 비슷한 성능을 내는 데 합성곱 신경망보다 데이터가 더 드는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    비전 트랜스포머에는 그림에 대한 귀납 편향이 없다. 국소성도 없고(조각이 1층부터 전역으로 주의한다) 옮김 동변성도 없다(자리 임베딩을 배운다). 합성곱 신경망은 이 사전 지식을 구조에 구워 넣는다. 데이터가 적으면 이 사전 지식이 합성곱 신경망의 일반화를 돕지만, 비전 트랜스포머는 데이터에서 그것을 배워야 하므로 그림 약 1억 장 이상이나 강한 데이터 불리기가 필요하다.

---

**연습문제 4.**
파이토치에서 비전 트랜스포머의 조각 임베딩 층을 구현하라.

??? success "연습문제 4 풀이"
    ```python
    class PatchEmbed(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
    ```

## 정리하며

이 마당은 동기: 왜 트랜스포머를 비전에 쓰는가、구조 개관、PyTorch 구현、모형의 변형을 차례로 짚었다.

**참고 문헌**

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021. (DeiT)
3. He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. (MAE)
4. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
5. Caron, M., et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021. (DINO)
