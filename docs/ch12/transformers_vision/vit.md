# 비전 트랜스포머 (ViT)

ViT는 이미지를 패치 토큰으로 투영하고 트랜스포머로 분류하는 구조다. 원 연구는 대규모 사전 학습의 효과를 보여 주었다. 패치·위치 표현·전역 어텐션을 살펴보며 CNN과의 차이를 보편적인 성능 우열이나 고정 데이터 임계로 해석하지 않는다.

## 1. 동기: 왜 트랜스포머를 비전에 쓰는가

CNN은 국소 커널과 계층적 처리를 사용한다. 깊이·팽창·풀링으로 먼 정보를 결합할 수 있지만 연결 범위의 설계가 필요하다. ViT는 패치 단위 전역 결합을 다른 방식으로 구성한다.

**컨볼루션 뉴럴 넷트웍의 제약:**

- 작은 커널 한 레이어의 직접 연결은 국소적이다.
- 먼 정보를 합치려면 깊이·다운샘플·팽창 등을 설계한다.
- 위치 공유·국소성은 유용한 귀납 바이어스며 모든 데이터에서 단점인 것은 아니다.

**트랜스포머의 이점:**

- 첫 전역 어텐션에서 모든 패치 사이의 직접 연결을 허용한다.
- 투영·내용·위치에 따른 웨이트로 정보를 결합한다.
- 모델·데이터·계산을 늘리는 방식은 실제 품질·비용으로 평가한다.

비전 트랜스포머의 핵심 통찰은 이미지를 패치으로 "토큰화"할 수 있고, 그러면 자연어 처리를 뒤바꾼 트랜스포머 구조에 그대로 맞아 들어간다는 것이다.

---

## 2. 구조 개관

비전 트랜스포머 구조는 주요 부품 넷으로 이루어진다.

```
Input Image → Patch Embedding → Transformer Encoder → Classification Head → Output
     ↓              ↓                    ↓                    ↓
  (H,W,C)    (N patches, D)      (N+1 tokens, D)         (classes)
```

### 수식으로 나타내기

입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$이 주어지면 비전 트랜스포머의 파이프라인은 다음과 같이 나아간다.

**1단계: 패치 샘플링**
이미지를 겹치지 않는 $P \times P$ 크기의 패치 $N = \frac{HW}{P^2}$개로 나눈다.

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, \ldots, N$$

$P = 16$인 표준 $224 \times 224$ 이미지에서는 패치이 $N = \frac{224 \times 224}{16 \times 16} = 196$개이고 패치마다 차원이 $P^2 \cdot C = 16^2 \cdot 3 = 768$이다.

**2단계: 선형 투영**
펼친 패치를 D차원으로 투영한다. 여기서는 위치 벡터를 아직 더하지 않고 다음 단계에서 CLS와 패치를 연결한 뒤 한 번만 더한다. 실제 선형 레이어는 바이어스를 포함할 수도 있다.

$$\mathbf{z}_0^i=\mathbf{x}_p^i\mathbf{E}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$

실제로는 이 투영을 커널 크기 $P$, 스트라이드 $P$의 2차원 컨볼루션으로 구현한다.

```python
# 조각을 펴고 선형 사영을 하는 것과 같다
self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
```

**3단계: 분류 토큰 앞에 붙이기**
학습되는 분류 토큰을 앞에 붙인다.

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \mathbf{z}_0^2; \ldots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

`[CLS]` 토큰은 셀프 어텐션으로 모든 패치의 정보를 모으며 마지막 분류 판단에 쓰인다.

**4단계: 트랜스포머 인코딩**
트랜스포머 인코더 레이어 $L$개를 적용한다.

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

요즘 트랜스포머 방식과 같은 **Pre-LN** 설계(하위 레이어마다 그 앞에 LayerNorm)임에 유의하라.

**5단계: 분류**
분류 토큰으로 예측한다.

$$\mathbf{y} = \text{MLP}_{head}(\text{LN}(\mathbf{z}_L^0))$$

### 위치 임베딩

비전 트랜스포머는 위치 인덱스을 벡터로 잇대는 간단한 찾기 표인 **학습된 1차원 위치 임베딩**을 쓴다.

$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

학습 위치 표는 행 우선으로 펼친 2차원 격자의 각 위치를 나타낼 수 있다. 학습 후 가까운 위치의 벡터가 유사한 무늬를 보이는지 분석할 수 있지만 모든 학습에서 격자 구조가 자동 복원된다고 보장하지 않는다.

행·열 표현을 분리하는 2차원 대안도 있다. 계수·결합 제약과 데이터에 따라 효과가 달라지므로 “약 0.5% 개선” 같은 출처·설정 없는 고정 차이를 일반화하지 않는다.

---

## 3. PyTorch 구현

아래는 VisionTransformer를 구성하는 패치이며 PatchEmbedding·TransformerBlock 정의가 이 코드 블록에 없다. 이 폴더의 vit_model 예제처럼 일치하는 생성자·출력 규약의 클래스를 함께 제공해야 한다. 팩터리 함수는 새 무작위 모델을 만들며 사전 학습 웨이트를 불러오지 않는다.

실제 패치 수가 생성자의 위치 표와 같아야 하며 입력 해상도 자동 보간·패딩 마스크는 없다. 반환값은 CLS의 클래스별 로짓이다. 코드 생성만으로 학습·성능 비교가 수행되는 것은 아니다.

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    """
    이미지 분류를 위한 비전 트랜스포머.
    
    핵심 혁신:
    1. 그림을 조각의 수열로 다룬다
    2. 이미지 분류에 트랜스포머 인코더를 쓴다
    3. 컨볼루션 뉴럴 넷트웍 방식의 입력 처리와 트랜스포머 구조를 잇는다
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
        
        # 조각 임베딩 레이어
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
        
        # 웨이트 초기화
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

## 4. 모델의 변형

비전 트랜스포머에는 표준 설정이 여럿 있다.

| 모델 | 파라메터 | 임베딩 차원 | 깊이 | 헤드 | 패치 크기 |
|-------|-----------|-----------|-------|-------|------------|
| ViT-Tiny | 5M | 192 | 12 | 3 | 16 |
| ViT-Small | 22M | 384 | 12 | 6 | 16 |
| ViT-Base | 86M | 768 | 12 | 12 | 16 |
| ViT-Large | 307M | 1024 | 24 | 16 | 16 |
| ViT-Huge | 632M | 1280 | 32 | 16 | 14 |

### 모델 생성 함수

```python
def create_vit_base(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Base: 파라메터 8600만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768,
        depth=12, n_heads=12, n_classes=n_classes
    )

def create_vit_large(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Large: 파라메터 3억 700만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024,
        depth=24, n_heads=16, n_classes=n_classes
    )
```

---

## 5. 핵심 통찰

### 1. 전역 리셉티브 필드
리셉티브 필드가 차츰 넓어지는 컨볼루션 뉴럴 넷트웍과 달리 비전 트랜스포머는 셀프 어텐션으로 첫 레이어부터 전역 리셉티브 필드를 가진다. 그래서 컨볼루션 뉴럴 넷트웍이라면 레이어를 많이 쌓아야 얻을 먼 거리의 공간 의존을 앞쪽 레이어에서 잡아낼 수 있다.

### 2. 데이터가 얼마나 드는가
데이터 요구량은 사전 학습·어그멘테이션·증류·모델 크기와 과제에 따라 달라진다. ViT에도 패치 분할·공유 투영·위치 표현 같은 귀납 바이어스가 있다. 이미지 1000만·1억 장을 CNN·ViT 우열의 보편적인 경계로 제시할 수 없다.

### 3. 계산 복잡도
전역 어텐션의 점수·가중합은 O(N²D), 투영은 O(ND²)이며 FFN 비용도 더해진다. 패치 수 196개에 CLS를 포함하면 점수는 197×197이다. 해상도·패치 크기·피처·레이어 수와 커널에 따라 실제 비용을 판단한다.

### 4. 전이 학습
사전 학습 ViT를 새 과제에 미세 조정할 수 있다. 해상도가 바뀌면 학습 위치 표의 패치 부분을 실제 2차원 격자에 맞춰 보간하는 방법이 있다. CLS를 분리하고 전처리·패치 수와 갱신할 계수를 함께 맞추며 전이 후 품질을 평가한다.

### 5. 섞은 구조
실용적인 절충은 컨볼루션 뉴럴 넷트웍 백본(이를테면 ResNet)로 피처 지도를 뽑은 뒤 그 피처를 "패치"으로 트랜스포머에 넣는 것이다.

$$\text{Image} \xrightarrow{\text{CNN}} \text{Feature Map} \xrightarrow{\text{Flatten}} \text{Patch Tokens} \xrightarrow{\text{Transformer}} \text{Output}$$

혼합 모델은 CNN 피처와 어텐션을 연결한다. 국소·전역 경로를 조합할 수 있지만 데이터 효율·속도·정확도의 개선은 같은 학습·평가 조건에서 확인해야 한다.

### 6. 자기 지도 사전 학습

**MAE(마스크 오토인코더)**: 원 연구의 대표 설정은 패치 75%를 숨기고 가벼운 디코더로 빠진 픽셀을 복원한다. BERT와 마스킹 아이디어를 공유하지만 어휘 분류가 아닌 픽셀 복원이며 인코더가 보이는 패치만 처리하는 등 계산도 다르다.

**DINO**: 학생이 EMA로 갱신하는 교사의 표현을 다른 이미지 뷰에서 맞추는 자기 지도 학습 방식이다. 일부 학습 피처·어텐션에서 물체 구조가 나타나는 결과가 보고되었지만 정확한 분할 경계를 항상 자동으로 얻는다는 보장은 아니다.

---

## 6. 비전 트랜스포머의 후예

| 변형·학습법 | 대표 설계 | 구분할 점 |
|---|---|---|
| DeiT | 학습 설정과 일부 변형의 증류 토큰 | 모든 변형이 교사를 쓰는 것은 아님 |
| Swin | 윈도·시프트와 계층 병합 | 고정 윈도에서 길이에 선형 가능 |
| BEiT | 시각 토큰 예측 | 토큰화·복원 목표를 확인 |
| MAE | 높은 비율의 패치 마스킹·픽셀 복원 | 보이는 패치 인코더와 복원 디코더 구분 |
| 대규모 ViT 계열 | 모델·데이터·학습 확장 | 특정 이름을 최고 성능 보장으로 해석하지 않음 |

---

## 7. 학습할 때 살필 점

**학습 구성 예시:**

아래 패치은 외부 model·epochs와 CosineAnnealingLR import를 가정한다. 스무딩·러닝레이트·감쇠·스케줄은 검증셋에서 선택하며 이 값들이 모든 모델의 최적 설정은 아니다.

```python
# 더 나은 일반화를 위한 이름표 매끄럽게 하기
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 웨이트 감쇠를 곁들인 AdamW 최적화기
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1
)

# 예열을 곁들인 코사인 담금질
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

**데이터 어그멘테이션:**

- RandAugment 또는 AutoAugment
- Mixup과 CutMix
- 무작위 지우기

---

## 8. 계량 금융에서의 쓰임

차트·문서·위성 영상처럼 이미지로 표현된 자료에 ViT를 적용하는 모델링 가설을 세울 수 있다. 예측 목표와 시점별 사용 가능한 정보를 정의하고 시간 분할·중복을 관리해야 한다.

차트 이미지를 만든다는 사실만으로 금융 예측력이 생기는 것은 아니다. 이 페이지는 실제 금융 데이터를 학습·검증하지 않으므로 수익성이나 분석 정확도를 주장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
ViT의 패치 투영과 위치 정보 적용 순서를 설명하라.

</div>

??? success "풀이"
    양 축이 P로 나누어떨어지는 H×W 이미지를 N=HW/P²개 패치로 나누어 각 CP² 값을 D차원에 투영한다. CLS를 붙인 N+1개 토큰에 위치 표를 한 번 더하고 인코더·최종 정규화·분류기를 적용한다. 패치 수식과 연결 수식에서 위치 벡터를 중복해 더하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
224×224, P=16에서 토큰 수와 점수 행렬 크기를 구하라.

</div>

??? success "풀이"
    패치 196개와 CLS 하나로 197토큰이다. 헤드·배치당 점수 행렬은 197×197이며 투영·FFN·웨이트 비용은 별도다. 이 수만으로 모든 장치에서 저렴하다거나 자연어 입력보다 짧다고 단정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ViT의 귀납 바이어스와 학습 데이터 요구량을 어떻게 해석해야 하는가?

</div>

??? success "풀이"
    전역 결합은 CNN보다 국소 연결 제약이 약할 수 있지만 패치 내부 투영·공유 계수·위치 표현 등의 가정은 남는다. 사전 학습·어그멘테이션·증류·과제에 따라 데이터 효율이 달라져 정해진 이미지 수만으로 성능 우위를 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
파이토치에서 비전 트랜스포머의 패치 임베딩 레이어를 구현하라.

</div>

??? success "풀이"
    ```python
    class PatchEmbed(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
    ```

## 정리하며

패치·CLS·위치 표와 인코더를 연결해 이미지 분류를 구성한다. 고정 위치 표와 외부 클래스의 규약, 원래 구조와 새 모델 초기화를 구분한다. 학습법·변형의 이점은 데이터·계산·평가 조건을 함께 보아야 한다.

**참고 문헌**

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021. (DeiT)
3. He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. (MAE)
4. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
5. Caron, M., et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021. (DINO)
