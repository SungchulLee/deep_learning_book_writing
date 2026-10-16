# 이미지을 위한 위치 임베딩

위치 정보가 없는 셀프 어텐션은 패치 순열에 따라 출력도 함께 재배열되는 순열 동변성을 갖는다. 토큰 출력이 순서와 무관하게 그대로인 것은 아니다. 위치 임베딩·상대 바이어스는 이미지의 공간 관계를 모델에 반영하는 방법이다.

## 1. 위치 정보가 왜 중요한가

두 상황을 생각해 보자.

1. 왼쪽 위 구석에 고양이가 있고 오른쪽 아래에 개가 있다
2. 왼쪽 위 구석에 개가 있고 오른쪽 아래에 고양이가 있다

위치 정보가 없는 전역 셀프 어텐션 뒤에 순열 불변 풀링을 쓰면 같은 패치 집합의 순서를 구분하지 못할 수 있다. 학습 위치 표는 패치 순서와 함께 더해 이 대칭을 바꾼다. 패치 자체의 내부 픽셀 순서나 경계 조건까지 모두 사라지는 것은 아니다.

---

## 2. 수학적 틀

### 학습되는 위치 임베딩

표준 비전 트랜스포머는 학습되는 위치 임베딩을 쓴다.

$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

여기서 $N$은 패치의 개수이고 $D$은 임베딩 차원이다. "+1"은 CLS 토큰 몫이다.

이 임베딩을 패치 임베딩에 더한다.

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \ldots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

### 사인·코사인 위치 임베딩

그 대신 (원래 트랜스포머처럼) 고정된 사인·코사인 임베딩을 쓸 수도 있다.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$

2차원 사인·코사인 표현의 한 방식은 피처를 두 부분으로 나누어 행·열 좌표 표현을 이어 붙이는 것이다. 두 표현의 차원 합을 최종 D에 맞춘다. 아래 구현은 각 좌표에 사인·코사인 쌍을 사용하므로 D가 4의 배수여야 한다.

$$PE_{(x, y)} = [PE_x; PE_y]$$

---

## 3. 구현

### 학습되는 1차원 위치 임베딩

표는 (1,N+1,D)이며 include_cls=False이면 N개다. 이 클래스는 CLS 토큰을 생성하지 않고 이미 구성된 같은 길이의 토큰 표현에 위치 표를 더한다. 토큰 수·피처 차원을 입력과 맞춰야 한다.

```python
import torch
import torch.nn as nn

class LearnablePositionEmbedding(nn.Module):
    """
    비전 트랜스포머에서 쓰는 표준 학습형 자리 임베딩.
    
    자리마다 고유한 학습 벡터를 얻어 그 자리의 조각 임베딩에
    더해진다.
    """
    def __init__(self, n_patches: int, embed_dim: int, include_cls: bool = True):
        super().__init__()
        n_positions = n_patches + 1 if include_cls else n_patches
        
        # 학습되는 자리 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, embed_dim))
        
        # 자른 정규 분포로 초기화한다
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력에 자리 임베딩을 더한다."""
        return x + self.pos_embed
```

### 2차원 사인·코사인 위치 임베딩

고정 격자 H×W를 행 우선으로 펼친 (1,H×W,D) 버퍼다. CLS 행은 없고 입력이 짧으면 앞부분을 자를 뿐 새로운 2차원 격자에 맞춰 재계산하지 않는다. 다른 가로·세로 비율에서 단순 슬라이스를 공간 좌표 대응으로 해석하면 안 된다.

```python
import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbedding2D(nn.Module):
    """
    2차원 사인파 자리 임베딩.
    
    x 좌표와 y 좌표의 임베딩을 따로 만들어 이어 붙여
    1차원 사인파 방식을 2차원으로 넓힌다.
    """
    def __init__(self, embed_dim: int, height: int, width: int, 
                 temperature: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        
        self.embed_dim = embed_dim
        pe = self._make_2d_sincos(height, width, embed_dim, temperature)
        self.register_buffer('pe', pe)
        
    def _make_2d_sincos(self, h: int, w: int, d: int, temp: float) -> torch.Tensor:
        """2차원 사인파 자리 임베딩을 만든다."""
        # 좌표 격자를 만든다
        y_pos = torch.arange(h).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w).unsqueeze(0).repeat(h, 1)
        
        # 차원 색인
        dim_t = torch.arange(d // 4)
        omega = 1.0 / (temp ** (dim_t / (d // 4)))
        
        # 임베딩을 셈한다
        # 좌표마다 sin 값 d/4개와 cos 값 d/4개를 보탠다
        y_embed = y_pos.flatten().unsqueeze(1) * omega.unsqueeze(0)
        x_embed = x_pos.flatten().unsqueeze(1) * omega.unsqueeze(0)
        
        pe = torch.cat([
            torch.sin(y_embed),
            torch.cos(y_embed),
            torch.sin(x_embed),
            torch.cos(x_embed)
        ], dim=1)
        
        return pe.unsqueeze(0)  # (1, h*w, d)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력에 자리 임베딩을 더한다."""
        return x + self.pe[:, :x.size(1)]
```

### 상대 위치 임베딩

RelativePositionBias는 한 변 M인 정사각 윈도의 오프셋마다 헤드별 바이어스를 학습한다. 표는 (2M−1)²×H개 파라메터, 반환값은 (1,H,M²,M²)다. 토큰 표현에 더하는 벡터가 아니라 어텐션 점수에 더하는 바이어스다.

```python
class RelativePositionBias(nn.Module):
    """
    스윈 트랜스포머에서 쓰는 상대 자리 바이어스.
    
    절대 자리 대신 조각 자리 사이의 상대 거리를
    담는다.
    """
    def __init__(self, window_size: int, n_heads: int):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        
        # 상대 자리 바이어스 표
        # 차원마다 (2*window-1)가지의 상대 자리가 있다
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # 쌍마다의 상대 자리 색인을 셈한다
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self) -> torch.Tensor:
        """주의를 위한 상대 자리 바이어스를 돌려준다."""
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size ** 2,
            self.window_size ** 2,
            -1
        )
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, N, N)
```

---

## 4. 위치 임베딩 보간

학습한 절대 위치 표를 새 패치 격자에 맞추는 방법으로 2차원 보간을 사용할 수 있다. 이는 가능한 방법 중 하나이며 모든 위치 방식에 필요한 것은 아니다.

아래 함수는 CLS 하나와 정사각 격자, 정수 patch_size 속성을 가정한다. old_size 메타데이터와 실제 표 길이가 맞아야 한다. 반환 텐서를 만드는 것만으로 model.pos_embed가 바뀌지는 않는다. Parameter를 교체하면 옵티마이저가 참조하는 계수와 모델의 패치 수·입력 검증도 함께 갱신해야 한다.

```python
def interpolate_pos_embed(model, new_size: int, old_size: int = 224):
    """
    다른 그림 크기를 위해 자리 임베딩의 사이를 채운다.
    
    인수:
        model: pos_embed 파라메터를 가진 비전 트랜스포머 모형
        new_size: 새 그림 크기
        old_size: 본디 그림 크기(기본 224)
    """
    pos_embed = model.pos_embed
    n_patches_new = (new_size // model.patch_embed.patch_size) ** 2
    n_patches_old = (old_size // model.patch_embed.patch_size) ** 2
    
    if n_patches_new == n_patches_old:
        return pos_embed
    
    # CLS 토큰과 조각 임베딩을 가른다
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    
    # 2차원 격자로 꼴을 바꾼다
    dim = patch_pos.shape[-1]
    h_old = w_old = int(n_patches_old ** 0.5)
    h_new = w_new = int(n_patches_new ** 0.5)
    
    patch_pos = patch_pos.reshape(1, h_old, w_old, dim).permute(0, 3, 1, 2)
    
    # 보간
    patch_pos = nn.functional.interpolate(
        patch_pos, 
        size=(h_new, w_new),
        mode='bicubic',
        align_corners=False
    )
    
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
    
    return torch.cat([cls_pos, patch_pos], dim=1)
```

---

## 5. 위치 표현 시각화

시각화는 CLS 행 하나를 제외한 위치 표를 가정한다. 코사인 정규화에서 0 노름을 처리하지 않아 영벡터에는 NaN이 생길 수 있다. 공간 시각화는 패치 수가 grid_size²와 같아야 하며 유사도 무늬만으로 인과적 공간 이해나 정확도를 판정할 수 없다.

### 위치 임베딩 눈으로 보기

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_position_embeddings(pos_embed: torch.Tensor, grid_size: int = 14):
    """
    학습된 자리 임베딩을 그려 본다.
    
    보이는 것:
    1. 자리 임베딩 벡터를 열지도로
    2. 자리 사이의 비슷함 행렬
    """
    # CLS 토큰을 없앤다
    pos_embed = pos_embed[0, 1:].detach().cpu()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 임베딩의 열지도
    im1 = axes[0].imshow(pos_embed.T, aspect='auto', cmap='coolwarm')
    axes[0].set_xlabel('Patch Position')
    axes[0].set_ylabel('Embedding Dimension')
    axes[0].set_title('Position Embedding Vectors')
    plt.colorbar(im1, ax=axes[0])
    
    # 비슷함 행렬
    pos_norm = pos_embed / pos_embed.norm(dim=1, keepdim=True)
    similarity = (pos_norm @ pos_norm.T).numpy()
    
    im2 = axes[1].imshow(similarity, cmap='viridis')
    axes[1].set_xlabel('Patch Position')
    axes[1].set_ylabel('Patch Position')
    axes[1].set_title('Position Similarity Matrix')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def visualize_spatial_similarity(pos_embed: torch.Tensor, grid_size: int = 14):
    """
    자리마다 다른 자리와 공간적으로 어떻게 얽히는지 보인다.
    
    기준 자리마다 다른 모든 자리와의 비슷함을 보인다.
    """
    pos_embed = pos_embed[0, 1:].detach().cpu()
    pos_norm = pos_embed / pos_embed.norm(dim=1, keepdim=True)
    similarity = (pos_norm @ pos_norm.T).numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 그려 볼 기준 자리
    refs = [
        (0, 0),                          # 왼쪽 위
        (0, grid_size // 2),             # 위 가운데
        (grid_size // 2, grid_size // 2),# Center
        (grid_size // 2, 0),             # 왼쪽 가운데
        (grid_size - 1, grid_size - 1),  # 오른쪽 아래
        (grid_size // 4, grid_size // 4) # 4분의 1 자리
    ]
    
    for ax, (row, col) in zip(axes.flatten(), refs):
        idx = row * grid_size + col
        sim_map = similarity[idx].reshape(grid_size, grid_size)
        
        im = ax.imshow(sim_map, cmap='viridis')
        ax.scatter([col], [row], c='red', s=100, marker='x')
        ax.set_title(f'Reference: ({row}, {col})')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Position Embedding Similarities')
    plt.tight_layout()
    plt.show()
```

---

## 6. 여러 방식 비교

| 방식 | 추가 위치 계수 | 새 해상도에서 확인할 점 |
|---|---|---|
| 학습 절대 위치 | 위치 수×D | 새 격자와 표의 보간·재학습 |
| 고정 1차원 사인·코사인 | 0 | 저장 길이와 펼침 순서 |
| 행·열 분리 학습 | (H+W)D 등, 결합 규약에 의존 | 행·열 범위와 공유 제약 |
| 고정 2차원 사인·코사인 | 0 | 실제 가로·세로 좌표 재계산 |
| 상대 바이어스 | 오프셋·헤드·버킷 수에 의존 | 거리 범위·윈도·마스크 |
| RoPE 기본형 | 회전 자체는 0 | 2차원 좌표 배치·주파수·캐시 |

상대 오프셋에 의존하는 바이어스라도 모델 전체의 이동 불변성을 보장하지는 않는다. 경계·윈도·패치 분할과 풀링도 영향을 준다. 어느 방식도 학습 해상도 밖의 품질을 자동 보장하지 않는다.

---

## 7. 실험으로 밝혀진 것

원래 ViT 연구는 여러 위치 표현을 비교하고 해상도 변경 시 위치 표 보간을 사용했다. 그 실험을 학습형이 항상 고정형보다 우수하거나 2차원 방식이 반드시 더 좋다는 순위로 일반화하지 않는다. 데이터·해상도·사전 학습·미세 조정 조건을 맞춰 평가해야 한다.

## 8. 구현 시 확인할 점

먼저 패치 펼침 순서와 위치 좌표를 맞춘다. 초기화·드롭아웃·보간은 모델의 학습 설정과 함께 선택하며, 아래 PositionEmbedding도 저장된 위치 수를 넘으면 덧셈 모양이 맞지 않는다. 드롭아웃의 학습·평가 모드도 구분한다.

```python
class PositionEmbedding(nn.Module):
    """좋은 방법을 갖춘 온전한 자리 임베딩 모듈."""
    def __init__(self, n_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # CLS 토큰 몫으로 +1
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, :x.size(1)]
        return self.dropout(x)
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
절대 위치·상대 바이어스·RoPE가 어느 텐서에 어떻게 적용되는지 설명하라.

</div>

??? success "연습문제 1 풀이"
    절대 위치 벡터는 토큰 표현에 더하고, 상대 바이어스는 쿼리·키 위치 차이에 따라 점수에 더한다. RoPE는 Q·K를 좌표별로 회전시켜 내적에 상대 오프셋을 반영한다. 내용 벡터와 경계 조건도 중요하며 길이·해상도 확장의 품질을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
2차원 위치 표현의 격자 표와 행·열 분리 표현의 계수·제약을 비교하라.

</div>

??? success "연습문제 2 풀이"
    H×W 격자에서 독립 D차원 표는 HWD개, D차원 행·열 표를 더하면 (H+W)D개다. 분리형은 계수를 줄이지만 임의 격자 표를 그대로 표현하는 자유도는 없다. 사인·코사인 연결형은 행·열 차원의 합을 D로 맞춰 좌표별로 계산한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
해상도 변경 시 위치 표와 다른 모델 요소를 어떻게 확인해야 하는가?

</div>

??? success "연습문제 3 풀이"
    새 패치 격자 수에 맞게 표를 보간하거나 재구성하고 CLS 등 특수 위치를 분리한다. 직사각형은 가로·세로를 따로 보존한다. 패치 크기·입력 검사·옵티마이저 참조와 미세 조정도 확인하며, 보간은 의미 보존이나 정확도를 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 위치 표 보간 코드 조각의 격자·특수 토큰 가정과 가져오기·모델 연결 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"
    CLS 하나와 정사각형의 기존 패치 격자를 가정하므로 N이 완전제곱인지 확인한다. F를 가져오지 않아 torch.nn.functional import가 별도로 필요하다. 앞의 같은 이름 함수와 인수 형식도 다르므로 같은 네임스페이스에서는 덮어쓴다. 반환값을 위치 Parameter로 적용하고 옵티마이저·입력 메타데이터를 맞추는 단계는 포함하지 않는다.

    ```python
    def interpolate_pos_embed(pos_embed, new_size):
        N = pos_embed.shape[1] - 1  # CLS 제외
        old_size = int(N**0.5)
        pos = pos_embed[:, 1:].reshape(1, old_size, old_size, -1).permute(0,3,1,2)
        pos = F.interpolate(pos, size=new_size, mode='bicubic')
        pos = pos.permute(0,2,3,1).reshape(1, -1, pos.shape[1])
        return torch.cat([pos_embed[:, :1], pos], dim=1)
    ```

## 정리하며

공간 위치를 넣는 위치·규약과 패치 격자 모양을 일치시킨다. 고정 수식 계산·표 보간·상대 바이어스의 거리 범위는 서로 다른 문제이며, 새로운 해상도에서의 품질과 모델 전체의 이동 성질은 별도로 확인한다.
