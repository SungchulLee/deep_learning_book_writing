# 그림을 위한 자리 임베딩

트랜스포머의 자기 주의는 순서를 바꾸어도 그대로여서 입력 토큰의 차례와 상관없이 같은 출력을 낸다. 이 성질이 반가울 때도 있지만 그림에서는 공간 정보를 지켜야 한다. 자리 임베딩은 조각마다 본디 그림의 어디에 있었는지를 담아 모형이 공간 관계를 이해하게 한다.

---

## 1. 자리 정보가 왜 중요한가

두 상황을 생각해 보자.

1. 왼쪽 위 구석에 고양이가 있고 오른쪽 아래에 개가 있다
2. 왼쪽 위 구석에 개가 있고 오른쪽 아래에 고양이가 있다

자리 임베딩이 없으면 자기 주의는 토큰의 자리가 아니라 토큰끼리의 관계만 따지므로 이 둘을 똑같이 다룬다. 자리 임베딩이 이 대칭을 깬다.

---

## 2. 수학적 틀

### 학습되는 자리 임베딩

표준 비전 트랜스포머는 학습되는 자리 임베딩을 쓴다.

$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

여기서 $N$은 조각의 개수이고 $D$은 임베딩 차원이다. "+1"은 CLS 토큰 몫이다.

이 임베딩을 조각 임베딩에 더한다.

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \ldots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

### 사인파 자리 임베딩

그 대신 (본디 트랜스포머처럼) 고정된 사인파 임베딩을 쓸 수도 있다.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$

2차원 그림에서는 다음으로 넓어진다.

$$PE_{(x, y)} = [PE_x; PE_y]$$

---

## 3. 구현

### 학습되는 1차원 자리 임베딩

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

### 2차원 사인파 자리 임베딩

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

### 상대 자리 임베딩

```python
class RelativePositionBias(nn.Module):
    """
    스윈 트랜스포머에서 쓰는 상대 자리 편향.
    
    절대 자리 대신 조각 자리 사이의 상대 거리를
    담는다.
    """
    def __init__(self, window_size: int, n_heads: int):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        
        # 상대 자리 편향 표
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
        """주의를 위한 상대 자리 편향을 돌려준다."""
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

## 4. 자리 임베딩 사이 채우기

어떤 그림 크기로 학습한 모형을 다른 크기에 쓸 때는 자리 임베딩의 사이를 채워야 한다.

```python
def interpolate_pos_embed(model, new_size: int, old_size: int = 224):
    """
    다른 그림 크기를 위해 자리 임베딩의 사이를 채운다.
    
    인수:
        model: pos_embed 매개변수를 가진 비전 트랜스포머 모형
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

## 5. 눈으로 보기와 분석

### 자리 임베딩 눈으로 보기

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

## 6. 여러 방식 견주기

| 방법 | 학습되는가 | 바깥으로 뻗기 | 좋은 점 | 나쁜 점 |
|--------|-----------|---------------|------------|---------------|
| 학습형 1차원 | 그렇다 | 나쁨 | 간단하고 잘 통한다 | 수열 길이가 고정된다 |
| 사인파 1차원 | 아니다 | 좋음 | 바깥으로 잘 뻗는다 | 최선이 아닐 수 있다 |
| 학습형 2차원 | 그렇다 | 보통 | 2차원 짜임을 지킨다 | 매개변수가 는다 |
| 사인파 2차원 | 아니다 | 좋음 | 그림에 자연스럽다 | 무늬가 고정된다 |
| 상대 | 그렇다 | 아주 좋음 | 옮김에 불변이다 | 더 까다롭다 |
| RoPE | 일부 | 아주 좋음 | 바깥으로 잘 뻗는다 | 구현이 까다롭다 |

---

## 7. 실험으로 밝혀진 것

연구로 비전 트랜스포머의 자리 임베딩에 대해 몇 가지가 밝혀졌다.

1. **학습형과 고정형**: 학습되는 임베딩이 고정된 사인파보다 조금 낫다
2. **1차원과 2차원**: 2차원을 고려한 임베딩은 아주 조금 나아진다
3. **공간 짜임**: 학습되는 임베딩이 저절로 2차원 공간 감각을 갖추어 간다
4. **사이 채우기**: 서로 다른 해상도에는 쌍삼차 사이 채우기가 잘 통한다

---

## 8. 모범 사례

1. **표준 선택**: 알맞게 초기화한 학습형 1차원 자리 임베딩을 쓴다
2. **여러 해상도**: 자유로움을 위해 자리 임베딩 사이 채우기를 구현한다
3. **초기화**: 표준편차 0.02의 자른 정규 분포를 쓴다
4. **드롭아웃**: 자리 임베딩을 더한 뒤 드롭아웃을 적용한다

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

**연습문제 1.**
비전 트랜스포머의 절대, 상대, 회전 자리 임베딩을 견주어라.

??? success "연습문제 1 풀이"
    절대(비전 트랜스포머)는 자리마다 학습된 임베딩을 두고 학습이 끝나면 고정된다. 상대(Swin)는 질의와 열쇠 사이의 상대 자리에 따른 학습된 편향을 쓴다. 회전(RoPE)은 질의와 열쇠 벡터를 자리에 비례하는 각만큼 돌려 내적에 상대 자리를 담는다. 상대와 회전이 본 적 없는 자리로 더 잘 일반화된다.

---

**연습문제 2.**
그림 조각에 2차원 자리 정보를 어떻게 담는지 설명하라.

??? success "연습문제 2 풀이"
    고를 수 있는 길은 이렇다. (1) 1차원으로 펴서 표준 자리 임베딩을 쓴다(비전 트랜스포머의 기본). (2) 행과 열 임베딩을 따로 두고 더한다. PE$(i,j) = $ PE$_{\text{row}}(i) +$ PE$_{\text{col}}(j)$. (3) 학습되는 2차원 격자 임베딩을 쓴다. (2)번이 매개변수를 아끼면서 공간 짜임을 잡아낸다.

---

**연습문제 3.**
자리 임베딩이 비전 트랜스포머가 서로 다른 그림 해상도를 다루는 능력을 제한할 수 있는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    학습된 절대 임베딩은 자리 수가 고정되어 있다(이를테면 224/16이면 196개). 해상도가 달라지면 조각의 수가 바뀐다. 해결책은 자리 임베딩을 새 격자 크기에 맞추어 쌍선형으로 사이를 채우는 것이다. 실제로 잘 통하지만 정확하지는 않다.

---

**연습문제 4.**
사전 학습된 비전 트랜스포머를 새 해상도에 맞추도록 2차원 위치 임베딩 사이 채우기를 구현하라.

??? success "연습문제 4 풀이"
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

자리 임베딩은 비전 트랜스포머가 공간 관계를 이해하는 데 꼭 필요하다. 여러 방식이 있지만 간단하고 잘 통하므로 학습형 1차원 자리 임베딩이 여전히 표준 선택이다. 자리 임베딩을 이해하는 것은 비전 트랜스포머를 다른 그림 크기에 맞추고 모형이 그림의 짜임에 대해 무엇을 배웠는지 읽어 내는 데 매우 중요하다.
