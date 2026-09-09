# 스윈 트랜스포머

Liu 외(2021)가 내놓은 스윈 트랜스포머(어긋난 창 트랜스포머)는 위계를 이루는 특징 지도와 국소 창 주의를 들여와 비전 트랜스포머의 계산 한계를 푼다. 이 설계는 자기 주의의 힘을 지키면서 그림 크기에 대해 일차 복잡도를 이룬다.

---

## 1. 왜 필요한가

표준 비전 트랜스포머는 빽빽한 예측 과제에 두 가지 큰 한계가 있다.

1. **이차 복잡도**: 모든 조각에 대한 자기 주의는 복잡도가 $O(N^2)$이다
2. **한 가지 크기의 특징**: 해상도 하나에서만 토큰을 낸다

이 한계 때문에 다음이 필요한 물체 탐지나 분할 같은 과제에 비전 트랜스포머를 쓰기 어렵다.

- 여러 크기의 특징 지도
- 해상도가 높은 출력
- 큰 그림의 효율적인 처리

스윈 트랜스포머는 위계를 이루는 구조와 창 주의로 두 문제를 모두 푼다.

---

## 2. 핵심 혁신

### 1. 위계를 이루는 특징 지도

스윈 트랜스포머는 합성곱 신경망처럼 공간 해상도를 차츰 줄이면서 통로 차원을 늘린다.

```
Stage 1: H/4 × W/4 × C
    ↓ (Patch Merging)
Stage 2: H/8 × W/8 × 2C
    ↓ (Patch Merging)
Stage 3: H/16 × W/16 × 4C
    ↓ (Patch Merging)
Stage 4: H/32 × W/32 × 8C
```

### 2. 창 기반 다중 머리 자기 주의 (W-MSA)

전역 주의 대신 국소 창 안에서 주의를 셈한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    창 기반 다중 머리 자기 주의.
    
    국소 창 안에서 주의를 셈하여 복잡도를 O(N²)에서
    O(N × M²)으로 줄인다. 여기서 M은 창 크기이다.
    """
    def __init__(self, dim: int, window_size: int, n_heads: int, 
                 qkv_bias: bool = True, attn_drop: float = 0., 
                 proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # 상대 자리 편향
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, n_heads)
        )
        
        # 상대 자리 색인을 만든다
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.flatten(1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        인수:
            x: (num_windows * B, window_size * window_size, C)
            mask: 어긋난 창을 위한 주의 가림
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 상대 자리 편향을 더한다
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 어긋난 창 주의를 위한 가림을 적용한다
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
```

### 3. 어긋난 창 나누기

창끼리 이어지게 하려고 보통 창 나누기와 어긋난 창 나누기를 번갈아 한다.

```python
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    그림을 겹치지 않는 창으로 나눈다.
    
    인수:
        x: (B, H, W, C)
        window_size: 창 크기
    반환값:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, 
                   H: int, W: int) -> torch.Tensor:
    """
    창 나누기를 되돌린다.
    
    인수:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: 창 크기
        H, W: 본디 높이와 너비
    반환값:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, 
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class ShiftedWindowAttention(nn.Module):
    """효율적인 돌려 옮기기를 쓰는 어긋난 창 주의."""
    
    def __init__(self, dim: int, window_size: int, shift_size: int, 
                 n_heads: int, input_resolution: tuple):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        
        self.attn = WindowAttention(dim, window_size, n_heads)
        
        # 어긋난 창을 위한 주의 가림을 만든다
        if shift_size > 0:
            H, W = input_resolution
            mask = self._create_mask(H, W)
            self.register_buffer("attn_mask", mask)
        else:
            self.attn_mask = None
            
    def _create_mask(self, H: int, W: int) -> torch.Tensor:
        """어긋난 창 주의를 위한 주의 가림을 만든다."""
        img_mask = torch.zeros((1, H, W, 1))
        
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        # 돌려 옮기기
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                                   dims=(1, 2))
        else:
            shifted_x = x
            
        # 창으로 나눈다
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # 창 주의
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # 창을 합친다
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # 돌려 옮긴 것을 되돌린다
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), 
                          dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        return x
```

---

## 3. 온전한 스윈 트랜스포머 블록

```python
class SwinTransformerBlock(nn.Module):
    """
    W-MSA와 SW-MSA를 갖춘 스윈 트랜스포머 블록.
    
    보통 창 주의(W-MSA)와 어긋난 창 주의(SW-MSA)를
    번갈아 한다.
    """
    def __init__(self, dim: int, input_resolution: tuple, n_heads: int,
                 window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0., 
                 drop_path: float = 0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(
            dim, window_size, shift_size, n_heads, input_resolution
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 창 주의
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 다층 퍼셉트론
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

---

## 4. 조각 합치기

공간 해상도를 줄이고 통로를 늘린다.

```python
class PatchMerging(nn.Module):
    """
    위계를 이루는 특징 지도를 위한 조각 합치기 층.
    
    공간 해상도를 절반으로 줄이고 통로를 두 배로 늘린다.
    합성곱 신경망의 걸음 있는 합성곱과 비슷하다.
    """
    def __init__(self, input_resolution: tuple, dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        # 한 칸씩 건너뛰며 행과 열을 고른다
        x0 = x[:, 0::2, 0::2, :]  # 왼쪽 위
        x1 = x[:, 1::2, 0::2, :]  # 왼쪽 아래
        x2 = x[:, 0::2, 1::2, :]  # 오른쪽 위
        x3 = x[:, 1::2, 1::2, :]  # 오른쪽 아래
        
        # 통로 차원을 따라 이어 붙인다
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2 * W/2, 2C)
        
        return x
```

---

## 5. 온전한 스윈 트랜스포머

```python
class SwinTransformer(nn.Module):
    """
    스윈 트랜스포머: 어긋난 창을 쓰는 위계 비전 트랜스포머.
    
    주요 기능:
    1. 위계를 이루는 특징 지도(합성곱 신경망처럼)
    2. 창 주의로 얻는 일차 복잡도 O(N)
    3. 어긋난 창으로 얻는 창끼리의 연결
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 96,
                 depths: tuple = (2, 2, 6, 2),
                 n_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7,
                 mlp_ratio: float = 4.,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.n_layers = len(depths)
        self.embed_dim = embed_dim
        self.n_features = int(embed_dim * 2 ** (self.n_layers - 1))
        
        # 조각 임베딩 (비전 트랜스포머보다 작은 조각)
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )
        patches_resolution = self.patch_embed.patches_resolution
        
        self.pos_drop = nn.Dropout(drop_rate)
        
        # 확률적 깊이 감쇠 규칙
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 단계를 세운다
        self.layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                n_heads=n_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if i_layer < self.n_layers - 1 else None
            )
            self.layers.append(layer)
            
        self.norm = nn.LayerNorm(self.n_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.n_features, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.flatten(1)
        x = self.head(x)
        
        return x
```

---

## 6. 계산 복잡도

### 표준 비전 트랜스포머 (전역 주의)

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

### 스윈 트랜스포머 (창 주의)

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

여기서 $M$은 창 크기이다(대개 7). 조각이 56×56인 224×224 그림에서는 다음과 같다.

- 비전 트랜스포머: $O(56^2 \times 56^2) = O(980만)$
- 스윈: $O(56^2 \times 7^2) = O(15만 3천)$

**주의 복잡도가 64배 줄어든다!**

---

## 7. 모형의 변형

| 모형 | 매개변수 | C | 깊이 | 머리 | ImageNet Top-1 |
|-------|--------|---|--------|-------|----------------|
| Swin-T | 29M | 96 | (2,2,6,2) | (3,6,12,24) | 81.3% |
| Swin-S | 50M | 96 | (2,2,18,2) | (3,6,12,24) | 83.0% |
| Swin-B | 88M | 128 | (2,2,18,2) | (4,8,16,32) | 83.5% |
| Swin-L | 197M | 192 | (2,2,18,2) | (6,12,24,48) | 87.3%* |

*ImageNet-22K로 사전 학습

---

## 8. 응용

스윈 트랜스포머는 빽빽한 예측 과제에서 뛰어나다.

### 물체 탐지
```python
# 탐지를 위한 스윈 등뼈
backbone = SwinTransformer(
    img_size=800,  # 더 큰 그림
    patch_size=4,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    n_heads=(4, 8, 16, 32)
)
# 출력: 1/4, 1/8, 1/16, 1/32의 여러 크기 특징
```

### 의미 분할
```python
# Swin-UNet 방식 구조
class SwinUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = SwinTransformer(...)
        self.decoder = UNetDecoder(...)
```

---

## 9. 스윈 V2의 개선

스윈 트랜스포머 V2는 다음을 들여온다.

1. **잔차 뒤 정규화**: 학습이 더 안정된다
2. **크기를 조정한 코사인 주의**: 서로 다른 해상도를 더 잘 다룬다
3. **로그 간격의 이어진 상대 자리 편향**: 더 잘 일반화된다

---

## 10. 금융에서의 쓰임

스윈 트랜스포머는 효율이 좋아 다음에 알맞다.

- **문서 분석**: 해상도가 높은 금융 문서 처리
- **위성 영상**: 하늘에서 찍은 그림으로 경제 활동 살피기
- **여러 크기의 도표**: 여러 해상도의 금융 도표 분석
- **영상 분석**: 금융 뉴스와 회견 영상 분석

---

## 연습문제

**연습문제 1.**
스윈 트랜스포머의 어긋난 창 얼개를 설명하고 그것이 왜 일차 복잡도를 이루게 하는지 밝혀라.

??? success "연습문제 1 풀이"
    스윈은 겹치지 않는 $M \times M$ 크기의 국소 창 안에서 주의를 셈하여 창마다 $O(M^2)$, 전체로는 (전역 주의의 $O(N^2)$에 견주어) $O(N)$이 든다. 창끼리 주고받게 하려고 층을 번갈아 가며 창을 $M/2$ 화소만큼 어긋나게 한다. 그러면 두 층에 걸쳐 모든 창이 이어진다.

---

**연습문제 2.**
계산 효율과 여러 크기의 특징 면에서 스윈 트랜스포머와 비전 트랜스포머를 견주어라.

??? success "연습문제 2 풀이"
    비전 트랜스포머는 전역 주의($O(N^2)$)에 한 가지 크기의 특징이며 붙박이 위계가 없다. 스윈은 국소 주의($O(N)$)에 (합성곱 신경망의 줄이기와 같은) 조각 합치기로 위계를 이루는 특징을 낸다. 스윈은 해상도가 높은 그림에 더 효율적이고 탐지와 분할에 쓸모 있는 여러 크기의 특징을 저절로 낸다.

---

**연습문제 3.**
스윈 트랜스포머의 창 나누기 연산을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    ```

---

**연습문제 4.**
스윈의 조각 합치기란 무엇이며 어떻게 위계를 이루는 표현을 만드는가?

??? success "연습문제 4 풀이"
    조각 합치기는 이웃한 2×2 조각의 특징을 이어 붙이고 선형 층으로 2C 차원에 사영하여, 공간 해상도를 반으로 줄이면서 통로 차원을 두 배로 만든다. 합성곱 신경망의 풀링이나 걸음을 흉내 내어 특징 피라미드를 만든다. 1단계는 H/4, 2단계는 H/8, 3단계는 H/16, 4단계는 H/32이다.

## 정리하며

스윈 트랜스포머는 다음을 훌륭히 아우른다.

- **트랜스포머의 모형화 힘**: 넉넉한 표현을 위한 자기 주의
- **합성곱 신경망의 효율**: 일차 복잡도의 위계를 이루는 특징
- **자유로움**: 분류와 빽빽한 예측 모두에 알맞다

그래서 요즘 컴퓨터 비전 체계의 두루 쓰이는 등뼈가 된다.

**참고 문헌**

1. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
2. Liu, Z., et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." CVPR 2022.
