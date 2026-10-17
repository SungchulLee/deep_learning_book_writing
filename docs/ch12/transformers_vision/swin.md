# 스윈 트랜스포머

Swin Transformer는 국소 윈도 어텐션, 시프트 윈도와 단계별 패치 병합을 결합한다. 고정된 윈도·피처 폭에서는 토큰 수에 선형인 어텐션을 구성할 수 있다. 아래 패치은 개념을 보여 주지만 독립 실행 가능한 완성 모델은 아니다.

## 1. 왜 필요한가

기본 ViT는 한 해상도의 패치 토큰을 전역 어텐션으로 처리한다. 고해상도 탐지·분할에 활용할 때는 점수 비용과 여러 해상도의 피처 구성 방법을 추가로 설계해야 한다. ViT를 이런 과제에 사용할 수 없다는 뜻은 아니다.

1. **이차 복잡도**: 모든 패치에 대한 셀프 어텐션은 복잡도가 $O(N^2)$이다
2. **한 가지 크기의 피처**: 해상도 하나에서만 토큰을 낸다

이 한계 때문에 다음이 필요한 물체 탐지나 분할 같은 과제에 비전 트랜스포머를 쓰기 어렵다.

- 여러 크기의 피처 지도
- 해상도가 높은 출력
- 큰 그림의 효율적인 처리

스윈 트랜스포머는 위계를 이루는 구조와 윈도 어텐션으로 두 문제를 모두 푼다.

---

## 2. 핵심 혁신

### 1. 위계를 이루는 피처 지도

스윈 트랜스포머는 컨볼루션 뉴럴 넷트웍처럼 공간 해상도를 차츰 줄이면서 채널 차원을 늘린다.

```
Stage 1: H/4 × W/4 × C
    ↓ (Patch Merging)
Stage 2: H/8 × W/8 × 2C
    ↓ (Patch Merging)
Stage 3: H/16 × W/16 × 4C
    ↓ (Patch Merging)
Stage 4: H/32 × W/32 × 8C
```

### 2. 창 기반 멀티헤드 셀프 어텐션 (W-MSA)

WindowAttention은 외부에서 분할한 (B×윈도 수,M²,C)를 받는다. 실제 N은 M²이어야 하고 C는 헤드 수로 나누어떨어져야 한다. 상대 바이어스는 점수에 더하며 그 자체가 패딩·시프트 경계를 차단하지는 않는다.

전역 어텐션 대신 국소 창 안에서 주의를 계산한다.

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
        
        # 상대 자리 바이어스
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
        
        # 상대 자리 바이어스를 더한다
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

### 3. 시프트 윈도 나누기

window_partition은 (B,H,W,C) 입력의 H·W가 window_size로 나누어떨어지는 경우만 처리한다. 시프트 마스크는 고정 input_resolution에 대해 생성하며 해상도를 자동 갱신하지 않는다. 순환 이동으로 붙은 반대편 경계는 별도 가산 마스크로 제한한다.

창끼리 이어지게 하려고 보통 창 나누기와 시프트 윈도 나누기를 번갈아 한다.

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

## 3. 스윈 트랜스포머 블록

아래 블록은 정의되지 않은 Mlp를 사용하며 drop_path>0이면 DropPath도 필요하다. qkv_bias·attn_drop 인수는 현재 ShiftedWindowAttention 경로로 전달하지 않아 호출자가 지정해도 반영되지 않는다. 코드 패치만으로 원래 Swin의 모든 설정을 재현하지 않는다.

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

## 4. 패치 합치기

패치 병합은 H·W가 짝수이고 L=H×W인 입력을 가정한다. 네 부분을 4C로 이어 정규화한 뒤 2C로 투영한다. 홀수 해상도에는 사전 패딩 등 별도 설계가 필요하다.

```python
class PatchMerging(nn.Module):
    """
    위계를 이루는 피처 지도를 위한 조각 합치기 레이어.
    
    공간 해상도를 절반으로 줄이고 통로를 두 배로 늘린다.
    컨볼루션 뉴럴 넷트웍의 스텝 있는 컨볼루션과 비슷하다.
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

## 5. 전체 모델의 연결

아래 SwinTransformer는 PatchEmbed·BasicLayer가 정의되어 있지 않아 그대로 생성할 수 없다. 앞 블록의 Mlp·DropPath도 함께 제공해야 하며 단계별 해상도·헤드·다운샘플을 맞춰야 한다. 이 클래스의 forward는 최종 풀링 분류만 반환해 여러 단계 피처를 자동으로 제공하지 않는다.

```python
class SwinTransformer(nn.Module):
    """
    스윈 트랜스포머: 어긋난 창을 쓰는 위계 비전 트랜스포머.
    
    주요 기능:
    1. 위계를 이루는 피처 지도(컨볼루션 뉴럴 넷트웍처럼)
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

### 표준 비전 트랜스포머 (전역 어텐션)

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

### 스윈 트랜스포머 (윈도 어텐션)

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

여기서 $M$은 창 크기이다(대개 7). 패치이 56×56인 224×224 이미지에서는 다음과 같다.

- 전역 점수 수는 헤드당 3136²=9,834,496개다.
- 윈도 64개에 각 49토큰이면 점수는 64×49²=153,664개다.
- 비율 64배는 점수 항목 수의 비교이며 투영·MLP·통신·전체 실행 시간의 같은 배율을 뜻하지 않는다.

윈도 하나의 점수·가중합은 O(M⁴C), 윈도 수는 hw/M²여서 전체는 O(hwM²C)다. M·C를 고정할 때 hw에 선형이다. 위 수식은 MSA의 투영까지 포함하지만 전체 모델의 MLP·정규화·병합은 별도다.

---

## 7. 모델의 변형

| 대표 구성 | 첫 단계 C | 단계별 깊이 | 단계별 헤드 |
|---|---|---|---|
| Swin-T | 96 | (2,2,6,2) | (3,6,12,24) |
| Swin-S | 96 | (2,2,18,2) | (3,6,12,24) |
| Swin-B | 128 | (2,2,18,2) | (4,8,16,32) |
| Swin-L | 192 | (2,2,18,2) | (6,12,24,48) |

계수 수와 ImageNet 정확도는 클래스 수·해상도·사전 학습·체크포인트에 따라 함께 확인한다. 이 페이지의 미완성 클래스가 이 구성들을 학습·평가한 결과는 아니다.

## 8. 응용

계층 피처는 탐지·분할의 백본으로 활용할 수 있다. 하지만 다음 분류 클래스 생성·SwinUNet 패치에는 피처 추출 인터페이스·탐지 헤드·정의되지 않은 UNetDecoder 등의 연결이 빠져 있다. img_size=800이면 첫 격자 200×200이 윈도 7로 나누어떨어지지 않아 현재 분할 함수에 패딩도 필요하다.

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
# 출력: 1/4, 1/8, 1/16, 1/32의 여러 크기 피처
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

Swin V2는 residual-post-norm, 스케일 코사인 어텐션과 연속 상대 위치 바이어스 등을 제안했다. 큰 모델·해상도 전이를 위한 설계이며 각각을 단독 적용해 항상 안정성·일반화가 개선된다고 보장하지 않는다. 아래 v1 코드가 v2를 구현하는 것도 아니다.

---

## 10. 금융에서의 쓰임

금융 문서·차트·위성 영상 등 이미지 자료에 계층 비전 백본을 적용하는 가설을 세울 수 있다. 과제별 데이터·시점·정답과 검증을 먼저 정해야 하며 윈도 구조만으로 금융 예측 품질이나 수익성이 보장되지는 않는다. 이 페이지는 해당 실험 결과를 제공하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
윈도 어텐션의 비용과 시프트가 바꾸는 연결을 설명하라.

</div>

??? success "풀이"
    M×M 윈도의 토큰 수는 M²이고 점수 수는 M⁴다. N/M²개 윈도 전체 비용은 피처 고정 시 O(NM²)이므로 M이 고정일 때 N에 선형이다. 시프트는 토큰 격자에서 floor(M/2) 정도 이동해 이웃 윈도 경계를 연결한다. 두 레이어만으로 모든 윈도가 전역적으로 정보 교환을 끝낸다는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기본 ViT와 Swin의 토큰 해상도·연결을 비교하고 조건을 설명하라.

</div>

??? success "풀이"
    기본 ViT는 같은 격자에서 전역 어텐션을 반복하고 Swin은 국소 윈도·시프트와 단계별 병합으로 해상도를 줄인다. 고정 윈도에서 어텐션의 길이 비용이 줄 수 있지만 전역 직접 접근이 제한된다. 실제 탐지·분할에는 단계별 피처 반환과 별도의 과제 헤드가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 윈도 분할의 입출력 모양과 필요한 나눗셈·메모리 배치 조건을 설명하라.

</div>

??? success "풀이"
    입력 (B,H,W,C)에서 H·W가 M으로 나누어떨어지면 (B×H/M×W/M,M,M,C)를 반환한다. view가 가능한 연속 메모리 배치도 확인한다. 임의 크기를 지원하려면 패딩·원래 크기 복원과 패딩 키 마스크를 함께 설계해야 한다.

    ```python
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
패치 병합의 입출력 모양과 단계별 다운샘플 조건을 설명하라.

</div>

??? success "풀이"
    (B,H×W,C)에서 2×2 이웃을 이어 (B,H×W/4,4C)로 만든 뒤 (B,H×W/4,2C)로 투영한다. H·W는 짝수여야 한다. 최초 패치 stride=4와 세 번 병합이면 H/4→H/8→H/16→H/32이며 입력 크기의 나눗셈·패딩 규칙을 맞춰야 한다.

## 정리하며

Swin은 윈도·시프트·병합으로 연결과 해상도를 설계한다. 선형 비용은 고정 윈도 등의 조건에서 성립하며 점수 수 절감과 전체 속도는 다르다. 이 페이지에는 빠진 클래스·고정 해상도 조건이 있어 완성된 백본·분할 시스템으로 해석해서는 안 된다.

**참고 문헌**

1. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
2. Liu, Z., et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." CVPR 2022.
