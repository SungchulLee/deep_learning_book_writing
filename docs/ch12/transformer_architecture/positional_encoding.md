# 위치 인코딩

---

## 1. 위치 정보가 필요한 이유

위치 정보와 위치에 의존하는 마스크가 없는 셀프 어텐션은 순열에 대해 동변이다. 입력 행을 순열 행렬 P로 재배열하면 출력 행도 같은 방식으로 재배열된다. 드롭아웃 같은 무작위 연산은 끄고 비교한다.

$$
\operatorname{Attention}(PX)=P\operatorname{Attention}(X)
$$

이는 출력 텐서가 입력 순서와 무관하게 같다는 뜻이 아니다. 토큰별 출력도 함께 이동한다. 순서에 무관한 풀링까지 적용하면 같은 토큰 집합을 다른 순서로 놓은 문장들을 구분하지 못할 수 있다. 위치 표현이나 인과 마스크는 위치에 따른 정보를 제공한다.

## 2. 사인·코사인 위치 인코딩

본디 트랜스포머 논문은 진동수가 다른 사인과 코사인 함수를 써서 사인·코사인 위치 인코딩을 들여온다.

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

여기서:

- $pos$은 시퀀스에서의 자리이다(0부터 센다)
- $i$은 차원 색인이다(0부터 $d_{\text{model}}/2 - 1$까지)
- $d_{\text{model}}$은 모델의 차원이다

### 진동수 분석

차원 쌍 $(2i, 2i+1)$마다 다음 파장의 사인·코사인에 해당한다.

$$
\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}
$$

짝수 모델 차원에서 파장은 $2\pi$부터 $2\pi\cdot10000^{1-2/d_{\text{model}}}$까지 등비수열을 이룬다. 최댓값은 유한한 차원에서 $2\pi\cdot10000$보다 작다.

### 상대 위치 성질

핵심 성질은 상대 위치를 선형 변환으로 나타낼 수 있다는 것이다. 고정된 어긋남 $k$에 대해 다음이 성립한다.

$$
\text{PE}_{pos+k} = f(\text{PE}_{pos})
$$

구체적으로는 다음과 같다.

$$
\begin{bmatrix} \sin((pos+k)\omega_i) \\ \cos((pos+k)\omega_i) \end{bmatrix} = 
\begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}
\begin{bmatrix} \sin(pos \cdot \omega_i) \\ \cos(pos \cdot \omega_i) \end{bmatrix}
$$

여기서 $\omega_i = 1/10000^{2i/d_{\text{model}}}$이다.

이 관계는 상대 위치를 표현하는 구조적 근거다. 토큰 벡터와 더한 뒤의 실제 어텐션 점수가 상대 위치만의 함수가 되거나, 모델이 그 관계를 반드시 학습한다는 뜻은 아니다.

---

## 3. 파이토치 구현: 사인·코사인 인코딩

이 구현은 짝수 d_model과 저장된 max_len 이내의 입력을 가정한다. 위치 표는 버퍼이며 드롭아웃은 학습 모드에서만 적용된다.

```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class SinusoidalPositionalEncoding(nn.Module):
    """
    'Attention Is All You Need'의 사인파 위치 인코딩.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        인수:
            d_model: 모형 차원(짝수여야 한다)
            max_len: 미리 셈해 둘 최대 수열 길이
            dropout: 드롭아웃 확률
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬을 만든다 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 자리 번호 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 진동수를 위한 나눗셈 항 [d_model/2]
        # 수치 안정성을 위해 exp와 log를 쓴다
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 짝수 색인에는 sin, 홀수 색인에는 cos을 적용한다
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 배치 차원을 더한다 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록한다 (파라메터는 아니지만 상태의 일부이다)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 임베딩에 위치 인코딩을 더한다.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        반환값:
            위치 인코딩을 더한 텐서 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # 위치 인코딩을 더한다 (배치 차원으로 퍼뜨린다)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """그려 보려고 위치 인코딩을 얻는다."""
        return self.pe[:, :seq_len, :].squeeze(0)

def visualize_positional_encoding(d_model: int = 128, max_len: int = 100):
    """사인파 위치 인코딩 행렬을 그려 본다."""
    
    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    encoding = pe.get_encoding(max_len).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 온전한 인코딩 행렬
    ax1 = axes[0, 0]
    im1 = ax1.imshow(encoding, aspect='auto', cmap='RdBu')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Position')
    ax1.set_title('Positional Encoding Matrix')
    plt.colorbar(im1, ax=ax1)
    
    # 특정 자리의 인코딩
    ax2 = axes[0, 1]
    positions_to_plot = [0, 10, 20, 50, 99]
    for pos in positions_to_plot:
        ax2.plot(encoding[pos, :50], label=f'pos={pos}', alpha=0.7)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Encoding at Different Positions (first 50 dims)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 특정 차원의 인코딩
    ax3 = axes[1, 0]
    dims_to_plot = [0, 1, 10, 11, 50, 51]
    for dim in dims_to_plot:
        ax3.plot(encoding[:, dim], label=f'dim={dim}', alpha=0.7)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Encoding Value')
    ax3.set_title('Encoding at Different Dimensions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 자리 사이의 비슷함 (내적)
    ax4 = axes[1, 1]
    similarity = encoding @ encoding.T
    im4 = ax4.imshow(similarity, cmap='viridis')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Position')
    ax4.set_title('Position Similarity (Dot Product)')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    plt.close()
    
    return encoding

# 사용 예
if __name__ == "__main__":
    # 인코딩을 만든다
    d_model = 512
    max_len = 100
    
    pe = SinusoidalPositionalEncoding(d_model, max_len)
    
    # 임베딩 배치로 시험한다
    batch_size = 32
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 상대 자리 성질을 확인한다
    encoding = pe.get_encoding(100)
    
    # PE[pos+k]를 PE[pos]에서 선형 변환으로 얻을 수 있는지 살핀다
    pos, k = 10, 5
    pe_pos = encoding[pos]
    pe_pos_k = encoding[pos + k]
    
    print(f"\nRelative position test:")
    print(f"PE[{pos}] shape: {pe_pos.shape}")
    print(f"PE[{pos + k}] shape: {pe_pos_k.shape}")
    
    # 시각화한다
    visualize_positional_encoding(d_model=128, max_len=100)
    print("\nVisualization saved to 'positional_encoding_visualization.png'")
```

**출력:**

```
Input shape: torch.Size([32, 50, 512])
Output shape: torch.Size([32, 50, 512])

Relative position test:
PE[10] shape: torch.Size([512])
PE[15] shape: torch.Size([512])

Visualization saved to 'positional_encoding_visualization.png'
```

---

## 4. 학습된 위치 인코딩

다른 방법은 위치 임베딩을 파라메터로 학습하는 것이다.

$$
\mathbf{P} = \text{Embedding}(\text{positions}) \in \mathbb{R}^{L \times d_{\text{model}}}
$$

### 좋은 점과 나쁜 점

| 측면 | 사인·코사인 | 학습형 |
|---|---|---|
| 학습 범위 밖의 위치 | 수식으로 계산 가능, 구현의 버퍼 확장 필요 | 테이블 확장·위치 변환 등 별도 설계 필요 |
| 과제에 따른 위치 표 갱신 | 고정 | 로스를 통해 학습 |
| 학습 파라메터 | 0 | $L\times d_{\text{model}}$ |
| 긴 입력의 예측 품질 | 별도 평가 필요 | 별도 평가 필요 |

### 파이토치 구현: 학습된 인코딩

기본형은 최대 위치를 검사한다. get_encoding은 CPU에서 위치 ID를 만들므로 임베딩이 GPU에 있으면 장치가 맞지 않을 수 있다. 보간 변형은 조회 길이를 확장하는 예제이며 확장 후 품질을 보여 주지 않는다.

```python
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """
    BERT와 GPT에서 쓰는 학습된 위치 인코딩.
    
    자리마다 학습되는 임베딩 벡터를 가진다.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        """
        인수:
            d_model: 모형 차원
            max_len: 순차열의 최대 길이
            dropout: 드롭아웃 확률
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 학습되는 자리 임베딩
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # 드롭아웃
        self.dropout = nn.Dropout(p=dropout)
        
        # 초기화한다
        self._init_weights()
    
    def _init_weights(self):
        """자리 임베딩을 작은 값으로 초기화한다."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 임베딩에 학습된 위치 인코딩을 더한다.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        반환값:
            위치 인코딩을 더한 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        # 자리 번호를 만든다 [seq_len]
        positions = torch.arange(seq_len, device=x.device)
        
        # 자리 임베딩을 얻는다 [seq_len, d_model]
        pos_embeddings = self.position_embeddings(positions)
        
        # 입력에 더한다 (배치 차원으로 퍼뜨린다)
        x = x + pos_embeddings
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """그려 보려고 학습된 인코딩을 얻는다."""
        positions = torch.arange(seq_len)
        return self.position_embeddings(positions)

class LearnedPositionalEncodingWithInterpolation(nn.Module):
    """
    길이가 제각각일 때를 위해 사이 채우기를 갖춘 학습된 위치 인코딩.
    
    사이 채우기로 max_len보다 긴 수열도 다룰 수 있다.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 학습되는 자리 임베딩
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        사이 채우기를 받치며 학습된 위치 인코딩을 더한다.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        반환값:
            위치 인코딩을 더한 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len <= self.max_len:
            # 임베딩을 그대로 쓴다
            pos_embeddings = self.position_embeddings[:, :seq_len, :]
        else:
            # 더 긴 수열을 위해 사이를 채운다
            pos_embeddings = torch.nn.functional.interpolate(
                self.position_embeddings.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_embeddings
        
        return self.dropout(x)

# 사용 예
if __name__ == "__main__":
    d_model = 256
    max_len = 512
    
    # 표준 학습형 인코딩
    learned_pe = LearnedPositionalEncoding(d_model, max_len)
    
    # 시험
    x = torch.randn(32, 100, d_model)
    output = learned_pe(x)
    print(f"Learned PE output shape: {output.shape}")
    
    # 사이 채우기와 함께
    learned_pe_interp = LearnedPositionalEncodingWithInterpolation(d_model, max_len)
    
    # 더 긴 수열로 시험한다
    x_long = torch.randn(32, 1000, d_model)
    output_long = learned_pe_interp(x_long)
    print(f"Interpolated PE output shape: {output_long.shape}")
```

**출력:**

```
Learned PE output shape: torch.Size([32, 100, 256])
Interpolated PE output shape: torch.Size([32, 1000, 256])
```

---

## 5. 회전 위치 임베딩 (RoPE)

RoFormer에서 나오고 LLaMA에 쓰인 RoPE는 복소 공간의 회전으로 자리를 인코딩한다.

$$
f_q(\mathbf{x}_m, m) = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m
$$

여기서 $\mathbf{R}_m$은 위치 $m$을 담은 회전 행렬이다.

### 주요 성질

1. **상대 위치의 반영**: 회전된 내적은 내용 벡터 q·k와 위치 차이에 의존한다.
2. **노름 보존**: 회전 자체는 벡터 노름을 바꾸지 않는다. 모든 내용 벡터에서 점수가 거리에 따라 단조 감소하는 것은 아니다.
3. **길이 확장**: 더 긴 위치의 회전은 계산할 수 있지만 버퍼·마스크와 학습 범위의 한계를 별도로 확인해야 한다.

### 구현

```python
import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    """
    LLaMA에서 쓰는 회전 위치 임베딩(RoPE).
    
    복소 공간의 회전으로 자리를 담는다.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        base: int = 10000
    ):
        """
        인수:
            d_model: 모형 차원(짝수여야 한다)
            max_len: 순차열의 최대 길이
            base: 진동수를 셈할 때 쓰는 밑
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # 진동수의 역수를 셈한다
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)
        
        # 회전 행렬을 미리 셈한다
        self._precompute_freqs(max_len)
    
    def _precompute_freqs(self, seq_len: int):
        """회전을 위한 sin과 cos을 미리 셈한다."""
        # 자리 번호 [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device)
        
        # 진동수 [seq_len, d_model/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # sin과 cos을 위해 두 벌로 만든다 [seq_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cos과 sin을 담아 둔다
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """차원의 절반을 돌린다."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None
    ) -> tuple:
        """
        질의와 열쇠에 회전 위치 임베딩을 적용한다.
        
        인수:
            q: 질의 텐서 [batch_size, num_heads, seq_len, head_dim]
            k: 열쇠 텐서 [batch_size, num_heads, seq_len, head_dim]
            seq_len: 수열 길이(선택. 주지 않으면 q에서 알아낸다)
            
        반환값:
            돌린 (q, k) 짝
        """
        if seq_len is None:
            seq_len = q.shape[2]
        
        # 담아 둔 값을 얻는다
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # 퍼뜨리기 좋게 꼴을 바꾼다 [1, 1, seq_len, d_model]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # 회전을 적용한다
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated

# 사용 예
if __name__ == "__main__":
    d_model = 64
    num_heads = 8
    head_dim = d_model // num_heads
    batch_size = 32
    seq_len = 100
    
    rope = RotaryPositionalEncoding(head_dim, max_len=512)
    
    # 질의와 열쇠를 만든다
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # RoPE를 적용한다
    q_rotated, k_rotated = rope(q, k)
    
    print(f"Query shape: {q_rotated.shape}")
    print(f"Key shape: {k_rotated.shape}")
```

**출력:**

```
Query shape: torch.Size([32, 8, 100, 8])
Key shape: torch.Size([32, 8, 100, 8])
```

---

## 6. ALiBi (선형 바이어스를 쓰는 주의)

ALiBi는 자리에 따라 달라지는 바이어스를 어텐션 점수에 곧바로 더한다.

$$
\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{m} \cdot [-(i-j)]\right)
$$

여기서 $\mathbf{m}$은 헤드마다 다른 그래디언트이다.

### 구현

```python
import torch
import torch.nn as nn
import math

class ALiBiPositionalBias(nn.Module):
    """
    선형 바이어스를 쓰는 주의(ALiBi).
    
    주의 점수에 선형 자리 바이어스를 더한다.
    """
    
    def __init__(self, num_heads: int, max_len: int = 2048):
        """
        인수:
            num_heads: 주의 머리의 수
            max_len: 순차열의 최대 길이
        """
        super().__init__()
        
        self.num_heads = num_heads
        
        # 머리마다 그래디언트를 셈한다
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # 자리 차이 행렬을 미리 셈한다
        self._precompute_bias(max_len)
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        머리마다 ALiBi 그래디언트를 얻는다.
        
        등비수열을 쓴다: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # 2의 거듭제곱이 아닌 머리 수를 다룬다
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            slopes = slopes + extra_slopes[:num_heads - closest_power_of_2]
        
        return torch.tensor(slopes).view(num_heads, 1, 1)
    
    def _precompute_bias(self, max_len: int):
        """자리 바이어스 행렬을 미리 셈한다."""
        # 자리 번호
        positions = torch.arange(max_len)
        
        # 상대 자리 [max_len, max_len]
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # ALiBi는 인과 주의에 음의 상대 자리를 쓴다
        # 바이어스는 인과가 아니면 -|i - j|, 인과이면 -(i - j)이다 (위 삼각은 가린다)
        bias = -torch.abs(relative_positions)
        
        self.register_buffer('alibi_bias', bias)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        주어진 수열 길이의 ALiBi 바이어스를 얻는다.
        
        인수:
            seq_len: 수열 길이
            
        반환값:
            바이어스 텐서 [num_heads, seq_len, seq_len]
        """
        # 지금 수열 길이의 바이어스를 얻는다
        bias = self.alibi_bias[:seq_len, :seq_len]
        
        # 그래디언트를 곱한다 [num_heads, seq_len, seq_len]
        return self.slopes * bias

class ALiBiAttention(nn.Module):
    """ALiBi 자리 바이어스를 갖춘 다중 머리 주의."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 사영
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ALiBi 바이어스
        self.alibi = ALiBiPositionalBias(num_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        ALiBi 자리 바이어스를 쓰는 피드포워드.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            mask: 주의 가림 [seq_len, seq_len]
            
        반환값:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 주의에 맞게 옮겨 놓는다: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 주의 점수를 셈한다
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # ALiBi 바이어스를 더한다 [1, num_heads, seq_len, seq_len]
        alibi_bias = self.alibi(seq_len).unsqueeze(0)
        scores = scores + alibi_bias
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스와 드롭아웃
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 값에 어텐션 적용
        output = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output

# 사용 예
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    batch_size = 32
    seq_len = 100
    
    alibi_attn = ALiBiAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = alibi_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

**출력:**

```
Input shape: torch.Size([32, 100, 512])
Output shape: torch.Size([32, 100, 512])
```

---

## 7. 위치 인코딩 방법 비교

| 방법 | 위치 정보를 넣는 곳 | 학습 파라메터 | 길이 확장 시 확인할 점 |
|---|---|---|---|
| 사인·코사인 | 입력 임베딩에 더함 | 0 | 위치 표와 마스크의 길이 |
| 학습형 | 입력 임베딩에 더함 | $L\times d$ | 테이블 범위와 새 위치의 학습 |
| RoPE | Q·K를 회전 | 기본형은 0 | 주파수, 위치 ID와 캐시 |
| ALiBi | 어텐션 점수에 거리 바이어스를 더함 | 기본형은 0 | 바이어스·마스크 구현과 긴 문맥 평가 |

---

## 8. RoPE 깊이 들여다보기

이 절의 코드는 앞 절과 별도의 구현이다. RoPEAttention은 캐시가 있을 때 현재 위치 ID에 과거 길이를 반영해야 하지만 기본값은 매 호출 0부터 시작한다. 캐시 확장 조건도 현재 입력 길이만 검사하므로 큰 position_ids를 안전하게 처리하지 못한다. 인과 마스크는 고정 크기여서 RoPE 캐시만 늘려도 긴 입력 전체를 처리할 수 있는 것은 아니다. 성분을 앞·뒤 절반으로 짝짓는 구현은 인접한 두 성분을 짝짓는 수식과 차원 배치가 다르다.

무작위 Q·K를 사용하는 거리별 그림은 학습된 거리 선호를 검증하지 않는다. 서로 독립적인 등방성 무작위 벡터는 회전해도 분포가 같으므로 평균 그림만으로 단조 거리 감쇠를 주장할 수 없다.

### 들어가며

RoFormer에서 나와 LLaMA, Mistral을 비롯한 요즘 대형 언어 모델이 받아들인 회전 위치 임베딩(RoPE)은 복소 공간의 회전 행렬로 위치 정보를 담는다. 더하는 방식의 위치 인코딩과 달리 RoPE는 어텐션을 셈하는 과정 안에서 상대 위치를 자연스레 잡아낸다.

### 왜 필요한가

전통적인 위치 인코딩에는 한계가 있다.

위치 표현은 입력에 벡터를 더하거나, Q·K를 회전하거나, 점수에 바이어스를 더하는 방식으로 구성할 수 있다. 연산량과 긴 문맥 품질은 이 선택뿐 아니라 구현, 모델의 학습 조건과 평가 길이에 따라 달라진다.

RoPE는 쿼리와 키를 위치별로 회전시켜 내적의 위치 의존성을 상대 오프셋으로 표현한다. 내용 벡터를 고정하면 $(R_mq)^T(R_nk)=q^TR_{n-m}k$이며, 점수는 q와 k에도 의존한다.

### 수학적 정식화

### 핵심 생각

2차원 벡터 $\mathbf{x} = [x_0, x_1]^T$을 각 $\theta$만큼 돌리면 다음과 같다.

$$
R_\theta \mathbf{x} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}
$$

RoPE는 자리에 따라 달라지는 회전을 적용한다.

$$
R_{\theta,m} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
$$

### 높은 차원으로 넓히기

$d$차원 벡터에서는 차원을 짝지어 서로 다른 진동수를 적용한다.

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1
$$

위치 $m$에 대한 온전한 회전 행렬은 다음과 같다.

$$
R_m = \begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

### 상대 위치 성질

핵심 성질은 이렇다. 위치 $m$과 $n$ 사이의 어텐션을 셈할 때 다음과 같다.

$$
(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
$$

위치에 대한 항은 상대 위치 $n-m$으로 정리된다. 내용 벡터 q와 k의 영향이 사라지는 것은 아니다.

### 효율적인 구현

온전한 회전 행렬을 세우는 대신 성분별 연산을 쓴다.

$$
\text{RoPE}(x, m)_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)
$$

$$
\text{RoPE}(x, m)_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

### 파이토치 구현

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class RotaryPositionEmbedding(nn.Module):
    """
    회전 위치 임베딩(RoPE).
    
    회전으로 절대 자리를 담아 주의 점수가 상대 자리에
    매이게 만든다.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000
    ):
        """
        인수:
            dim: 임베딩의 차원(짝수여야 한다)
            max_seq_len: 미리 셈해 둘 최대 수열 길이
            base: 진동수를 셈할 때 쓰는 밑
        """
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 진동수 띠를 미리 셈한다
        # i = 0, 1, ..., d/2-1에 대해 theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # sin과 cos 캐시를 미리 셈한다
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """sin과 cos 값을 미리 셈한다."""
        # 자리 번호: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # 바깥곱: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # 온전한 차원을 위해 이어 붙인다: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cos과 sin을 담아 둔다
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        숨은 차원의 절반을 돌린다.
        
        [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        질의와 열쇠에 회전 위치 임베딩을 적용한다.
        
        인수:
            q: 질의 텐서 [batch, num_heads, seq_len, head_dim]
            k: 열쇠 텐서 [batch, num_heads, seq_len, head_dim]
            position_ids: 선택으로 주는 자리 색인 [batch, seq_len]
            
        반환값:
            돌린 (q, k) 짝
        """
        seq_len = q.shape[2]
        
        # 필요하면 캐시를 늘린다
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # 담아 둔 값을 얻는다
        if position_ids is not None:
            # 따로 정한 자리 (생성 중 KV 캐시를 위해)
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            # 표준 차례 자리
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # 회전을 적용한다: x * cos + rotate_half(x) * sin
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated

class RoPEAttention(nn.Module):
    """
    회전 위치 임베딩을 갖춘 다중 머리 주의.
    
    LLaMA, Mistral을 비롯한 요즘 대형 언어 모형이 쓴다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        rope_base: int = 10000
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 사영
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 인과 가림
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        피드포워드.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            attention_mask: 선택으로 주는 가림
            position_ids: RoPE를 위한 자리 색인
            past_key_value: 생성을 위해 담아 둔 KV
            use_cache: 고친 캐시를 돌려줄지 여부
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 사영한다
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Q와 K에 RoPE를 적용한다
        q, k = self.rope(q, k, position_ids)
        
        # KV 캐시를 다룬다
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        
        # 주의 점수
        kv_seq_len = k.size(2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 인과 가림을 적용한다
        if seq_len > 1:  # 토큰 하나를 만들 때는 건너뛴다
            causal_mask = self.causal_mask[
                kv_seq_len - seq_len:kv_seq_len,
                :kv_seq_len
            ]
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 주의 가림이 있으면 적용한다
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # 소프트맥스를 하고 값에 적용한다
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 출력을 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output, present_key_value

class RoPEWithNTKScaling(RotaryPositionEmbedding):
    """
    길이를 더 잘 뻗도록 NTK를 고려해 크기를 조정한 RoPE.
    
    학습 때 본 것보다 긴 수열을 다루려고 밑 진동수의
    크기를 조정한다.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        # NTK 크기 조정을 위해 밑을 고친다
        adjusted_base = base * (scaling_factor ** (dim / (dim - 2)))
        super().__init__(dim, max_seq_len, int(adjusted_base))
        self.scaling_factor = scaling_factor

class RoPEWithLinearScaling(RotaryPositionEmbedding):
    """
    길이를 뻗으려고 선형 사이 채우기를 쓰는 RoPE.
    
    자리 색인의 크기를 조정해 학습 범위 안에 맞춘다.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        super().__init__(dim, max_seq_len, base)
        self.scaling_factor = scaling_factor
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """자리에 선형 크기 조정을 적용한다."""
        seq_len = q.shape[2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device)
        
        # 자리의 크기를 조정한다
        scaled_positions = position_ids.float() / self.scaling_factor
        
        # 크기를 조정한 자리의 cos과 sin을 셈한다
        freqs = torch.outer(scaled_positions.flatten(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().view(1, 1, seq_len, -1)
        sin = emb.sin().view(1, 1, seq_len, -1)
        
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated

# 보이기와 그리기
def visualize_rope_properties():
    """RoPE의 상대 자리 성질을 그려 본다."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    dim = 64
    seq_len = 100
    rope = RotaryPositionEmbedding(dim, seq_len)
    
    # 무작위 질의와 열쇠를 만든다
    q = torch.randn(1, 1, seq_len, dim)
    k = torch.randn(1, 1, seq_len, dim)
    
    # RoPE를 적용한다
    q_rot, k_rot = rope(q, k)
    
    # 주의 무늬를 셈한다 (상대 자리에 매임을 보인다)
    attn = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 주의 무늬
    ax1 = axes[0]
    im1 = ax1.imshow(attn[0, 0].detach().numpy(), cmap='viridis')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    ax1.set_title('Attention Pattern with RoPE')
    plt.colorbar(im1, ax=ax1)
    
    # 상대 자리에 따른 잦아듦
    ax2 = axes[1]
    # 상대 자리의 함수로 본 평균 주의 웨이트
    relative_weights = []
    for offset in range(-seq_len+1, seq_len):
        weights = []
        for i in range(seq_len):
            j = i + offset
            if 0 <= j < seq_len:
                weights.append(attn[0, 0, i, j].item())
        if weights:
            relative_weights.append((offset, np.mean(weights)))
    
    offsets, weights = zip(*relative_weights)
    ax2.plot(offsets, weights)
    ax2.set_xlabel('Relative Position (k - q)')
    ax2.set_ylabel('Average Attention Weight')
    ax2.set_title('Attention Decay with Relative Distance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rope_visualization.png', dpi=150)
    plt.close()

# 사용 예
if __name__ == "__main__":
    # 설정
    d_model = 512
    num_heads = 8
    head_dim = d_model // num_heads
    batch_size = 2
    seq_len = 128
    
    # RoPE 모듈을 곧바로 시험한다
    rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    q_rot, k_rot = rope(q, k)
    
    print(f"Query shape: {q.shape} -> {q_rot.shape}")
    print(f"Key shape: {k.shape} -> {k_rot.shape}")
    
    # 온전한 주의 레이어를 시험한다
    attention = RoPEAttention(d_model, num_heads, max_seq_len=2048)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, cache = attention(x, use_cache=True)
    
    print(f"\nAttention input: {x.shape}")
    print(f"Attention output: {output.shape}")
    print(f"KV cache shapes: K={cache[0].shape}, V={cache[1].shape}")
    
    # 점진 생성을 시험한다
    print("\n--- Testing Incremental Generation ---")
    
    # 첫 번째: 프롬프트를 처리한다
    prompt_len = 10
    prompt = torch.randn(1, prompt_len, d_model)
    _, kv_cache = attention(prompt, use_cache=True)
    
    # 토큰을 하나씩 만든다
    for step in range(5):
        new_token = torch.randn(1, 1, d_model)
        position_ids = torch.tensor([[prompt_len + step]])
        
        output, kv_cache = attention(
            new_token,
            position_ids=position_ids,
            past_key_value=kv_cache,
            use_cache=True
        )
        
        print(f"Step {step}: output={output.shape}, cache_len={kv_cache[0].size(2)}")
    
    # 파라메터
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # 시각화한다
    visualize_rope_properties()
    print("\nVisualization saved to 'rope_visualization.png'")
```

**출력:**

```
Query shape: torch.Size([2, 8, 128, 64]) -> torch.Size([2, 8, 128, 64])
Key shape: torch.Size([2, 8, 128, 64]) -> torch.Size([2, 8, 128, 64])

Attention input: torch.Size([2, 128, 512])
Attention output: torch.Size([2, 128, 512])
KV cache shapes: K=torch.Size([2, 8, 128, 64]), V=torch.Size([2, 8, 128, 64])

--- Testing Incremental Generation ---
Step 0: output=torch.Size([1, 1, 512]), cache_len=11
Step 1: output=torch.Size([1, 1, 512]), cache_len=12
Step 2: output=torch.Size([1, 1, 512]), cache_len=13
Step 3: output=torch.Size([1, 1, 512]), cache_len=14
Step 4: output=torch.Size([1, 1, 512]), cache_len=15

Total parameters: 1,048,576

Visualization saved to 'rope_visualization.png'
```

### 다른 방법과 비교

### 더하기와 곱하기

| 측면 | 더하기 방식 | RoPE |
|---|---|---|
| 적용 | 토큰 표현에 위치 벡터를 더함 | Q·K의 성분 쌍을 회전 |
| 상대 위치 | 삼각함수의 이동 관계를 활용할 수 있음 | 회전 내적에 상대 오프셋이 나타남 |
| 거리별 점수 | 내용·학습 웨이트에 의존 | 내용·주파수·상대 위치에 의존하며 단조 감소를 보장하지 않음 |

### 길이를 늘리는 RoPE 변형

| 방법 | 조정 대상 | 해석 |
|---|---|---|
| 선형 스케일링 | 위치를 양의 배율로 나눔 | 같은 회전 범위에 더 많은 토큰을 배치 |
| NTK 기반 스케일링 | 주파수의 밑 | 차원별 회전 속도를 변경 |
| 복합 확장 방법 | 주파수·스케일과 학습 조건 | 방법별 설계와 평가를 확인 |

확장 배율에 보편적인 상한이나 품질 보장은 없다. 아래 코드의 NTK 식은 dim=2에서 분모가 0이 되며, 선형 스케일링도 양의 scaling_factor와 위치 ID의 모양을 확인해야 한다.

### 핵심 내용

RoPE는 Q·K의 회전 내적에 상대 위치를 반영한다. 회전 연산 자체는 학습 파라메터를 추가하지 않지만 점수 계산과 캐시 저장 비용은 남는다. 긴 위치를 계산하는 기능, 캐시의 올바른 사용과 긴 문맥에서의 예측 품질을 나누어 확인한다.

위치별 회전을 쓰더라도 모든 거리에서 다른 회전이 수학적으로 보장되거나 어텐션이 거리에만 의존하는 것은 아니다.

### 참고 문헌

1. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
2. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases."
3. Chen, S., et al. (2023). "Extending Context Window of Large Language Models via Positional Interpolation."

---

## 9. ALiBi 깊이 들여다보기

앞 절의 간단한 ALiBiAttention은 자동 인과 마스크가 없다. 이 절의 causal=True 구현은 인과 마스크를 적용하지만 마스크 크기를 자동 확장하지 않아 max_seq_len보다 긴 입력 시연은 모양 오류가 날 수 있다. 바이어스 캐시 재생성도 장치를 확인해야 한다.

get_bias_for_kv_cache는 query_len 인수를 사용하지 않고 마지막 위치 한 개 기준의 바이어스를 만든다. 양의 그래디언트와 인과 마스킹에서 여러 쿼리 행 간 차이가 행별 상수라면 소프트맥스가 이를 상쇄하지만, 함수가 일반적인 상대 거리 행렬을 반환하는 것은 아니다. 추가 attention_mask는 점수에 더하는 가산 마스크다.

### 들어가며

ALiBi는 어텐션 점수에 헤드별 선형 거리 바이어스를 더하는 방식이다. 학습 가능한 위치 테이블이 없어 새 거리의 바이어스를 계산할 수 있지만, 학습 길이를 넘은 품질은 데이터와 모델별로 평가해야 한다.

### 핵심 통찰

ALiBi는 입력 임베딩에 자리를 담는 대신 자리에 따른 패널티를 어텐션 점수에 더한다.

$$
\text{softmax}\left(\mathbf{q}_i \mathbf{K}^T + m \cdot [-(i-j)]\right)
$$

여기서 $m$은 먼 자리에 주의하는 것에 벌을 주는, 헤드마다 다른 그래디언트이다.

### 수학적 정식화

### 어텐션 점수 고치기

위치 $i$의 쿼리와 위치 $j$의 키에 대해 다음과 같다.

$$
a_{ij} = \mathbf{q}_i^T \mathbf{k}_j - m \cdot |i - j|
$$

인과(디코더) 어텐션에서는 다음과 같다.

$$
a_{ij} = \begin{cases}
\mathbf{q}_i^T \mathbf{k}_j - m \cdot (i - j) & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

### 헤드마다 다른 그래디언트

헤드별 그래디언트를 다르게 정한다. H가 2의 거듭제곱일 때 대표 구성은 다음 식과 같다.

$$
m_h = \frac{1}{2^{8h/H}}
$$

H가 2의 거듭제곱일 때 위 식은 h=1,…,H를 사용한다. H=8이면 기울기는 $\{2^{-1},\ldots,2^{-8}\}$이다. 다른 헤드 수는 아래 get_alibi_slopes의 구성 규칙을 따른다.

그래디언트 m이 클수록 같은 거리에 더 큰 음의 바이어스를 주어 가까운 위치를 강하게 선호한다. m이 작으면 거리 패널티가 완만하다. 실제 웨이트는 내용 점수와 소프트맥스에도 의존한다.

### 바이어스 행렬

시퀀스 길이가 $n$일 때 바이어스 행렬 $B$은 다음과 같다.

$$
B = \begin{bmatrix}
0 & -\infty & -\infty & \cdots \\
-1 & 0 & -\infty & \cdots \\
-2 & -1 & 0 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix} \times m
$$

### 파이토치 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    주의 머리마다 ALiBi 그래디언트를 셈한다.
    
    등비수열을 쓴다: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    
    인수:
        num_heads: 주의 머리의 수
        
    반환값:
        그래디언트 텐서 [num_heads]
    """
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    if math.log2(num_heads).is_integer():
        # 2의 거듭제곱: 표준 등비수열을 쓴다
        slopes = get_slopes_power_of_2(num_heads)
    else:
        # 2의 거듭제곱이 아닐 때: 사이를 채운다
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base_slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
        slopes = base_slopes + extra_slopes[:num_heads - closest_power_of_2]
    
    return torch.tensor(slopes)

class ALiBiPositionalBias(nn.Module):
    """
    ALiBi(선형 바이어스를 쓰는 주의) 위치 인코딩.
    
    자리에 따라 달라지는 선형 바이어스를 주의 점수에 더한다.
    학습되는 파라메터가 없다. 등비 그래디언트뿐이다.
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 4096):
        """
        인수:
            num_heads: 주의 머리의 수
            max_seq_len: 미리 셈해 둘 최대 수열 길이
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # 머리마다 그래디언트를 얻는다
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes.view(num_heads, 1, 1))
        
        # 바이어스 행렬을 미리 셈한다
        self._build_alibi_bias(max_seq_len)
    
    def _build_alibi_bias(self, seq_len: int):
        """ALiBi 바이어스 행렬을 세운다."""
        # 자리 번호
        positions = torch.arange(seq_len)
        
        # 상대 자리 행렬: [seq_len, seq_len]
        # relative_pos[i, j] = j - i
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # 인과 주의에서는 j <= i만 따진다
        # 허락된 자리의 바이어스는 -(i - j) = j - i이다
        # 행마다 0, -1, -2, -3, ...이 된다
        relative_pos = relative_pos.float()
        
        # 퍼뜨리기 좋게 [1, seq_len, seq_len]으로 담는다
        self.register_buffer('alibi_bias_base', relative_pos.unsqueeze(0))
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        주어진 수열 길이의 ALiBi 바이어스를 얻는다.
        
        인수:
            seq_len: 지금의 수열 길이
            
        반환값:
            바이어스 텐서 [num_heads, seq_len, seq_len]
        """
        if seq_len > self.max_seq_len:
            self._build_alibi_bias(seq_len)
        
        # 지금 수열 길이의 바이어스를 얻는다
        bias = self.alibi_bias_base[:, :seq_len, :seq_len]
        
        # 머리마다 다른 그래디언트를 곱한다: [num_heads, seq_len, seq_len]
        return self.slopes * bias
    
    def get_bias_for_kv_cache(
        self,
        query_len: int,
        kv_len: int
    ) -> torch.Tensor:
        """
        KV 캐시를 쓰는 점진 디코딩을 위한 ALiBi 바이어스를 얻는다.
        
        인수:
            query_len: 질의 자리의 수(대개 1)
            kv_len: 캐시를 넣은 열쇠·값의 전체 길이
            
        반환값:
            바이어스 텐서 [num_heads, query_len, kv_len]
        """
        # 자리 kv_len-1의 질의 하나가 열쇠 kv_len개 모두에 주의할 때
        # 상대 자리: [-(kv_len-1), -(kv_len-2), ..., -1, 0]
        relative_pos = torch.arange(kv_len, device=self.slopes.device).float()
        relative_pos = relative_pos - (kv_len - 1)  # 마지막 자리가 0이 되도록 민다
        
        # 질의 차원으로 넓힌다: [1, kv_len]
        relative_pos = relative_pos.unsqueeze(0)
        
        # 그래디언트를 곱한다: [num_heads, 1, kv_len]
        return self.slopes * relative_pos

class ALiBiAttention(nn.Module):
    """
    ALiBi 자리 바이어스를 갖춘 다중 머리 주의.
    
    BLOOM, MPT를 비롯한 모형이 쓴다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        causal: bool = True
    ):
        """
        인수:
            d_model: 모형 차원
            num_heads: 주의 머리의 수
            max_seq_len: 최대 수열 길이
            dropout: 주의 드롭아웃
            causal: 인과 가림을 쓸지 여부
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # 사영
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # ALiBi 바이어스
        self.alibi = ALiBiPositionalBias(num_heads, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # 인과 가림
        if causal:
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
            self.register_buffer('causal_mask', mask.bool())
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        ALiBi를 쓰는 피드포워드.
        
        인수:
            x: 입력 [batch, seq_len, d_model]
            attention_mask: 선택으로 주는 주의 가림
            past_key_value: 생성을 위해 담아 둔 KV
            use_cache: 고친 캐시를 돌려줄지 여부
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV 캐시를 다룬다
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        kv_len = k.size(2)
        
        # 주의 점수를 셈한다
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # ALiBi 바이어스를 더한다
        if past_key_value is not None:
            # 점진 디코딩: 질의가 자리 하나이다
            alibi_bias = self.alibi.get_bias_for_kv_cache(seq_len, kv_len)
        else:
            alibi_bias = self.alibi(kv_len)
        
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)
        
        # 인과 가림을 적용한다
        if self.causal and seq_len > 1:
            causal_mask = self.causal_mask[kv_len - seq_len:kv_len, :kv_len]
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 추가 주의 가림을 적용한다
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # 소프트맥스와 드롭아웃
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 값에 적용한다
        output = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output, present_key_value

class ALiBiTransformerBlock(nn.Module):
    """ALiBi 주의를 갖춘 트랜스포머 블록."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = ALiBiAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """앞 정규화 구조의 피드포워드."""
        # 어텐션
        residual = x
        x = self.norm1(x)
        attn_out, present = self.attention(x, past_key_value=past_key_value, use_cache=use_cache)
        x = residual + self.dropout(attn_out)
        
        # 순전파 뉴럴 넷트웍
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)
        
        return x, present

def visualize_alibi_bias(num_heads: int = 8, seq_len: int = 64):
    """머리마다의 ALiBi 바이어스 무늬를 그려 본다."""
    import matplotlib.pyplot as plt
    
    alibi = ALiBiPositionalBias(num_heads, seq_len)
    bias = alibi(seq_len)  # [num_heads, seq_len, seq_len]
    
    # 그림을 위해 인과 가림을 적용한다
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    bias = bias.masked_fill(causal_mask, float('nan'))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    slopes = get_alibi_slopes(num_heads)
    
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(bias[h].numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Head {h+1}, slope={slopes[h]:.4f}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('ALiBi Bias Patterns (Different Slopes per Head)')
    plt.tight_layout()
    plt.savefig('alibi_visualization.png', dpi=150)
    plt.close()

# 사용 예
if __name__ == "__main__":
    # 설정
    d_model = 512
    num_heads = 8
    max_seq_len = 2048
    batch_size = 2
    seq_len = 128
    
    # ALiBi 주의를 시험한다
    attention = ALiBiAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, cache = attention(x, use_cache=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cache shapes: K={cache[0].shape}, V={cache[1].shape}")
    
    # 길이 바깥으로 뻗기를 시험한다
    print("\n--- Testing Length Extrapolation ---")
    long_seq = torch.randn(1, max_seq_len * 2, d_model)
    long_output, _ = attention(long_seq)
    print(f"Long sequence input: {long_seq.shape}")
    print(f"Long sequence output: {long_output.shape}")
    
    # 점진 생성을 시험한다
    print("\n--- Testing Incremental Generation ---")
    prompt = torch.randn(1, 10, d_model)
    _, kv_cache = attention(prompt, use_cache=True)
    
    for step in range(5):
        new_token = torch.randn(1, 1, d_model)
        output, kv_cache = attention(
            new_token,
            past_key_value=kv_cache,
            use_cache=True
        )
        print(f"Step {step}: cache_len={kv_cache[0].size(2)}")
    
    # 파라메터
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("(Note: ALiBi adds 0 extra parameters!)")
    
    # 시각화한다
    visualize_alibi_bias(num_heads=8, seq_len=64)
    print("\nVisualization saved to 'alibi_visualization.png'")
```

### ALiBi의 이점

### 1. 파라메터가 더 들지 않는다

ALiBi는 미리 셈해 둔 그래디언트만 있으면 된다. 학습되는 임베딩이 없다.

| 방식 | 위치 표현의 추가 학습 파라메터 |
|---|---|
| 학습형 절대 위치 | $L\times d$ |
| 사인·코사인 | 0 |
| 상대 위치 바이어스 | 헤드 수, 거리 범위 또는 버킷 수에 따라 다름 |
| ALiBi 기본형 | 0 |

고정 버퍼와 전체 거리 행렬도 메모리를 사용하므로 추가 파라메터가 0이라는 사실을 저장 비용이 0이라는 뜻으로 해석하지 않는다.

### 2. 길이를 아주 잘 뻗는다

ALiBi 원 논문은 학습보다 긴 입력에 대한 실험을 보고했다. 그 결과를 임의의 모델·데이터와 길이에 적용할 수는 없다. 같은 토큰화와 로스 집계 방식으로 길이별 품질 및 메모리를 측정해야 한다.

이 페이지에는 길이별 퍼플렉시티 증가율을 재현하는 학습·평가 실험이 없다. 아래 긴 무작위 입력 시연은 예측 품질 평가가 아니다.

### 3. 끼워 넣기 쉽다

점수에 바이어스를 더하되 인과 마스크와 패딩 마스크를 함께 맞춰야 한다. 기본 ALiBi의 거리 바이어스는 미래 토큰을 차단하지 않는다. 기존 사전 학습 모델의 위치 방식을 바꾸는 일도 단순한 교체만으로 같은 품질을 보장하지 않는다.

```python
attn_scores = q @ k.T / sqrt(d_k)
attn_scores = attn_scores + alibi_bias  # 여기만 바뀐다!
attn_weights = softmax(attn_scores)
```

### 다른 방법과 비교

| 방법 | 위치 정보의 형태 | 구현에서 확인할 저장 비용 |
|---|---|---|
| 사인·코사인 | 입력에 더하는 고정 벡터 | 미리 저장하면 $O(nd)$ |
| 학습형 | 위치 테이블 | $O(Ld)$ |
| 버킷 상대 바이어스 | 상대 거리 버킷별 값 | 헤드·버킷별 표와 적용 시 점수 텐서 |
| RoPE | Q·K 회전 | 주파수·삼각함수 캐시 |
| ALiBi | 거리의 선형 바이어스 | 직접 만들면 헤드별 $O(n^2)$ 거리 바이어스 |

이는 위치 표현의 저장 방식 비교다. 전체 어텐션의 연산·메모리 복잡도를 대신하는 표는 아니다.

### 핵심 내용

ALiBi는 고정 그래디언트로 상대 거리에 대한 선호를 점수에 넣는다. 그래디언트 크기, 마스킹과 캐시 위치를 함께 이해해야 하며, 긴 문맥 성능은 별도의 검증 대상이다.

고정 거리 바이어스와 인과 마스크는 역할이 다르다. 위치 표현에 학습 계수가 없더라도 Q·K·V 투영과 다른 모델 파라메터는 학습한다.

### 참고 문헌

1. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR.
2. Scao, T., et al. (2022). "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model."
3. MosaicML (2023). "MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs."

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
사인·코사인 위치 벡터의 상대 이동 관계를 설명하고 그 해석의 한계를 서술하라.

</div>

??? success "연습문제 1 풀이"
    각 성분 쌍은 각도 덧셈 공식에 의해 고정 오프셋 k의 선형 회전으로 이동한다. 이 성질은 상대 위치 표현의 근거이지만, 토큰 표현과 합쳐 학습한 어텐션이 상대 위치만의 함수가 된다는 보장은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사인·코사인과 학습 가능한 위치 임베딩의 파라메터, 구현 범위와 길이 확장을 비교하라.

</div>

??? success "연습문제 2 풀이"
    사인·코사인은 고정 수식이며 학습형은 위치 테이블을 로스로 갱신한다. 전자는 저장 버퍼를, 후자는 조회 범위를 확인해야 한다. 버퍼 재계산이나 테이블 보간으로 긴 입력을 허용해도 그 길이에서의 예측 품질은 별도 평가가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
위치 정보가 없는 셀프 어텐션의 순열 동변성과 RNN의 순서 의존성을 비교하라.

</div>

??? success "연습문제 3 풀이"
    위치 마스크 등이 없는 결정적 셀프 어텐션은 $A(PX)=PA(X)$이므로 입력 재배열에 맞춰 출력도 재배열된다. 출력이 그대로라는 불변성과 다르다. RNN은 앞선 상태를 순차적으로 갱신하므로 입력 순서가 상태에 영향을 준다. 인과 마스크나 상대 위치 바이어스도 위치 정보를 제공할 수 있어 절대 위치 벡터만이 유일한 방법은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
파이토치에서 사인·코사인 위치 인코딩을 구현하라.

</div>

??? success "연습문제 4 풀이"
    ```python
    def positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
    ```

## 정리하며

위치 표현은 입력에 벡터를 더하거나 Q·K를 회전하거나 점수에 바이어스를 더해 구성한다. 순열 동변성, 상대 위치의 수식과 실제 구현의 길이 제한을 구분한다. 새로운 위치를 계산할 수 있다는 사실만으로 긴 문맥에서 좋은 예측이 보장되지는 않는다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Shaw, P., et al. (2018). "Self-Attention with Relative Position Representations." NAACL.
3. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
4. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases." ICLR.

---
