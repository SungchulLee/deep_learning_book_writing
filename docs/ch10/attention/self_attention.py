"""self_attention — self_attention 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch09/attention/self_attention.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SelfAttention(nn.Module):
    """
    자기 어텐션 층
    
    질의와 열쇠와 값이 모두 같은 입력에서 나오는 어텐션을 계산한다.
    트랜스포머 부호기 층에서 쓴다.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_k: Optional[int] = None, 
        d_v: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.scale = self.d_k ** -0.5
        
        # Q, K, V의 선형 사영
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        
        # 출력 사영
        self.out_proj = nn.Linear(self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, d_model)
            mask: 선택적인 어텐션 가림막 (배치 크기, seq_len, seq_len)
                  0은 가릴 자리를 뜻한다
            
        반환값:
            output: 자기 어텐션을 거친 출력 (배치 크기, seq_len, d_model)
            attention_weights: 어텐션 행렬 (배치 크기, seq_len, seq_len)
        """
        # Q, K, V로 사영 (모두 같은 입력 x에서 나온다)
        Q = self.W_q(x)  # (배치, seq_len, d_k)
        K = self.W_k(x)  # (배치, seq_len, d_k)
        V = self.W_v(x)  # (배치, seq_len, d_v)
        
        # 배율 조정 내적 어텐션 점수 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 열쇠(마지막 차원)에 대해 소프트맥스
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값의 가중합
        attended = torch.matmul(attention_weights, V)
        
        # 출력 사영
        output = self.out_proj(attended)
        
        return output, attention_weights

def demonstrate_self_attention():
    """기본 자기 어텐션 시연."""
    d_model = 512
    seq_len = 10
    batch_size = 2
    
    self_attn = SelfAttention(d_model)
    X = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = self_attn(X)
    
    print(f"Input shape:     {X.shape}")        # (2, 10, 512)
    print(f"Output shape:    {output.shape}")   # (2, 10, 512)
    print(f"Attention shape: {weights.shape}")  # (2, 10, 10)
    print(f"\nAttention matrix is square: {weights.shape[-2]} x {weights.shape[-1]}")
    print(f"Each row sums to 1: {weights[0, 0].sum().item():.4f}")

class CausalSelfAttention(nn.Module):
    """
    다중 머리를 지원하는 인과(가림막) 자기 어텐션
    
    자리마다 뒤의 자리에 주목하지 못하게 막는다.
    자기회귀 생성을 하는 GPT 같은 복호기 전용 모델에서 쓴다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int = 2048, 
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 효율을 위해 QKV 사영을 합침 (행렬 곱 세 번 대신 한 번)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 인과 가림막을 버퍼로 등록 (매개변수는 아니지만 모델과 함께 움직인다)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len))
        
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, embed_dim)
            return_attention: 어텐션 가중치를 돌려줄지 여부
            
        반환값:
            output: 어텐션을 거친 출력 (배치 크기, seq_len, embed_dim)
            attention_weights: 선택적으로 (배치 크기, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 효율적인 연산 한 번으로 Q, K, V 사영
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, 배치, heads, seq, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 어텐션 점수 계산: (배치, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 인과 가림막 씌우기 (자리마다 과거와 현재에만 주목할 수 있다)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스와 선택적인 드롭아웃
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값에 어텐션 적용
        attended = torch.matmul(attention_weights, V)
        
        # 머리를 이어 붙이고 되사영
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended)
        
        if return_attention:
            return output, attention_weights
        return output, None

def demonstrate_causal_attention():
    """인과 가림막의 무늬를 보인다."""
    batch_size, seq_len, embed_dim, num_heads = 1, 5, 64, 4
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    causal_attn = CausalSelfAttention(embed_dim, num_heads)
    
    output, weights = causal_attn(x)
    
    print("Causal Attention Pattern (first head):")
    print("Each row shows what that position attends to.")
    print("Position i can only attend to positions <= i (lower triangular).\n")
    print(weights[0, 0].detach().numpy().round(3))
    print("\nNote: Upper triangle is 0 (future positions masked)")

if __name__ == "__main__":
    demonstrate_causal_attention()

class TransformerEncoderBlock(nn.Module):
    """
    양방향 자기 어텐션이 있는 트랜스포머 부호기 블록 하나.
    
    구조: 자기 어텐션 → 더하기와 정규화 → 순방향 신경망 → 더하기와 정규화
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 다중 머리 자기 어텐션 (양방향)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 자리별 순방향 신경망
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인수:
            x: 입력 (배치, seq_len, embed_dim)
            src_mask: 어텐션 가림막 (seq_len, seq_len)
            src_key_padding_mask: 덧댐 가림막 (배치, seq_len)
        """
        # 잔차 연결이 있는 자기 어텐션 (사전 층 정규화 판본)
        attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # 잔차 연결이 있는 순방향 신경망
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class VisionSelfAttention(nn.Module):
    """
    이미지 조각을 위한 자기 어텐션 (비전 트랜스포머 방식).
    
    이미지를 조각으로 나누어 펼친 뒤 순차열로 다룬다.
    조각마다 다른 모든 조각에 전역적으로 주목할 수 있다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            patches: 펼친 이미지 조각 (배치, num_patches, embed_dim)
                     대체로 첫 자리에 [CLS] 토큰이 있다
        
        반환값:
            output: 자기 어텐션을 거친 조각 (배치, num_patches, embed_dim)
            weights: 어텐션 가중치 (배치, num_patches, num_patches)
        """
        # 조각마다 (CLS 토큰을 포함한) 다른 모든 조각에 주목한다
        attended, weights = self.attention(patches, patches, patches)
        
        # 잔차 연결
        output = self.norm(patches + attended)
        
        return output, weights
