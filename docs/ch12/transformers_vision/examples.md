# 예제

ViT의 순전파·패치·어텐션, 모델 크기와 전이 학습 설정을 살펴보는 예제 모음이다. 대부분 무작위 모델·입력을 사용하며 메시지에 사전 학습·성능 개선이 언급되어도 실제 코드가 그 작업을 수행하는지 구분해야 한다.

## 1. 코드

```python
"""
비전 트랜스포머의 예제와 데모
흔한 쓰임새를 위한 빠른 시작 코드 조각
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import cv2
import torch.nn.functional as F
from typing import Tuple

# --- cnn_vs_vit ---
class SimpleCNN(nn.Module):
    """견주기 위한 전통적인 합성곱 신경망 구조."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # 블록 1: 224 → 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 2: 112 → 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 3: 56 → 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 4: 28 → 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class HybridCNNViT(nn.Module):
    """합성곱 밑동 + 트랜스포머 — 둘을 섞은 모형.

    비전 트랜스포머는 그림을 16x16 조각으로 잘라 곧바로 선형 사영한다.
    조각 안쪽을 볼 장치가 없으므로 가장자리나 결 같은 낮은 수준의 무늬를
    처음부터 어텐션으로 배워야 하고, 그래서 자료가 많이 든다.

    섞은 모형은 앞쪽 몇 겹을 합성곱으로 둔다. 합성곱이 낮은 수준의 무늬를
    맡고, 그 위에서 어텐션이 먼 관계를 맡는다. 조각을 자르는 일도
    합성곱의 stride가 대신하므로 따로 자를 필요가 없다.

    224x224 -> (stride 4, 4번) -> 14x14 = 196 토큰, ViT-Base와 같은 수다.
    """

    def __init__(self, n_classes: int = 10, embed_dim: int = 192,
                 depth: int = 4, num_heads: int = 3):
        super().__init__()
        # 합성곱 밑동. 네 번 반으로 줄여 224 -> 14
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.stem(x)                       # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)       # (B, 196, embed_dim) — 조각 자르기를 대신한다
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.shape[1] + 1]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))   # 맨 앞 토큰 하나로 가른다


# --- visualizations ---
"""
비전 트랜스포머를 위한 시각화 도구
비전 트랜스포머가 합성곱 신경망과 어떻게 다르게 그림을 처리하는지 이해하도록 돕는다
"""


# ========================================================================
# 메인
# ========================================================================


class AttentionVisualizer:
    """
    비전 트랜스포머의 주의 지도를 그려 본다.
    모형이 그림의 어느 부분에 집중하는지 보인다.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.attention_maps = []
        
        # 주의를 붙잡으려고 후크를 등록한다
        self._register_hooks()
    
    def _register_hooks(self):
        """주의 가중치를 붙잡으려고 앞먹임 후크를 등록한다"""
        
        def hook_fn(module, input, output):
            # MultiHeadAttention에서 주의 가중치를 붙잡는다
            if hasattr(module, 'attn'):
                self.attention_maps.append(module.attn.detach().cpu())
        
        # 모든 트랜스포머 블록에 후크를 등록한다
        for block in self.model.blocks:
            block.attn.register_forward_hook(
                lambda m, i, o: self.attention_maps.append(
                    m.attn if hasattr(m, 'attn') else None
                )
            )
    
    def get_attention_maps(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        그림의 주의 지도를 뽑아낸다.
        
        인수:
            image: (1, 3, H, W) 텐서
        반환값:
            층마다의 주의 지도 목록
        """
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(image.to(self.device))
        
        return self.attention_maps
    
    def visualize_attention(self, 
                          image: torch.Tensor,
                          layer_idx: int = -1,
                          head_idx: Optional[int] = None,
                          save_path: Optional[str] = None):
        """
        그림 위에 겹친 주의 지도를 그려 본다.
        
        인수:
            image: 입력 그림 텐서 (1, 3, H, W)
            layer_idx: 어느 트랜스포머 층을 그릴지
            head_idx: 어느 주의 머리를 그릴지(None이면 모두 평균)
            save_path: 그림을 저장할 경로
        """
        # 주의 지도를 얻는다
        attn_maps = self.get_attention_maps(image)
        
        if len(attn_maps) == 0:
            print("No attention maps captured!")
            return
        
        # 층을 고른다
        attn = attn_maps[layer_idx]  # (batch, heads, seq_len, seq_len)
        
        # 따로 정하지 않으면 머리에 걸쳐 평균 낸다
        if head_idx is None:
            attn = attn.mean(dim=1)  # (batch, seq_len, seq_len)
        else:
            attn = attn[:, head_idx]  # (batch, seq_len, seq_len)
        
        # CLS 토큰에서 모든 조각으로 가는 주의를 얻는다
        attn = attn[0, 0, 1:]  # (n_patches,)
        
        # 공간 격자로 꼴을 바꾼다
        n_patches = int(np.sqrt(len(attn)))
        attn = attn.reshape(n_patches, n_patches).numpy()
        
        # 그림을 마련한다
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        # 주의 지도를 그림 크기로 바꾼다
        attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
        
        # 시각화 만들기
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원래 이미지
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 주의 지도
        axes[1].imshow(attn_resized, cmap='jet')
        axes[1].set_title(f"Attention Map (Layer {layer_idx})")
        axes[1].axis('off')
        
        # 겹쳐 놓기
        axes[2].imshow(img)
        axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()


def visualize_patch_embedding(image: torch.Tensor, 
                             patch_size: int = 16,
                             save_path: Optional[str] = None):
    """
    그림이 조각으로 어떻게 나뉘는지 그려 본다.
    이어진 그림에서 띄엄띄엄한 토큰으로 잇는 다리를 보인다.
    """
    # 넘파이로 바꾼다
    img = image[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    H, W = img.shape[:2]
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 원래 이미지
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 조각 격자를 얹은 그림
    axes[1].imshow(img)
    
    # 격자를 그린다
    for i in range(n_patches_h + 1):
        axes[1].axhline(y=i*patch_size, color='red', linewidth=2)
    for j in range(n_patches_w + 1):
        axes[1].axvline(x=j*patch_size, color='red', linewidth=2)
    
    axes[1].set_title(f"Patches ({n_patches_h}×{n_patches_w} = {n_patches_h*n_patches_w} tokens)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def compare_receptive_fields():
    """
    합성곱 신경망과 비전 트랜스포머가 받는 영역의 차이를 그려 본다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 합성곱 신경망이 받는 영역
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    
    # 위계를 이루는 수용 영역을 그린다
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    sizes = [8, 6, 4, 2]
    
    for i, (size, color) in enumerate(zip(sizes, colors)):
        circle = plt.Circle((5, 5), size/2, color=color, alpha=0.5, 
                          label=f'Layer {i+1}')
        axes[0].add_patch(circle)
    
    axes[0].plot(5, 5, 'ro', markersize=10, label='Target pixel')
    axes[0].set_aspect('equal')
    axes[0].set_title('CNN: Hierarchical Local Receptive Field')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 비전 트랜스포머가 받는 영역
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    
    # 그림 전체에 대한 주의를 그린다
    rect = plt.Rectangle((0, 0), 10, 10, color='lightblue', 
                         alpha=0.3, label='Global attention')
    axes[1].add_patch(rect)
    
    # 조각 격자를 그린다
    for i in range(4):
        for j in range(4):
            x, y = i * 2.5, j * 2.5
            rect = plt.Rectangle((x, y), 2.5, 2.5, 
                               fill=False, edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)
            
            # 가운데 조각에서 나가는 주의 선을 그린다
            if i == 1 and j == 1:
                axes[1].plot(x+1.25, y+1.25, 'ro', markersize=10)
            else:
                axes[1].plot([1.25*2.5, x+1.25], [1.25*2.5, y+1.25], 
                           'b-', alpha=0.3, linewidth=1)
    
    axes[1].set_aspect('equal')
    axes[1].set_title('ViT: Global Self-Attention from Layer 1')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('receptive_fields_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved receptive field comparison to 'receptive_fields_comparison.png'")


def visualize_positional_encoding(n_patches: int = 196, embed_dim: int = 768):
    """
    비전 트랜스포머에서 쓰는 위치 인코딩을 그려 본다.
    자리 정보가 어떻게 담기는지 보인다.
    """
    # 위치 임베딩을 만든다
    pos_embed = torch.randn(1, n_patches, embed_dim)
    
    # 열지도로 그린다
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 위치 임베딩을 그린다
    im1 = axes[0].imshow(pos_embed[0].T, aspect='auto', cmap='coolwarm')
    axes[0].set_xlabel('Patch Position')
    axes[0].set_ylabel('Embedding Dimension')
    axes[0].set_title('Learned Positional Embeddings')
    plt.colorbar(im1, ax=axes[0])
    
    # 자리 사이의 비슷함을 셈한다
    pos_embed_norm = pos_embed / pos_embed.norm(dim=-1, keepdim=True)
    similarity = (pos_embed_norm[0] @ pos_embed_norm[0].T).numpy()
    
    im2 = axes[1].imshow(similarity, cmap='viridis')
    axes[1].set_xlabel('Patch Position')
    axes[1].set_ylabel('Patch Position')
    axes[1].set_title('Positional Similarity Matrix')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    print("Saved positional encoding visualization to 'positional_encoding.png'")


def plot_training_comparison(cnn_history: dict, vit_history: dict):
    """
    합성곱 신경망과 비전 트랜스포머를 견주는 학습 곡선을 그린다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_cnn = range(1, len(cnn_history['train_loss']) + 1)
    epochs_vit = range(1, len(vit_history['train_loss']) + 1)
    
    # 학습 손실
    axes[0, 0].plot(epochs_cnn, cnn_history['train_loss'], 'b-', label='CNN')
    axes[0, 0].plot(epochs_vit, vit_history['train_loss'], 'r-', label='ViT')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 검증 손실
    axes[0, 1].plot(epochs_cnn, cnn_history['val_loss'], 'b-', label='CNN')
    axes[0, 1].plot(epochs_vit, vit_history['val_loss'], 'r-', label='ViT')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 학습 정확도
    axes[1, 0].plot(epochs_cnn, cnn_history['train_acc'], 'b-', label='CNN')
    axes[1, 0].plot(epochs_vit, vit_history['train_acc'], 'r-', label='ViT')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 검증 정확도
    axes[1, 1].plot(epochs_cnn, cnn_history['val_acc'], 'b-', label='CNN')
    axes[1, 1].plot(epochs_vit, vit_history['val_acc'], 'r-', label='ViT')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved training comparison to 'training_comparison.png'")


if __name__ == "__main__":
    print("Generating visualizations...")
    
    # 개념을 보여 주는 그림을 만든다
    compare_receptive_fields()
    visualize_positional_encoding()
    
    print("\nVisualization utilities ready!")
    print("Use AttentionVisualizer class to visualize attention maps on actual images.")


# --- vit_model ---
"""
비전 트랜스포머(ViT) 구현
이미지 분류에서 합성곱 신경망과 트랜스포머를 잇는다
"""


# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """
    그림을 조각으로 바꾸고 임베딩으로 사영한다.
    합성곱 신경망 방식의 입력과 트랜스포머 처리를 잇는 다리이다.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 편 조각의 선형 사영 (합성곱 신경망의 합성곱과 비슷하다)
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, channels, height, width)
        반환값:
            (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    트랜스포머에서 온 다중 머리 자기 주의 얼개.
    모형이 그림의 여러 부분에 한꺼번에 주의하게 해 준다.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, seq_len, embed_dim)
        반환값:
            (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Q, K, V를 만든다
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 주의 점수
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 값에 어텐션 적용
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    트랜스포머 블록에서 쓰는 순전파 신경망.
    비선형성과 특징 변환을 준다.
    """
    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    자기 주의와 다층 퍼셉트론을 갖춘 표준 트랜스포머 인코더 블록.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, 
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 잔차 연결을 곁들인 주의
        x = x + self.attn(self.norm1(x))
        # 잔차 연결을 곁들인 다층 퍼셉트론
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    비전 트랜스포머(ViT) 모형.
    
    핵심 혁신:
    1. 그림을 조각의 수열로 다룬다
    2. (본디 자연어 처리에서 온) 트랜스포머 인코더를 이미지 분류에 쓴다
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
        
        # 조각 임베딩 층 (그림에서 토큰으로 잇는 다리)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # 분류 토큰 (학습되며 수열 앞에 붙인다)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 위치 임베딩 (학습되는 것)
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
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, channels, height, width)
        반환값:
            (batch_size, n_classes)
        """
        B = x.shape[0]
        
        # 그림을 조각 임베딩으로 바꾼다
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # 분류 토큰을 앞에 붙인다
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # 위치 임베딩을 더한다
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 트랜스포머 블록을 적용한다
        for block in self.blocks:
            x = block(x)
            
        # 분류 토큰으로 분류한다
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 분류 토큰만 쓴다
        x = self.head(cls_token_final)
        
        return x


def create_vit_tiny(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Tiny: 매개변수 500만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, 
        depth=12, n_heads=3, n_classes=n_classes
    )


def create_vit_small(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Small: 매개변수 2200만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384,
        depth=12, n_heads=6, n_classes=n_classes
    )


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


if __name__ == "__main__":
    # 사용 예
    model = create_vit_base(n_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========================================================================
# 메인
# ========================================================================


def example_1_basic_inference():
    """
    보기 1: 비전 트랜스포머로 하는 기본 추론
    """
    print("\n" + "="*60)
    print("Example 1: Basic Inference with Vision Transformer")
    print("="*60 + "\n")
    
    # 모델 생성
    model = create_vit_tiny(n_classes=10)
    model.eval()
    
    # 무작위 입력을 만든다 (batch_size=1, channels=3, height=224, width=224)
    image = torch.randn(1, 3, 224, 224)
    
    # 순전파
    with torch.no_grad():
        output = model(image)
    
    # 예측을 얻는다
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Top probability: {probabilities.max().item():.4f}")
    
    # 모형 정보
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")


def example_2_compare_models():
    """
    보기 2: 컨볼루션 뉴럴 넷트웍과 비전 트랜스포머와 섞은 모형 견주기
    """
    print("\n" + "="*60)
    print("Example 2: Comparing Different Architectures")
    print("="*60 + "\n")
    
    # 모델 만들기
    cnn = SimpleCNN(n_classes=10)
    vit = create_vit_tiny(n_classes=10)
    hybrid = HybridCNNViT(n_classes=10)
    
    # 입력
    x = torch.randn(1, 3, 224, 224)
    
    # 비교
    models = {"CNN": cnn, "ViT": vit, "Hybrid": hybrid}
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:10} | Parameters: {n_params:>10,} | Output shape: {output.shape}")


def example_3_patch_visualization():
    """
    보기 3: 조각 임베딩 과정 그려 보기
    """
    print("\n" + "="*60)
    print("Example 3: Visualizing Patch Embeddings")
    print("="*60 + "\n")
    
    
    # 조각 임베딩 레이어를 만든다
    patch_embed = PatchEmbedding(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    )
    
    # 무작위 그림
    image = torch.randn(1, 3, 224, 224)
    
    # 조각으로 바꾼다
    patches = patch_embed(image)
    
    print(f"Original image shape: {image.shape}")
    print(f"  - Dimensions: (batch, channels, height, width)")
    print(f"  - Size: {image.numel()} values")
    
    print(f"\nPatches shape: {patches.shape}")
    print(f"  - Dimensions: (batch, n_patches, embed_dim)")
    print(f"  - Number of patches: {patch_embed.n_patches}")
    print(f"  - Each patch is: 16×16 pixels = 256 pixels")
    print(f"  - Projected to: {patches.shape[-1]} dimensions")
    
    print("\nThis shows how ViT bridges continuous images to discrete tokens!")


def example_4_attention_mechanism():
    """
    보기 4: 자기 주의 이해하기
    """
    print("\n" + "="*60)
    print("Example 4: Self-Attention Mechanism")
    print("="*60 + "\n")
    
    
    # 어텐션 모듈 만들기
    attention = MultiHeadAttention(embed_dim=384, n_heads=6)
    
    # 무작위 수열 (batch=1, seq_len=197, embed_dim=384)
    # 197 = 조각 196개 + CLS 토큰 1개
    x = torch.randn(1, 197, 384)
    
    # 주의를 적용한다
    output = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nNumber of attention heads: {attention.n_heads}")
    print(f"Dimension per head: {attention.head_dim}")
    
    print("\nKey insight:")
    print("• Each token attends to ALL other tokens simultaneously")
    print("• This is different from CNN which only sees local neighbors")
    print("• Enables global context from the first layer")


def example_5_transfer_learning():
    """
    보기 5: 비전 트랜스포머로 하는 전이 학습
    """
    print("\n" + "="*60)
    print("Example 5: Transfer Learning Setup")
    print("="*60 + "\n")
    
    # 사전 학습된 모형을 싣는다 (흉내)
    model = create_vit_base(n_classes=1000)  # ImageNet 부류
    
    print("Step 1: Load pretrained model")
    print(f"  - Original classes: 1000 (ImageNet)")
    
    # 새 과제를 위해 분류 머리를 바꾼다
    n_new_classes = 10
    model.head = torch.nn.Linear(model.head.in_features, n_new_classes)
    
    print(f"\nStep 2: Replace classification head")
    print(f"  - New classes: {n_new_classes}")
    
    # 등뼈를 얼린다 (선택)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False
    
    print(f"\nStep 3: Freeze backbone layers")
    print(f"  - Only train classification head")
    
    # 학습 가능한 파라메터 세기
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\nResult:")
    print(f"  - Total parameters: {total:,}")
    print(f"  - Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  - Frozen parameters: {total-trainable:,} ({100*(total-trainable)/total:.1f}%)")


def example_6_model_variants():
    """
    보기 6: 여러 크기의 비전 트랜스포머 모형
    """
    print("\n" + "="*60)
    print("Example 6: ViT Model Variants")
    print("="*60 + "\n")
    
    
    models = {
        "ViT-Tiny": create_vit_tiny,
        "ViT-Small": create_vit_small,
        "ViT-Base": create_vit_base,
        "ViT-Large": create_vit_large,
    }
    
    print(f"{'Model':<15} {'Parameters':<15} {'Embed Dim':<12} {'Depth':<10} {'Heads'}")
    print("-" * 70)
    
    for name, create_fn in models.items():
        model = create_fn(n_classes=1000)
        n_params = sum(p.numel() for p in model.parameters())
        
        embed_dim = model.patch_embed.proj.out_channels
        depth = len(model.blocks)
        n_heads = model.blocks[0].attn.n_heads
        
        print(f"{name:<15} {n_params:>12,}   {embed_dim:<12} {depth:<10} {n_heads}")


def example_7_hybrid_architecture():
    """
    보기 7: 컨볼루션 뉴럴 넷트웍과 트랜스포머를 섞은 모형
    """
    print("\n" + "="*60)
    print("Example 7: Hybrid CNN-Transformer Architecture")
    print("="*60 + "\n")
    
    model = HybridCNNViT(n_classes=10)
    
    print("Architecture Pipeline:")
    print("\n1. CNN Stem (ResNet-style)")
    print("   • Input: 224×224×3 image")
    print("   • Convolutions for local feature extraction")
    print("   • Output: 28×28×384 feature maps")
    
    print("\n2. Reshape for Transformer")
    print("   • Flatten spatial dimensions")
    print("   • 28×28 = 784 tokens")
    print("   • Each token: 384 dimensions")
    
    print("\n3. Transformer Encoder")
    print("   • Self-attention across all 784 tokens")
    print("   • Global reasoning on CNN features")
    
    print("\n4. Classification Head")
    print("   • Global average pooling")
    print("   • Linear layer to classes")
    
    print("\nAdvantages:")
    print("• Combines local CNN features with global Transformer reasoning")
    print("• More data-efficient than pure ViT")
    print("• Better inductive biases for vision tasks")


def example_8_key_differences():
    """
    보기 8: 컨볼루션 뉴럴 넷트웍과 비전 트랜스포머의 핵심 차이
    """
    print("\n" + "="*60)
    print("Example 8: CNN vs ViT - Key Differences")
    print("="*60 + "\n")
    
    print("1. INPUT PROCESSING")
    print("   CNN: Sliding window convolutions")
    print("   ViT: Divide into patches, linear projection")
    
    print("\n2. RECEPTIVE FIELD")
    print("   CNN: Grows gradually with depth")
    print("   ViT: Global from first layer")
    
    print("\n3. INDUCTIVE BIAS")
    print("   CNN: Strong (locality, translation equivariance)")
    print("   ViT: Weak (learns from data)")
    
    print("\n4. DATA REQUIREMENTS")
    print("   CNN: Works well with small datasets")
    print("   ViT: Needs large datasets (or pretraining)")
    
    print("\n5. COMPUTATIONAL COMPLEXITY")
    print("   CNN: O(k²·C·H·W) where k=kernel size")
    print("   ViT: O(N²·D) where N=number of patches")
    
    print("\n6. INTERPRETATION")
    print("   CNN: Activation maps, filter visualization")
    print("   ViT: Attention maps, token importance")


def run_all_examples():
    """모든 보기를 돌린다"""
    example_1_basic_inference()
    example_2_compare_models()
    example_3_patch_visualization()
    example_4_attention_mechanism()
    example_5_transfer_learning()
    example_6_model_variants()
    example_7_hybrid_architecture()
    example_8_key_differences()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_examples()
```

## 2. 논의

예제 1은 무작위 Tiny 모델과 이미지에서 (1,10) 로짓을 계산한다. 소프트맥스 최댓값은 실제 클래스 정확도나 보정된 신뢰도를 뜻하지 않는다. 예제 2·6의 계수·출력 모양 비교에도 학습·정확도·속도 측정은 없다.

예제 3의 패치 토큰은 정수의 이산 토큰 ID가 아니라 실수 피처 벡터다. 16×16 RGB 패치는 256개의 픽셀 위치, 768개의 채널 값을 갖는다. 패치 투영은 이를 피처 공간에 매핑한다. 예제 4는 어텐션 모듈이 학습 모드라 드롭아웃도 켜져 있고 의미 관계를 학습한 결과가 아니다.

**전이 설정의 한계:** 예제 5는 create_vit_base로 새 모델을 생성하며 사전 학습 웨이트를 불러오지 않는다. patch_embed와 blocks만 동결해 cls_token·pos_embed·최종 norm은 계속 학습 가능하다. 따라서 “분류기만 학습”도 실제 동작과 다르다. 동결과 eval 전환은 별개의 설정이다.

예제 7의 메시지는 HybridCNNViT의 실제 구조와 다르다. 현재 외부 클래스는 224→14 해상도, 기본 192채널, 패치 196개에 CLS 하나를 붙이며 CLS 출력으로 분류한다. 메시지의 28×28×384·784토큰·전역 평균은 이 구현에 해당하지 않는다. CNN 특성이면 데이터 효율이 더 좋다는 주장도 이 예제에서 평가하지 않는다.

AttentionVisualizer를 가져오기 때문에 실제 사용하지 않아도 visualizations의 cv2 등 의존성이 필요하다. 같은 이름의 모델·외부 파일 규약과 설치 환경을 먼저 확인한다. 예제 8의 정성적 비교 메시지는 계수·투영·패딩·장치 등 조건을 생략한 설명이며 보편적 성능 순위가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
예제의 “사전 학습 모델 불러오기”와 “분류기만 학습” 메시지가 코드와 다른 이유를 설명하라.

</div>

??? success "풀이"
    팩터리는 무작위 모델을 만들고 head를 교체할 뿐 웨이트를 로드하지 않는다. 동결한 patch_embed·blocks 외에 CLS·위치 표·최종 norm도 requires_grad=True다. 실제 전이 학습에는 호환 체크포인트·전처리를 불러오고 갱신할 계수 목록을 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
패치와 어텐션 시연이 언어 토큰·학습된 시각 관계와 다른 점을 설명하라.

</div>

??? success "풀이"
    패치 벡터는 실수 피처가며 어휘의 정수 ID가 아니다. 무작위 모델·입력은 모양과 계산 규약을 보여 줄 뿐 의미를 학습한 결과가 아니다. 전역 연결 가능성과 실제로 유용한 전역 관계를 학습하는 것도 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
예제 7 메시지의 토큰 수와 현재 HybridCNNViT의 실제 토큰 수를 비교하라.

</div>

??? success "풀이"
    메시지는 28×28=784개를 말하지만 외부 클래스는 네 번 stride=2로 224→14가 되어 196패치와 CLS 하나를 쓴다. 기본 폭은 192이며 분류는 CLS다. 출력 로그의 설명을 실제 계산 그래프와 대조해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_examples():
        model = Examples(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

예제의 출력 메시지를 수행 사실로 그대로 읽지 않는다. 새 모델 생성과 사전 학습 로드, 일부 동결과 분류기 전용 학습, 가상의 구조 설명과 실제 외부 모듈을 구분한다. 이번 편집은 코드 실행·그림 생성 없이 설명만 검토했다.
