# 예제

ViT의 순전파·패치·어텐션, 모델 크기와 전이 학습 설정을 살펴보는 예제 모음이다. 대부분 무작위 모델·입력을 사용하며 메시지에 사전 트레인·성능 개선이 언급되어도 실제 코드가 그 작업을 수행하는지 구분해야 한다.

## 1. 코드

```python
"""
비전 트랜스포머의 예제와 데모
흔한 용례를 위한 빠른 시작 코드 조각
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
    """견주기 위한 전통적인 컨볼루션 뉴럴 네트워크 구조."""
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
    """컨볼루션 스템 + 트랜스포머 — 둘을 결합한 하이브리드 모델.

    비전 트랜스포머는 이미지를 16x16 패치로 잘라 곧바로 선형 투영한다.
    패치 안쪽을 볼 장치가 없으므로 에지나 텍스처 같은 저수준 패턴을
    처음부터 어텐션으로 배워야 하고, 그래서 데이터가 많이 든다.

    하이브리드 모델은 앞쪽 몇 레이어를 컨볼루션으로 둔다. 컨볼루션이 저수준 패턴을
    맡고, 그 위에서 어텐션이 장거리 관계를 맡는다. 패치를 자르는 일도
    컨볼루션의 stride가 대신하므로 따로 자를 필요가 없다.

    224x224 -> (stride 4, 4번) -> 14x14 = 196 토큰, ViT-Base와 같은 수다.
    """

    def __init__(self, n_classes: int = 10, embed_dim: int = 192,
                 depth: int = 4, num_heads: int = 3):
        super().__init__()
        # 컨볼루션 스템. 네 번 반으로 줄여 224 -> 14
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
        x = x.flatten(2).transpose(1, 2)       # (B, 196, embed_dim) — 패치 분할을 대신한다
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.shape[1] + 1]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))   # 맨 앞 토큰 하나로 분류한다


# --- visualizations ---
"""
비전 트랜스포머를 위한 시각화 도구
비전 트랜스포머가 컨볼루션 뉴럴 네트워크와 어떻게 다르게 이미지를 처리하는지 이해하도록 돕는다
"""


# ========================================================================
# 메인
# ========================================================================


class AttentionVisualizer:
    """
    비전 트랜스포머의 어텐션 맵을 그려 본다.
    모델이 이미지의 어느 부분에 집중하는지 보인다.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.attention_maps = []
        
        # 어텐션을 붙잡으려고 후크를 등록한다
        self._register_hooks()
    
    def _register_hooks(self):
        """어텐션 웨이트를 붙잡으려고 순방향 후크를 등록한다"""
        
        def hook_fn(module, input, output):
            # MultiHeadAttention에서 어텐션 웨이트를 붙잡는다
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
        이미지의 어텐션 맵을 뽑아낸다.
        
        인수:
            image: (1, 3, H, W) 텐서
        반환값:
            레이어마다의 어텐션 맵 목록
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
        이미지 위에 겹친 어텐션 맵을 그려 본다.
        
        인수:
            image: 입력 이미지 텐서 (1, 3, H, W)
            layer_idx: 어느 트랜스포머 레이어를 그릴지
            head_idx: 어느 어텐션 헤드를 그릴지(None이면 모두 평균)
            save_path: 그림을 저장할 경로
        """
        # 어텐션 맵을 얻는다
        attn_maps = self.get_attention_maps(image)
        
        if len(attn_maps) == 0:
            print("No attention maps captured!")
            return
        
        # 레이어를 고른다
        attn = attn_maps[layer_idx]  # (batch, heads, seq_len, seq_len)
        
        # 따로 정하지 않으면 헤드에 걸쳐 평균 낸다
        if head_idx is None:
            attn = attn.mean(dim=1)  # (batch, seq_len, seq_len)
        else:
            attn = attn[:, head_idx]  # (batch, seq_len, seq_len)
        
        # CLS 토큰에서 모든 패치로 가는 어텐션을 얻는다
        attn = attn[0, 0, 1:]  # (n_patches,)
        
        # 공간 격자로 형태를 바꾼다
        n_patches = int(np.sqrt(len(attn)))
        attn = attn.reshape(n_patches, n_patches).numpy()
        
        # 이미지를 준비한다
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        # 어텐션 맵을 이미지 크기로 바꾼다
        attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
        
        # 시각화 만들기
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원래 이미지
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 어텐션 맵
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
    이미지가 패치로 어떻게 나뉘는지 그려 본다.
    연속적인 이미지에서 이산 토큰으로 넘어가는 과정을 보인다.
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
    
    # 패치 격자를 얹은 이미지
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
    컨볼루션 뉴럴 네트워크와 비전 트랜스포머의 리셉티브 필드 차이를 그려 본다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 컨볼루션 뉴럴 네트워크의 리셉티브 필드
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    
    # 계층적 리셉티브 필드를 그린다
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
    
    # 비전 트랜스포머의 리셉티브 필드
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    
    # 이미지 전체에 대한 어텐션을 그린다
    rect = plt.Rectangle((0, 0), 10, 10, color='lightblue', 
                         alpha=0.3, label='Global attention')
    axes[1].add_patch(rect)
    
    # 패치 격자를 그린다
    for i in range(4):
        for j in range(4):
            x, y = i * 2.5, j * 2.5
            rect = plt.Rectangle((x, y), 2.5, 2.5, 
                               fill=False, edgecolor='red', linewidth=2)
            axes[1].add_patch(rect)
            
            # 가운데 패치에서 나가는 어텐션 선을 그린다
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
    위치 정보가 어떻게 담기는지 보인다.
    """
    # 위치 임베딩을 만든다
    pos_embed = torch.randn(1, n_patches, embed_dim)
    
    # 히트맵으로 그린다
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 위치 임베딩을 그린다
    im1 = axes[0].imshow(pos_embed[0].T, aspect='auto', cmap='coolwarm')
    axes[0].set_xlabel('Patch Position')
    axes[0].set_ylabel('Embedding Dimension')
    axes[0].set_title('Learned Positional Embeddings')
    plt.colorbar(im1, ax=axes[0])
    
    # 위치 사이의 유사도를 계산한다
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
    컨볼루션 뉴럴 네트워크와 비전 트랜스포머를 견주는 트레인 곡선을 그린다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_cnn = range(1, len(cnn_history['train_loss']) + 1)
    epochs_vit = range(1, len(vit_history['train_loss']) + 1)
    
    # 트레인 로스
    axes[0, 0].plot(epochs_cnn, cnn_history['train_loss'], 'b-', label='CNN')
    axes[0, 0].plot(epochs_vit, vit_history['train_loss'], 'r-', label='ViT')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 검증 로스
    axes[0, 1].plot(epochs_cnn, cnn_history['val_loss'], 'b-', label='CNN')
    axes[0, 1].plot(epochs_vit, vit_history['val_loss'], 'r-', label='ViT')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 트레인 정확도
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
이미지 분류에서 컨볼루션 뉴럴 네트워크와 트랜스포머를 잇는다
"""


# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """
    이미지를 패치로 바꾸고 임베딩으로 투영한다.
    컨볼루션 뉴럴 네트워크 방식의 입력과 트랜스포머 처리를 잇는 다리이다.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 펼친 패치의 선형 투영 (컨볼루션 뉴럴 네트워크의 컨볼루션과 비슷하다)
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
    트랜스포머에서 온 멀티헤드 셀프 어텐션 메커니즘.
    모델이 이미지의 여러 부분에 한꺼번에 어텐션을 주게 해 준다.
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
        
        # 어텐션 점수
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 밸류에 어텐션 적용
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    트랜스포머 블록에서 쓰는 피드포워드 네트워크.
    비선형성과 피처 변환을 준다.
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
    셀프 어텐션과 다층 퍼셉트론을 갖춘 표준 트랜스포머 인코더 블록.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, 
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 잔차 연결을 더한 어텐션
        x = x + self.attn(self.norm1(x))
        # 잔차 연결을 더한 다층 퍼셉트론
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    비전 트랜스포머(ViT) 모델.
    
    핵심 혁신:
    1. 이미지를 패치의 시퀀스로 다룬다
    2. (원래 자연어 처리에서 온) 트랜스포머 인코더를 이미지 분류에 쓴다
    3. 컨볼루션 뉴럴 네트워크 방식의 입력 처리와 트랜스포머 구조를 잇는다
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
        
        # 패치 임베딩 레이어 (이미지에서 토큰으로 잇는 다리)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # 분류 토큰 (트레인되며 시퀀스 앞에 붙인다)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 위치 임베딩 (트레인되는 것)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 트랜스포머 인코더 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 분류 헤드
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # 웨이트 초기화
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
        
        # 이미지를 패치 임베딩으로 바꾼다
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
    """ViT-Tiny: 파라메터 500만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, 
        depth=12, n_heads=3, n_classes=n_classes
    )


def create_vit_small(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Small: 파라메터 2200만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384,
        depth=12, n_heads=6, n_classes=n_classes
    )


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
    예제 1: 비전 트랜스포머로 하는 기본 추론
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
    
    # 모델 정보
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")


def example_2_compare_models():
    """
    예제 2: 컨볼루션 뉴럴 네트워크와 비전 트랜스포머와 하이브리드 모델 견주기
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
    예제 3: 패치 임베딩 과정 그려 보기
    """
    print("\n" + "="*60)
    print("Example 3: Visualizing Patch Embeddings")
    print("="*60 + "\n")
    
    
    # 패치 임베딩 레이어를 만든다
    patch_embed = PatchEmbedding(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    )
    
    # 무작위 이미지
    image = torch.randn(1, 3, 224, 224)
    
    # 패치로 바꾼다
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
    예제 4: 셀프 어텐션 이해하기
    """
    print("\n" + "="*60)
    print("Example 4: Self-Attention Mechanism")
    print("="*60 + "\n")
    
    
    # 어텐션 모듈 만들기
    attention = MultiHeadAttention(embed_dim=384, n_heads=6)
    
    # 무작위 시퀀스 (batch=1, seq_len=197, embed_dim=384)
    # 197 = 패치 196개 + CLS 토큰 1개
    x = torch.randn(1, 197, 384)
    
    # 어텐션을 적용한다
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
    예제 5: 비전 트랜스포머로 하는 전이 학습
    """
    print("\n" + "="*60)
    print("Example 5: Transfer Learning Setup")
    print("="*60 + "\n")
    
    # 사전 트레인된 모델을 불러온다 (흉내만 낸다)
    model = create_vit_base(n_classes=1000)  # ImageNet 클래스
    
    print("Step 1: Load pretrained model")
    print(f"  - Original classes: 1000 (ImageNet)")
    
    # 새 과제를 위해 분류 헤드를 바꾼다
    n_new_classes = 10
    model.head = torch.nn.Linear(model.head.in_features, n_new_classes)
    
    print(f"\nStep 2: Replace classification head")
    print(f"  - New classes: {n_new_classes}")
    
    # 백본을 동결한다 (선택)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False
    
    print(f"\nStep 3: Freeze backbone layers")
    print(f"  - Only train classification head")
    
    # 트레인 가능한 파라메터 세기
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\nResult:")
    print(f"  - Total parameters: {total:,}")
    print(f"  - Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  - Frozen parameters: {total-trainable:,} ({100*(total-trainable)/total:.1f}%)")


def example_6_model_variants():
    """
    예제 6: 여러 크기의 비전 트랜스포머 모델
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
    예제 7: 컨볼루션 뉴럴 네트워크와 트랜스포머를 결합한 하이브리드 모델
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
    예제 8: 컨볼루션 뉴럴 네트워크와 비전 트랜스포머의 핵심 차이
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
    """모든 예제를 실행한다"""
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

예제 1은 무작위 Tiny 모델과 이미지에서 (1,10) 로짓을 계산한다. 소프트맥스 최댓값은 실제 클래스 정확도나 보정된 신뢰도를 뜻하지 않는다. 예제 2·6의 파라메터·출력 형태 비교에도 트레인·정확도·속도 측정은 없다.

예제 3의 패치 토큰은 정수의 이산 토큰 ID가 아니라 실수 피처 벡터다. 16×16 RGB 패치는 256개의 픽셀 위치, 768개의 채널 값을 갖는다. 패치 투영은 이를 임베딩 공간에 매핑한다. 예제 4는 어텐션 모듈이 트레인 모드라 드롭아웃도 켜져 있고 의미 관계를 학습한 결과가 아니다.

**전이 설정의 한계:** 예제 5는 create_vit_base로 새 모델을 생성하며 사전 트레인 웨이트를 불러오지 않는다. patch_embed와 blocks만 동결해 cls_token·pos_embed·최종 norm은 계속 트레인 가능하다. 따라서 “분류기만 트레인”도 실제 동작과 다르다. 동결과 eval 전환은 별개의 설정이다.

예제 7의 메시지는 HybridCNNViT의 실제 구조와 다르다. 이 쪽 코드의 HybridCNNViT는 224→14 해상도, 기본 192채널, 패치 196개에 CLS 하나를 붙이며 CLS 출력으로 분류한다. 메시지의 28×28×384·784토큰·전역 평균은 이 구현에 해당하지 않는다. CNN 피처를 쓰면 데이터 효율이 더 좋다는 주장도 이 예제에서 평가하지 않는다.

코드 맨 위에서 cv2를 import하므로 AttentionVisualizer를 쓰지 않아도 OpenCV가 설치되어 있어야 한다. 또 코드에 `if __name__ == "__main__":` 블록이 셋 있어, 파일로 돌리면 시각화 그림 두 장 저장과 ViT-Base 순전파가 run_all_examples보다 먼저 실행된다. 예제 8의 정성적 비교 메시지는 파라메터·투영·패딩·장치 등 조건을 생략한 설명이며 보편적 성능 순위가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
예제의 “사전 트레인 모델 불러오기”와 “분류기만 트레인” 메시지가 코드와 다른 이유를 설명하라.

</div>

??? success "풀이"
    팩터리는 무작위 모델을 만들고 head를 교체할 뿐 웨이트를 로드하지 않는다. 동결한 patch_embed·blocks 외에 CLS·위치 표·최종 norm도 requires_grad=True다. 실제 전이 학습에는 호환 체크포인트·전처리를 불러오고 갱신할 파라메터 목록을 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
패치와 어텐션 시연이 언어 토큰·트레인된 시각 관계와 다른 점을 설명하라.

</div>

??? success "풀이"
    패치 벡터는 실수 피처이며 어휘의 정수 ID가 아니다. 무작위 모델·입력은 형태와 계산 규약을 보여 줄 뿐 의미를 학습한 결과가 아니다. 전역 연결 가능성과 실제로 유용한 전역 관계를 학습하는 것도 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
예제 7 메시지의 토큰 수와 현재 HybridCNNViT의 실제 토큰 수를 비교하라.

</div>

??? success "풀이"
    메시지는 28×28=784개를 말하지만 실제 클래스는 네 번 stride=2로 224→14가 되어 196패치와 CLS 하나를 쓴다. 기본 폭은 192이며 분류는 CLS다. 출력 로그의 설명을 실제 계산 그래프와 대조해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 형태 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

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
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 쪽에서 점검할 대상은 `SimpleCNN`·`HybridCNNViT`·`VisionTransformer`와 예제 함수들이다. 실제 함수의 입력·출력 형태, 허용 자료형, 유한한 값과 그래디언트의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 그래디언트가 나와야 한다고 가정하지 않는다.

## 정리하며

예제의 출력 메시지를 수행 사실로 그대로 읽지 않는다. 새 모델 생성과 사전 트레인 로드, 일부 동결과 분류기 전용 트레인, 메시지가 말하는 구조와 실제 코드의 구조를 구분한다.
