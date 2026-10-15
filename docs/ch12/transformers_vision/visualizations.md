# 시각화

패치 격자·위치 벡터·어텐션과 학습 이력을 시각화하는 도구를 살펴본다. 실제 학습 모델에서 추출한 값과 무작위·도식 예제를 구분하며, 그림만으로 예측의 인과적 근거를 확정하지 않는다.

## 1. 코드

```python
"""
비전 트랜스포머를 위한 시각화 도구
비전 트랜스포머가 합성곱 신경망과 어떻게 다르게 그림을 처리하는지 이해하도록 돕는다
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import cv2

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
```

## 2. 논의

**어텐션 추출의 미완성:** AttentionVisualizer는 model.blocks의 block.attn에 훅을 달고 m.attn 속성을 읽는다. 하지만 이 폴더 MultiHeadAttention은 확률을 지역 변수 attn에만 두고 속성에 저장하지 않는다. 따라서 목록에 None이 들어가고 visualize_attention의 mean·인덱싱이 실패할 수 있다. 별도로 정의한 hook_fn은 실제 등록에 사용되지 않는다.

훅 핸들을 저장·제거하지 않아 같은 모델에 시각화 객체를 반복 생성하면 훅이 누적될 수 있다. 실제 확률을 반환하거나 저장하는 명시적 인터페이스와 장치·detach·훅 수명 관리가 필요하다. 기본 장치 CUDA의 사용 가능 여부도 확인한다.

패치 격자 함수는 픽셀 영역의 경계를 그릴 뿐 학습한 패치 임베딩 벡터를 표시하지 않는다. visualize_positional_encoding은 torch.randn으로 새 표를 생성하므로 “Learned”라는 그림 제목과 달리 학습 모델의 위치 표를 추출하지 않는다. compare_receptive_fields도 계산 그래프에서 측정한 수용 영역이 아니라 도식이며 CNN의 뒤 층 원이 작게 그려져 깊이별 실제 수용 영역으로 해석할 수 없다.

이미지와 어텐션의 min-max 정규화는 상수 입력에서 분모가 0이 된다. 어텐션은 배치 첫 이미지·CLS 하나·정사각 패치 격자를 가정한다. 학습 이력은 키별 길이·정확도 단위와 같은 분할·손실 집계인지 확인해야 비교할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
AttentionVisualizer가 현재 ViT에서 확률을 얻지 못하는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    훅이 기대하는 모듈 속성 m.attn이 없고 실제 확률은 forward 지역 변수다. 훅 출력은 모듈 최종 특징이지 자동으로 확률 행렬이 되는 것도 아니다. 따라서 속성·반환 규약을 맞추고 None을 검증해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
무작위 위치 표와 학습한 위치 표의 시각화를 어떻게 구분하는가?

</div>

??? success "연습문제 2 풀이"
    해당 함수는 새 난수 표를 만들어 그리므로 학습된 공간 패턴을 검증하지 않는다. 실제 model.pos_embed를 분리해 같은 격자 순서로 분석하고 0 노름·상수 값을 처리해야 한다. 유사도만으로 분류 성능을 판정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 시각화에 입력·자료형·수명 검증이 필요한 사례를 세 가지 들어라.

</div>

??? success "연습문제 3 풀이"
    상수 이미지의 min-max 분모, CLS를 뺀 패치 수의 정사각 조건, 반환된 None·CPU/GPU 텐서와 numpy 변환을 확인한다. 훅 핸들을 제거하고 그림 객체를 닫는 수명 관리도 필요하다. 시각화 함수에 학습용 경사가 항상 있어야 한다는 가정은 적절하지 않다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_attentionvisualizer():
        model = AttentionVisualizer(...)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

시각화의 데이터 출처와 훅 인터페이스를 먼저 확인한다. 이 예제에는 어텐션 추출의 미완성 부분과 무작위·도식 그림이 있으므로 학습된 설명으로 해석하지 않는다. 기존 그림은 이번 편집에서 다시 생성하지 않았다.
