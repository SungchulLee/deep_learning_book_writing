# 앞선 켜

앞선 켜: 눈 변환기(ViT)의 눈길 그리기. 눈 변환기에 맞춘 그림 그리기로, 자리 눈길을 보인다

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 눈길 그림 그리기 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
앞선 켜: 눈 변환기(ViT)의 눈길 그리기

눈 변환기에 맞춘 그림 그리기로, 그림 조각에 걸친 자리 눈길
결을 보인다.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import warnings

# ========================================================================
# 메인
# ========================================================================
warnings.filterwarnings('ignore')

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class ViTAttentionVisualizer:
    """
    눈 변환기의 눈길 결을 그리는 개.

    ViT은 그림을 조각으로 나누고 변환기를 건다. 이 클래스는 모형이
    자리마다 어떻게 눈길을 두는지 그리도록 돕는다.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16):
        """
        매개변수:
        ----------
        image_size : int
            들임 그림 크기(네모라고 여긴다)
        patch_size : int
            조각 하나의 크기
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

    def visualize_patch_attention(self,
                                  attention: torch.Tensor,
                                  image: Optional[torch.Tensor] = None,
                                  focus_patch: int = 0,
                                  save_path: Optional[str] = None):
        """
        조각 하나가 다른 모든 조각을 어떻게 보는지 그린다.

        매개변수:
        ----------
        attention : torch.Tensor
            눈길 행렬, 꼴: (조각 수+1, 조각 수+1)
            (+1은 CLS 낱말)
        image : torch.Tensor, 없어도 됨
            본디 그림 텐서, 꼴: (3, H, W)
        focus_patch : int
            눈여겨볼 조각(0이면 CLS 낱말)
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()

        # 눈여겨보는 조각에서 뻗는 눈길을 뽑는다
        patch_attention = attention[focus_patch, :]

        # 2차원 격자로 꼴을 바꾼다(CLS 낱말은 뺀다)
        grid_size = int(np.sqrt(self.num_patches))

        if focus_patch == 0:  # CLS 낱말
            # 그릴 때 CLS 낱말은 건너뛴다
            spatial_attention = patch_attention[1:].reshape(grid_size, grid_size)
            title = "CLS 낱말이 조각에 두는 눈길"
        else:
            spatial_attention = patch_attention[1:].reshape(grid_size, grid_size)
            title = f"조각 {focus_patch-1}의 눈길"

        # 그림을 만든다
        if image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 본디 그림을 보인다
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
                if img_np.shape[0] == 3:  # CHW 꼴
                    img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = image

            axes[0].imshow(img_np)
            axes[0].set_title('본디 그림', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # 눈길 그림을 겹쳐 보인다
            axes[1].imshow(img_np, alpha=0.5)

            # 눈길 그림을 그림 크기로 키운다
            attn_resized = F.interpolate(
                torch.tensor(spatial_attention).unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()

            im = axes[1].imshow(attn_resized, cmap='jet', alpha=0.5, vmin=0, vmax=spatial_attention.max())
            axes[1].set_title(title, fontsize=12, fontweight='bold')
            axes[1].axis('off')

            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(spatial_attention, cmap='viridis', aspect='auto')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('조각 세로줄', fontsize=11)
            ax.set_ylabel('조각 가로줄', fontsize=11)
            plt.colorbar(im, ax=ax, label='눈길 짐')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

    def visualize_attention_map(self,
                               attention: torch.Tensor,
                               layer_idx: int = -1,
                               head_idx: int = 0):
        """
        정한 켜와 머리의 온 눈길 그림을 그린다.

        자리 눈길 결을 보이는 격자를 만든다.
        """
        if attention.dim() == 4:  # (묶음, 머리, 열, 열)
            attention = attention[0, head_idx]  # 정한 머리를 집는다

        attention = attention.cpu().numpy()

        # 눈길 행렬을 그린다
        fig, ax = plt.subplots(figsize=(10, 9))

        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        ax.set_xlabel('열쇠 조각', fontsize=12, fontweight='bold')
        ax.set_ylabel('물음 조각', fontsize=12, fontweight='bold')
        ax.set_title(f'ViT 눈길 - 켜 {layer_idx}, 머리 {head_idx}',
                    fontsize=14, fontweight='bold', pad=20)

        plt.colorbar(im, ax=ax, label='눈길 짐')
        plt.tight_layout()
        plt.show()

def example_vit_attention():
    """보기: ViT 자리 눈길 그리기."""
    print("=" * 70)
    print("눈 변환기 눈길 그리기")
    print("=" * 70)

    # 지어낸 ViT 눈길을 만든다
    image_size = 224
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2

    # 지어낸 눈길을 만든다(CLS 낱말 몫으로 조각 수+1)
    seq_len = num_patches + 1
    attention = torch.zeros(seq_len, seq_len)

    # CLS 낱말은 모든 조각을 본다
    attention[0, 1:] = torch.softmax(torch.randn(num_patches), dim=0)
    attention[0, 0] = 0.1

    # 다른 조각은 그 자리를 본다
    for i in range(1, seq_len):
        # 그 자리 눈길 결을 만든다
        distances = torch.abs(torch.arange(1, seq_len) - i)
        attn_logits = -distances.float() * 0.5
        attention[i, 1:] = torch.softmax(attn_logits, dim=0) * 0.9
        attention[i, 0] = 0.05  # CLS에도 얼마쯤 눈길
        attention[i, i] = 0.05  # 제 눈길

    # 그림으로 보인다
    viz = ViTAttentionVisualizer(image_size=image_size, patch_size=patch_size)

    print("\nCLS 낱말의 눈길을 그린다(모형이 어디에 눈길을 두는가):")
    viz.visualize_patch_attention(attention, focus_patch=0)

    print("\n가운데 조각의 눈길을 그린다:")
    center_patch = num_patches // 2
    viz.visualize_patch_attention(attention, focus_patch=center_patch)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_vit_attention()

    print("\n고갱이 깨침:")
    print("  - CLS 낱말이 모든 조각의 소식을 한데 모은다")
    print("  - 자리 눈길이 그림의 어느 자리가 중요한지 드러낸다")
    print("  - 그 자리 조각은 흔히 가까운 자리를 본다")
    print("  - 눈길 그림이 두드러진 물체를 짚어 줄 수 있다")```

## 2. 논의

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 나타냄, 모여 가는 결, 따짐 자를 들여다보게 해서 손에 잡히지 않던 셈을 눈에 보이게 한다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 눈길 그림 그리기에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율을 0.1으로 잡아라. 눈길 드롭아웃이 정칙화에 왜 도움이 되는지 밝혀라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 몇몇을 아무렇게나 0으로 만들어, 모형이 특정 낱말끼리의 얽힘에 지나치게 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 나타냄을 배우게 되는데, 여느 드롭아웃이 신경 세포끼리 함께 굳는 것을 막는 것과 같은 결이다.

---

**연습문제 3.**
제 눈길의 셈 복잡도를 열 길이 $n$과 모형 차원 $d$의 함수로 밝혀라. 이것이 왜 긴 열에 Longformer이나 Linformer 같은 얼개를 부르는가?

??? success "연습문제 3 풀이"
    여느 제 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때가 $O(n^2 d)$이고 눈길 짐에 드는 기억이 $O(n^2)$이다. 열이 길면(보기로 $n = 4096$) 감당할 수 없다. Longformer는 그 자리 미끄럼 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 온 세상 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 낮은 차원 $k \ll n$으로 쏘아 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 나타내는 힘을 얼마쯤 내주고 긴 들임에서의 쓸모를 얻는다.

---
**연습문제 4.**
앞선 켜 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_vitattentionvisualizer():
        model = 앞선 켜(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 앞선 켜

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다.

고갱이 갈래는 `ViTAttentionVisualizer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
