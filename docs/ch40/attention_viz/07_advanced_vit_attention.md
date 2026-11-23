# ViT 어텐션 심화

Vision Transformer(ViT)의 어텐션을 그리는 심화 예제다. ViT는 이미지를 16×16 같은 패치로 잘라 토큰처럼 트랜스포머에 넣으므로, 한 토큰의 어텐션을 패치 격자로 되돌려 놓으면 이미지 위의 공간 어텐션 맵이 된다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 어텐션 시각화 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
심화: 비전 트랜스포머(ViT)의 어텐션 시각화

비전 트랜스포머에 맞춘 시각화로, 이미지 패치에 걸친 공간 어텐션
패턴을 보인다.
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
    비전 트랜스포머의 어텐션 패턴을 시각화하는 클래스.

    ViT는 이미지를 패치로 나누고 트랜스포머를 적용한다. 이 클래스는 모델이
    위치마다 어떻게 어텐션을 두는지 그리도록 돕는다.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16):
        """
        파라메터:
        ----------
        image_size : int
            입력 이미지 크기(정사각형이라고 가정한다)
        patch_size : int
            패치 하나의 크기
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
        패치 하나가 다른 모든 패치를 어떻게 보는지 그린다.

        파라메터:
        ----------
        attention : torch.Tensor
            어텐션 행렬, 형태: (패치 수+1, 패치 수+1)
            (+1은 CLS 토큰)
        image : torch.Tensor, 선택
            원본 이미지 텐서, 형태: (3, H, W)
        focus_patch : int
            살펴볼 패치(0이면 CLS 토큰)
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()

        # 살펴보는 패치에서 나가는 어텐션을 뽑는다
        patch_attention = attention[focus_patch, :]

        # 2차원 격자로 형태를 바꾼다(CLS 토큰은 뺀다)
        grid_size = int(np.sqrt(self.num_patches))

        if focus_patch == 0:  # CLS 토큰
            # 그릴 때 CLS 토큰은 건너뛴다
            spatial_attention = patch_attention[1:].reshape(grid_size, grid_size)
            title = "CLS 토큰이 패치에 두는 어텐션"
        else:
            spatial_attention = patch_attention[1:].reshape(grid_size, grid_size)
            title = f"패치 {focus_patch-1}의 어텐션"

        # figure를 만든다
        if image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 원본 이미지를 보인다
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
                if img_np.shape[0] == 3:  # CHW 형태
                    img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = image

            axes[0].imshow(img_np)
            axes[0].set_title('원본 이미지', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # 어텐션 맵을 겹쳐 보인다
            axes[1].imshow(img_np, alpha=0.5)

            # 어텐션 맵을 이미지 크기로 키운다
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
            ax.set_xlabel('패치 열', fontsize=11)
            ax.set_ylabel('패치 행', fontsize=11)
            plt.colorbar(im, ax=ax, label='어텐션 웨이트')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

    def visualize_attention_map(self,
                               attention: torch.Tensor,
                               layer_idx: int = -1,
                               head_idx: int = 0):
        """
        지정한 레이어와 헤드의 전체 어텐션 맵을 그린다.

        공간 어텐션 패턴을 보이는 격자를 만든다.
        """
        if attention.dim() == 4:  # (배치, 헤드, 시퀀스, 시퀀스)
            attention = attention[0, head_idx]  # 지정한 헤드를 고른다

        attention = attention.cpu().numpy()

        # 어텐션 행렬을 그린다
        fig, ax = plt.subplots(figsize=(10, 9))

        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        ax.set_xlabel('키 패치', fontsize=12, fontweight='bold')
        ax.set_ylabel('쿼리 패치', fontsize=12, fontweight='bold')
        ax.set_title(f'ViT 어텐션 - 레이어 {layer_idx}, 헤드 {head_idx}',
                    fontsize=14, fontweight='bold', pad=20)

        plt.colorbar(im, ax=ax, label='어텐션 웨이트')
        plt.tight_layout()
        plt.show()

def example_vit_attention():
    """예제: ViT 공간 어텐션 시각화."""
    print("=" * 70)
    print("비전 트랜스포머 어텐션 시각화")
    print("=" * 70)

    # 합성한 ViT 어텐션을 만든다
    image_size = 224
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2

    # 합성한 어텐션을 만든다(CLS 토큰 때문에 패치 수+1)
    seq_len = num_patches + 1
    attention = torch.zeros(seq_len, seq_len)

    # CLS 토큰은 모든 패치를 본다
    attention[0, 1:] = torch.softmax(torch.randn(num_patches), dim=0)
    attention[0, 0] = 0.1

    # 다른 패치는 주변을 본다
    for i in range(1, seq_len):
        # 국소 어텐션 패턴을 만든다
        distances = torch.abs(torch.arange(1, seq_len) - i)
        attn_logits = -distances.float() * 0.5
        attention[i, 1:] = torch.softmax(attn_logits, dim=0) * 0.9
        attention[i, 0] = 0.05  # CLS에도 약간의 어텐션
        attention[i, i] = 0.05  # 자기 자신에 대한 어텐션

    # 시각화한다
    viz = ViTAttentionVisualizer(image_size=image_size, patch_size=patch_size)

    print("\nCLS 토큰의 어텐션을 그린다(모델이 어디에 어텐션을 두는가):")
    viz.visualize_patch_attention(attention, focus_patch=0)

    print("\n가운데 패치의 어텐션을 그린다:")
    center_patch = num_patches // 2
    viz.visualize_patch_attention(attention, focus_patch=center_patch)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_vit_attention()

    print("\n핵심 정리:")
    print("  - CLS 토큰이 모든 패치의 정보를 한데 모은다")
    print("  - 공간 어텐션이 이미지의 어느 영역이 중요한지 드러낸다")
    print("  - 패치는 흔히 가까운 위치를 본다")
    print("  - 어텐션 맵이 눈에 띄는 물체의 위치를 알려 줄 수 있다")
```

**출력:**

```
======================================================================
비전 트랜스포머 어텐션 시각화
======================================================================

CLS 토큰의 어텐션을 그린다(모델이 어디에 어텐션을 두는가):

가운데 패치의 어텐션을 그린다:

핵심 정리:
  - CLS 토큰이 모든 패치의 정보를 한데 모은다
  - 공간 어텐션이 이미지의 어느 영역이 중요한지 드러낸다
  - 패치는 흔히 가까운 위치를 본다
  - 어텐션 맵이 눈에 띄는 물체의 위치를 알려 줄 수 있다
```

## 2. 논의

출력에는 그림을 그린다는 안내와 핵심 문장만 찍힌다. 그림은 `plt.show()` 로 띄울 뿐 파일로 저장하지 않고, 어텐션도 트레인된 ViT에서 나온 것이 아니라 코드가 지어낸 것이다. 그러니 출력 끝의 「핵심 정리」 네 줄은 이 실행이 보여 준 결과가 아니라 미리 적어 둔 문장이다.

지어낸 어텐션을 읽을 때 짚을 점이 셋 있다. 첫째, 「가까운 패치를 본다」는 패턴은 패치 번호를 한 줄로 늘어놓은 뒤의 거리 `|j - i|` 로 만들었다. 14×14 격자에서 번호가 이웃한 두 패치는 대개 옆 칸이지만, 한 줄의 끝과 다음 줄의 처음도 번호로는 이웃이고, 바로 위아래 칸은 번호가 14나 떨어진다. 그래서 격자로 되돌려 그리면 둥근 이웃이 아니라 가로로 길게 번진 띠가 나온다. 둘째, 각 행은 소프트맥스에 0.9를 곱한 뒤 CLS와 자기 칸을 0.05로 덮어써서 합이 정확히 1이 아니고, 자기 자신에 대한 어텐션이 바로 옆 패치보다 오히려 작다. 셋째, `focus_patch=center_patch` 는 98번째 토큰이고 CLS가 0번이므로 제목에는 「패치 97」이 찍힌다. 실제 ViT라면 `output_attentions=True` 로 얻은 웨이트를 같은 함수에 넘기면 된다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 어텐션 시각화에 알맞은지 설명하라.

</div>

??? success "풀이"
    `ViTAttentionVisualizer` 에서 셋을 들 수 있다. (1) **CLS 토큰을 빼고 패치 격자로 되돌리기**(`patch_attention[1:].reshape(grid_size, grid_size)`) -- 시퀀스의 0번은 CLS 토큰이라 공간 위치가 없으므로 빼야 나머지 196개가 14×14 격자에 맞는다. (2) **쌍선형 보간으로 이미지 크기까지 키우기** -- 14×14 맵을 224×224 이미지 위에 겹칠 수 있게 한다. 대신 패치 경계 안쪽까지 값이 번져 실제 해상도보다 세밀해 보인다. (3) **CLS 토큰의 어텐션을 기본으로 보기**(`focus_patch=0`) -- 분류 헤드가 CLS 토큰의 표현을 쓰므로, CLS가 어느 패치를 보는지가 분류와 가장 가깝다. 다만 한 레이어·한 헤드의 웨이트일 뿐이라 레이어를 거친 정보 흐름은 Attention Rollout 같은 방법으로 따로 봐야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
어텐션 웨이트 뒤(밸류와 곱하기 전)에 드롭아웃 레이어를 더하여라. 트레인하는 동안 드롭아웃 비율을 0.1로 잡아라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 트레인하는 동안 어텐션 웨이트 일부를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 상관에 지나치게 기대는 것을 막는다. 그래서 모델이 어텐션을 더 고르게 나누고 더 강건한 표현을 배우게 되는데, 보통의 드롭아웃이 뉴런끼리의 공적응(co-adaptation)을 막는 것과 같은 원리다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 밝혀라. 이것이 왜 긴 시퀀스에 Longformer나 Linformer 같은 구조를 필요로 하는가?

</div>

??? success "풀이"
    보통의 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간이 $O(n^2 d)$이고 어텐션 웨이트에 드는 메모리가 $O(n^2)$이다. 시퀀스가 길면(예를 들어 $n = 4096$) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 고른 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 밸류를 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---
<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ViTAttentionVisualizer` 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_vitattentionvisualizer():
        model = ViTAttentionVisualizer(...)
        # 보통 입력
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 트레인을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — ViT 어텐션 심화

ViT의 어텐션을 패치 격자로 되돌려 이미지 위에 겹쳐 그리는 법을 보았다. 지어낸 어텐션의 「국소성」이 1차원 번호 거리로 만들어져 격자에서는 띠로 보인다는 점도 짚었다.

핵심 클래스는 `ViTAttentionVisualizer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
