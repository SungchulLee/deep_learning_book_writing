# 예제

비전 트랜스포머의 예제와 데모. 흔한 쓰임새를 위한 빠른 시작 코드 조각

트랜스포머에 바탕을 둔 구조는 자연어 처리를 뒤바꾸어 놓았다. 이 구현은 트랜스포머의 개념을 살피며, 갖가지 과제에서 최고 수준의 성능을 내게 하는 주의 얼개와 구조의 본을 보여 준다.

## 1. 코드

```python
"""
비전 트랜스포머의 예제와 데모
흔한 쓰임새를 위한 빠른 시작 코드 조각
"""

import torch
from vit_model import create_vit_tiny, create_vit_base, VisionTransformer
from cnn_vs_vit import SimpleCNN, HybridCNNViT
from visualizations import AttentionVisualizer, visualize_patch_embedding

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
    보기 2: 합성곱 신경망과 비전 트랜스포머와 섞은 모형 견주기
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
    
    from vit_model import PatchEmbedding
    
    # 조각 임베딩 층을 만든다
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
    
    from vit_model import MultiHeadAttention
    
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
    
    # 학습 가능한 매개변수 세기
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
    
    from vit_model import create_vit_tiny, create_vit_small, create_vit_base, create_vit_large
    
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
    보기 7: 합성곱 신경망과 트랜스포머를 섞은 모형
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
    보기 8: 합성곱 신경망과 비전 트랜스포머의 핵심 차이
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
    run_all_examples()```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 파이토치 코드로 트랜스포머의 핵심 개념을 보여 준다. 모듈 방식의 짜임 덕분에 낱낱의 부품을 살펴보고 다른 과제나 데이터셋에 맞추어 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 설계 결정을 짚어라. 구체적인 구현 선택 세 가지를 들고 각각이 트랜스포머에 왜 알맞은지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
예제 구현을 검증하는 두루 갖춘 시험 함수를 작성하라. 빈 입력, 원소가 하나인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 모서리 경우를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 예제

이 구현은 깔끔하고 읽기 쉬운 파이토치 코드로 트랜스포머의 핵심 개념을 보여 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
