# 예제

ViT의 순전파·패치·어텐션, 모델 크기와 전이 학습 설정을 살펴보는 예제 모음이다. 대부분 무작위 모델·입력을 사용하며 메시지에 사전 학습·성능 개선이 언급되어도 실제 코드가 그 작업을 수행하는지 구분해야 한다.

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
    
    from vit_model import PatchEmbedding
    
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

??? success "연습문제 1 풀이"
    팩터리는 무작위 모델을 만들고 head를 교체할 뿐 웨이트를 로드하지 않는다. 동결한 patch_embed·blocks 외에 CLS·위치 표·최종 norm도 requires_grad=True다. 실제 전이 학습에는 호환 체크포인트·전처리를 불러오고 갱신할 계수 목록을 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
패치와 어텐션 시연이 언어 토큰·학습된 시각 관계와 다른 점을 설명하라.

</div>

??? success "연습문제 2 풀이"
    패치 벡터는 실수 피처가며 어휘의 정수 ID가 아니다. 무작위 모델·입력은 모양과 계산 규약을 보여 줄 뿐 의미를 학습한 결과가 아니다. 전역 연결 가능성과 실제로 유용한 전역 관계를 학습하는 것도 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
예제 7 메시지의 토큰 수와 현재 HybridCNNViT의 실제 토큰 수를 비교하라.

</div>

??? success "연습문제 3 풀이"
    메시지는 28×28=784개를 말하지만 외부 클래스는 네 번 stride=2로 224→14가 되어 196패치와 CLS 하나를 쓴다. 기본 폭은 192이며 분류는 CLS다. 출력 로그의 설명을 실제 계산 그래프와 대조해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
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
