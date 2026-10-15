# 인스턴스 정규화

인스턴스 정규화는 이미지의 표본·채널마다 공간 축의 통계량을 사용한다. NumPy 순전파와 이미지 변환 네트워크를 통해 배치 정규화와의 차이를 살펴본다.

## 1. 코드

??? note "코드 (399줄)"

    ```python
    """
    사례 정규화의 구현과 예제
    ===================================================

    사례 정규화는 표본마다, 채널마다 따로 정규화한다.
    배치 통계가 섞이면 안 되는 양식 전이와 GAN에서 널리 쓰인다.

    논문: "Instance Normalization: The Missing Ingredient for Fast Stylization"
           (Ulyanov et al., 2016)
    """

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ========================================================================
    # 메인
    # ========================================================================


    class InstanceNorm2dNumPy:
        """
        NumPy로 바닥부터 구현한 사례 정규화.
        표본마다, 채널마다 따로 정규화한다.
        """

        def __init__(self, num_features, eps=1e-5, affine=True):
            """
            인수:
                num_features: 채널의 수 (C)
                eps: 수치 안정성을 위한 작은 상수
                affine: True이면 gamma와 beta 매개변수를 배운다
            """
            self.num_features = num_features
            self.eps = eps
            self.affine = affine

            if self.affine:
                # 채널마다의 학습 가능한 매개변수
                self.gamma = np.ones((1, num_features, 1, 1))
                self.beta = np.zeros((1, num_features, 1, 1))

        def forward(self, x):
            """
            사례 정규화의 순전파.

            인수:
                x: 모양이 (N, C, H, W)인 입력

            반환값:
                같은 모양의 정규화된 출력
            """
            # 사례별, 채널별로 평균과 분산 계산
            # 각 (N, C)에 대해 공간 차원 (H, W)으로 평균
            # 축 (2, 3)은 높이와 너비다. 곧 배치 축도 채널 축도 건드리지
            # 않고, 표본 하나의 채널 하나 안에서만 통계를 낸다.
            # 네 정규화가 갈리는 지점이 바로 이 축의 선택이다.
            #   배치 정규화: (0, 2, 3) — 배치를 가로질러 채널마다
            #   층 정규화:   (1, 2, 3) — 표본마다 채널을 통틀어
            #   인스턴스:    (2, 3)    — 표본마다 채널마다 따로
            #   그룹 정규화: 채널을 몇 묶음으로 나눈 중간형
            # 이미지 하나의 한 채널이 지닌 평균과 분산은 대체로 그 그림의
            # 밝기와 대비, 곧 화풍에 해당한다. 그것을 지워 버리므로
            # 화풍 옮기기에 알맞다
            mean = np.mean(x, axis=(2, 3), keepdims=True)
            var = np.var(x, axis=(2, 3), keepdims=True)

            # 정규화
            x_normalized = (x - mean) / np.sqrt(var + self.eps)

            # 켜져 있으면 아핀 변환 적용
            if self.affine:
                x_normalized = self.gamma * x_normalized + self.beta

            return x_normalized


    class StyleTransferNetwork(nn.Module):
        """
        사례 정규화를 쓰는 양식 전이 신경망.
        사례 정규화는 사례별 대비 정보를 없애 양식 전이를 더 효과적으로 만들므로
        양식 전이에서 매우 중요하다.
        """

        def __init__(self):
            super(StyleTransferNetwork, self).__init__()

            # 인코더
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
                # affine=True를 명시해야 gamma와 beta를 배운다.
                # nn.InstanceNorm2d의 기본값은 False라 그냥 두면 정규화만 하고
                # 되돌릴 손잡이가 없다. 배치 정규화와 층 정규화는 기본값이
                # True이므로 이 층만 다르다는 점을 기억해 두어야 한다.
                # 화풍 옮기기에서는 이 gamma와 beta가 곧 "입힐 화풍"의
                # 통계 노릇을 하므로 더욱 중요하다
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(inplace=True),
            )

            # 잔차 블록
            self.residual_blocks = nn.Sequential(
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
            )

            # 디코더
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.residual_blocks(x)
            x = self.decoder(x)
            return x


    class ResidualBlock(nn.Module):
        """
        사례 정규화를 갖춘 잔차 블록.
        """

        def __init__(self, channels):
            super(ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(channels, affine=True),
            )

        def forward(self, x):
            # 23_deep_network.md의 잔차 블록과 달리 더한 뒤에 ReLU를 걸지
            # 않는다. 화풍 옮기기의 생성기는 음수 값도 그대로 흘려보내야
            # 하는데, 여기서 ReLU를 걸면 신호의 절반이 잘려 나간다.
            # 제자리 연산(+=)이 아니라 새 텐서를 만드는 형태라 블록이
            # 무엇으로 끝나든 안전하다
            return x + self.conv_block(x)


    class GeneratorWithInstanceNorm(nn.Module):
        """
        사례 정규화를 쓰는 GAN 생성기.
        이미지 대 이미지 변환에서 흔하다 (예: CycleGAN, Pix2Pix).
        """

        def __init__(self, input_channels=3, output_channels=3, ngf=64):
            super(GeneratorWithInstanceNorm, self).__init__()

            # 첫 합성곱
            model = [
                nn.Conv2d(input_channels, ngf, kernel_size=7, padding=3),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True)
            ]

            # 하향 표본화
            n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(ngf * mult * 2),
                    nn.ReLU(inplace=True)
                ]

            # 잔차 블록
            mult = 2 ** n_downsampling
            for i in range(9):
                model += [ResidualBlock(ngf * mult)]

            # 상향 표본화
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                model += [
                    nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                      kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(int(ngf * mult / 2)),
                    nn.ReLU(inplace=True)
                ]

            # 출력층
            model += [
                nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
                nn.Tanh()
            ]

            self.model = nn.Sequential(*model)

        def forward(self, x):
            return self.model(x)


    def demonstrate_instance_norm():
        """
        사례 정규화가 어떻게 작동하는지 보인다.
        """
        print("=" * 60)
        print("Instance Normalization Demonstration")
        print("=" * 60)

        np.random.seed(42)

        # 예시 데이터 만들기: 이미지 2장, 채널 3개, 공간 4x4
        batch_size, channels, height, width = 2, 3, 4, 4

        # 이미지별, 채널별로 통계가 다른 데이터 만들기
        x = np.random.randn(batch_size, channels, height, width)

        # 채널마다 척도를 다르게 하기
        x[0, 0] *= 10   # 이미지 1, 채널 1: 큰 값
        x[0, 1] *= 1    # 이미지 1, 채널 2: 보통 값
        x[0, 2] *= 0.1  # 이미지 1, 채널 3: 작은 값

        x[1, 0] *= 5    # 이미지 2, 채널 1: 중간 값
        x[1, 1] *= 15   # 이미지 2, 채널 2: 아주 큰 값
        x[1, 2] *= 2    # 이미지 2, 채널 3: 보통 값

        print("\nOriginal data statistics:")
        for n in range(batch_size):
            print(f"\nImage {n}:")
            for c in range(channels):
                mean = np.mean(x[n, c])
                std = np.std(x[n, c])
                print(f"  Channel {c}: mean={mean:6.2f}, std={std:6.2f}")

        # 사례 정규화 적용
        instance_norm = InstanceNorm2dNumPy(channels)
        x_normalized = instance_norm.forward(x)

        print("\nAfter Instance Normalization:")
        for n in range(batch_size):
            print(f"\nImage {n}:")
            for c in range(channels):
                mean = np.mean(x_normalized[n, c])
                std = np.std(x_normalized[n, c])
                print(f"  Channel {c}: mean={mean:6.2f}, std={std:6.2f}")

        print("\nKey observations:")
        print("- Each (image, channel) pair is normalized independently")
        print("- Mean ≈ 0 and Std ≈ 1 for EACH channel of EACH image")
        print("- No mixing of statistics across samples or channels")


    def compare_all_normalizations():
        """
        배치 정규화, 층 정규화, 사례 정규화를 나란히 비교한다.
        """
        print("\n" + "=" * 60)
        print("Comparing All Normalization Methods")
        print("=" * 60)

        torch.manual_seed(42)

        # 예시 데이터 만들기: (N=2, C=3, H=4, W=4)
        x = torch.randn(2, 3, 4, 4) * 10

        print("\nInput shape: (N=2, C=3, H=4, W=4)")
        print("N=batch, C=channels, H=height, W=width")

        # 배치 정규화 (각 C에 대해 N, H, W로 정규화)
        bn = nn.BatchNorm2d(3)
        bn.eval()
        x_bn = bn(x)

        # 층 정규화 (각 N에 대해 C, H, W로 정규화)
        ln = nn.LayerNorm([3, 4, 4])
        x_ln = ln(x)

        # 사례 정규화 (각 N, C에 대해 H, W로 정규화)
        instance_norm = nn.InstanceNorm2d(3, affine=False)
        x_in = instance_norm(x)

        # G=3인 그룹 정규화 (채널마다 자기 그룹)
        gn = nn.GroupNorm(3, 3)  # 채널마다 한 그룹이면 사례 정규화와 비슷하다
        x_gn = gn(x)

        print("\n" + "-" * 60)
        print("Statistics after normalization:")
        print("-" * 60)

        print("\nBatch Norm:")
        print(f"  Normalizes over: (N, H, W) for each C")
        print(f"  Mean per channel: {x_bn.mean(dim=(0, 2, 3))}")
        print(f"  Std per channel:  {x_bn.std(dim=(0, 2, 3))}")

        print("\nLayer Norm:")
        print(f"  Normalizes over: (C, H, W) for each N")
        print(f"  Mean per sample: {x_ln.mean(dim=(1, 2, 3))}")
        print(f"  Std per sample:  {x_ln.std(dim=(1, 2, 3))}")

        print("\nInstance Norm:")
        print(f"  Normalizes over: (H, W) for each N and C")
        for n in range(2):
            print(f"  Sample {n}:")
            for c in range(3):
                mean = x_in[n, c].mean()
                std = x_in[n, c].std()
                print(f"    Channel {c}: mean={mean:.4f}, std={std:.4f}")

        print("\n" + "=" * 60)
        print("Summary of Normalization Methods")
        print("=" * 60)

        comparison = """
        방법            | 정규화 대상 축  | 쓰임새
        ----------------|-----------------|----------------------------------
        배치 정규화     | N, H, W         | CNN, 큰 배치
        층 정규화       | C, H, W         | RNN, 트랜스포머, 작은 배치
        사례 정규화     | H, W            | 양식 전이, GAN
        그룹 정규화     | (H, W, C/G)     | 작은 배치, 배치 정규화가 안 통할 때
        """
        print(comparison)


    def demonstrate_style_transfer_example():
        """
        왜 양식 전이에서 사례 정규화가 중요한지 보인다.
        """
        print("\n" + "=" * 60)
        print("Why Instance Norm for Style Transfer?")
        print("=" * 60)

        torch.manual_seed(42)

        # 내용 이미지 모의실험 (밝음)
        content = torch.randn(1, 3, 32, 32) + 2.0

        # 양식 이미지 모의실험 (어두움)
        style = torch.randn(1, 3, 32, 32) - 2.0

        print("\nOriginal statistics:")
        print(f"Content image mean: {content.mean():.4f}, std: {content.std():.4f}")
        print(f"Style image mean:   {style.mean():.4f}, std: {style.std():.4f}")

        # 배치 정규화를 쓸 때 (이미지 사이에서 통계가 섞인다)
        bn = nn.BatchNorm2d(3)
        bn.eval()
        combined_bn = torch.cat([content, style], dim=0)
        normalized_bn = bn(combined_bn)

        print("\nWith Batch Normalization (not ideal):")
        print(f"Normalized content mean: {normalized_bn[0].mean():.4f}")
        print(f"Normalized style mean:   {normalized_bn[1].mean():.4f}")
        print("→ Statistics are mixed across images!")

        # 사례 정규화를 쓸 때 (서로 독립)
        instance_norm = nn.InstanceNorm2d(3, affine=False)
        content_in = instance_norm(content)
        style_in = instance_norm(style)

        print("\nWith Instance Normalization (ideal):")
        print(f"Normalized content mean: {content_in.mean():.4f}")
        print(f"Normalized style mean:   {style_in.mean():.4f}")
        print("→ Each image normalized independently!")

        print("\nKey insight:")
        print("Instance Norm removes instance-specific contrast information,")
        print("allowing the network to focus on transferring style features")
        print("without being influenced by the original image's brightness/contrast.")


    if __name__ == "__main__":
        demonstrate_instance_norm()
        compare_all_normalizations()
        demonstrate_style_transfer_example()

        print("\n" + "=" * 60)
        print("When to use Instance Normalization:")
        print("=" * 60)
        print("✓ Style transfer networks")
        print("✓ GANs (especially image-to-image translation)")
        print("✓ When each sample should be processed independently")
        print("✓ When batch statistics shouldn't mix")
        print("✓ Real-time applications (no running statistics needed)")
    ```


??? note "전체 출력 (108줄)"

    ```
    ============================================================
    Instance Normalization Demonstration
    ============================================================

    Original data statistics:

    Image 0:
      Channel 0: mean= -0.25, std=  9.40
      Channel 1: mean= -0.25, std=  0.91
      Channel 2: mean= -0.03, std=  0.09

    Image 1:
      Channel 0: mean= -0.75, std=  4.08
      Channel 1: mean=  2.19, std= 16.88
      Channel 2: mean= -0.10, std=  1.49

    After Instance Normalization:

    Image 0:
      Channel 0: mean=  0.00, std=  1.00
      Channel 1: mean=  0.00, std=  1.00
      Channel 2: mean= -0.00, std=  1.00

    Image 1:
      Channel 0: mean= -0.00, std=  1.00
      Channel 1: mean= -0.00, std=  1.00
      Channel 2: mean=  0.00, std=  1.00

    Key observations:
    - Each (image, channel) pair is normalized independently
    - Mean ≈ 0 and Std ≈ 1 for EACH channel of EACH image
    - No mixing of statistics across samples or channels

    ============================================================
    Comparing All Normalization Methods
    ============================================================

    Input shape: (N=2, C=3, H=4, W=4)
    N=batch, C=channels, H=height, W=width

    ------------------------------------------------------------
    Statistics after normalization:
    ------------------------------------------------------------

    Batch Norm:
      Normalizes over: (N, H, W) for each C
      Mean per channel: tensor([-0.8787,  3.8724, -1.6152], grad_fn=<MeanBackward1>)
      Std per channel:  tensor([10.8478,  9.4152,  9.8867], grad_fn=<StdBackward0>)

    Layer Norm:
      Normalizes over: (C, H, W) for each N
      Mean per sample: tensor([ 2.1110e-08, -1.4901e-08], grad_fn=<MeanBackward1>)
      Std per sample:  tensor([1.0106, 1.0106], grad_fn=<StdBackward0>)

    Instance Norm:
      Normalizes over: (H, W) for each N and C
      Sample 0:
        Channel 0: mean=0.0000, std=1.0328
        Channel 1: mean=-0.0000, std=1.0328
        Channel 2: mean=-0.0000, std=1.0328
      Sample 1:
        Channel 0: mean=-0.0000, std=1.0328
        Channel 1: mean=0.0000, std=1.0328
        Channel 2: mean=0.0000, std=1.0328

    ============================================================
    Summary of Normalization Methods
    ============================================================

        방법            | 정규화 대상 축  | 쓰임새
        ----------------|-----------------|----------------------------------
        배치 정규화     | N, H, W         | CNN, 큰 배치
        층 정규화       | C, H, W         | RNN, 트랜스포머, 작은 배치
        사례 정규화     | H, W            | 양식 전이, GAN
        그룹 정규화     | (H, W, C/G)     | 작은 배치, 배치 정규화가 안 통할 때
        

    ============================================================
    Why Instance Norm for Style Transfer?
    ============================================================

    Original statistics:
    Content image mean: 2.0052, std: 0.9937
    Style image mean:   -1.9841, std: 1.0164

    With Batch Normalization (not ideal):
    Normalized content mean: 2.0052
    Normalized style mean:   -1.9840
    → Statistics are mixed across images!

    With Instance Normalization (ideal):
    Normalized content mean: -0.0000
    Normalized style mean:   0.0000
    → Each image normalized independently!

    Key insight:
    Instance Norm removes instance-specific contrast information,
    allowing the network to focus on transferring style features
    without being influenced by the original image's brightness/contrast.

    ============================================================
    When to use Instance Normalization:
    ============================================================
    ✓ Style transfer networks
    ✓ GANs (especially image-to-image translation)
    ✓ When each sample should be processed independently
    ✓ When batch statistics shouldn't mix
    ✓ Real-time applications (no running statistics needed)
    ```


## 2. 논의

입력이 `(N, C, H, W)`이면 각 `(N, C)` 조합에서 H와 W를 평균낸다. 서로 다른 표본이나 채널의 통계량을 섞지 않고 출력 모양을 유지한다. 이 과정이 표본별 대비 정보의 일부를 제거하므로 과제에 따라 유익하거나 필요한 정보를 잃을 수 있다.

NumPy 구현은 기본 affine=True이고 PyTorch InstanceNorm2d의 기본은 affine=False이므로 비교할 때 설정을 맞춘다. PyTorch는 기본적으로 이동 통계량을 추적하지 않아 학습과 평가 모두 현재 표본의 통계를 사용한다.

StyleTransferNetwork와 GeneratorWithInstanceNorm은 별도의 구조 예이다. ResidualBlock은 그 안에서 재사용되지만, NumPy 클래스와 세 모듈을 모두 하나의 파이프라인으로 연결하는 것은 아니다. 스타일 손실이나 GAN의 전체 학습 과정까지 제공하는 예제로 해석하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 `(4, 3, 64, 64)`의 통계량과 출력 모양을 구하라.

</div>

??? success "연습문제 1 풀이"
    keepdims=True인 평균과 분산은 `(4, 3, 1, 1)`이다. affine=True의 배율·이동은 `(1, 3, 1, 1)`이며 출력은 `(4, 3, 64, 64)`이다. 정규화 자체는 합성곱이나 풀링을 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
RGB 64×64 입력에 NumPy 정규화를 적용하려면 무엇을 지정해야 하는가?

</div>

??? success "연습문제 2 풀이"
    num_features를 3으로 두고 NCHW NumPy 배열을 forward에 전달한다. PyTorch 텐서나 모델 호출·parameters 인터페이스를 기대하면 안 된다. 스타일 변환 모델은 이미 입력 채널이 3이며 정규화 클래스와 별도 구조이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배치 크기 1에서 BatchNorm2d와 InstanceNorm2d가 같은가?

</div>

??? success "연습문제 3 풀이"
    학습 중 통계 축은 이 경우 같아질 수 있다. 하지만 기본 affine와 이동 통계량 추적 설정, 평가 시 통계량이 다르므로 일반적으로 같은 연산이라고 할 수 없다. 모든 관련 설정을 맞춰야 수치 비교가 가능하다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력층과 출력층은 과제의 차원을 유지하고, 은닉층 사이에 필요한 비선형성을 넣어야 한다. 선형층만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

인스턴스 정규화는 표본·채널별 공간 통계량을 사용한다. 기본 설정과 통계 축을 맞춰 비교하고, 정규화로 제거되는 정보가 목표 과제에 필요한지도 살핀다.
