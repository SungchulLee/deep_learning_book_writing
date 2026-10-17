# 정규화 기법 비교

배치·레이어·인스턴스·그룹 정규화의 축과 배치 의존성을 비교한다. 비교 도우미 클래스는 학습 모델이 아니라 여러 정규화 레이어를 보관하고 측정하는 일반 파이썬 클래스이다.

## 1. 코드

??? note "코드 (458줄)"

    ```python
    """
    정규화 레이어의 종합 비교
    =================================================

    이 파일은 배치 정규화, 레이어 정규화, 사례 정규화, 그룹 정규화를
    나란히 비교하고 실용적인 예제를 제공한다.
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib

    # ========================================================================
    # 메인
    # ========================================================================
    matplotlib.use('Agg')  # 비대화형 백엔드
    import matplotlib.pyplot as plt


    class NormalizationComparison:
        """
        여러 정규화 기법을 비교하기 위한 클래스.
        """

        def __init__(self):
            self.normalizations = {
                'BatchNorm': nn.BatchNorm2d(3, affine=False),
                'LayerNorm': nn.LayerNorm([3, 4, 4], elementwise_affine=False),
                'InstanceNorm': nn.InstanceNorm2d(3, affine=False),
                # 채널 3개에 대한 그룹 3개.
                # 그룹 정규화는 채널을 몇 묶음으로 나누어 배치 안에서만
                # 통계를 낸다. 그래서 양 끝이 다른 방법과 만난다.
                # 그룹 수 = 채널 수이면 채널마다 따로이니 인스턴스 정규화와 같고,
                # 그룹 수 = 1이면 채널을 통틀어 보니 레이어 정규화와 같다.
                # 여기는 채널이 3개인데 그룹도 3개이므로 사실상 인스턴스
                # 정규화와 같은 값이 나온다. 아래 표에서 두 줄이 똑같이
                # 찍히더라도 이상이 아니다
                'GroupNorm': nn.GroupNorm(3, 3, affine=False),
            }

            # 모두 평가 모드로 두기
            for norm in self.normalizations.values():
                if hasattr(norm, 'eval'):
                    norm.eval()

        def visualize_normalization_axes(self):
            """
            각 정규화 방법이 어느 축에 작용하는지 시각화한다.
            """
            print("=" * 70)
            print("Normalization Axes Visualization")
            print("=" * 70)

            print("\nInput tensor shape: (N, C, H, W) = (Batch, Channels, Height, Width)")
            print("\nNormalization axes (what dimensions are averaged over):")
            print("-" * 70)

            visualizations = {
                'BatchNorm':     "Axes: [0, 2, 3] → (N, H, W) | Per channel across batch",
                'LayerNorm':     "Axes: [1, 2, 3] → (C, H, W) | Per sample across features",
                'InstanceNorm':  "Axes: [2, 3]    → (H, W)   | Per sample per channel",
                'GroupNorm':     "Axes: [2, 3]    → (H, W)   | Per sample per group",
            }

            for name, desc in visualizations.items():
                print(f"{name:15s}: {desc}")

            print("\n" + "=" * 70)

        def compare_on_sample_data(self):
            """
            같은 입력에 대해 모든 정규화 방법을 비교한다.
            """
            print("\n" + "=" * 70)
            print("Comparing Normalizations on Sample Data")
            print("=" * 70)

            torch.manual_seed(42)

            # 예시 입력 만들기: 이미지 2장, 채널 3개, 공간 4x4
            x = torch.randn(2, 3, 4, 4)

            # 표본과 채널마다 척도를 다르게 하기
            x[0] *= 5   # 첫 이미지가 더 큰 값을 갖는다
            x[1] *= 0.5  # 둘째 이미지가 더 작은 값을 갖는다
            x[:, 0] *= 2  # 첫 채널이 더 큰 값을 갖는다

            print(f"\nInput shape: {x.shape}")
            print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")

            print("\nOriginal data statistics:")
            for n in range(2):
                for c in range(3):
                    mean = x[n, c].mean()
                    std = x[n, c].std()
                    print(f"  Sample {n}, Channel {c}: mean={mean:7.3f}, std={std:7.3f}")

            print("\n" + "-" * 70)
            print("After normalization:")
            print("-" * 70)

            for name, norm_layer in self.normalizations.items():
                with torch.no_grad():
                    x_norm = norm_layer(x)

                print(f"\n{name}:")
                print(f"  Overall: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")

                # 무엇을 정규화해야 하는지에 맞추어 통계 보이기
                if name == 'BatchNorm':
                    print("  Per channel (averaged over batch, H, W):")
                    for c in range(3):
                        mean = x_norm[:, c].mean()
                        std = x_norm[:, c].std()
                        print(f"    Channel {c}: mean={mean:.4f}, std={std:.4f}")

                elif name == 'LayerNorm':
                    print("  Per sample (averaged over C, H, W):")
                    for n in range(2):
                        mean = x_norm[n].mean()
                        std = x_norm[n].std()
                        print(f"    Sample {n}: mean={mean:.4f}, std={std:.4f}")

                elif name == 'InstanceNorm':
                    print("  Per sample per channel (averaged over H, W):")
                    for n in range(2):
                        for c in range(3):
                            mean = x_norm[n, c].mean()
                            std = x_norm[n, c].std()
                            print(f"    Sample {n}, Channel {c}: mean={mean:.4f}, std={std:.4f}")

        def test_batch_size_sensitivity(self):
            """
            배치 크기가 달라질 때 정규화 방법마다 어떻게 대응하는지 시험한다.

            핵심은 "같은 표본"을 서로 다른 크기의 배치에 넣어 보고, 그 표본의
            출력이 얼마나 달라지는지를 재는 것이다. 배치 정규화가 배치 크기에
            민감하다는 말의 뜻이 바로 이것이다. 출력의 평균이 0에서 벗어난다는
            뜻이 아니라(어느 배치 크기에서든 배치 전체의 평균은 0이 된다),
            곁에 어떤 표본이 놓이느냐에 따라 같은 표본의 출력이 달라진다는
            뜻이다.
            """
            print("\n" + "=" * 70)
            print("Batch Size Sensitivity Test")
            print("=" * 70)

            torch.manual_seed(42)

            # 이 표본 하나를 모든 배치에 똑같이 넣고, 그 출력만 지켜본다
            # 공간 크기를 4x4로 두어야 위에서 만든
            # nn.LayerNorm([3, 4, 4])이 그대로 받는다
            probe = torch.randn(1, 3, 4, 4)
            batch_sizes = [2, 4, 8, 32, 128]
            n_trials = 20

            print("\n같은 표본을 서로 다른 배치에 20번씩 넣었을 때,")
            print("그 표본의 출력이 시행마다 얼마나 달라지는가 (표준편차):\n")
            print(f"{'batch':>6s} " + "".join(f"{n:>14s}" for n in self.normalizations))

            for batch_size in batch_sizes:
                spreads = []
                for name in self.normalizations:
                    outputs = []
                    for _ in range(n_trials):
                        # 곁에 놓이는 표본만 매번 새로 뽑는다
                        others = torch.randn(batch_size - 1, 3, 4, 4)
                        batch = torch.cat([probe, others], dim=0)

                        if name == 'BatchNorm':
                            # train() 모드여야 이동 통계가 아니라 이 배치의
                            # 통계로 정규화한다. 갓 만든 레이어를 eval()로 두면
                            # running_mean=0, running_var=1이라 입력이 거의
                            # 그대로 지나가 아무것도 보이지 않는다
                            layer = nn.BatchNorm2d(3, affine=False)
                            layer.train()
                        else:
                            layer = self.normalizations[name]

                        with torch.no_grad():
                            # 배치 전체를 넣되 첫 표본의 출력만 꺼내 둔다
                            outputs.append(layer(batch)[0])

                    # 같은 표본인데 시행마다 출력이 얼마나 흩어지는가.
                    # 0이면 곁 표본에 전혀 영향받지 않는다는 뜻이다
                    spreads.append(torch.stack(outputs).std(dim=0).mean().item())

                print(f"{batch_size:6d} " + "".join(f"{s:14.4f}" for s in spreads))

            print("\nObservations:")
            print("- BatchNorm: spread shrinks as batch grows -> depends on batch companions")
            print("- LayerNorm / InstanceNorm / GroupNorm: spread is exactly 0 at every size")
            print("  (each sample is normalized on its own, so neighbours cannot matter)")


    def create_comparison_network():
        """
        비교를 위해 정규화 레이어가 서로 다른 뉴럴 넷트웍을 만든다.
        """
        print("\n" + "=" * 70)
        print("Example Networks with Different Normalizations")
        print("=" * 70)

        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, norm_type='batch'):
                super(ConvBlock, self).__init__()

                self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

                # 정규화 고르기
                if norm_type == 'batch':
                    self.norm = nn.BatchNorm2d(out_channels)
                elif norm_type == 'layer':
                    # 2차원 데이터에 LayerNorm을 쓰려면 모양을 지정해야 한다
                    # 이는 간략한 판본이다
                    # 그룹 하나로 묶으면 채널 전체를 함께 보므로 레이어 정규화가 된다.
                    # nn.LayerNorm을 쓰려면 (C, H, W)를 미리 알아야 하는데
                    # 컨볼루션에서는 입력 크기에 따라 H와 W가 달라지므로,
                    # 채널 수만 알면 되는 이쪽이 실용적이다
                    self.norm = nn.GroupNorm(1, out_channels)
                elif norm_type == 'instance':
                    # 주의: nn.InstanceNorm2d의 affine 기본값은 False다.
                    # 위의 BatchNorm2d나 GroupNorm과 달리 gamma와 beta를
                    # 배우지 않으므로, 이 비교에서 인스턴스 정규화만
                    # 학습되는 파라메터가 없다
                    self.norm = nn.InstanceNorm2d(out_channels)
                elif norm_type == 'group':
                    # 그룹 8개.
                    # out_channels가 8로 나누어떨어져야 한다. 64와 128은
                    # 괜찮지만 채널 수를 3처럼 바꾸면 오류가 난다.
                    # 8은 널리 쓰이는 어림값이며, 원 논문은 그룹당 채널
                    # 16개 언저리를 권한다
                    self.norm = nn.GroupNorm(8, out_channels)
                else:
                    self.norm = nn.Identity()

                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                # 컨볼루션 → 정규화 → 활성화. 어느 정규화를 골랐든 자리는
                # 같다. 정규화가 ReLU 앞에 오므로 활성화에 들어가는 값이
                # 0 언저리에 모이고, 바로 앞 컨볼루션의 바이어스는 곧 지워질
                # 값이라 사실상 쓸모가 없다
                x = self.conv(x)
                x = self.norm(x)
                x = self.relu(x)
                return x

        # 정규화 방식이 다른 뉴럴 넷트웍 만들기
        networks = {
            'BatchNorm': nn.Sequential(
                ConvBlock(3, 64, 'batch'),
                ConvBlock(64, 128, 'batch'),
            ),
            'InstanceNorm': nn.Sequential(
                ConvBlock(3, 64, 'instance'),
                ConvBlock(64, 128, 'instance'),
            ),
            'GroupNorm': nn.Sequential(
                ConvBlock(3, 64, 'group'),
                ConvBlock(64, 128, 'group'),
            ),
        }

        # 예시 입력으로 시험
        x = torch.randn(4, 3, 32, 32)

        print("\nTesting networks with input shape:", x.shape)

        for name, net in networks.items():
            net.eval()
            with torch.no_grad():
                out = net(x)
            print(f"{name:15s}: output shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")


    def performance_comparison():
        """
        정규화 방법들의 계산 성능을 비교한다.
        """
        print("\n" + "=" * 70)
        print("Performance Characteristics")
        print("=" * 70)

        characteristics = {
            'BatchNorm': {
                'Speed': 'Fast',
                'Memory': 'Low (stores running stats)',
                'Batch dependency': 'Yes (sensitive to batch size)',
                'Train/Eval difference': 'Yes (uses different stats)',
            },
            'LayerNorm': {
                'Speed': 'Fast',
                'Memory': 'Low',
                'Batch dependency': 'No (batch independent)',
                'Train/Eval difference': 'No (same computation)',
            },
            'InstanceNorm': {
                'Speed': 'Fast',
                'Memory': 'Low',
                'Batch dependency': 'No (batch independent)',
                'Train/Eval difference': 'No (same computation)',
            },
            'GroupNorm': {
                'Speed': 'Fast',
                'Memory': 'Low',
                'Batch dependency': 'No (batch independent)',
                'Train/Eval difference': 'No (same computation)',
            },
        }

        for norm_name, chars in characteristics.items():
            print(f"\n{norm_name}:")
            for key, value in chars.items():
                print(f"  {key:25s}: {value}")


    def practical_recommendations():
        """
        정규화 방법을 고르기 위한 실용적인 권고를 제시한다.
        """
        print("\n" + "=" * 70)
        print("Practical Recommendations")
        print("=" * 70)

        recommendations = """
        과제 종류                    | 권장 정규화       | 이유
        -----------------------------|------------------|-------------------------
        이미지 분류 (CNN)            | BatchNorm         | 큰 배치에서 잘 통한다
        물체 검출                    | GroupNorm/SyncBN  | 작은 배치에 더 낫다
        의미 분할                    | BatchNorm/GroupNorm | 배치 크기에 달렸다
        양식 전이                    | InstanceNorm      | 사례별 정보를 없앤다
        GAN (이미지 대 이미지)       | InstanceNorm      | 표본을 따로따로 처리한다
        트랜스포머 (자연어)          | LayerNorm         | 표준 선택, 배치와 무관하다
        RNN/LSTM                     | LayerNorm         | 길이가 변하는 순차열을 잘 다룬다
        온라인 학습 (batch=1)        | LayerNorm/InstanceNorm | 배치와 무관하다
        작은 배치 학습               | GroupNorm/LayerNorm | 배치 크기에 민감하지 않다
        영상 처리                    | GroupNorm         | 시간 차원을 잘 다룬다

        특수한 경우:
        - 배치가 작으면(< 8): 그룹 정규화나 레이어 정규화를 쓴다
        - 학습이 불안정하면: 그룹 정규화를 시도한다
        - 학습과 평가에서 완전히 같은 거동이 필요하면: 레이어 정규화나 사례 정규화를 쓴다
        - 다중 GPU 학습이면: SyncBatchNorm을 쓴다 (GPU 사이에서 통계를 맞춘다)
        """

        print(recommendations)


    def common_mistakes():
        """
        정규화 레이어를 쓸 때 흔한 실수를 짚는다.
        """
        print("\n" + "=" * 70)
        print("Common Mistakes to Avoid")
        print("=" * 70)

        mistakes = """
        1. model.eval() 호출을 잊는 것
           - 배치 정규화는 학습 모드와 평가 모드에서 다르게 작동한다
           - 추론 전에는 언제나 model.eval()을 부르라!

        2. 배치 크기가 1인데 배치 정규화를 쓰는 것
           - 배치 정규화는 통계를 내려면 표본이 여럿 필요하다
           - 대신 레이어 정규화나 사례 정규화를 쓰라

        3. 정규화를 활성화 앞에 두는 것
           - 표준: 컨볼루션 → 정규화 → 활성화
           - 정규화 → 컨볼루션 → 활성화가 더 낫다는 실험도 있다
           - 구조 안에서 일관되게 하라

        4. 배치 정규화의 모멘텀을 조정하지 않는 것
           - 기본 모멘텀(0.1)이 최적이 아닐 수 있다
           - 데이터셋이 작으면 더 작은 모멘텀(0.01)을 시도하라

        5. 과제에 맞지 않는 정규화를 쓰는 것
           - 양식 전이에 배치 정규화를 쓰지 마라 (사례 정규화를 쓰라)
           - 분류에 사례 정규화를 쓰지 마라 (배치 정규화를 쓰라)

        6. 배치 정규화 레이어를 잘못 얼리는 것
           - 미세 조정할 때는 배치 정규화 레이어를 조심하라
           - 평가 모드로 두거나 이동 통계를 갱신해야 할 수 있다

        7. 다중 GPU 학습을 고려하지 않는 것
           - 표준 배치 정규화는 GPU마다 따로 통계를 낸다
           - GPU 사이에서 더 좋은 결과를 얻으려면 SyncBatchNorm을 쓰라

        8. 아핀 파라메터를 무시하는 것
           - affine=True는 배율과 이동을 학습한다는 뜻이다
           - 보통 True로 두되, 순수한 정규화만 원하면 끈다
        """

        print(mistakes)


    def quick_reference():
        """
        정규화 레이어의 간단한 참고 안내.
        """
        print("\n" + "=" * 70)
        print("Quick Reference Guide")
        print("=" * 70)

        reference = """
        PyTorch 구현:

        # 배치 정규화
        nn.BatchNorm1d(num_features)      # 1차원/선형 레이어용
        nn.BatchNorm2d(num_channels)      # 2차원/컨볼루션 레이어용
        nn.BatchNorm3d(num_channels)      # 3차원 데이터용

        # 레이어 정규화
        nn.LayerNorm(normalized_shape)    # 정규화할 모양 지정
        nn.LayerNorm([C, H, W])          # 2차원 데이터용

        # 사례 정규화
        nn.InstanceNorm1d(num_features)   # 1차원 데이터용
        nn.InstanceNorm2d(num_channels)   # 2차원/이미지용
        nn.InstanceNorm3d(num_channels)   # 3차원 데이터용

        # 그룹 정규화
        nn.GroupNorm(num_groups, num_channels)  # 채널을 그룹으로 나누기

        공통 파라메터:
        - eps: 수치 안정성을 위한 작은 값 (기본값: 1e-5)
        - momentum: 배치 정규화의 이동 통계용 (기본값: 0.1)
        - affine: 학습 가능한 배율/이동 파라메터 (기본값: True)
        - track_running_stats: 배치 정규화용 (기본값: True)

        기억할 것:
        - 배치 정규화로 추론할 때는 언제나 model.eval()을 부르라
        - 레이어 정규화/사례 정규화: 학습과 평가에서 거동이 같다
        - 모드를 바꾸려면 .train()과 .eval()을 쓰라
        """

        print(reference)


    if __name__ == "__main__":
        comp = NormalizationComparison()

        # 모든 비교 실행
        comp.visualize_normalization_axes()
        comp.compare_on_sample_data()
        comp.test_batch_size_sensitivity()

        create_comparison_network()
        performance_comparison()
        practical_recommendations()
        common_mistakes()
        quick_reference()

        print("\n" + "=" * 70)
        print("For more details, see individual files:")
        print("  - batch_normalization.py")
        print("  - layer_normalization.py")
        print("  - instance_normalization.py")
        print("=" * 70)
    ```


??? note "전체 출력 (219줄)"

    ```
    ======================================================================
    Normalization Axes Visualization
    ======================================================================

    Input tensor shape: (N, C, H, W) = (Batch, Channels, Height, Width)

    Normalization axes (what dimensions are averaged over):
    ----------------------------------------------------------------------
    BatchNorm      : Axes: [0, 2, 3] → (N, H, W) | Per channel across batch
    LayerNorm      : Axes: [1, 2, 3] → (C, H, W) | Per sample across features
    InstanceNorm   : Axes: [2, 3]    → (H, W)   | Per sample per channel
    GroupNorm      : Axes: [2, 3]    → (H, W)   | Per sample per group

    ======================================================================

    ======================================================================
    Comparing Normalizations on Sample Data
    ======================================================================

    Input shape: torch.Size([2, 3, 4, 4])
    Input mean: 0.0335, std: 5.7866

    Original data statistics:
      Sample 0, Channel 0: mean= -1.367, std= 12.344
      Sample 0, Channel 1: mean=  2.864, std=  3.968
      Sample 0, Channel 2: mean= -1.329, std=  5.472
      Sample 1, Channel 0: mean= -0.039, std=  0.950
      Sample 1, Channel 1: mean=  0.101, std=  0.531
      Sample 1, Channel 2: mean= -0.029, std=  0.447

    ----------------------------------------------------------------------
    After normalization:
    ----------------------------------------------------------------------

    BatchNorm:
      Overall: mean=0.0335, std=5.7866
      Per channel (averaged over batch, H, W):
        Channel 0: mean=-0.7031, std=8.6381
        Channel 1: mean=1.4825, std=3.1184
        Channel 2: mean=-0.6789, std=3.8757

    LayerNorm:
      Overall: mean=0.0000, std=1.0052
      Per sample (averaged over C, H, W):
        Sample 0: mean=0.0000, std=1.0106
        Sample 1: mean=0.0000, std=1.0106

    InstanceNorm:
      Overall: mean=-0.0000, std=1.0052
      Per sample per channel (averaged over H, W):
        Sample 0, Channel 0: mean=0.0000, std=1.0328
        Sample 0, Channel 1: mean=-0.0000, std=1.0328
        Sample 0, Channel 2: mean=-0.0000, std=1.0328
        Sample 1, Channel 0: mean=-0.0000, std=1.0328
        Sample 1, Channel 1: mean=0.0000, std=1.0328
        Sample 1, Channel 2: mean=0.0000, std=1.0328

    GroupNorm:
      Overall: mean=0.0000, std=1.0052

    ======================================================================
    Batch Size Sensitivity Test
    ======================================================================

    같은 표본을 서로 다른 배치에 20번씩 넣었을 때,
    그 표본의 출력이 시행마다 얼마나 달라지는가 (표준편차):

     batch      BatchNorm     LayerNorm  InstanceNorm     GroupNorm
         2         0.1447        0.0000        0.0000        0.0000
         4         0.1224        0.0000        0.0000        0.0000
         8         0.1017        0.0000        0.0000        0.0000
        32         0.0531        0.0000        0.0000        0.0000
       128         0.0245        0.0000        0.0000        0.0000

    Observations:
    - BatchNorm: spread shrinks as batch grows -> depends on batch companions
    - LayerNorm / InstanceNorm / GroupNorm: spread is exactly 0 at every size
      (each sample is normalized on its own, so neighbours cannot matter)

    ======================================================================
    Example Networks with Different Normalizations
    ======================================================================

    Testing networks with input shape: torch.Size([4, 3, 32, 32])
    BatchNorm      : output shape=torch.Size([4, 128, 32, 32]), mean=0.0989, std=0.1395
    InstanceNorm   : output shape=torch.Size([4, 128, 32, 32]), mean=0.3970, std=0.5846
    GroupNorm      : output shape=torch.Size([4, 128, 32, 32]), mean=0.3955, std=0.5849

    ======================================================================
    Performance Characteristics
    ======================================================================

    BatchNorm:
      Speed                    : Fast
      Memory                   : Low (stores running stats)
      Batch dependency         : Yes (sensitive to batch size)
      Train/Eval difference    : Yes (uses different stats)

    LayerNorm:
      Speed                    : Fast
      Memory                   : Low
      Batch dependency         : No (batch independent)
      Train/Eval difference    : No (same computation)

    InstanceNorm:
      Speed                    : Fast
      Memory                   : Low
      Batch dependency         : No (batch independent)
      Train/Eval difference    : No (same computation)

    GroupNorm:
      Speed                    : Fast
      Memory                   : Low
      Batch dependency         : No (batch independent)
      Train/Eval difference    : No (same computation)

    ======================================================================
    Practical Recommendations
    ======================================================================

        과제 종류                    | 권장 정규화       | 이유
        -----------------------------|------------------|-------------------------
        이미지 분류 (CNN)            | BatchNorm         | 큰 배치에서 잘 통한다
        물체 검출                    | GroupNorm/SyncBN  | 작은 배치에 더 낫다
        의미 분할                    | BatchNorm/GroupNorm | 배치 크기에 달렸다
        양식 전이                    | InstanceNorm      | 사례별 정보를 없앤다
        GAN (이미지 대 이미지)       | InstanceNorm      | 표본을 따로따로 처리한다
        트랜스포머 (자연어)          | LayerNorm         | 표준 선택, 배치와 무관하다
        RNN/LSTM                     | LayerNorm         | 길이가 변하는 순차열을 잘 다룬다
        온라인 학습 (batch=1)        | LayerNorm/InstanceNorm | 배치와 무관하다
        작은 배치 학습               | GroupNorm/LayerNorm | 배치 크기에 민감하지 않다
        영상 처리                    | GroupNorm         | 시간 차원을 잘 다룬다
        
        특수한 경우:
        - 배치가 작으면(< 8): 그룹 정규화나 레이어 정규화를 쓴다
        - 학습이 불안정하면: 그룹 정규화를 시도한다
        - 학습과 평가에서 완전히 같은 거동이 필요하면: 레이어 정규화나 사례 정규화를 쓴다
        - 다중 GPU 학습이면: SyncBatchNorm을 쓴다 (GPU 사이에서 통계를 맞춘다)
        

    ======================================================================
    Common Mistakes to Avoid
    ======================================================================

        1. model.eval() 호출을 잊는 것
           - 배치 정규화는 학습 모드와 평가 모드에서 다르게 작동한다
           - 추론 전에는 언제나 model.eval()을 부르라!
        
        2. 배치 크기가 1인데 배치 정규화를 쓰는 것
           - 배치 정규화는 통계를 내려면 표본이 여럿 필요하다
           - 대신 레이어 정규화나 사례 정규화를 쓰라
        
        3. 정규화를 활성화 앞에 두는 것
           - 표준: 컨볼루션 → 정규화 → 활성화
           - 정규화 → 컨볼루션 → 활성화가 더 낫다는 실험도 있다
           - 구조 안에서 일관되게 하라
        
        4. 배치 정규화의 모멘텀을 조정하지 않는 것
           - 기본 모멘텀(0.1)이 최적이 아닐 수 있다
           - 데이터셋이 작으면 더 작은 모멘텀(0.01)을 시도하라
        
        5. 과제에 맞지 않는 정규화를 쓰는 것
           - 양식 전이에 배치 정규화를 쓰지 마라 (사례 정규화를 쓰라)
           - 분류에 사례 정규화를 쓰지 마라 (배치 정규화를 쓰라)
        
        6. 배치 정규화 레이어를 잘못 얼리는 것
           - 미세 조정할 때는 배치 정규화 레이어를 조심하라
           - 평가 모드로 두거나 이동 통계를 갱신해야 할 수 있다
        
        7. 다중 GPU 학습을 고려하지 않는 것
           - 표준 배치 정규화는 GPU마다 따로 통계를 낸다
           - GPU 사이에서 더 좋은 결과를 얻으려면 SyncBatchNorm을 쓰라
        
        8. 아핀 파라메터를 무시하는 것
           - affine=True는 배율과 이동을 학습한다는 뜻이다
           - 보통 True로 두되, 순수한 정규화만 원하면 끈다
        

    ======================================================================
    Quick Reference Guide
    ======================================================================

        PyTorch 구현:
        
        # 배치 정규화
        nn.BatchNorm1d(num_features)      # 1차원/선형 레이어용
        nn.BatchNorm2d(num_channels)      # 2차원/컨볼루션 레이어용
        nn.BatchNorm3d(num_channels)      # 3차원 데이터용
        
        # 레이어 정규화
        nn.LayerNorm(normalized_shape)    # 정규화할 모양 지정
        nn.LayerNorm([C, H, W])          # 2차원 데이터용
        
        # 사례 정규화
        nn.InstanceNorm1d(num_features)   # 1차원 데이터용
        nn.InstanceNorm2d(num_channels)   # 2차원/이미지용
        nn.InstanceNorm3d(num_channels)   # 3차원 데이터용
        
        # 그룹 정규화
        nn.GroupNorm(num_groups, num_channels)  # 채널을 그룹으로 나누기
        
        공통 파라메터:
        - eps: 수치 안정성을 위한 작은 값 (기본값: 1e-5)
        - momentum: 배치 정규화의 이동 통계용 (기본값: 0.1)
        - affine: 학습 가능한 배율/이동 파라메터 (기본값: True)
        - track_running_stats: 배치 정규화용 (기본값: True)
        
        기억할 것:
        - 배치 정규화로 추론할 때는 언제나 model.eval()을 부르라
        - 레이어 정규화/사례 정규화: 학습과 평가에서 거동이 같다
        - 모드를 바꾸려면 .train()과 .eval()을 쓰라
        

    ======================================================================
    For more details, see individual files:
      - batch_normalization.py
      - layer_normalization.py
      - instance_normalization.py
    ======================================================================
    ```


## 2. 논의

일반적인 NCHW 입력에서 BatchNorm2d는 채널마다 N·H·W를 사용한다. LayerNorm은 지정한 normalized_shape에 따라 달라지며, 이 예제의 `[C, H, W]` 설정은 표본 전체 피처를 사용한다. InstanceNorm은 H·W를, GroupNorm은 그룹 안의 채널과 H·W를 사용한다.

이 코드는 GroupNorm(3, 3)으로 채널마다 그룹 하나를 둔다. 따라서 affine=False인 이 설정에서는 인스턴스 정규화와 같은 통계 축이지만, 일반적인 그룹 정규화가 항상 채널별 정규화인 것은 아니다.

샘플 비교 함수는 BatchNorm을 eval 모드로 두고 학습되지 않은 초기 이동 평균 0과 분산 1을 사용한다. 따라서 이 출력은 현재 배치에서 정규화한 결과가 아니다. 뒤의 배치 민감도 실험은 새 BatchNorm을 학습 모드로 만들어 같은 표본의 출력이 동반 표본에 따라 달라지는지를 잰다. 두 실험의 모드를 구분한다.

출력 표준편차를 torch.std의 기본 불편 보정으로 재면 정규화에 쓴 분모와 다를 수 있다. epsilon과 아핀 설정도 있어 출력 표준편차가 항상 정확히 1이어야 하는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
그룹 정규화의 일반적인 통계 축과 이 코드의 특수 설정을 설명하라.

</div>

??? success "풀이"
    표본마다 채널을 G개 그룹으로 나눈 뒤 그룹 안 채널과 공간 축을 함께 사용한다. 여기서는 G=C=3이어서 각 그룹의 채널이 하나이므로 공간 축만 남는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
같은 표본을 다른 배치에 섞어 비교하는 실험이 배치 의존성을 보여 주는 이유는 무엇인가?

</div>

??? success "풀이"
    대상 표본을 고정하고 동반 표본만 바꾸므로 출력 변화가 다른 표본의 통계 영향인지 볼 수 있다. BatchNorm은 학습 모드에서 배치 통계를 쓰지만 나머지 설정은 표본별 통계를 사용한다. 평가 모드의 고정 이동 통계량은 다른 조건이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
NormalizationComparison을 nn.Module처럼 사용하면 안 되는 이유를 설명하라.

</div>

??? success "풀이"
    이 클래스는 일반 측정 도우미로 forward와 parameters를 정의하지 않는다. 실제 정규화 모듈은 딕셔너리에 저장되어 있으며 각 모듈을 직접 호출한다. 저장·장치 이동을 nn.Module 등록처럼 자동 처리한다고 가정하면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

정규화 비교는 이름보다 통계 축, 아핀 계수, 이동 통계량과 모드를 기준으로 한다. 동일 입력에서도 설정이 다르면 출력이 달라지므로 실험 조건을 함께 기록한다.
