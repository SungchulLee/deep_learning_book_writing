# ResNet 구현

BasicBlock과 Bottleneck을 네 단계로 쌓아 ResNet-18·34·50·101·152를 구성한다. _make_layer는 단계 시작에서 채널과 공간 크기를 맞추고 필요한 투영 지름길을 만든다. 블록 하나에 분류 헤드를 연결할 수도 있지만, 이 예제는 여러 단계의 피처를 전역 풀링과 완전 연결 레이어로 분류한다.

기록된 파라메터 수는 1000클래스 설정의 ResNet-18에서 11689512개, ResNet-50에서 25557032개이다. 구조와 계수 계산을 확인하는 예제가며 사전 학습 웨이트나 학습 후 정확도를 제공하는 실험은 아니다.

## 1. 코드

```python
"""
완전한 ResNet 구현
===============================
ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152를 모두 구현
바탕: "Deep Residual Learning for Image Recognition" (He 등, 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class BasicBlock(nn.Module):
    """
    기본 잔차 블록 (ResNet-18과 ResNet-34에서 쓴다)
    건너뛰기 연결이 있는 3x3 컨볼루션 두 개

    지름길은 downsample=None이면 손대지 않은 x이고,
    아니면 1x1 컨볼루션 + BN으로 모양을 맞춘 사영이다.
    """
    expansion = 1  # 블록의 출력 채널 = out_channels * expansion

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 뒤에 BN이 오므로 bias=False (BN이 평균을 빼 치우침을 지운다)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    병목 블록 (ResNet-50, ResNet-101, ResNet-152에서 쓴다)
    컨볼루션 세 개: 1x1, 3x3, 1x1 (채널을 줄였다가 늘린다)
    3x3 컨볼루션을 좁은 채널에서만 돌리므로 파라메터가 적게 든다
    """
    expansion = 4  # 블록의 출력 채널 = out_channels * 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 차원을 줄이는 1x1 컨볼루션
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 컨볼루션 (주된 계산, stride는 여기서 준다)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 차원을 늘리는 1x1 컨볼루션
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 줄이기
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 3x3 컨볼루션
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # 늘리기
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 구조

    인수:
        block: BasicBlock 또는 Bottleneck
        layers: 네 단계마다의 블록 수를 담은 리스트
        num_classes: 출력 부류의 수
        in_channels: 입력 채널의 수 (RGB 이미지는 3)
    """

    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()

        # 다음 단계의 입력 채널 수. _make_layer가 단계마다 갱신한다.
        self.in_channels = 64

        # 첫 컨볼루션 (7x7 컨볼루션, 스트라이드 2)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 잔차 단계 넷 (첫 단계만 스트라이드 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 전역 평균 풀링과 완전 연결 레이어
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 웨이트 초기화
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        잔차 블록 여러 개로 한 단계를 만든다

        사영 지름길은 모양이 어긋나는 첫 블록에만 넘어간다.
        둘째 블록부터는 입출력 모양이 같으므로 downsample=None이고,
        지름길이 손대지 않은 항등이 된다.
        """
        downsample = None

        # 스트라이드가 1이 아니거나 채널 수가 바뀌면 지름길을 사영으로 바꾼다
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # 첫 블록만 스트라이드와 사영 지름길을 받는다
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        # 뒤따르는 블록을 위해 in_channels 갱신
        self.in_channels = out_channels * block.expansion

        # 남은 블록들 (스트라이드 1, 지름길은 항등)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        카이밍 초기화로 웨이트를 초기화한다
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 처음 레이어들
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 잔차 단계들
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 분류 머리
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000, in_channels=3):
    """ResNet-18: 기본 블록 [2, 2, 2, 2]"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes=1000, in_channels=3):
    """ResNet-34: 기본 블록 [3, 4, 6, 3]"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50(num_classes=1000, in_channels=3):
    """ResNet-50: 병목 블록 [3, 4, 6, 3]"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def resnet101(num_classes=1000, in_channels=3):
    """ResNet-101: 병목 블록 [3, 4, 23, 3]"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def resnet152(num_classes=1000, in_channels=3):
    """ResNet-152: 병목 블록 [3, 8, 36, 3]"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


BUILDERS = [
    ("ResNet-18", resnet18, [2, 2, 2, 2]),
    ("ResNet-34", resnet34, [3, 4, 6, 3]),
    ("ResNet-50", resnet50, [3, 4, 6, 3]),
    ("ResNet-101", resnet101, [3, 4, 23, 3]),
    ("ResNet-152", resnet152, [3, 8, 36, 3]),
]

# ========================================================================
# 파라메터와 계산량 세기
# ========================================================================


def count_params(model):
    """모델의 파라메터 개수"""
    return sum(p.numel() for p in model.parameters())


def compare_models():
    """
    다섯 모델의 파라메터를 1000 부류와 10 부류에서 견준다.
    1000 부류의 수는 논문과 torchvision이 싣는 공식 값과 맞아야 한다.
    """
    print("=" * 76)
    print("Parameter counts")
    print("=" * 76)
    print(f"\n{'model':<12}{'1000-class':>16}{'10-class':>16}{'difference':>16}")
    print("-" * 76)
    for name, builder, _ in BUILDERS:
        n1000 = count_params(builder(num_classes=1000))
        n10 = count_params(builder(num_classes=10))
        print(f"{name:<12}{n1000:>16,}{n10:>16,}{n1000 - n10:>16,}")

    print("\nThe difference is the classifier alone: 990 extra rows of")
    print("512*expansion weights plus 990 biases.")
    print(f"  ResNet-18  990 * 512  + 990 = {990 * 512 + 990:,}")
    print(f"  ResNet-50  990 * 2048 + 990 = {990 * 2048 + 990:,}")
    print("=" * 76)


def count_depth():
    """
    이름에 붙은 수가 어디서 오는지 센다.
    주 경로의 컨볼루션 + 완전 연결 레이어 하나가 그 수와 같아야 한다.
    1x1 사영 지름길은 세지 않는다.
    """
    print("\n" + "=" * 76)
    print("Where the number in the name comes from")
    print("=" * 76)
    print(f"\n{'model':<12}{'blocks':>8}{'main conv':>11}{'fc':>5}"
          f"{'sum':>6}{'proj conv':>11}{'all conv':>10}")
    print("-" * 76)
    for name, builder, layers in BUILDERS:
        model = builder(num_classes=1000)
        convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        main = [n for n in convs if "downsample" not in n]
        fcs = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        print(f"{name:<12}{sum(layers):>8}{len(main):>11}{len(fcs):>5}"
              f"{len(main) + len(fcs):>6}{len(convs) - len(main):>11}{len(convs):>10}")
    print("\n'sum' matches the number in the name for every row.")
    print("The projection convolutions are not counted in that number.")
    print("=" * 76)


def compare_block_cost():
    """
    같은 256 -> 256 사상을 기본 블록과 병목 블록으로 각각 만들어 견준다
    """
    print("\n" + "=" * 76)
    print("Basic block vs bottleneck at 256 -> 256 channels")
    print("=" * 76)

    basic = BasicBlock(256, 256)
    proj = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                         nn.BatchNorm2d(256))
    bottle = Bottleneck(256, 64, downsample=None)

    nb = count_params(basic)
    nt = count_params(bottle)
    print(f"\n  BasicBlock(256, 256)   {nb:>10,}   (18*256^2 + 4*256 = "
          f"{18 * 256 ** 2 + 4 * 256:,})")
    print(f"  Bottleneck(256, 64)    {nt:>10,}   ratio {nb / nt:.2f}x")
    print(f"    conv1 1x1 256->64  {bottle.conv1.weight.numel():>8,}")
    print(f"    conv2 3x3  64->64  {bottle.conv2.weight.numel():>8,}")
    print(f"    conv3 1x1 64->256  {bottle.conv3.weight.numel():>8,}")
    bn = sum(p.numel() for p in
             list(bottle.bn1.parameters()) + list(bottle.bn2.parameters())
             + list(bottle.bn3.parameters()))
    print(f"    bn1+bn2+bn3        {bn:>8,}")
    print("  Both map (N, 256, H, W) to (N, 256, H, W).")
    print(f"  A 1x1 projection shortcut at 256 channels costs "
          f"{count_params(proj):,} by itself,")
    print("  which is almost the whole bottleneck block.")
    print("=" * 76)


def stage_macs(model, size=224):
    """
    단계마다 컨볼루션의 곱셈-덧셈 횟수를 센다.
    컨볼루션 하나의 횟수는 웨이트 개수 x 출력 격자의 칸 수다.
    """
    counts = {}
    handles = []

    def make_hook(stage):
        def hook(module, inputs, output):
            cost = module.weight.numel() * output.shape[-1] * output.shape[-2]
            counts[stage] = counts.get(stage, 0) + cost
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name.split(".")[0])))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in handles:
        h.remove()
    return counts


def compare_compute():
    """
    파라메터는 뒤쪽 단계에 몰리는데 계산은 그렇지 않음을 보인다
    """
    print("\n" + "=" * 76)
    print("Parameters vs multiply-adds, by stage (224x224 input)")
    print("=" * 76)

    for name, builder, _ in [BUILDERS[0], BUILDERS[2]]:
        model = builder(num_classes=1000)
        macs = stage_macs(model)
        total_p = count_params(model)
        total_m = sum(macs.values())
        print(f"\n  {name}   {total_p:,} params, {total_m / 1e9:.2f}G multiply-adds")
        print(f"  {'stage':<10}{'params':>14}{'% of params':>13}"
              f"{'MACs':>12}{'% of MACs':>11}")
        for stage in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            p = sum(q.numel() for q in getattr(model, stage).parameters())
            m = macs[stage]
            print(f"  {stage:<10}{p:>14,}{100 * p / total_p:>12.1f}%"
                  f"{m / 1e9:>11.3f}G{100 * m / total_m:>10.1f}%")

    print("\nParameters pile up in the last stage; the multiply-adds do not,")
    print("because each stage halves the grid while doubling the channels.")
    print("=" * 76)


# ========================================================================
# 모양과 지름길 확인
# ========================================================================


def trace_stages(model, name, input_size=(3, 224, 224)):
    """
    단계마다 텐서의 모양과 사영 지름길이 붙은 자리를 찍는다
    """
    print("\n" + "=" * 76)
    print(f"Stage-by-stage trace: {name}")
    print("=" * 76)

    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, *input_size)

    with torch.no_grad():
        h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        print(f"\n  input                {tuple(x.shape)}")
        print(f"  after stem           {tuple(h.shape)}")
        for i in range(1, 5):
            stage = getattr(model, f"layer{i}")
            h = stage(h)
            flags = ["proj" if b.downsample is not None else "id" for b in stage]
            params = sum(p.numel() for p in stage.parameters())
            total = count_params(model)
            print(f"  after layer{i}         {tuple(h.shape)}"
                  f"   shortcuts {flags}")
            print(f"    params {params:>12,}  ({100 * params / total:4.1f}% of model)")
        h = torch.flatten(model.avgpool(h), 1)
        print(f"  after avgpool        {tuple(h.shape)}")
        print(f"  after fc             {tuple(model.fc(h).shape)}")

    ds = sum(p.numel() for n, p in model.named_parameters() if "downsample" in n)
    print(f"\n  projection shortcuts in total: {ds:,} "
          f"({100 * ds / count_params(model):.2f}% of the model)")
    print("=" * 76)


def check_shortcut_is_identity():
    """
    지름길이 정말 손대지 않은 x인지 확인한다.
    마지막 BN의 크기와 치우침을 0으로 두면 F(x) = 0이므로
    블록의 출력은 ReLU(x)와 정확히 같아야 한다.
    """
    print("\n" + "=" * 76)
    print("Is the shortcut the untouched input?")
    print("=" * 76)

    torch.manual_seed(0)
    basic = BasicBlock(64, 64).eval()
    with torch.no_grad():
        basic.bn2.weight.zero_()
        basic.bn2.bias.zero_()
        xb = torch.randn(2, 64, 8, 8)
        yb = basic(xb)

    bottle = Bottleneck(256, 64).eval()
    with torch.no_grad():
        bottle.bn3.weight.zero_()
        bottle.bn3.bias.zero_()
        xt = torch.randn(2, 256, 8, 8)
        yt = bottle(xt)

    for tag, x, y in [("BasicBlock", xb, yb), ("Bottleneck", xt, yt)]:
        print(f"\n  {tag} with F(x) = 0")
        print(f"    max |y - relu(x)| = {(y - torch.relu(x)).abs().max().item():.2e}")
        print(f"    max |y - x|       = {(y - x).abs().max().item():.2e}")
        print(f"    negative entries of x: {100 * (x < 0).float().mean().item():.2f}%")

    print("\n  y equals relu(x) exactly, so nothing touches the shortcut branch.")
    print("  It is not equal to x because the last ReLU sits after the addition.")
    print("=" * 76)


def model_summary(model, name, input_size=(3, 224, 224)):
    """
    파라메터 개수와 순전파 모양으로 모델 요약을 출력한다
    """
    print("\n" + "=" * 76)
    print(f"Model summary: {name}")
    print("=" * 76)

    total = count_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    torch.manual_seed(0)
    x = torch.randn(2, *input_size)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"\n  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print("=" * 76)


if __name__ == "__main__":
    print("\n" + "=" * 76)
    print("RESNET IMPLEMENTATION")
    print("=" * 76)

    compare_models()
    count_depth()
    compare_block_cost()

    trace_stages(resnet18(num_classes=1000), "ResNet-18")
    trace_stages(resnet50(num_classes=1000), "ResNet-50")
    compare_compute()

    check_shortcut_is_identity()

    # 웨이트까지 재현되도록 모델을 만들기 직전에 씨앗을 심는다
    torch.manual_seed(0)
    model_summary(resnet18(num_classes=10), "ResNet-18, 10-class head")

    print("\n" + "=" * 76)
    print("Next: 03_training_comparison.md trains these against plain networks")
    print("=" * 76 + "\n")
```

**출력:**

```
============================================================================
RESNET IMPLEMENTATION
============================================================================
============================================================================
Parameter counts
============================================================================

model             1000-class        10-class      difference
----------------------------------------------------------------------------
ResNet-18         11,689,512      11,181,642         507,870
ResNet-34         21,797,672      21,289,802         507,870
ResNet-50         25,557,032      23,528,522       2,028,510
ResNet-101        44,549,160      42,520,650       2,028,510
ResNet-152        60,192,808      58,164,298       2,028,510

The difference is the classifier alone: 990 extra rows of
512*expansion weights plus 990 biases.
  ResNet-18  990 * 512  + 990 = 507,870
  ResNet-50  990 * 2048 + 990 = 2,028,510
============================================================================

============================================================================
Where the number in the name comes from
============================================================================

model         blocks  main conv   fc   sum  proj conv  all conv
----------------------------------------------------------------------------
ResNet-18          8         17    1    18          3        20
ResNet-34         16         33    1    34          3        36
ResNet-50         16         49    1    50          4        53
ResNet-101        33        100    1   101          4       104
ResNet-152        50        151    1   152          4       155

'sum' matches the number in the name for every row.
The projection convolutions are not counted in that number.
============================================================================

============================================================================
Basic block vs bottleneck at 256 -> 256 channels
============================================================================

  BasicBlock(256, 256)    1,180,672   (18*256^2 + 4*256 = 1,180,672)
  Bottleneck(256, 64)        70,400   ratio 16.77x
    conv1 1x1 256->64    16,384
    conv2 3x3  64->64    36,864
    conv3 1x1 64->256    16,384
    bn1+bn2+bn3             768
  Both map (N, 256, H, W) to (N, 256, H, W).
  A 1x1 projection shortcut at 256 channels costs 66,048 by itself,
  which is almost the whole bottleneck block.
============================================================================

============================================================================
Stage-by-stage trace: ResNet-18
============================================================================

  input                (1, 3, 224, 224)
  after stem           (1, 64, 56, 56)
  after layer1         (1, 64, 56, 56)   shortcuts ['id', 'id']
    params      147,968  ( 1.3% of model)
  after layer2         (1, 128, 28, 28)   shortcuts ['proj', 'id']
    params      525,568  ( 4.5% of model)
  after layer3         (1, 256, 14, 14)   shortcuts ['proj', 'id']
    params    2,099,712  (18.0% of model)
  after layer4         (1, 512, 7, 7)   shortcuts ['proj', 'id']
    params    8,393,728  (71.8% of model)
  after avgpool        (1, 512)
  after fc             (1, 1000)

  projection shortcuts in total: 173,824 (1.49% of the model)
============================================================================

============================================================================
Stage-by-stage trace: ResNet-50
============================================================================

  input                (1, 3, 224, 224)
  after stem           (1, 64, 56, 56)
  after layer1         (1, 256, 56, 56)   shortcuts ['proj', 'id', 'id']
    params      215,808  ( 0.8% of model)
  after layer2         (1, 512, 28, 28)   shortcuts ['proj', 'id', 'id', 'id']
    params    1,219,584  ( 4.8% of model)
  after layer3         (1, 1024, 14, 14)   shortcuts ['proj', 'id', 'id', 'id', 'id', 'id']
    params    7,098,368  (27.8% of model)
  after layer4         (1, 2048, 7, 7)   shortcuts ['proj', 'id', 'id']
    params   14,964,736  (58.6% of model)
  after avgpool        (1, 2048)
  after fc             (1, 1000)

  projection shortcuts in total: 2,776,576 (10.86% of the model)
============================================================================

============================================================================
Parameters vs multiply-adds, by stage (224x224 input)
============================================================================

  ResNet-18   11,689,512 params, 1.81G multiply-adds
  stage             params  % of params        MACs  % of MACs
  conv1              9,408         0.1%      0.118G       6.5%
  layer1           147,968         1.3%      0.462G      25.5%
  layer2           525,568         4.5%      0.411G      22.7%
  layer3         2,099,712        18.0%      0.411G      22.7%
  layer4         8,393,728        71.8%      0.411G      22.7%

  ResNet-50   25,557,032 params, 4.09G multiply-adds
  stage             params  % of params        MACs  % of MACs
  conv1              9,408         0.0%      0.118G       2.9%
  layer1           215,808         0.8%      0.668G      16.3%
  layer2         1,219,584         4.8%      1.028G      25.1%
  layer3         7,098,368        27.8%      1.464G      35.8%
  layer4        14,964,736        58.6%      0.809G      19.8%

Parameters pile up in the last stage; the multiply-adds do not,
because each stage halves the grid while doubling the channels.
============================================================================

============================================================================
Is the shortcut the untouched input?
============================================================================

  BasicBlock with F(x) = 0
    max |y - relu(x)| = 0.00e+00
    max |y - x|       = 3.84e+00
    negative entries of x: 49.48%

  Bottleneck with F(x) = 0
    max |y - relu(x)| = 0.00e+00
    max |y - x|       = 4.14e+00
    negative entries of x: 50.45%

  y equals relu(x) exactly, so nothing touches the shortcut branch.
  It is not equal to x because the last ReLU sits after the addition.
============================================================================

============================================================================
Model summary: ResNet-18, 10-class head
============================================================================

  Total parameters:     11,181,642
  Trainable parameters: 11,181,642

  Input shape:  (2, 3, 224, 224)
  Output shape: (2, 10)
  Output range: [-2.4744, 2.7939]
============================================================================

============================================================================
Next: 03_training_comparison.md trains these against plain networks
============================================================================
```

## 2. 논의

**단계 구성**: 단계의 첫 블록에서만 이전 단계와 채널·해상도가 달라질 수 있다. _make_layer는 실제 모양 변경이 필요한 경우에만 투영을 만들고 이후 블록은 downsample=None으로 둔다. 첫 단계의 첫 블록도 입출력 채널이 같으면 투영이 필요하지 않다.

```python
layers.append(block(self.in_channels, out_channels, stride, downsample))
self.in_channels = out_channels * block.expansion
for _ in range(1, blocks):
    layers.append(block(self.in_channels, out_channels))
```

출력의 단계별 추적이 이것을 그대로 보인다. ResNet-18의 `layer2`는 `['proj', 'id']`, `layer3`과 `layer4`도 마찬가지다. 투영이 붙은 블록은 뉴럴 넷트웍 전체에서 셋뿐이며, 그 파라메터는 173,824개로 모델의 1.49%다. 앞 페이지의 `BasicBlock`이 지름길을 스스로 만들어 `self.shortcut`에 담았던 것과 달리 여기서는 바깥에서 `downsample` 인자로 받는데, 만드는 자리만 다를 뿐 계산하는 함수는 같다.

**스트라이드가 1인데도 투영이 필요한 자리.** 투영을 만드는 조건은 `stride != 1 or self.in_channels != out_channels * block.expansion`이고, 두 번째 항이 없으면 안 된다. ResNet-50의 `layer1`은 스트라이드가 1이지만 출력의 추적이 `['proj', 'id', 'id']`를 찍는다. 스템가 내놓는 채널은 64인데 병목 블록의 출력 채널은 $64 \times 4 = 256$이라 채널 축에서 어긋나기 때문이다. 채널 조건을 빼고 돌리면 다음이 난다.

```
RuntimeError: The size of tensor a (256) must match the size of tensor b (64)
at non-singleton dimension 1
```

256은 주 경로의 채널 수, 64는 손대지 않은 입력의 채널 수다. 차원 1이 채널 축이다. 앞 페이지 연습문제 3의 오류가 차원 3(너비)에서 났던 것과 비교하여 보라. 같은 더하기가 공간 크기로도, 채널 수로도 어긋날 수 있다.

**이름에 붙은 수.** 출력의 둘째 표는 모델마다 컨볼루션과 완전 연결 레이어를 세어 본 것이다. 주 경로의 컨볼루션에 완전 연결 레이어 하나를 더하면 이름의 수와 정확히 맞는다.

| 모델 | 블록 | 주 경로 컨볼루션 | 완전 연결 | 합 | 투영 컨볼루션 |
|---|---|---|---|---|---|
| ResNet-18 | 8 | 17 | 1 | 18 | 3 |
| ResNet-34 | 16 | 33 | 1 | 34 | 3 |
| ResNet-50 | 16 | 49 | 1 | 50 | 4 |
| ResNet-101 | 33 | 100 | 1 | 101 | 4 |
| ResNet-152 | 50 | 151 | 1 | 152 | 4 |

ResNet-18은 스템의 7×7 컨볼루션 하나에 블록 8개의 3×3 컨볼루션 16개, 그리고 완전 연결 레이어 하나다. ResNet-50은 스템 하나에 블록 16개 × 컨볼루션 3개 = 48개, 그리고 완전 연결 레이어 하나다. **투영 지름길의 1×1 컨볼루션은 이 수에 들어가지 않는다.** 그래서 `all conv` 열은 언제나 이름의 수보다 크다. 최대 풀링과 평균 풀링은 파라메터가 없으므로 애초에 세지 않는다.

**병목 블록이 아끼는 것.** 채널이 $4C$인 텐서를 다시 $4C$로 보내는 일을 두 가지로 할 수 있다. 기본 블록으로 하면 3×3 컨볼루션 두 개를 $4C$ 폭에서 돌리므로, 앞 페이지의 식 $18C'^2 + 4C'$에 $C' = 4C$를 넣어

$$18(4C)^2 + 4(4C) = 288C^2 + 16C$$

개가 든다. 병목 블록은 1×1로 $4C \to C$로 줄이고, 3×3을 좁은 $C$ 폭에서만 돌리고, 1×1로 $C \to 4C$로 되돌린다. 컨볼루션 세 개가 $4C^2 + 9C^2 + 4C^2 = 17C^2$개, 배치 정규화 셋이 $2C + 2C + 8C = 12C$개이므로

$$17C^2 + 12C$$

C=64이면 기본 블록 1180672개, 병목 블록 70400개로 비는 약 16.77이다. 이 비교는 동일한 외부 폭 4C를 유지하고 투영이 없는 두 블록을 비교한 것이다. 병목이 임의의 기본 블록과 같은 함수를 더 적은 계수로 항상 표현한다는 뜻은 아니다. 추가 투영의 비용도 별도로 계산한다.

**계수와 연산량의 분포**: stage_macs는 배치 1에서 컨볼루션 웨이트 수×출력 위치 수를 MAC으로 센다. BatchNorm·ReLU·풀링·완전 연결 레이어의 연산은 포함하지 않는다. ResNet-18의 뒤쪽 단계에 파라메터가 집중되지만 컨볼루션 MAC은 비교적 비슷한 비중을 갖는다. 채널 증가와 공간 크기 감소의 상쇄가 한 이유이며 블록 수가 다르면 비중도 달라진다. 실제 시간을 줄일 위치는 이 표만으로 정하지 않고 장치의 연산·메모리 병목을 측정해야 한다.

투영 지름길의 몫도 두 모델이 크게 다르다. ResNet-18은 1.49%인데 ResNet-50은 10.86%다. 병목의 투영이 $4C$ 폭으로 넓어진 데다 단계마다 하나씩 넷이 있기 때문이다.

**지름길이 정말 손대지 않은 입력인가.** 잔차 뉴럴 넷트웍의 주장은 지름길이 항등이라는 데 기대고 있으므로, 구현이 지름길에 무언가를 걸고 있지 않은지 재어 보는 편이 낫다. 마지막 배치 정규화의 크기와 바이어스를 0으로 두면 $F(x) = 0$이 되고, 이때 블록의 출력은 $\operatorname{ReLU}(0 + x)$여야 한다. 기본 블록과 병목 블록 모두에서 출력이 찍는 값은

$$\max_i \lvert y_i - \operatorname{ReLU}(x)_i \rvert = 0$$

기록된 검사는 차원을 유지하는 블록에서 마지막 BatchNorm의 스케일·바이어스를 0으로 만든 경우이다. 구조적으로 잔차가 0이고 지름길이 항등이면 출력은 ReLU(x)이다. 출력이 0.00e+00으로 표시된다는 사실만으로 모든 블록의 항등성을 증명하는 것은 아니다. 투영이 있는 블록이나 사전 활성화 블록에는 각각의 규약을 적용한다.

**클래스 수를 바꾸면 달라지는 것.** 첫 표의 두 열은 같은 뉴럴 넷트웍에 출력 클래스만 1000개와 10개로 달리 준 것이다. 차이는 완전 연결 레이어 하나가 전부다. ResNet-18은 $990 \times 512 + 990 = 507{,}870$개, ResNet-50은 $990 \times 2048 + 990 = 2{,}028{,}510$개이고, 출력의 두 줄이 이 뺄셈을 그대로 찍는다. 다만 클래스를 10개로 줄였다고 이것이 CIFAR용 ResNet이 되지는 않는다. 스템가 여전히 7×7 스트라이드 2 컨볼루션과 최대 풀링이라 $32 \times 32$ 입력을 잔차 단계가 시작하기도 전에 $8 \times 8$로 깎아 놓기 때문이다. 논문이 CIFAR-10에 쓴 ResNet-20/32/44/56/110은 3×3 스템에 단계가 셋뿐인 별개의 계열이다(연습문제 4).

**다음 쪽으로.** 여기까지는 구조를 세우고 세어 본 것일 뿐, 이 뉴럴 넷트웍이 정말 더 잘 배우는지는 재지 않았다. 같은 깊이의 평범한 뉴럴 넷트웍과 학습 곡선을 비교하는 일은 [학습 비교](03_training_comparison.md)에서 이어 본다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
ResNet-18과 ResNet-50에서 출력 클래스를 1000개에서 10개로 줄이면 파라메터가 각각 몇 개 줄어드는가? 손으로 셈한 뒤 `sum(p.numel() for p in m.parameters())`로 확인하라.

</div>

??? success "연습문제 1 풀이"
    줄어드는 것은 `fc` 하나뿐이다. `nn.Linear(in_features, num_classes)`는 `in_features` $\times$ `num_classes`개의 웨이트와 `num_classes`개의 바이어스를 갖는다. 클래스가 1000개에서 10개로 줄면 990줄이 사라지므로 990 $\times$ `in_features` $+ \; 990$개가 준다. `fc`의 입력 차원은 $512 \times \text{expansion}$이므로 ResNet-18은 512, ResNet-50은 2048이다.

    - ResNet-18: $990 \times 512 + 990 = 506{,}880 + 990 = 507{,}870$
    - ResNet-50: $990 \times 2048 + 990 = 2{,}027{,}520 + 990 = 2{,}028{,}510$

    ```python
    for f in (resnet18, resnet50):
        a = sum(p.numel() for p in f(num_classes=1000).parameters())
        b = sum(p.numel() for p in f(num_classes=10).parameters())
        print(a, b, a - b)
    # 11689512 11181642 507870
    # 25557032 23528522 2028510
    ```

    ResNet-34는 ResNet-18과, ResNet-101·ResNet-152는 ResNet-50과 차이가 같다. 블록의 종류가 같으면 `fc`의 입력 차원도 같기 때문이다. 출력의 `difference` 열이 이 두 값만 갖는 까닭이 그것이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
ResNet-50의 `layer1`은 스트라이드가 1인데도 첫 블록에 투영 지름길이 붙는다. 왜인가? `_make_layer`의 조건에서 `self.in_channels != out_channels * block.expansion`를 빼고 `resnet50()`에 $224 \times 224$ 이미지를 넣으면 무엇이 나는가?

</div>

??? success "연습문제 2 풀이"
    `layer1`에 들어올 때 `self.in_channels`는 스템가 내놓는 64다. 그런데 병목 블록의 출력 채널은 `out_channels * expansion` $= 64 \times 4 = 256$이다. 스트라이드가 1이라 공간 크기는 맞지만 채널 수가 64와 256으로 어긋나므로 더할 수가 없다.

    채널 조건을 빼면 `layer1`의 첫 블록이 `downsample=None`이 되어 다음이 난다.

    ```
    RuntimeError: The size of tensor a (256) must match the size of tensor b (64)
    at non-singleton dimension 1
    ```

    NCHW 텐서의 차원 1은 채널 축이다. layer1에서 기본 블록의 expansion은 1이라 64채널을 유지하지만 병목은 4라 256채널을 출력한다. 이 조건은 특정 블록 이름 때문이 아니라 실제 입력·출력 채널 수를 맞추기 위해 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`_make_layer`를 고쳐 **모든** 블록에 투영 지름길을 붙이면 파라메터가 얼마나 늘어나는가? ResNet-18과 ResNet-50에서 각각 재고, 두 값의 차이가 왜 그렇게 큰지 설명하라.

</div>

??? success "연습문제 3 풀이"
    `_make_layer`를 블록마다 투영을 만들도록 바꾼다(첫 블록만 스트라이드를 받는다).

    ```python
    class ResNetAllProj(ResNet):
        def _make_layer(self, block, out_channels, blocks, stride=1):
            layers = []
            for i in range(blocks):
                s = stride if i == 0 else 1
                ds = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion,
                              kernel_size=1, stride=s, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion))
                layers.append(block(self.in_channels, out_channels, s, ds))
                self.in_channels = out_channels * block.expansion
            return nn.Sequential(*layers)
    ```

    재어 보면 다음과 같다.

    ```
    ResNet-18  11,689,512 -> 12,043,816   extra    354,304  (3.0%)
    ResNet-50  25,557,032 -> 40,128,552   extra 14,571,520  (57.0%)
    ```

    차이가 큰 까닭은 두 가지다. 첫째, 투영은 1×1 컨볼루션이라 파라메터가 $C_{\text{in}} C_{\text{out}}$개인데, 병목을 쓰는 단계에서는 그 폭이 $4C$라 기본 블록의 네 배 제곱, 즉 열여섯 배다. 둘째, ResNet-50은 뒤쪽 단계에 블록이 더 많다. 예컨대 `layer3`의 블록 여섯 개 가운데 첫째는 이미 투영을 갖고 있으므로 다섯 개가 새로 $1024 \times 1024$ 투영을 얻는데, 그것만으로 $5 \times (1024^2 + 2048) = 5{,}253{,}120$개가 더 든다.

    지름길을 모두 투영으로 바꾸는 것은 파라메터뿐 아니라 학습 경로도 바꾸는 설계이다. 논문의 A·B·C 비교는 각각 차원 변경 시 제로 패딩, 필요한 경우 투영, 모든 블록에 투영을 사용하는 조건을 구분한다. 이 구현은 필요한 경우만 투영하는 형태이며 다른 조건에서의 우열과 비용은 따로 평가한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
`resnet18(num_classes=10)`에 $3 \times 32 \times 32$ 짜리 CIFAR 이미지를 넣어 단계마다의 모양을 추적하라. 오류 없이 돌아가는가? 그래도 이것을 CIFAR용 ResNet이라 부르면 안 되는 까닭은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    아래 예의 배치와 평가 조건에서는 AdaptiveAvgPool2d가 마지막 공간 크기를 1×1로 만들어 분류기 차원을 맞춘다. 그러나 모든 작은 입력과 학습 모드에서 동작을 보장하는 것은 아니다. 예를 들어 학습 모드 BatchNorm에서 채널당 관측값이 하나뿐인 배치는 오류를 낼 수 있다.

    ```
    입력      (1, 3, 32, 32)
    줄기 뒤   (1, 64, 8, 8)
    layer1    (1, 64, 8, 8)
    layer2    (1, 128, 4, 4)
    layer3    (1, 256, 2, 2)
    layer4    (1, 512, 1, 1)
    출력      (1, 10)
    ```

    문제는 스템다. 7×7 스트라이드 2 컨볼루션이 $32 \to 16$, 최대 풀링이 $16 \to 8$로 줄여 버려 잔차 단계들이 시작하기도 전에 한 변이 4분의 1이 된다. $224 \times 224$에서 스템 뒤가 $56 \times 56$이었던 것과 비교하여 보라. 그래서 `layer4`는 $1 \times 1$ 격자 위에서 3×3 컨볼루션을 돌리게 되는데, 패딩 덕분에 값이 나오기는 하지만 이웃을 볼 것이 없다.

    논문이 CIFAR-10에 쓴 것은 다른 계열이다. 3×3 스트라이드 1 컨볼루션으로 시작해 최대 풀링을 두지 않고, 단계를 넷이 아니라 셋($32 \times 32$, $16 \times 16$, $8 \times 8$)만 두며, 채널도 16·32·64로 훨씬 좁다. ResNet-20/32/44/56/110이 그것이고, 이름의 수는 단계마다 블록을 $n$개씩 둘 때 $6n + 2$로 정해진다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
채널이 $4C$인 텐서를 $4C$로 보내는 블록의 파라메터를 기본 블록과 병목 블록에 대해 각각 $C$의 식으로 유도하라(모든 컨볼루션은 `bias=False`, 배치 정규화는 채널마다 파라메터 둘). $C \to \infty$에서 비는 얼마로 가는가? $C = 64$에서 식과 실제 값이 맞는지 확인하라.

</div>

??? success "연습문제 5 풀이"
    **기본 블록.** 폭 $4C$에서 3×3 컨볼루션 두 개와 배치 정규화 두 개를 쓰므로, 앞 페이지의 식 $18C'^2 + 4C'$에 $C' = 4C$를 넣어

    $$18(4C)^2 + 4(4C) = 288C^2 + 16C$$

    **병목 블록.** 컨볼루션 셋은 $1 \cdot 4C \cdot C$, $9 \cdot C \cdot C$, $1 \cdot C \cdot 4C$로 $4C^2 + 9C^2 + 4C^2 = 17C^2$개다. 배치 정규화 셋은 채널이 각각 $C$, $C$, $4C$이므로 $2C + 2C + 8C = 12C$개다. 합쳐서

    $$17C^2 + 12C$$

    **비.** $C \to \infty$에서 1차 항이 사라지므로

    $$\lim_{C \to \infty} \frac{288C^2 + 16C}{17C^2 + 12C} = \frac{288}{17} = 16.94$$

    **확인.** $C = 64$에서 $288 \cdot 4096 + 1024 = 1{,}180{,}672$이고 $17 \cdot 4096 + 768 = 70{,}400$이다. 비는 16.77로 극한값 16.94보다 조금 작은데, 1차 항이 분모에서 상대적으로 더 무겁기 때문이다.

    ```python
    print(sum(p.numel() for p in BasicBlock(256, 256).parameters()))  # 1180672
    print(sum(p.numel() for p in Bottleneck(256, 64).parameters()))   # 70400
    ```

    유도에서 짚어 둘 것은 절약이 1×1 컨볼루션 자체에서 오지 않는다는 점이다. 1×1 두 개가 $8C^2$개를 쓰므로 3×3 하나($9C^2$개)와 거의 같다. 절약은 **3×3을 $4C$가 아니라 $C$ 폭에서 돌린 것**에서 온다. 3×3을 원래 폭에서 돌렸다면 $9 \cdot (4C)^2 = 144C^2$개로 지금의 $9C^2$개보다 열여섯 배였을 것이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
이 구현의 `Bottleneck`은 스트라이드를 가운데 3×3 컨볼루션에 준다. 스트라이드를 첫 1×1 컨볼루션으로 옮긴 판본을 만들어라. 파라메터 개수와 곱셈-덧셈 횟수가 각각 어떻게 달라지는가? `stage_macs`로 재어 답하라.

</div>

??? success "연습문제 6 풀이"
    `Bottleneck`을 물려받아 두 컨볼루션만 바꿔 끼운다.

    ```python
    class BottleneckStrideFirst(Bottleneck):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__(in_channels, out_channels, stride, downsample)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)

    a = resnet50(num_classes=1000)
    b = ResNet(BottleneckStrideFirst, [3, 4, 6, 3], num_classes=1000)
    for tag, m in [("stride on 3x3", a), ("stride on 1x1", b)]:
        print(tag, f"{count_params(m):,}",
              f"{sum(stage_macs(m).values()) / 1e9:.2f}G")
    # stride on 3x3 25,557,032 4.09G
    # stride on 1x1 25,557,032 3.86G
    ```

    **파라메터는 한 개도 달라지지 않는다.** 스트라이드는 웨이트의 모양에 들어가지 않기 때문이다. 1×1 컨볼루션의 웨이트는 언제나 $C_{\text{out}} \times C_{\text{in}} \times 1 \times 1$이고, 3×3은 $C_{\text{out}} \times C_{\text{in}} \times 3 \times 3$이다.

    **계산은 달라진다.** 단계 전환의 스트라이드 2를 첫 1×1로 옮기면 그 1×1의 출력 격자가 먼저 작아져 해당 레이어의 MAC이 줄어든다. 가운데 3×3의 출력 격자는 두 배치 방식 모두 같은 크기이므로 그 레이어의 MAC 감소가 주된 이유라는 설명은 틀리다. 기록된 컨볼루션 MAC은 약 4.09G에서 3.86G로 줄지만 BatchNorm 등 다른 연산과 실제 지연은 별도이다. 스트라이드 위치가 달라지면 샘플링과 함수 표현도 달라진다. 이 구현은 3×3에 스트라이드를 두는 변형이다.

---

## 정리하며

_make_layer는 입출력 모양이 바뀌는 첫 블록에 투영을 연결한다. 병목 블록은 좁은 채널에서 공간 컨볼루션을 수행하며 외부 채널 폭과 깊이 규약을 함께 읽어야 한다. MAC 표는 제한된 연산 집계이고 실제 지연과 다르다. 클래스 수만 바꾼 모델은 원래 CIFAR용 계열과 구분하며, 같은 계수 수라도 스트라이드 위치가 다르면 계산과 함수가 달라진다.
