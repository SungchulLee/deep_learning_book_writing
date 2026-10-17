# Squeeze-and-Excitation(SE) 블록

SE 블록은 피처 맵의 채널별 전역 평균을 구하고 작은 뉴럴 넷트웍으로 채널마다 입력에 따른 배율을 계산한다. 이 배율을 공간 전체에 공유해 곱하는 채널 어텐션이다. 일반 컨볼루션도 입력에 따라 다른 엑티베이션을 만들지만, SE는 배율을 별도로 계산해 곱하는 경로를 명시적으로 추가한다.

이 페이지는 수식과 그래디언트, 계수 수와 연산량 추정치를 살펴본다. 파라메터 증가, 연산 수 증가와 실제 지연은 서로 다른 지표이다.

## 1. 핵심 개념

- **채널 어텐션**: 전역 맥락에 따라 피처 채널의 웨이트를 다시 매기기
- **압축 연산**: 공간 차원만 눌러 채널마다 한 수씩 남기기
- **여기 연산**: 완전 연결 뉴럴 넷트웍 두 개로 채널의 중요도를 학습하기
- **적응형 재보정**: 입력에 따라 피처 맵을 그때그때 고치기
- **비용 구분**: 아래 구성의 계수 증가는 약 9.90%이다. MAC과 일부 스칼라 연산을 합산한 코드의 0.331% 추정치는 전체 FLOP이나 실행 시간 증가율과 구분한다.

---

## 2. SE 블록의 구조

### 핵심 설계

SE 블록은 압축과 여기 두 단계로 움직인다.

$$\text{SE}(\mathbf{X}) = \mathbf{X} \odot \sigma\left(W_2\,\delta(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2\right)$$

여기서 각 기호는 다음과 같다.

- $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$은 입력 피처 맵이다
- $\mathbf{z} \in \mathbb{R}^{C}$은 압축된 채널 기술자이다
- $W_1 \in \mathbb{R}^{(C/r) \times C}$와 $W_2 \in \mathbb{R}^{C \times (C/r)}$은 여기 장치의 두 웨이트 행렬이고, $\mathbf{b}_1 \in \mathbb{R}^{C/r}$과 $\mathbf{b}_2 \in \mathbb{R}^{C}$은 그 바이어스다
- $\delta$은 ReLU 활성화이다
- $\sigma$은 시그모이드 게이트이다
- $\odot$은 채널별 곱을 뜻한다

마지막 기호는 주의해서 읽어야 한다. 오른쪽 인수는 길이 $C$인 벡터이고 왼쪽은 $C \times H \times W$짜리 텐서이므로, $\odot$은 같은 꼴 두 개를 성분별로 곱하는 보통의 아다마르 곱이 아니라 **채널 배율 $s_c$를 그 채널의 $H \times W$ 칸 전체에 퍼뜨려 곱하는 것**이다.

$$Y_{c,i,j} = s_c \cdot X_{c,i,j}$$

배율은 채널의 모든 위치에 같은 값으로 적용된다. 실수 수학에서 시그모이드 배율은 0과 1 사이지만 부동소수점에서는 끝값으로 포화될 수 있다. 배율을 안다고 해도 0에 가깝거나 수치적으로 0이면 나눗셈을 통한 복원은 불안정하거나 불가능하다. 입력이 0인 위치에서 출력/입력의 비를 바로 계산해서도 안 된다.

### 압축 연산

압축 연산은 전역 평균 풀링으로 **공간 차원만** 눌러 담는다. 채널 축은 건드리지 않으므로 $C$개의 채널은 그대로 $C$개의 수로 남는다.

$$z_c = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{c,i,j}$$

전역 평균은 피처 맵 전체의 통계에 의존한다. 따라서 배율 계산에는 전역 문맥이 들어가지만 공간 위치 정보는 평균에서 사라진다. 컨볼루션도 충분한 깊이·스트라이드·팽창률을 사용하면 전체 입력 범위에 닿을 수 있으므로 컨볼루션으로는 전역 문맥을 얻을 수 없다는 설명은 부정확하다.

### 여기 연산

압축이 준 전역 맥락을 받아, 여기 장치는 채널마다의 중요도 웨이트를 배운다.

$$s_c = \sigma\left(W_2\,\delta(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2\right)_c$$

$\delta$은 성분별 ReLU이다.

$$\delta(\mathbf{u}) = \max(0, \mathbf{u})$$

**차원 축소**: 중간 차원은 보통 $C/r$이며 여기서 $r$은 축소 비율(흔히 16)이다.

$$\text{Excitation}: \mathbb{R}^{C} \xrightarrow{W_1} \mathbb{R}^{C/r} \xrightarrow{\delta} \mathbb{R}^{C/r} \xrightarrow{W_2} \mathbb{R}^{C} \xrightarrow{\sigma} \mathbb{R}^{C}$$

C가 r로 나누어떨어지면 두 선형 레이어의 웨이트는 2C²/r개이고 바이어스는 C/r+C개이다. 단일 C→C레이어의 웨이트는 C²개이다. C=256,r=16에서는 웨이트만 8192개 대 65536개이다. 실제 코드는 channels//r을 사용하므로 나누어떨어지지 않는 경우 중간 차원을 그 정수값으로 계산해야 하며, r>C이면 중간 차원 0이 되는 퇴화한 설정을 피해야 한다.

---

## 3. 수학적 성질

### 게이팅

SE 블록은 학습된 게이트 값으로 게이팅를 구현한다. 여기 연산의 마지막 시그모이드가 이미 게이트 값을 만들었으므로, 문은 다시 무엇을 씌우지 않고 그대로 쓴다.

$$\text{Gate}_c = s_c \in (0, 1)$$

!!! warning "시그모이드를 두 번 먹이지 말 것"
    $s_c$가 이미 $\sigma(\cdot)$의 출력인데 여기에 다시 $\sigma$을 씌워 $\sigma(s_c)$로 적는 실수가 흔하다. 그러면 값이 $(0, 1)$이 아니라 $\sigma(0)$과 $\sigma(1)$ 사이, 곧 **$0.5$와 $0.7311$ 사이**에 갇힌다. 4절의 출력이 이 범위를 직접 찍는다. 채널을 끌 수 없는 문은 문이 아니다.

값이 구간 $(0, 1)$에 놓이므로 채널을 부드럽게 고를 수 있다. 하드 어텐션처럼 채널을 완전히 끄고 켜는 대신 연속적인 배율을 쓰므로 미분이 되고, 따라서 역전파로 함께 학습된다.

### 그래디언트의 흐름

$X_{c,i,j}$은 출력에 두 클래스로 닿는다. 곧바로 곱해지는 길과, 평균 $z_c$을 거쳐 문 $\mathbf{s}$를 바꾸는 길이다. 연쇄 법칙은 두 길을 더한다.

$$\frac{\partial \mathcal{L}}{\partial X_{c,i,j}} = s_c \cdot \frac{\partial \mathcal{L}}{\partial Y_{c,i,j}} + \frac{1}{HW} \sum_{c'=1}^{C} \frac{\partial \mathcal{L}}{\partial s_{c'}} \cdot \frac{\partial s_{c'}}{\partial z_c}$$

$$\frac{\partial \mathcal{L}}{\partial s_{c'}} = \sum_{i=1}^{H} \sum_{j=1}^{W} X_{c',i,j} \cdot \frac{\partial \mathcal{L}}{\partial Y_{c',i,j}}$$

둘째 항에서 $c'$에 대한 합을 빠뜨리면 안 된다. 여기 장치의 두 레이어는 완전 연결이므로 채널 $c$의 평균 하나가 **모든** 채널의 게이트 값을 움직인다. 4절의 코드가 이 합을 뺀 대각선 근사를 자동 미분과 비교하여, 그래디언트 크기가 최대 1.4249인 문제에서 최대 0.0174만큼 어긋남을 보인다. 빠뜨린 것은 $c' \neq c$인 $C - 1$개의 기여이므로, 채널이 여덟 개인 그 예에서 이미 0이 아니고 채널이 많아질수록 빠뜨리는 항의 수도 늘어난다. 같은 코드가 $X_{c,i,j}$을 둘째 항에 곱해 넣은 식도 함께 견주는데, 그쪽의 어긋남은 2.7972로 그래디언트 자체(최대 1.4249)보다 크다. 곧 틀린 정도가 아니라 부호와 크기가 모두 다른 값이다.

!!! note "직접 곱셈 경로와 전체 그래디언트는 다르다"
    직접 경로에는 배율 s가 곱해지지만 전체 그래디언트에는 배율 자체가 입력에 의존하는 두 번째 항도 있다. 따라서 SE가 전체 그래디언트를 반드시 감소시킨다고 결론 내릴 수 없다. 잔차 분기 안에 SE를 두면 바깥의 직접 경로를 유지할 수 있지만, 잔차 연결 역시 전체 그래디언트 노름의 하한을 보장하지는 않는다.

---

## 4. 계산량 분석

### 복잡도

채널이 $C$개이고 공간 차원이 $H \times W$인 피처 맵에 SE 블록을 적용할 때는 다음과 같다.

**압축**: $O(CHW)$ (전역 평균 풀링)

**여기**: $O\!\left(2 \times C \times \frac{C}{r}\right) = O\!\left(\frac{2C^2}{r}\right)$ (완전 연결 레이어 두 개)

**게이팅**: $O(CHW)$ (채널별 곱)

**합계**: $O\!\left(CHW + \frac{2C^2}{r}\right)$

둘 중 어느 쪽이 큰지는 해상도가 정한다. $CHW > 2C^2/r$은 곧 $HW > 2C/r$이므로, $r = 16$이면 **$HW$가 $C/8$을 넘는 동안은 풀링이 더 크고, 그 아래로 내려가면 완전 연결 레이어가 더 크다**. ResNet-50의 네 단계 가운데 셋은 풀링 쪽이다.

| 단계 | $C$ | $H \times W$ | $CHW$ | $2C^2/r$ | 큰 쪽 |
|---|---|---|---|---|---|
| layer1 | 256 | $56 \times 56$ | 802,816 | 8,192 | 풀링 (98배) |
| layer2 | 512 | $28 \times 28$ | 401,408 | 32,768 | 풀링 (12배) |
| layer3 | 1024 | $14 \times 14$ | 200,704 | 131,072 | 풀링 (1.5배) |
| layer4 | 2048 | $7 \times 7$ | 100,352 | 524,288 | 완전 연결 레이어 (5.2배) |

아래 코드는 ResNet-50의 컨볼루션·선형 레이어 MAC을 약 4.0892G와 4.0917G로 센다. 별도의 성분별 연산 항도 기록한다. 이 작은 MAC 증가가 실제 실행 시간의 같은 비율 증가를 뜻하지 않는다.

!!! warning "MAC과 스칼라 연산의 집계 규약"
    코드가 출력하는 0.331%는 컨볼루션·선형 레이어의 MAC 한 번과 일부 스칼라 연산 한 번을 같은 단위로 더한 추정치이다. 일반적으로 1 MAC을 2 FLOP으로 세는 규약의 전체 FLOP 증가율과 같지 않다. 풀링·채널 곱을 포함할 때 각 연산의 단위를 맞춰야 하며 시그모이드·ReLU, 메모리 이동 등도 이 추정에 모두 들어 있지는 않다. 따라서 이 수치를 완전한 연산량이나 실행 시간의 증가율로 인용하지 않는다.

### 축소 비율의 트레이드오프

$r$이 정하는 것은 파라메터의 수다. $C = 256$을 놓고 실제 `nn.Module`에서 센 값은 다음과 같다.

| $r$ | 파라메터 | 웨이트 $2C^2/r$ | 바이어스 | $3 \times 3$ 컨볼루션 대비 |
|---|---|---|---|---|
| 2 | 65,920 | 65,536 | 384 | 11.176% |
| 8 | 16,672 | 16,384 | 288 | 2.827% |
| 16 | 8,464 | 8,192 | 272 | 1.435% |
| 32 | 4,360 | 4,096 | 264 | 0.739% |

!!! warning "바이어스는 $r$에 따라 줄지 않는다"
    $2C^2/r$만 외우면 $r$을 두 배로 할 때 파라메터가 정확히 반이 된다고 생각하게 된다. 그렇지 않다. $\mathbf{b}_2$의 성분 수는 $C$로 $r$과 무관하고, 그래서 $r = 16 \to 32$에서 웨이트는 8,192에서 4,096으로 정확히 반이 되지만 합계는 8,464에서 4,360으로 1.94배 줄어드는 데 그친다. 이 페이지에 실린 수는 모두 `sum(p.numel() for p in m.parameters())`로 센 값이지 공식을 믿고 적은 값이 아니다.

$r$을 키우면 문이 쓸 수 있는 중간 차원 $C/r$이 좁아지므로 채널 사이의 관계를 그만큼 거칠게밖에 나타내지 못한다. 어디까지 좁혀도 되는지는 데이터셋과 구조에 달린 문제이고, 이 페이지는 정확도를 재지 않으므로 여기에 순위를 매기지 않는다.

!!! note "흔한 설정"
    원 논문과 이후의 구현들은 대개 $r = 16$을 쓴다. 위 표에서 보듯 파라메터를 컨볼루션의 1.435%로 눌러 주는 값이다.

### 주장을 실제 모듈에 물어보기

지금까지 적은 것 — 압축은 공간 차원만 누른다, 여기는 채널별 곱이다, 파라메터는 $2C^2/r$에 바이어스를 더한 값이다, 역전파 식의 둘째 항은 채널 전체에 걸친 합이다 — 은 모두 코드로 확인할 수 있는 주장이다. 아래 프로그램이 하나씩 확인한다.

```python
"""SE 블록이 말하는 것을 하나씩 재어 본다.

압축이 공간 차원에 대한 전역 평균인지, 여기가 채널마다의 곱셈인지,
파라메터 부담이 정말 2C^2/r 인지, 그리고 역전파 식이 맞는지를
자동 미분과 실제 모듈에 물어서 확인한다.
"""

import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)

# === SE 블록 ==========================================================

class SEBlock(nn.Module):
    """압축(전역 평균 풀링) → 여기(완전 연결 두 개) → 채널별 곱."""

    def __init__(self, channels, r=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // r)
        self.fc2 = nn.Linear(channels // r, channels)

    def gate(self, x):
        z = x.mean(dim=(2, 3))                       # 압축: (N, C, H, W) → (N, C)
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))   # 여기: (N, C)

    def forward(self, x):
        s = self.gate(x)
        return x * s.view(x.size(0), -1, 1, 1)       # 채널별 곱 (더하기가 아니다)


# === 1. 압축은 공간 차원만 누른다 =====================================

def check_squeeze():
    x = torch.randn(2, 256, 7, 7)
    se = SEBlock(256)
    z_manual = x.mean(dim=(2, 3))
    z_pool = nn.AdaptiveAvgPool2d(1)(x).flatten(1)
    print("압축 출력 꼴:", tuple(z_manual.shape), "(배치와 채널은 남는다)")
    print("AdaptiveAvgPool2d(1)와 같은가:", torch.allclose(z_manual, z_pool, atol=1e-6))
    y = se(x)
    s = se.gate(x)
    print("여기 출력 꼴:", tuple(s.shape))
    print("출력 = 입력 * s 인가(채널별 곱):", torch.allclose(y, x * s[:, :, None, None]))
    # 한 채널만 골라 확인한다 -- 곱이면 비가 일정하고, 더하기면 차가 일정하다.
    ratio = (y[0, 3] / x[0, 3])
    print("채널 3의 출력/입력 비가 한 값인가:",
          bool((ratio - ratio.flatten()[0]).abs().max() < 1e-5),
          "  그 값 s_3 = %.4f" % s[0, 3].item())


# === 2. 파라메터 부담 =================================================

def check_params():
    se = SEBlock(256, r=16)
    total = sum(p.numel() for p in se.parameters())
    weights = se.fc1.weight.numel() + se.fc2.weight.numel()
    biases = se.fc1.bias.numel() + se.fc2.bias.numel()
    print("SE(C=256, r=16) 파라메터 합계:", total)
    print("  웨이트 2C^2/r =", weights, " 치우침 C/r + C =", biases)
    conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
    n_conv = sum(p.numel() for p in conv.parameters())
    print("3x3 컨볼루션(256→256, 치우침 없음):", n_conv)
    print("SE / 컨볼루션 = %.3f%%   (2C^2/r 만 세면 %.3f%%)"
          % (100 * total / n_conv, 100 * weights / n_conv))
    print("축소 비율별 (C=256):")
    for r in (2, 8, 16, 32):
        n = sum(p.numel() for p in SEBlock(256, r).parameters())
        print("  r=%2d  파라메터 %6d  = 2C^2/r %6d + 치우침 %4d   컨볼루션의 %6.3f%%"
              % (r, n, 2 * 256 * 256 // r, 256 // r + 256, 100 * n / n_conv))


# === 3. ResNet-50에 넣어 본다 =========================================

class SEBottleneck(nn.Module):
    """torchvision Bottleneck의 잔차 가지 **끝**에 SE를 달고, 그 뒤에 더한다."""

    def __init__(self, blk, r=16):
        super().__init__()
        self.blk = blk
        self.se = SEBlock(blk.conv3.out_channels, r)

    def forward(self, x):
        idt = x if self.blk.downsample is None else self.blk.downsample(x)
        o = self.blk.relu(self.blk.bn1(self.blk.conv1(x)))
        o = self.blk.relu(self.blk.bn2(self.blk.conv2(o)))
        o = self.blk.bn3(self.blk.conv3(o))
        return self.blk.relu(self.se(o) + idt)       # y = x + SE(F(x))


def macs(model):
    """224x224 입력 한 장에 드는 곱셈-덧셈 횟수.

    둘로 나누어 센다. 흔히 GFLOPs 로 보고되는 값은 컨볼루션과 완전 연결만
    센 것인데, SE 가 더하는 일 가운데 풀링과 채널별 곱은 거기에 들어가지
    않는다. 빼놓고 세면 SE 가 실제보다 싸 보이므로 따로 센다.
    """
    total = [0, 0]                       # [컨볼루션·완전 연결, 성분별]
    hooks = []

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            total[0] += (m.in_channels // m.groups) * m.kernel_size[0] \
                * m.kernel_size[1] * m.out_channels * o.shape[-1] * o.shape[-2]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features
        else:                            # SEBlock: 풀링 CHW + 채널별 곱 CHW
            total[1] += 2 * o.numel()

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, SEBlock)):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    for h in hooks:
        h.remove()
    return total[0], total[1]


def check_resnet():
    r50 = torchvision.models.resnet50(weights=None)
    se50 = torchvision.models.resnet50(weights=None)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(se50, name)
        setattr(se50, name, nn.Sequential(*[SEBottleneck(b) for b in stage]))

    p1 = sum(p.numel() for p in r50.parameters())
    p2 = sum(p.numel() for p in se50.parameters())
    (m1, e1), (m2, e2) = macs(r50), macs(se50)
    print("ResNet-50      파라메터 %10d   컨볼루션·완전연결 %.4f G   성분별 %.4f G"
          % (p1, m1 / 1e9, e1 / 1e9))
    print("SE-ResNet-50   파라메터 %10d   컨볼루션·완전연결 %.4f G   성분별 %.4f G"
          % (p2, m2 / 1e9, e2 / 1e9))
    print("늘어난 몫      파라메터 +%.2f%%   컨볼루션·완전연결 +%.3f%%   둘을 합치면 +%.3f%%"
          % (100 * (p2 - p1) / p1, 100 * (m2 - m1) / m1,
             100 * (m2 + e2 - m1 - e1) / (m1 + e1)))
    print("단계별 SE 파라메터:")
    for c, k in ((256, 3), (512, 4), (1024, 6), (2048, 3)):
        per = sum(p.numel() for p in SEBlock(c).parameters())
        print("  C=%4d  블록 하나 %7d  x %d = %8d" % (c, per, k, per * k))


# === 4. 압축과 여기 중 어느 쪽이 큰가 =================================

def check_cost():
    print("  %6s %7s %10s %10s" % ("C", "HxW", "CHW", "2C^2/r"))
    for c, hw in ((256, 56), (512, 28), (1024, 14), (2048, 7)):
        print("  %6d %3dx%-3d %10d %10d" % (c, hw, hw, c * hw * hw, 2 * c * c // 16))


# === 5. 역전파 식 =====================================================

def check_backward():
    """dL/dX = s * dL/dY + (1/HW) * (dL/ds) J,  J = ds/dz."""
    torch.manual_seed(1)
    C, H, W, r = 8, 4, 5, 4
    fc1 = nn.Linear(C, C // r).double()
    fc2 = nn.Linear(C // r, C).double()

    def excite(z):
        return torch.sigmoid(fc2(torch.relu(fc1(z))))

    x = torch.randn(1, C, H, W, dtype=torch.double, requires_grad=True)
    z = x.mean(dim=(2, 3))
    s = excite(z)
    y = x * s.view(1, C, 1, 1)
    loss = 0.5 * (y ** 2).sum() + 0.3 * y.sum()

    g_x, = torch.autograd.grad(loss, x, retain_graph=True)
    g_y, g_s = torch.autograd.grad(loss, [y, s], retain_graph=True)
    J = torch.autograd.functional.jacobian(excite, z)[0, :, 0, :]   # (C, C)

    right = g_y * s.view(1, C, 1, 1) + (g_s[0] @ J).view(1, C, 1, 1) / (H * W)
    wrong = g_y * s.view(1, C, 1, 1) + x * g_s.view(1, C, 1, 1) / (H * W)
    diag = g_y * s.view(1, C, 1, 1) \
        + (g_s[0] * J.diagonal()).view(1, C, 1, 1) / (H * W)

    print("자동 미분과 맞는가 (바른 식):", torch.allclose(right, g_x, atol=1e-12))
    print("그래디언트 크기 |dL/dX| 최대: %.4f" % g_x.abs().max())
    print("  X_{c,i,j} dL/ds_c 로 적은 식의 최대 오차: %.4f" % (wrong - g_x).abs().max())
    print("  대각선만 쓴 식(채널 사이 결합 무시)의 최대 오차: %.4f"
          % (diag - g_x).abs().max())
    print("문의 값 s 범위: %.4f ~ %.4f  (첫째 항은 이 값만큼 줄어든다)"
          % (s.min(), s.max()))
    t = torch.linspace(-20, 20, 100001)
    ss = torch.sigmoid(torch.sigmoid(t))
    print("시그모이드를 두 번 먹이면: %.4f ~ %.4f" % (ss.min(), ss.max()))


# === 실행 =============================================================

if __name__ == "__main__":
    print("[1] 압축과 여기")
    check_squeeze()
    print("\n[2] 파라메터 부담")
    check_params()
    print("\n[3] ResNet-50 vs SE-ResNet-50")
    check_resnet()
    print("\n[4] 압축과 여기의 계산량 (ResNet-50의 네 단계)")
    check_cost()
    print("\n[5] 역전파")
    check_backward()
```

**출력:**

```
[1] 압축과 여기
압축 출력 꼴: (2, 256) (배치와 채널은 남는다)
AdaptiveAvgPool2d(1)와 같은가: True
여기 출력 꼴: (2, 256)
출력 = 입력 * s 인가(채널별 곱): True
채널 3의 출력/입력 비가 한 값인가: True   그 값 s_3 = 0.4635

[2] 파라메터 부담
SE(C=256, r=16) 파라메터 합계: 8464
  웨이트 2C^2/r = 8192  치우침 C/r + C = 272
3x3 컨볼루션(256→256, 치우침 없음): 589824
SE / 컨볼루션 = 1.435%   (2C^2/r 만 세면 1.389%)
축소 비율별 (C=256):
  r= 2  파라메터  65920  = 2C^2/r  65536 + 치우침  384   컨볼루션의 11.176%
  r= 8  파라메터  16672  = 2C^2/r  16384 + 치우침  288   컨볼루션의  2.827%
  r=16  파라메터   8464  = 2C^2/r   8192 + 치우침  272   컨볼루션의  1.435%
  r=32  파라메터   4360  = 2C^2/r   4096 + 치우침  264   컨볼루션의  0.739%

[3] ResNet-50 vs SE-ResNet-50
ResNet-50      파라메터   25557032   컨볼루션·완전연결 4.0892 G   성분별 0.0000 G
SE-ResNet-50   파라메터   28088024   컨볼루션·완전연결 4.0917 G   성분별 0.0110 G
늘어난 몫      파라메터 +9.90%   컨볼루션·완전연결 +0.062%   둘을 합치면 +0.331%
단계별 SE 파라메터:
  C= 256  블록 하나    8464  x 3 =    25392
  C= 512  블록 하나   33312  x 4 =   133248
  C=1024  블록 하나  132160  x 6 =   792960
  C=2048  블록 하나  526464  x 3 =  1579392

[4] 압축과 여기의 계산량 (ResNet-50의 네 단계)
       C     HxW        CHW     2C^2/r
     256  56x56      802816       8192
     512  28x28      401408      32768
    1024  14x14      200704     131072
    2048   7x7       100352     524288

[5] 역전파
자동 미분과 맞는가 (바른 식): True
그래디언트 크기 |dL/dX| 최대: 1.4249
  X_{c,i,j} dL/ds_c 로 적은 식의 최대 오차: 2.7972
  대각선만 쓴 식(채널 사이 결합 무시)의 최대 오차: 0.0174
문의 값 s 범위: 0.3418 ~ 0.6783  (첫째 항은 이 값만큼 줄어든다)
시그모이드를 두 번 먹이면: 0.5000 ~ 0.7311
```

!!! note "파라메터가 9.90% 느는 곳은 어디인가"
    [3]의 단계별 표가 답을 준다. 늘어난 2,530,992개 가운데 1,579,392개, 곧 62.4%가 layer4 세 블록에서 나온다. $C = 2048$이라 블록 하나가 526,464개를 쓰는데, 이는 layer1 블록 하나(8,464개)의 62.2배다. 채널이 8배이고 파라메터의 으뜸항이 $C^2$이니 64배가 나와야 할 것 같지만, 바이어스가 $C$에만 비례해 함께 커지지 않으므로 62.2배에 그친다. 어느 쪽으로 읽든 요점은 같다 — SE의 부담은 채널 수의 제곱으로 불어나므로 깊은 단계에 몰린다. 뒤쪽 단계에서만 $r$을 키우는 변형이 있는 까닭이 이것이다.

---

## 5. CNN 구조에 넣기

SE는 채널과 공간 축을 가진 여러 CNN 구성에 결합할 수 있다. 적용 위치와 배율을 정하는 입력이 달라지면 계산 함수도 달라지므로 실제 구조와 목적에 맞춰 비교한다.

**ResNet-SE**: SE는 잔차 가지의 출력에 적용하고, 그 결과를 스킵 연결에 더한다.

$$\mathbf{y} = \mathbf{x} + \text{SE}\big(F_3(F_2(F_1(\mathbf{x})))\big)$$

잔차 더하기 전에 SE를 두면 입력을 직접 더하는 바깥 경로를 남긴다. 더한 뒤에 적용하면 그 경로에도 배율이 곱해지는 다른 구조가 된다. 후자가 수학적으로 금지된 것은 아니며 효과는 별도로 검증해야 한다. 아래 SEBottleneck은 잔차 분기 안에 적용하고 마지막에 ReLU를 수행하므로 블록 전체의 그래디언트에 ReLU 마스크도 반영해야 한다.

**DenseNet-SE**: 이어 붙이기 전에 조밀 블록의 출력에 적용한다.

**EfficientNet-SE**: 기본 구조 설계에 아예 들어 있는 부품이다. 깊이별 분리 컨볼루션 뒤에 SE를 두어, [깊이별 분리 컨볼루션](../cnn/depthwise_separable.md)이 채널 사이의 섞임을 $1 \times 1$ 컨볼루션 하나로만 처리하는 것을 채널 어텐션으로 보완한다.

---

## 6. 실험으로 드러난 성질

이 페이지의 코드는 정확도나 지연을 측정하지 않고 구성과 수치 비교를 제공한다. 논문에서 보고한 성능은 해당 데이터·학습 설정의 외부 실험 결과로 인용하며 여기서 재현했다고 설명하지 않는다. 한 실행도 관찰값이지만 시드 변동과 유한 표본의 불확실성을 다 담는 것은 아니다. 차이가 표준편차보다 작다는 사실만으로 차이가 없다고 판정하는 일반 규칙도 없다.

SE 배율은 채널의 상대적 감쇠를 보여 주지만 곧바로 원인적 중요도를 뜻하지 않는다. 공간적으로 같은 배율을 공유하므로 어느 픽셀을 보았는지 설명하는 공간 지도도 아니다. 공간 어텐션 역시 그 값만으로 판단의 인과적 근거가 증명되는 것은 아니다. 관련 연산은 [비국소 뉴럴 넷트웍](non_local.md)에서 살펴본다.

---

## 7. 변형과 확장

**공간과 채널을 함께 쓰는 SE (scSE)**: SE의 채널 문에 공간 문($1 \times 1$ 컨볼루션으로 $H \times W$짜리 지도를 만들어 곱하는 것)을 나란히 두고 둘을 합친다.

**효과적인 SE (ESE)**: 완전 연결 레이어 두 개를 하나로 줄인 변형이다. 병목을 없애므로 파라메터는 $2C^2/r$에서 $C^2$으로 오히려 늘지만, 중간의 차원 축소에서 잃는 정보가 없어진다.

**컨볼루션 블록 어텐션 모듈 (CBAM)**: 채널 어텐션과 공간 어텐션을 차례로 적용하고, 채널 쪽에서는 평균 풀링과 최대 풀링을 함께 쓴다. [CBAM](../../appendix/cnn/cbam.md)에서 자세히 다룬다.

**좌표 어텐션**: 전역 평균 풀링이 공간 정보를 통째로 버리는 것을 보완하려고, 가로와 세로 방향으로 따로 풀링하여 위치 정보를 문 안에 남긴다.

---

## 8. 관련 주제

- [컨볼루션 뉴럴 넷트웍에서 트랜스포머로](cnn_to_vit_bridge.md) — SE가 순수 CNN과 ViT 사이의 어느 단계에 놓이는지
- [비국소 뉴럴 넷트웍](non_local.md) — 같은 절의 공간 어텐션 쪽 짝
- [CBAM](../../appendix/cnn/cbam.md) — 채널 어텐션과 공간 어텐션을 잇대어 쓰는 확장
- [ResNet 구현](../residual/02_resnet_implementation.md) — 5절이 SE를 끼워 넣는 병목 블록
- [항등 사상](../residual/identity_mapping.md) — 3절의 "항등 연결이 없다"가 무엇을 잃는지
- [셀프 어텐션](../../ch11/attention/self_attention.md) — 채널이 아니라 위치 사이의 관계를 보는 어텐션

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
채널이 $C = 256$개이고 축소 비율이 $r = 16$인 SE 블록의 파라메터를 세어라. 바이어스까지 포함한 값과 $2C^2/r$만 센 값을 모두 구하고, 같은 채널 수의 $3 \times 3$ 컨볼루션과 비교하여라.

</div>

??? success "연습문제 1 풀이"
    웨이트는 $W_1$이 $(C/r) \times C = 16 \times 256 = 4{,}096$개, $W_2$이 $C \times (C/r) = 256 \times 16 = 4{,}096$개로 모두 $2C^2/r = 8{,}192$개다. 여기에 바이어스 $\mathbf{b}_1$이 $C/r = 16$개, $\mathbf{b}_2$이 $C = 256$개, 합해서 272개가 더 붙는다. 따라서 `nn.Linear` 두 개로 만든 실제 모듈의 파라메터는 **8,464개**다. 4절의 출력 [2]가 이 수를 찍는다.

    편향 없는 $3 \times 3$ 컨볼루션은 $256 \times 256 \times 3 \times 3 = 589{,}824$개이므로 비는 $8{,}464 / 589{,}824 = 1.435\%$다. $2C^2/r$만 세면 1.389%가 나오는데, 두 값 모두 "약 1.4%"로 반올림되므로 어느 쪽을 적어도 틀려 보이지 않는다. 그렇기 때문에 더 위험하다 — 채널이 적은 블록에서는 $C$가 $2C^2/r$에 비교하여 상대적으로 크므로 바이어스의 몫이 커진다. $C = 64$, $r = 16$이면 웨이트 512개에 바이어스 68개로, 바이어스가 합계의 11.7%를 차지한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SE 블록을 처음부터 유도하라. 전역 평균 풀링, 축소 비율이 $r$인 완전 연결 레이어 두 개, 시그모이드 게이트가 각각 무엇을 맡는지 밝혀라.

</div>

??? success "연습문제 2 풀이"
    입력은 피처 맵 $X \in \mathbb{R}^{C \times H \times W}$이다.

    (1) **압축**: $z_c = \frac{1}{HW}\sum_{i,j} X_{c,i,j}$. 채널마다 공간 전체의 평균을 잡아 $\mathbf{z} \in \mathbb{R}^{C}$을 만든다. 공간 축만 없애고 채널 축은 남긴다. 이 한 스텝이 하는 일은 리셉티브 필드를 피처 맵 전체로 넓히는 것이다 — 뒤의 문이 "이 이미지 전체에서" 어떤 채널이 중요한지 판단할 수 있게 된다.

    (2) **여기**: $\mathbf{s} = \sigma(W_2\,\delta(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2)$이며 $W_1 \in \mathbb{R}^{(C/r) \times C}$, $W_2 \in \mathbb{R}^{C \times (C/r)}$이다. 완전 연결 레이어이므로 모든 채널의 통계량이 모든 채널의 게이트 값에 영향을 준다 — 이것이 "채널 사이의 상호 의존"의 구체적인 뜻이다. 사이에 ReLU를 넣어야 두 레이어가 하나의 선형 사상으로 무너지지 않는다. 병목 $C/r$은 파라메터를 $C^2$에서 $2C^2/r$로 줄인다.

    (3) **배율 적용**: s_c를 해당 채널 전체에 곱한다. 시그모이드 출력이 0~1이므로 직접적인 크기를 키우는 연산은 아니며 채널 간 상대적인 감쇠를 조절한다. 후속 레이어가 이를 보완하도록 학습될 수 있지만 충분한 보완이나 성능 유지가 보장되는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$r = 16$일 때 압축(전역 평균 풀링)보다 여기(완전 연결 레이어 두 개)의 계산량이 더 커지는 조건을 $C$와 $HW$로 나타내어라. ResNet-50의 네 단계 가운데 어느 단계가 그 조건을 만족하는가?

</div>

??? success "연습문제 3 풀이"
    여기서는 풀링의 스칼라 연산 수를 CHW로 근사하고 두 선형 레이어의 MAC을 2C²/r로 놓아 규모를 비교한다. 둘의 단위가 같지 않으므로 다음 부등식은 코드의 집계 규약에서의 경향이며 실제 FLOP이나 지연의 정확한 경계가 아니다. 정밀한 비교에는 MAC당 FLOP 규약, 평균의 나눗셈과 활성화 비용도 포함한다.

    $$\frac{2C^2}{r} > CHW \iff HW < \frac{2C}{r}$$

    이고, $r = 16$이면 $HW < C/8$이다. ResNet-50의 네 단계를 대입하면

    | 단계 | $C$ | $HW$ | $C/8$ | $HW < C/8$ |
    |---|---|---|---|---|
    | layer1 | 256 | 3,136 | 32 | 아니다 |
    | layer2 | 512 | 784 | 64 | 아니다 |
    | layer3 | 1024 | 196 | 128 | 아니다 |
    | layer4 | 2048 | 49 | 256 | **그렇다** |

    네 단계 가운데 layer4 하나뿐이다. 4절 출력 [4]의 수와 맞는다 — layer3에서는 200,704 대 131,072로 아직 풀링이 1.5배 크고, layer4에서 100,352 대 524,288로 뒤집힌다. 해상도가 절반이 되면 $HW$은 $1/4$로 줄고 $C$은 두 배가 되므로, 단계를 하나 지날 때마다 이 부등식의 양변 비가 8배씩 여기 쪽으로 기운다. 그래서 뒤집힘은 한 번만, 그것도 갑작스럽게 일어난다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
SE 블록을 PyTorch로 구현하고 ResNet 병목에 넣어라. SE를 더하기 앞에 두는 것과 뒤에 두는 것이 왜 다른지 밝혀라.

</div>

??? success "연습문제 4 풀이"
    ```python
    class SE(nn.Module):
        def __init__(self, c, r=16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.excite = nn.Sequential(
                nn.Linear(c, c // r), nn.ReLU(),
                nn.Linear(c // r, c), nn.Sigmoid(),
            )

        def forward(self, x):
            s = self.excite(self.pool(x).flatten(1))   # (N, C)
            return x * s.view(x.size(0), -1, 1, 1)     # 채널별 곱

    class SEBottleneck(nn.Module):
        def __init__(self, blk, r=16):                 # torchvision Bottleneck
            super().__init__()
            self.blk, self.se = blk, SE(blk.conv3.out_channels, r)

        def forward(self, x):
            idt = x if self.blk.downsample is None else self.blk.downsample(x)
            o = self.blk.relu(self.blk.bn1(self.blk.conv1(x)))
            o = self.blk.relu(self.blk.bn2(self.blk.conv2(o)))
            o = self.blk.bn3(self.blk.conv3(o))
            return self.blk.relu(self.se(o) + idt)     # y = x + SE(F(x))
    ```

    `flatten(1)`은 $(N, C, 1, 1)$을 $(N, C)$로 펴고, `view(N, C, 1, 1)`은 다시 곱하기 좋게 되돌린다. 이 두 번의 모양 바꿈을 건너뛰고 $(N, C, 1, 1)$짜리 문을 그대로 곱해도 브로드캐스팅으로 같은 결과가 나오지만, 완전 연결 레이어가 마지막 축을 쓰므로 중간에 한 번은 펴야 한다.

    **적용 위치**: 잔차 분기에만 적용하면 x+SE(F(x)), 더한 뒤 적용하면 SE(x+F(x))가 된다. 후자는 x의 직접 경로에도 배율이 들어가지만 배율이 입력에 의존하는 미분 항도 포함한다. 한 예제의 배율을 0.5로 대체해 16번 곱한 값은 실제 16블록 전체의 그래디언트를 예측하지 못한다. 이 구현의 사후 ReLU와 투영도 함께 고려해야 한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
SE 블록은 어텐션 메커니즘과 어떤 관계인가? 어떤 종류의 어텐션을 구현하며, 트랜스포머의 셀프 어텐션과 무엇이 다른가?

</div>

??? success "연습문제 5 풀이"
    SE는 **채널 어텐션**을 구현한다. 전역 맥락에 따라 피처 맵의 채널마다 중요도를 매기는 법을 배운다. 공간 어텐션(어디를 볼지)이나 셀프 어텐션(쌍의 관계)과 달리 채널 어텐션은 "이 입력에는 어떤 피처가 중요한가"에 답한다.

    셀프 어텐션과 견주면 세 가지가 다르다.

    - **웨이트의 개수**: 셀프 어텐션은 $N$개 위치에 대해 $N \times N$짜리 웨이트를 만들지만, SE는 $C$개의 수만 만든다. 그래서 계산량이 $N$에 이차가 아니라 $HW$에 일차다 — 4절의 $O(CHW + 2C^2/r)$이 그것이다.
    - **질의가 없다**: 셀프 어텐션은 위치마다 질의를 만들어 다른 위치와 견주지만, SE에는 질의가 없고 전역 평균 하나가 모든 채널의 문을 정한다. 위치에 따라 다른 문을 쓰지 못한다는 뜻이다.
    - **곱의 대상**: 셀프 어텐션의 웨이트는 값들의 볼록 결합을 만드는 계수(합이 1)이지만, SE의 $s_c$는 각 채널에 독립적으로 붙는 배율이라 합이 1일 까닭이 없다. 소프트맥스가 아니라 시그모이드를 쓰는 것이 이 차이의 표시다.

    그런 뜻에서 SE는 트랜스포머의 온전한 어텐션에 앞선 가벼운 선구자이며, [컨볼루션 뉴럴 넷트웍에서 트랜스포머로](cnn_to_vit_bridge.md)가 그 사이의 단계들을 늘어놓는다.

---

## 정리하며

SE는 공간 평균으로 채널별 배율을 계산하고 피처 맵에 곱한다. 직접 곱셈과 배율 의존성을 통한 그래디언트를 모두 고려해야 한다. 계수 수와 MAC·스칼라 연산·실제 지연은 서로 구분하고, 잔차 분기 안팎의 적용 위치도 다른 구조로 비교한다. 배율 시각화만으로 채널의 인과적 중요도를 확정하지 않는다.
