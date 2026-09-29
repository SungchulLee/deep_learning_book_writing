# 압축-여기 신경망

합성곱 층은 공간 방향으로는 이웃을 섞지만 채널 방향으로는 모든 채널을 똑같이 내보낸다. 어떤 입력에서는 결을 잡는 채널이 중요하고 다른 입력에서는 색을 잡는 채널이 중요한데, 합성곱 하나만으로는 그 차이를 입력마다 다르게 반영할 길이 없다. 압축-여기(squeeze-and-excitation, SE) 블록은 특징 맵 전체를 채널마다 한 수로 요약한 뒤, 그 수를 보고 채널마다 0과 1 사이의 배율을 정해 곱한다. 채널 어텐션이라고 부르는 것이 이것이다.

이 장치는 값이 싸다. 다만 "싸다"가 무엇을 뜻하는지는 두 갈래로 갈린다 — 이 쪽에서 직접 재어 보면 ResNet-50에 SE를 달 때 **곱셈-덧셈 횟수는 0.331%밖에 늘지 않지만 매개변수는 9.90% 는다**. 흔히 한 덩어리로 뭉뚱그리는 이 둘을 4절에서 따로 떼어 센다.

---

## 1. 핵심 개념

- **채널 어텐션**: 전역 맥락에 따라 특징 채널의 가중치를 다시 매기기
- **압축 연산**: 공간 차원만 눌러 채널마다 한 수씩 남기기
- **여기 연산**: 완전 연결 신경망 두 개로 채널의 중요도를 배우기
- **적응형 재보정**: 입력에 따라 특징 맵을 그때그때 고치기
- **부담의 두 얼굴**: ResNet-50 기준으로 곱셈-덧셈은 +0.331%, 매개변수는 +9.90%

---

## 2. SE 블록의 구조

### 핵심 설계

SE 블록은 압축과 여기 두 단계로 움직인다.

$$\text{SE}(\mathbf{X}) = \mathbf{X} \odot \sigma\left(W_2\,\delta(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2\right)$$

여기서 각 기호는 다음과 같다.

- $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$은 입력 특징 맵이다
- $\mathbf{z} \in \mathbb{R}^{C}$은 압축된 채널 기술자이다
- $W_1 \in \mathbb{R}^{(C/r) \times C}$와 $W_2 \in \mathbb{R}^{C \times (C/r)}$은 여기 장치의 두 가중치 행렬이고, $\mathbf{b}_1 \in \mathbb{R}^{C/r}$과 $\mathbf{b}_2 \in \mathbb{R}^{C}$은 그 치우침이다
- $\delta$은 ReLU 활성화이다
- $\sigma$은 시그모이드 문이다
- $\odot$은 채널별 곱을 뜻한다

마지막 기호는 주의해서 읽어야 한다. 오른쪽 인수는 길이 $C$인 벡터이고 왼쪽은 $C \times H \times W$짜리 텐서이므로, $\odot$은 같은 꼴 두 개를 성분별로 곱하는 보통의 아다마르 곱이 아니라 **채널 배율 $s_c$를 그 채널의 $H \times W$ 칸 전체에 퍼뜨려 곱하는 것**이다.

$$Y_{c,i,j} = s_c \cdot X_{c,i,j}$$

곱이지 합이 아니라는 점도 중요하다. 곱이므로 $s_c$가 0에 가까우면 그 채널이 통째로 눌리고(시그모이드이므로 정확히 0이 되지는 않는다), 출력을 $s_c$로 나누면 입력이 그대로 돌아온다. 더하기였다면 채널 하나를 끄기 위해 그 채널의 모든 칸에 서로 다른 값을 더해야 하는데, 채널마다 한 수만 가지고는 그럴 수 없다. 4절의 코드가 한 채널을 골라 출력을 입력으로 나누어 보고, 그 비가 칸마다 바뀌지 않는 한 수($s_3 = 0.4635$)임을 확인한다.

### 압축 연산

압축 연산은 전역 평균 풀링으로 **공간 차원만** 눌러 담는다. 채널 축은 건드리지 않으므로 $C$개의 채널은 그대로 $C$개의 수로 남는다.

$$z_c = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{c,i,j}$$

이렇게 전역 채널 통계량을 담은 채널 기술자 $\mathbf{z} \in \mathbb{R}^{C}$이 나온다. 이 한 걸음으로 수용 영역이 특징 맵 전체가 된다는 점이 요점이다 — $3 \times 3$ 합성곱을 아무리 쌓아도 얻지 못하는 전역 맥락을 풀링 한 번이 준다.

### 여기 연산

압축이 준 전역 맥락을 받아, 여기 장치는 채널마다의 중요도 가중치를 배운다.

$$s_c = \sigma\left(W_2\,\delta(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2\right)_c$$

$\delta$은 성분별 ReLU이다.

$$\delta(\mathbf{u}) = \max(0, \mathbf{u})$$

**차원 축소**: 중간 차원은 보통 $C/r$이며 여기서 $r$은 축소 비율(흔히 16)이다.

$$\text{Excitation}: \mathbb{R}^{C} \xrightarrow{W_1} \mathbb{R}^{C/r} \xrightarrow{\delta} \mathbb{R}^{C/r} \xrightarrow{W_2} \mathbb{R}^{C} \xrightarrow{\sigma} \mathbb{R}^{C}$$

완전 연결층을 하나만 두면 매개변수가 $C^2$개 들지만, 병목을 거치면 $2C^2/r$개로 줄어든다. $C = 256$, $r = 16$이면 65,536개가 8,192개가 되는 셈이다. 그러면서도 비선형 $\delta$을 사이에 끼워 넣어 채널 사이의 관계를 선형보다 복잡하게 쓸 수 있다.

---

## 3. 수학적 성질

### 문 장치

SE 블록은 학습된 문 값으로 문 장치를 구현한다. 여기 연산의 마지막 시그모이드가 이미 문 값을 만들었으므로, 문은 다시 무엇을 씌우지 않고 그대로 쓴다.

$$\text{Gate}_c = s_c \in (0, 1)$$

!!! warning "시그모이드를 두 번 먹이지 말 것"
    $s_c$가 이미 $\sigma(\cdot)$의 출력인데 여기에 다시 $\sigma$을 씌워 $\sigma(s_c)$로 적는 실수가 흔하다. 그러면 값이 $(0, 1)$이 아니라 $\sigma(0)$과 $\sigma(1)$ 사이, 곧 **$0.5$와 $0.7311$ 사이**에 갇힌다. 4절의 출력이 이 범위를 직접 찍는다. 채널을 끌 수 없는 문은 문이 아니다.

값이 구간 $(0, 1)$에 놓이므로 채널을 부드럽게 고를 수 있다. 하드 어텐션처럼 채널을 완전히 끄고 켜는 대신 연속적인 배율을 쓰므로 미분이 되고, 따라서 역전파로 함께 학습된다.

### 기울기의 흐름

$X_{c,i,j}$은 출력에 두 갈래로 닿는다. 곧바로 곱해지는 길과, 평균 $z_c$을 거쳐 문 $\mathbf{s}$를 바꾸는 길이다. 연쇄 법칙은 두 길을 더한다.

$$\frac{\partial \mathcal{L}}{\partial X_{c,i,j}} = s_c \cdot \frac{\partial \mathcal{L}}{\partial Y_{c,i,j}} + \frac{1}{HW} \sum_{c'=1}^{C} \frac{\partial \mathcal{L}}{\partial s_{c'}} \cdot \frac{\partial s_{c'}}{\partial z_c}$$

$$\frac{\partial \mathcal{L}}{\partial s_{c'}} = \sum_{i=1}^{H} \sum_{j=1}^{W} X_{c',i,j} \cdot \frac{\partial \mathcal{L}}{\partial Y_{c',i,j}}$$

둘째 항에서 $c'$에 대한 합을 빠뜨리면 안 된다. 여기 장치의 두 층은 완전 연결이므로 채널 $c$의 평균 하나가 **모든** 채널의 문 값을 움직인다. 4절의 코드가 이 합을 뺀 대각선 근사를 자동 미분과 견주어, 기울기 크기가 최대 1.4249인 문제에서 최대 0.0174만큼 어긋남을 보인다. 빠뜨린 것은 $c' \neq c$인 $C - 1$개의 기여이므로, 채널이 여덟 개인 그 예에서 이미 0이 아니고 채널이 많아질수록 빠뜨리는 항의 수도 늘어난다. 같은 코드가 $X_{c,i,j}$을 둘째 항에 곱해 넣은 식도 함께 견주는데, 그쪽의 어긋남은 2.7972로 기울기 자체(최대 1.4249)보다 크다. 곧 틀린 정도가 아니라 부호와 크기가 모두 다른 값이다.

!!! warning "SE 블록에는 항등 연결이 없다"
    첫째 항의 계수는 $1$이 아니라 $s_c < 1$이다. 잔차 블록의 항등 연결은 기울기에 $1$을 곱해 그대로 흘려보내지만, SE의 곱셈 문은 기울기를 **줄인다**. 4절의 예에서 $s$는 $0.3418$에서 $0.6783$ 사이였으므로 그 길로 오는 기울기는 절반 안팎으로 깎인다. SE를 잔차 가지 안에 두는 관행(5절)이 중요한 까닭이 이것이다 — 기울기를 지키는 것은 SE가 아니라 그것을 감싸는 잔차 연결이다.

---

## 4. 계산량 분석

### 복잡도

채널이 $C$개이고 공간 차원이 $H \times W$인 특징 맵에 SE 블록을 적용할 때는 다음과 같다.

**압축**: $O(CHW)$ (전역 평균 풀링)

**여기**: $O\!\left(2 \times C \times \frac{C}{r}\right) = O\!\left(\frac{2C^2}{r}\right)$ (완전 연결층 두 개)

**문 달기**: $O(CHW)$ (채널별 곱)

**합계**: $O\!\left(CHW + \frac{2C^2}{r}\right)$

둘 중 어느 쪽이 큰지는 해상도가 정한다. $CHW > 2C^2/r$은 곧 $HW > 2C/r$이므로, $r = 16$이면 **$HW$가 $C/8$을 넘는 동안은 풀링이 더 크고, 그 아래로 내려가면 완전 연결층이 더 크다**. ResNet-50의 네 단계 가운데 셋은 풀링 쪽이다.

| 단계 | $C$ | $H \times W$ | $CHW$ | $2C^2/r$ | 큰 쪽 |
|---|---|---|---|---|---|
| layer1 | 256 | $56 \times 56$ | 802,816 | 8,192 | 풀링 (98배) |
| layer2 | 512 | $28 \times 28$ | 401,408 | 32,768 | 풀링 (12배) |
| layer3 | 1024 | $14 \times 14$ | 200,704 | 131,072 | 풀링 (1.5배) |
| layer4 | 2048 | $7 \times 7$ | 100,352 | 524,288 | 완전 연결층 (5.2배) |

어느 쪽이든 같은 자리의 $3 \times 3$ 합성곱에 견주면 무시할 만하다. 아래에서 재는 값으로 ResNet-50 전체의 합성곱과 완전 연결은 4.0892 G번이고 SE를 다 달아도 4.0917 G번, 곧 0.062% 증가에 그친다.

!!! warning "GFLOPs 표에는 SE의 절반이 빠져 있다"
    흔히 보고되는 GFLOPs 는 합성곱과 완전 연결만 센 값이다. SE가 하는 일 가운데 풀링($CHW$번의 덧셈)과 채널별 곱($CHW$번의 곱셈)은 그 셈에 들어가지 않는데, 위 표에서 보듯 **앞쪽 세 단계에서는 빠지는 쪽이 더 큰 쪽**이다. 실제로 아래 출력 [3]에서 성분별 연산은 0.0110 G번으로 완전 연결층이 더하는 0.0025 G번의 4.4배이고, 둘을 함께 세면 증가분은 0.062%가 아니라 **0.331%**가 된다. 결론은 바뀌지 않지만(여전히 1%가 안 된다), 어느 쪽 수를 인용하고 있는지는 알고 써야 한다.

### 축소 비율의 맞바꿈

$r$이 정하는 것은 매개변수의 수다. $C = 256$을 놓고 실제 `nn.Module`에서 센 값은 다음과 같다.

| $r$ | 매개변수 | 가중치 $2C^2/r$ | 치우침 | $3 \times 3$ 합성곱 대비 |
|---|---|---|---|---|
| 2 | 65,920 | 65,536 | 384 | 11.176% |
| 8 | 16,672 | 16,384 | 288 | 2.827% |
| 16 | 8,464 | 8,192 | 272 | 1.435% |
| 32 | 4,360 | 4,096 | 264 | 0.739% |

!!! warning "치우침은 $r$에 따라 줄지 않는다"
    $2C^2/r$만 외우면 $r$을 두 배로 할 때 매개변수가 정확히 반이 된다고 생각하게 된다. 그렇지 않다. $\mathbf{b}_2$의 성분 수는 $C$로 $r$과 무관하고, 그래서 $r = 16 \to 32$에서 가중치는 8,192에서 4,096으로 정확히 반이 되지만 합계는 8,464에서 4,360으로 1.94배 줄어드는 데 그친다. 이 쪽에 실린 수는 모두 `sum(p.numel() for p in m.parameters())`로 센 값이지 공식을 믿고 적은 값이 아니다.

$r$을 키우면 문이 쓸 수 있는 중간 차원 $C/r$이 좁아지므로 채널 사이의 관계를 그만큼 거칠게밖에 나타내지 못한다. 어디까지 좁혀도 되는지는 데이터셋과 구조에 달린 문제이고, 이 쪽은 정확도를 재지 않으므로 여기에 순위를 매기지 않는다.

!!! note "흔한 설정"
    원 논문과 이후의 구현들은 대개 $r = 16$을 쓴다. 위 표에서 보듯 매개변수를 합성곱의 1.435%로 눌러 주는 값이다.

### 주장을 실제 모듈에 물어보기

지금까지 적은 것 — 압축은 공간 차원만 누른다, 여기는 채널별 곱이다, 매개변수는 $2C^2/r$에 치우침을 더한 값이다, 역전파 식의 둘째 항은 채널 전체에 걸친 합이다 — 은 모두 코드로 확인할 수 있는 주장이다. 아래 프로그램이 하나씩 확인한다.

```python
"""SE 블록이 말하는 것을 하나씩 재어 본다.

압축이 공간 차원에 대한 전역 평균인지, 여기가 채널마다의 곱셈인지,
매개변수 부담이 정말 2C^2/r 인지, 그리고 역전파 식이 맞는지를
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


# === 2. 매개변수 부담 =================================================

def check_params():
    se = SEBlock(256, r=16)
    total = sum(p.numel() for p in se.parameters())
    weights = se.fc1.weight.numel() + se.fc2.weight.numel()
    biases = se.fc1.bias.numel() + se.fc2.bias.numel()
    print("SE(C=256, r=16) 매개변수 합계:", total)
    print("  가중치 2C^2/r =", weights, " 치우침 C/r + C =", biases)
    conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
    n_conv = sum(p.numel() for p in conv.parameters())
    print("3x3 합성곱(256→256, 치우침 없음):", n_conv)
    print("SE / 합성곱 = %.3f%%   (2C^2/r 만 세면 %.3f%%)"
          % (100 * total / n_conv, 100 * weights / n_conv))
    print("축소 비율별 (C=256):")
    for r in (2, 8, 16, 32):
        n = sum(p.numel() for p in SEBlock(256, r).parameters())
        print("  r=%2d  매개변수 %6d  = 2C^2/r %6d + 치우침 %4d   합성곱의 %6.3f%%"
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

    둘로 나누어 센다. 흔히 GFLOPs 로 보고되는 값은 합성곱과 완전 연결만
    센 것인데, SE 가 더하는 일 가운데 풀링과 채널별 곱은 거기에 들어가지
    않는다. 빼놓고 세면 SE 가 실제보다 싸 보이므로 따로 센다.
    """
    total = [0, 0]                       # [합성곱·완전 연결, 성분별]
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
    print("ResNet-50      매개변수 %10d   합성곱·완전연결 %.4f G   성분별 %.4f G"
          % (p1, m1 / 1e9, e1 / 1e9))
    print("SE-ResNet-50   매개변수 %10d   합성곱·완전연결 %.4f G   성분별 %.4f G"
          % (p2, m2 / 1e9, e2 / 1e9))
    print("늘어난 몫      매개변수 +%.2f%%   합성곱·완전연결 +%.3f%%   둘을 합치면 +%.3f%%"
          % (100 * (p2 - p1) / p1, 100 * (m2 - m1) / m1,
             100 * (m2 + e2 - m1 - e1) / (m1 + e1)))
    print("단계별 SE 매개변수:")
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
    print("기울기 크기 |dL/dX| 최대: %.4f" % g_x.abs().max())
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
    print("\n[2] 매개변수 부담")
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

[2] 매개변수 부담
SE(C=256, r=16) 매개변수 합계: 8464
  가중치 2C^2/r = 8192  치우침 C/r + C = 272
3x3 합성곱(256→256, 치우침 없음): 589824
SE / 합성곱 = 1.435%   (2C^2/r 만 세면 1.389%)
축소 비율별 (C=256):
  r= 2  매개변수  65920  = 2C^2/r  65536 + 치우침  384   합성곱의 11.176%
  r= 8  매개변수  16672  = 2C^2/r  16384 + 치우침  288   합성곱의  2.827%
  r=16  매개변수   8464  = 2C^2/r   8192 + 치우침  272   합성곱의  1.435%
  r=32  매개변수   4360  = 2C^2/r   4096 + 치우침  264   합성곱의  0.739%

[3] ResNet-50 vs SE-ResNet-50
ResNet-50      매개변수   25557032   합성곱·완전연결 4.0892 G   성분별 0.0000 G
SE-ResNet-50   매개변수   28088024   합성곱·완전연결 4.0917 G   성분별 0.0110 G
늘어난 몫      매개변수 +9.90%   합성곱·완전연결 +0.062%   둘을 합치면 +0.331%
단계별 SE 매개변수:
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
기울기 크기 |dL/dX| 최대: 1.4249
  X_{c,i,j} dL/ds_c 로 적은 식의 최대 오차: 2.7972
  대각선만 쓴 식(채널 사이 결합 무시)의 최대 오차: 0.0174
문의 값 s 범위: 0.3418 ~ 0.6783  (첫째 항은 이 값만큼 줄어든다)
시그모이드를 두 번 먹이면: 0.5000 ~ 0.7311
```

!!! note "매개변수가 9.90% 느는 곳은 어디인가"
    [3]의 단계별 표가 답을 준다. 늘어난 2,530,992개 가운데 1,579,392개, 곧 62.4%가 layer4 세 블록에서 나온다. $C = 2048$이라 블록 하나가 526,464개를 쓰는데, 이는 layer1 블록 하나(8,464개)의 62.2배다. 채널이 8배이고 매개변수의 으뜸항이 $C^2$이니 64배가 나와야 할 것 같지만, 치우침이 $C$에만 비례해 함께 커지지 않으므로 62.2배에 그친다. 어느 쪽으로 읽든 요점은 같다 — SE의 부담은 채널 수의 제곱으로 불어나므로 깊은 단계에 몰린다. 뒤쪽 단계에서만 $r$을 키우는 변형이 있는 까닭이 이것이다.

---

## 5. CNN 구조에 넣기

SE 블록은 어떤 CNN 구조에도 끼워 넣을 수 있다. 다만 잔차 구조에서는 **어디에 넣는가**가 중요하다.

**ResNet-SE**: SE는 잔차 가지의 출력에 적용하고, 그 결과를 건너뛰기 연결에 더한다.

$$\mathbf{y} = \mathbf{x} + \text{SE}\big(F_3(F_2(F_1(\mathbf{x})))\big)$$

더한 뒤에 SE를 씌우면 안 된다. 그렇게 하면 $\mathbf{y} = \mathbf{s} \odot (\mathbf{x} + F(\mathbf{x}))$가 되어 항등 경로까지 $s_c < 1$로 깎이고, 3절에서 본 대로 항등 연결이 기울기에 $1$을 곱해 주던 성질이 사라진다. 위 식에서는 $\partial \mathbf{y} / \partial \mathbf{x}$의 항등 부분이 그대로 남는다. 4절의 `SEBottleneck`이 이 자리를 쓴다.

**DenseNet-SE**: 이어 붙이기 전에 조밀 블록의 출력에 적용한다.

**EfficientNet-SE**: 기본 구조 설계에 아예 들어 있는 부품이다. 깊이별 분리 합성곱 뒤에 SE를 두어, [깊이별 분리 합성곱](../cnn/depthwise_separable.md)이 채널 사이의 섞임을 $1 \times 1$ 합성곱 하나로만 처리하는 것을 채널 어텐션으로 보완한다.

---

## 6. 실험으로 드러난 성질

!!! warning "이 쪽이 재지 않은 것"
    Hu 등(2018, CVPR)은 ImageNet에서 ResNet-50의 top-1 오류율 24.80%가 SE-ResNet-50에서 23.29%로 내려갔다고 보고한다. 1.51%p 차이다. 그러나 이것은 **씨앗 하나로 한 번 학습한 값**이고, 씨앗을 바꾸어 가며 잰 퍼짐은 함께 보고되지 않았다. 이 책의 규칙은 한 번 돌린 것은 잰 것이 아니며 퍼짐보다 작은 차이는 차이가 아니라는 것이다([CIFAR-10 데이터셋](../cnn/03_cifar10_dataset.md)의 사다리가 그 보기다). 이 쪽은 ImageNet 학습을 돌리지 않으므로 위 값을 이 책이 잰 값으로 내세우지 않는다. 4절에서 이 쪽이 직접 잰 것은 매개변수와 곱셈-덧셈 횟수뿐이며, 그 둘은 씨앗과 무관하게 정확히 재현된다.

정확도 이야기와 달리, **값이 싸다는 쪽은 이 쪽에서 확인된다**. 매개변수 +9.90%에 연산량 +0.331%라는 비대칭이 SE가 널리 쓰이게 된 실제 이유다 — 곱셈-덧셈 횟수는 거의 그대로이고, 늘어나는 것은 주로 저장 공간이다. 다만 벽시계 시간은 이 쪽에서 재지 않았다. 성분별 곱과 풀링은 곱셈-덧셈이 적어도 메모리를 훑는 일이라 연산량에 비례하지 않으며, 그 값은 기계마다 다르다.

**해석 가능성**: 채널 중요도 가중치 $s_c$를 그려 보면 어느 채널이 눌리고 어느 채널이 키워지는지 볼 수 있다. 다만 채널 번호는 그 자체로 뜻을 갖지 않으므로, 이 그림을 "모델이 이미지의 어디를 본다"로 읽으면 안 된다. 그 물음에 답하는 것은 공간 어텐션이다([비국소 신경망](non_local.md)).

**전이 가능성**: SE로 보강한 신경망은 전이 학습에서도 잘 쓰이지만, 이 역시 원 논문과 후속 연구의 보고이지 이 쪽에서 잰 값은 아니다.

---

## 7. 변형과 확장

**공간과 채널을 함께 쓰는 압축-여기 (scSE)**: SE의 채널 문에 공간 문($1 \times 1$ 합성곱으로 $H \times W$짜리 지도를 만들어 곱하는 것)을 나란히 두고 둘을 합친다.

**효과적인 압축-여기 (ESE)**: 완전 연결층 두 개를 하나로 줄인 변형이다. 병목을 없애므로 매개변수는 $2C^2/r$에서 $C^2$으로 오히려 늘지만, 중간의 차원 축소에서 잃는 정보가 없어진다.

**합성곱 블록 어텐션 모듈 (CBAM)**: 채널 어텐션과 공간 어텐션을 차례로 적용하고, 채널 쪽에서는 평균 풀링과 최대 풀링을 함께 쓴다. [CBAM](../../appendix/cnn/cbam.md)에서 자세히 다룬다.

**좌표 어텐션**: 전역 평균 풀링이 공간 정보를 통째로 버리는 것을 보완하려고, 가로와 세로 방향으로 따로 풀링하여 위치 정보를 문 안에 남긴다.

---

## 8. 관련 주제

- [합성곱 신경망에서 트랜스포머로](cnn_to_vit_bridge.md) — SE가 순수 CNN과 ViT 사이의 어느 단계에 놓이는지
- [비국소 신경망](non_local.md) — 같은 절의 공간 어텐션 쪽 짝
- [CBAM](../../appendix/cnn/cbam.md) — 채널 어텐션과 공간 어텐션을 잇대어 쓰는 확장
- [ResNet 구현](../residual/02_resnet_implementation.md) — 5절이 SE를 끼워 넣는 병목 블록
- [항등 사상](../residual/identity_mapping.md) — 3절의 "항등 연결이 없다"가 무엇을 잃는지
- [자기 어텐션](../../ch11/attention/self_attention.md) — 채널이 아니라 위치 사이의 관계를 보는 어텐션

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
채널이 $C = 256$개이고 축소 비율이 $r = 16$인 SE 블록의 매개변수를 세어라. 치우침까지 포함한 값과 $2C^2/r$만 센 값을 모두 구하고, 같은 채널 수의 $3 \times 3$ 합성곱과 견주어라.

</div>

??? success "연습문제 1 풀이"
    가중치는 $W_1$이 $(C/r) \times C = 16 \times 256 = 4{,}096$개, $W_2$이 $C \times (C/r) = 256 \times 16 = 4{,}096$개로 모두 $2C^2/r = 8{,}192$개다. 여기에 치우침 $\mathbf{b}_1$이 $C/r = 16$개, $\mathbf{b}_2$이 $C = 256$개, 합해서 272개가 더 붙는다. 따라서 `nn.Linear` 두 개로 만든 실제 모듈의 매개변수는 **8,464개**다. 4절의 출력 [2]가 이 수를 찍는다.

    치우침 없는 $3 \times 3$ 합성곱은 $256 \times 256 \times 3 \times 3 = 589{,}824$개이므로 비는 $8{,}464 / 589{,}824 = 1.435\%$다. $2C^2/r$만 세면 1.389%가 나오는데, 두 값 모두 "약 1.4%"로 반올림되므로 어느 쪽을 적어도 틀려 보이지 않는다. 그렇기 때문에 더 위험하다 — 채널이 적은 블록에서는 $C$가 $2C^2/r$에 견주어 상대적으로 크므로 치우침의 몫이 커진다. $C = 64$, $r = 16$이면 가중치 512개에 치우침 68개로, 치우침이 합계의 11.7%를 차지한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SE 블록을 처음부터 유도하라. 전역 평균 풀링, 축소 비율이 $r$인 완전 연결층 두 개, 시그모이드 문이 각각 무엇을 맡는지 밝혀라.

</div>

??? success "연습문제 2 풀이"
    입력은 특징 맵 $X \in \mathbb{R}^{C \times H \times W}$이다.

    (1) **압축**: $z_c = \frac{1}{HW}\sum_{i,j} X_{c,i,j}$. 채널마다 공간 전체의 평균을 잡아 $\mathbf{z} \in \mathbb{R}^{C}$을 만든다. 공간 축만 없애고 채널 축은 남긴다. 이 한 걸음이 하는 일은 수용 영역을 특징 맵 전체로 넓히는 것이다 — 뒤의 문이 "이 이미지 전체에서" 어떤 채널이 중요한지 판단할 수 있게 된다.

    (2) **여기**: $\mathbf{s} = \sigma(W_2\,\delta(W_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2)$이며 $W_1 \in \mathbb{R}^{(C/r) \times C}$, $W_2 \in \mathbb{R}^{C \times (C/r)}$이다. 완전 연결층이므로 모든 채널의 통계량이 모든 채널의 문 값에 영향을 준다 — 이것이 "채널 사이의 상호 의존"의 구체적인 뜻이다. 사이에 ReLU를 넣어야 두 층이 하나의 선형 사상으로 무너지지 않는다. 병목 $C/r$은 매개변수를 $C^2$에서 $2C^2/r$로 줄인다.

    (3) **배율 적용**: $\tilde{X}_{c,i,j} = s_c \cdot X_{c,i,j}$. $s_c$을 그 채널의 $H \times W$ 칸 전체에 퍼뜨려 곱한다. 시그모이드가 $s_c$을 $(0, 1)$에 가두므로 문은 채널을 줄이기만 하고 키우지는 않는다. '여기(excitation)'라는 이름과 달리 이 장치는 **상대적으로 덜 누르는 것**이지 절대적으로 키우는 것이 아니다. 모든 채널이 함께 작아져 출력의 크기가 줄어드는 것 자체는 문제가 되지 않는다 — 뒤따르는 층의 가중치가 그만큼 커지는 쪽으로 함께 학습되기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$r = 16$일 때 압축(전역 평균 풀링)보다 여기(완전 연결층 두 개)의 계산량이 더 커지는 조건을 $C$와 $HW$로 나타내어라. ResNet-50의 네 단계 가운데 어느 단계가 그 조건을 만족하는가?

</div>

??? success "연습문제 3 풀이"
    압축은 $CHW$번, 여기는 $2C^2/r$번의 곱셈-덧셈을 쓴다. 여기가 더 크다는 것은

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

    `flatten(1)`은 $(N, C, 1, 1)$을 $(N, C)$로 펴고, `view(N, C, 1, 1)`은 다시 곱하기 좋게 되돌린다. 이 두 번의 모양 바꿈을 건너뛰고 $(N, C, 1, 1)$짜리 문을 그대로 곱해도 방송으로 같은 결과가 나오지만, 완전 연결층이 마지막 축을 쓰므로 중간에 한 번은 펴야 한다.

    **자리의 차이**: `self.se(o) + idt`은 $\mathbf{y} = \mathbf{x} + \mathbf{s} \odot F(\mathbf{x})$이고, `self.se(o + idt)`은 $\mathbf{y} = \mathbf{s} \odot (\mathbf{x} + F(\mathbf{x}))$이다. 앞의 것은 $\partial \mathbf{y} / \partial \mathbf{x}$에 항등 부분이 그대로 남지만, 뒤의 것은 항등 경로까지 $s_c$로 곱해져 깎인다. 4절의 실험에서 $s$는 0.3418에서 0.6783 사이였으므로, 뒤의 방식으로 ResNet-50의 병목 블록 $3+4+6+3 = 16$개를 쌓으면 항등 경로의 기울기가 $0.5^{16} \approx 1.5 \times 10^{-5}$ 규모로 줄어든다. ResNet이 애써 만든 성질을 SE가 도로 무너뜨리는 셈이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
SE 블록은 어텐션 장치와 어떤 관계인가? 어떤 종류의 어텐션을 구현하며, 트랜스포머의 자기 어텐션과 무엇이 다른가?

</div>

??? success "연습문제 5 풀이"
    SE는 **채널 어텐션**을 구현한다. 전역 맥락에 따라 특징 맵의 채널마다 중요도를 매기는 법을 배운다. 공간 어텐션(어디를 볼지)이나 자기 어텐션(쌍의 관계)과 달리 채널 어텐션은 "이 입력에는 어떤 특징이 중요한가"에 답한다.

    자기 어텐션과 견주면 세 가지가 다르다.

    - **가중치의 개수**: 자기 어텐션은 $N$개 위치에 대해 $N \times N$짜리 가중치를 만들지만, SE는 $C$개의 수만 만든다. 그래서 계산량이 $N$에 이차가 아니라 $HW$에 일차다 — 4절의 $O(CHW + 2C^2/r)$이 그것이다.
    - **질의가 없다**: 자기 어텐션은 위치마다 질의를 만들어 다른 위치와 견주지만, SE에는 질의가 없고 전역 평균 하나가 모든 채널의 문을 정한다. 위치에 따라 다른 문을 쓰지 못한다는 뜻이다.
    - **곱의 대상**: 자기 어텐션의 가중치는 값들의 볼록 결합을 만드는 계수(합이 1)이지만, SE의 $s_c$는 각 채널에 독립적으로 붙는 배율이라 합이 1일 까닭이 없다. 소프트맥스가 아니라 시그모이드를 쓰는 것이 이 차이의 표시다.

    그런 뜻에서 SE는 트랜스포머의 온전한 어텐션에 앞선 가벼운 선구자이며, [합성곱 신경망에서 트랜스포머로](cnn_to_vit_bridge.md)가 그 사이의 단계들을 늘어놓는다.

---

## 정리하며

SE 블록은 공간 차원을 전역 평균으로 눌러 채널마다 한 수를 얻고($\mathbf{z}$), 완전 연결층 두 개와 시그모이드로 채널마다의 배율을 정하고($\mathbf{s}$), 그 배율을 채널별로 곱한다. 더하지 않고 곱하기 때문에 채널을 거의 0까지 누를 수 있고, 같은 이유로 그 길로 오는 기울기가 $s_c < 1$배로 깎이므로 잔차 가지 **안**에 두어야 한다.

값이 싸다는 말은 나누어 읽어야 한다. ResNet-50에 다 달았을 때 합성곱과 완전 연결의 곱셈-덧셈은 4.0892 G에서 4.0917 G로 0.062%, 흔히 빠뜨리는 풀링과 채널별 곱까지 세도 0.331%만 늘지만, 매개변수는 25,557,032개에서 28,088,024개로 9.90% 늘며 그중 62.4%가 $C = 2048$인 layer4에서 나온다. 매개변수 공식 $2C^2/r$은 치우침 $C/r + C$을 빼놓은 값이라, $C = 256$, $r = 16$에서 8,192개가 아니라 8,464개가 맞다. 이 쪽의 수는 모두 실제 모듈에서 센 것이며, 정확도 향상은 이 쪽이 재지 않았으므로 논문의 보고로만 적어 두었다.
