# 게이트·잔차·스킵 연결을 사용하는 WaveNet

이 예제는인과적 확장 합성곱에 게이트와 잔차·스킵 연결을 추가한다. [WaveNet 원 논문](https://arxiv.org/abs/1609.03499)의 구조적 아이디어를 살펴보되, 실제 학습은 인공 연속 시계열의 다음 값을 MSE로 예측한다. 오디오 샘플의 확률 분포나 음성 합성 시스템 전체를 구현하는 것은 아니다.

## 1. 구현

```python
"""문 달린 WaveNet."""
# ---
# title: "WaveNet: 문 달린 남은 덩이"
# description: "문 달린 깨움과 건너뛰기 이음을 갖춘 온전한 WaveNet 구조"
# ---
#
# 이 대본은 온전한 WaveNet 구조를 PyTorch로 짜며 다음을 담는다:
#   1. 문 달린 깨움 낱개(tanh ⊙ 시그모이드)
#   2. 늘린 인과 겹말기를 갖춘 남은 덩이
#   3. 모든 층에 걸쳐 모은 건너뛰기 이음
#
# 문 달린 깨움이 여느 늘린 겹말기와 다른 핵심이다:
#
#       z = tanh(W_f * x) ⊙ σ(W_g * x)
#
# 여기서 W_f과 W_g은 따로 있는 거르개와 문 겹말기 무게이다.
#
# 바탕: O'Reilly Hands-On ML 15장(TF → PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


# ─── 인과 Conv1d ───────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


# ─── 문 달린 깨움 낱개 ────────────────────────────────────────────────────
class GatedActivationUnit(nn.Module):
    """채널 차원을 반으로 갈라 다음을 쓴다.
       내놓기 = tanh(앞 반) * 시그모이드(뒤 반)
    """

    def forward(self, x):
        # x: (배치, 2*n_filters, 차례 길이)
        n_filters = x.size(1) // 2
        return torch.tanh(x[:, :n_filters]) * torch.sigmoid(x[:, n_filters:])


# ─── WaveNet 남은 덩이 ────────────────────────────────────────────────────
class WaveNetResidualBlock(nn.Module):
    """남은 덩이 하나:
       늘린 겹말기 → 문 달린 깨움 → 1x1 겹말기 → 남은 것 더하기
       1x1 내놓기는 건너뛰기 이음으로도 돌려준다.
    """

    def __init__(self, n_filters, kernel_size=2, dilation=1):
        super().__init__()
        self.dilated_conv = CausalConv1d(n_filters, 2 * n_filters, kernel_size, dilation)
        self.gate = GatedActivationUnit()
        self.residual_conv = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.skip_conv = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x):
        z = self.dilated_conv(x)
        z = self.gate(z)
        skip = self.skip_conv(z)
        residual = self.residual_conv(z) + x
        return residual, skip


# ─── 온전한 WaveNet ────────────────────────────────────────────────────────
class WaveNet(nn.Module):
    """늘린 인과 겹말기 덩이 여럿과
    문 달린 깨움, 남은 이음 + 건너뛰기 이음을 갖춘 온전한 WaveNet.

    인수:
        in_channels:        들임 특징의 수(한 변수이면 1)
        n_filters:          층마다 숨은 채널의 수
        n_layers_per_block: 덩이 안에서 늘림 비율이 1, 2, …, 2^(n-1)으로 간다
        n_blocks:           늘림 비율 돌기의 수
        n_outputs:          내놓기 채널의 수(예컨대 μ법이면 256)
    """

    def __init__(self, in_channels=1, n_filters=32, n_layers_per_block=3,
                 n_blocks=1, n_outputs=10, kernel_size=2):
        super().__init__()
        self.input_conv = CausalConv1d(in_channels, n_filters, kernel_size)

        self.residual_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            for i in range(n_layers_per_block):
                dilation = 2 ** i
                self.residual_blocks.append(
                    WaveNetResidualBlock(n_filters, kernel_size, dilation)
                )

        self.post_conv1 = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.post_conv2 = nn.Conv1d(n_filters, n_outputs, kernel_size=1)

    def forward(self, x):
        z = self.input_conv(x)
        skips = []
        for block in self.residual_blocks:
            z, skip = block(z)
            skips.append(skip)
        # 건너뛰기 이음을 모두 더한다
        s = torch.stack(skips, dim=0).sum(dim=0)
        s = F.relu(s)
        s = F.relu(self.post_conv1(s))
        return self.post_conv2(s)

    def receptive_field(self):
        """이론상 받는 자리의 크기를 셈한다."""
        rf = 1
        for block in self.residual_blocks:
            dilation = block.dilated_conv.conv.dilation[0]
            kernel = block.dilated_conv.conv.kernel_size[0]
            rf += (kernel - 1) * dilation
        return rf


# ─── 자료 ──────────────────────────────────────────────────────────────────
def generate_time_series(batch_size, n_steps):
    freq1, freq2 = np.random.uniform(0.1, 0.5, (2, batch_size, 1))
    off1, off2 = np.random.uniform(0, 2 * np.pi, (2, batch_size, 1))
    time = np.linspace(0, 1, n_steps + 10).reshape(1, -1)
    series = 0.5 * np.sin((time - off1) * (n_steps * freq1))
    series += 0.2 * np.sin((time - off2) * (n_steps * freq2))
    series += 0.1 * (np.random.randn(batch_size, n_steps + 10) - 0.5)
    return series.astype(np.float32)


np.random.seed(42)
torch.manual_seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps)
X_train = torch.tensor(series[:7000, :n_steps]).unsqueeze(1)
Y_train = torch.tensor(series[:7000, 1:n_steps + 1]).unsqueeze(1)
X_valid = torch.tensor(series[7000:9000, :n_steps]).unsqueeze(1)
Y_valid = torch.tensor(series[7000:9000, 1:n_steps + 1]).unsqueeze(1)

# ─── 모델 ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WaveNet(
    in_channels=1,
    n_filters=32,         # 본디 논문에서는 128
    n_layers_per_block=3, # 본디 논문에서는 10
    n_blocks=1,           # 본디 논문에서는 3
    n_outputs=1,          # 되돌아 기대기(내놓기 채널 1개)
).to(device)

print(f"WaveNet parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Receptive field:    {model.receptive_field()} time steps")
print(f"Device:             {device}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ─── 익히기 ────────────────────────────────────────────────────────────────
epochs = 25
batch_size = 128
train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss, n = 0.0, 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i : i + batch_size]
        xb = X_train[idx].to(device)
        yb = Y_train[idx].to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n += 1
    train_losses.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        vp = model(X_valid.to(device))
        vl = loss_fn(vp, Y_valid.to(device)).item()
    val_losses.append(vl)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}")

# ─── 그려 보기 ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 손실 곡선들
ax1.plot(train_losses, label="train")
ax1.plot(val_losses, label="valid")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.set_title("WaveNet Training Curve")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 표본 헤아림
model.eval()
with torch.no_grad():
    sample_x = X_valid[:3].to(device)
    sample_pred = model(sample_x).cpu().squeeze(1).numpy()
    sample_gt = Y_valid[:3].squeeze(1).numpy()

for i in range(3):
    ax2.plot(sample_gt[i], alpha=0.4, label=f"true {i}" if i == 0 else None, color="C0")
    ax2.plot(sample_pred[i], alpha=0.7, label=f"pred {i}" if i == 0 else None, color="C1", linestyle="--")
ax2.set_title("WaveNet Predictions (3 samples)")
ax2.set_xlabel("Time step")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("wavenet_gated_results.png", dpi=150)
plt.show()
print("Done.")


if __name__ == "__main__":
    pass
```

CausalConv1d는 왼쪽 패딩으로 길이를 유지한다. 각 잔차 블록의 확장 합성곱은 H채널을 2H채널로 만들고, 이를 반씩 나눠 tanh(a)⊙sigmoid(b)를 계산해 다시 H채널로 만든다. 게이트는 특징의 통과 크기를 조절하며 곱셈만으로 순환 기억·장기 안정성을 보장하지는 않는다.

게이트 특징은 1×1 합성곱 두 개로 나뉜다. 잔차 출력은 입력에 더해 다음 블록으로 보내고 스킵 출력은 마지막에 모아 예측 헤드로 보낸다. 모든 경로가 미래를 보지 않으면 이 합도 인과성을 유지한다. 마지막 블록의 잔차 출력은 이후 사용되지 않아 해당 residual_conv 파라미터는 최종 손실에 기여하지 않지만 파라미터 집계에는 포함된다.

**기존 출력 기록:**

```
WaveNet parameters: 20,001
Receptive field:    8 time steps
Device:             cpu

Epoch   1  train=0.06757  val=0.02158
Epoch   5  train=0.02100  val=0.02083
Epoch  10  train=0.02070  val=0.02067
Epoch  15  train=0.02050  val=0.02037
Epoch  20  train=0.02020  val=0.02016
Epoch  25  train=0.02004  val=0.02010
Done.
```

기존 출력·그림 기록은 보존했으며 이 편집에서 학습을 실행하거나 재생성하지 않았다. 특히 아래 수용 영역 메서드의 누락 때문에 출력된 수용 영역 숫자를 실제 전체 경로의 값과 구분해야 한다.

## 2. 모양·파라미터와 수용 영역

실행 설정은 H=32, dilation 1·2·4의 블록 3개와 출력 1채널이다. 입력 `(B,1,50)`은 input_conv 뒤 `(B,32,50)`, 각 확장 합성곱 뒤 `(B,64,50)`, 게이트·잔차·스킵 뒤 `(B,32,50)`이고 최종 출력은 `(B,1,50)`이다. 학습 파라미터 수는 20,001개다. 클래스 기본 n_outputs=10과 실행의 1은 다르다.

보폭 1·동일 커널 k에서 전체 최대 수용 영역은 입력 합성곱까지 포함해 다음과 같다.

$$R=1+(k-1)\left[1+B\sum_{i=0}^{L-1}2^i\right]
=1+(k-1)[1+B(2^L-1)].$$

여기서 B는 반복 스택 수 n_blocks, L은 스택당 확장 층 수다. 배치 크기와 다른 기호다. 실행의 k=2,B=1,L=3에서는 R=9다. receptive_field 메서드는 잔차 블록만 더해 8을 반환하며 input_conv의 기여 1을 빠뜨린다. 스킵 중 가장 깊은 경로의 실제 문맥은 9개다.

## 3. 타깃과 학습 설정

입력은 시점 0~49, 타깃은 1~50으로 한 칸 이동한다. 각 위치의 출력은 현재 입력과 과거를 보고 다음 값을 예측하므로 미래 타깃을 직접 보지 않는다. 이번 손실은 마지막 10개가 아니라 **모든 50개 시간 위치**에 대한 평균 MSE다. 초반 위치에는 왼쪽 0 패딩이 포함된다.

이는 [기본 모델](01_wavenet_basic.md)의 출력 10채널·마지막 10개 위치 반복 타깃 손실과 다르다. 두 손실 숫자를 게이트·잔차 구조만 바꾼 동일 조건 실험으로 비교할 수 없다. 공정한 구조 비교에는 출력·타깃·유효 시간 위치를 먼저 맞춰야 한다.

독립 시퀀스 10,000개 중 앞 7,000개를 학습, 다음 2,000개를 검증에 쓴다. 나머지 1,000개에 대한 테스트는 없다. NumPy·PyTorch 시드를 지정하지만 모든 장치·버전의 동일 결과까지 보장하지 않는다. Adam 0.001로 25에포크 학습하며 학습 곡선은 배치 평균들의 단순 평균, 검증은 전체 자료 평균이다.

## 4. 결과의 범위

오른쪽 예측 그림은 각 시점에 실제 과거 입력을 준 한 스텝 예측이다. 예측을 다시 입력에 넣어 여러 미래를 만드는 자유 롤아웃이나 새 오디오 생성이 아니다. 확률적 생성에는 조건부 분포·초기 문맥·샘플링 규칙을 별도로 정의해야 한다.

자료 생성의 잡음은 평균 −0.05이며 나머지 구성은 기본 예제와 같다. 코드의 자료 생성·학습·표시는 최상위에서 실행되어 import에도 부수 효과가 있다. 체크포인트·조기 종료·독립 테스트 성능은 이 페이지에서 구현하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
실행 모델의 블록 모양과 파라미터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    배치 4·길이 50에서는 input_conv가 `(4,1,50)`→`(4,32,50)`이다. 각 확장 합성곱은 64채널, 게이트는 32채널이고 잔차·스킵 길이도 같다. 최종 출력은 `(4,1,50)`이다. 입력 합성곱 96개, 각 잔차 블록 6,272개씩 3개, post_conv1 1,056개와 post_conv2 33개로 총 20,001개다. 풀링·2차원 합성곱은 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
전체 수용 영역을 계산하고 메서드가 반환하는 값과 비교하라.

</div>

??? success "연습문제 2 풀이"

    실제 최대 경로에는 input_conv의 커널 2·dilation 1과 잔차 블록의 dilation 1·2·4가 있다. 따라서 R=1+1+1+2+4=9다. 메서드는 입력 합성곱을 세지 않아 8을 반환한다. 모든 위치의 타깃이 존재하도록 한 칸 이동을 유지해야 하며 2차원 RGB 구조 변경은 이 시계열 Conv1d 예제의 직접적인 확장 문제가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
게이트 블록의 비용을 단순 합성곱과 비교할 때 무엇을 포함해야 하는가?

</div>

??? success "연습문제 3 풀이"

    게이트용 확장 Conv1d는 H→2H이므로 가중치가 2H²k개다. 두 1×1 경로는 각각 H²개이며 편향도 더한다. H채널의 단순 합성곱 하나와 비교하면 게이트·잔차·스킵 비용이 추가된다. Depthwise 분리 대안을 쓴다면 1차원 커널 k의 식을 사용하고 각 경로의 채널 수를 맞춰 계산해야 한다. 비용 변화가 같은 예측 성능이나 속도 변화를 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
깊이 설정을 바꾸는 기존 인수와 유효 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    WaveNet에는 이미 n_layers_per_block와 n_blocks가 있고 residual_blocks는 ModuleList로 등록된다. 각 스택에서 dilation이 다시 1로 시작한다. 둘 중 하나를 0으로 만들어 스킵 목록이 비면 torch.stack이 실패하므로 양수 조건이 필요하다. CausalConv1d에는 num_layers 인수가 없으며 아래 YourBlock 골격만 추가한다고 전체 모델이 확장되지 않는다. 깊이별 실제 수용 영역·파라미터 수·유효 학습 위치를 함께 비교한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

게이트·잔차·스킵 경로는 인과적 합성곱 특징을 조합한다. 이 실행은 1채널 다음 값 회귀이며 실제 수용 영역은 입력 합성곱까지 포함한 9다. 한 스텝 예측 그림과 확률적 자유 생성, 다른 손실 설정의 기본 모델 비교를 구분해야 한다.
