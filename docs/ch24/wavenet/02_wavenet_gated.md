# 문 달린 WaveNet

문 달린 WaveNet.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 코드

```python
"""문 달린 WaveNet."""
# ---
# title: "WaveNet: 문 달린 남은 덩이"
# description: "문 달린 깨움과 건너뛰기 이음을 갖춘 온전한 WaveNet 얼개"
# ---
#
# 이 대본은 온전한 WaveNet 얼개를 PyTorch로 짜며 다음을 담는다:
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
        # x: (묶음, 2*n_filters, 차례 길이)
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
    pass```

## 논의

이 짜기는 갈래 4개(`CausalConv1d`, `GatedActivationUnit`, `WaveNetResidualBlock`, `WaveNet`)를 뜻매김하며 이들이 함께 온전한 자기 되돌이 모델 얼개를 이룬다. 갈래마다 뚜렷이 구분되는 부품을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 자동 미분에 쓰는 셈 그래프를 뜻매김한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 소리 만들어 내기 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`CausalConv1d` 앞먹임을 지나는 텐서 꼴을 좇아라. 기본 매개변수로 들임 표본 4개 묶음에 대해 큰 셈(겹말기, 모으기, 선형 층)마다 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 겹말기 층의 `in_channels`을 지금 값에서 3으로 바꾸어라. 공식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 겹말기와 모으기 층마다 뒤의 공간 차원을 다시 셈하라. 첫 선형 층의 `in_features`을 마지막 겹말기/모으기 층의 펼친 내놓기에 맞게 고쳐라. `model = CausalConv1d(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
`CausalConv1d`을 층이나 덩이의 수를 맞출 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 2, 4, 8층으로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이하라. (여느 파이썬 목록이 아니라) `nn.ModuleList`을 쓰면 PyTorch가 모든 매개변수를 가장 좋게 하기에 등록한다. `for n in [2, 4, 8]: model = CausalConv1d(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험하라.
