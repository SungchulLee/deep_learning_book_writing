# WaveNet 기본

WaveNet 기본.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 코드

```python
"""WaveNet 기본."""
# ---
# title: "WaveNet: 기본 늘린 인과 겹말기"
# description: "차례 나타내기를 위한 늘린 인과 겹말기의 PyTorch 짜기"
# ---
#
# WaveNet(van den Oord 외, 2016)은 늘린 인과 겹말기를 쌓아
# 받는 자리를 지수로 키우면서도
# 매개변수가 지수로 늘지 않게 한다.
#
# 이 대본은 핵심 벽돌을 PyTorch로 짠다:
#   1. 인과 Conv1d(앞날이 새지 않도록 왼쪽을 채운다)
#   2. 단순한 늘린 겹말기 쌓기(문 없음)
#   3. 인공 시계열 내다보기 과제로 익히기
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
    """때 t의 내놓기가 때 ≤ t의 들임에만 매이도록
    인과(왼쪽) 채우기를 한 Conv1d."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,          # 손수 채운다
        )

    def forward(self, x):
        # x: (배치, 채널, 순차열 길이)
        x = F.pad(x, (self.padding, 0))   # 왼쪽만 채운다
        return self.conv(x)


# ─── 단순한 늘린 쌓기 ──────────────────────────────────────────────────────
class SimpleDilatedStack(nn.Module):
    """ReLU 깨움을 갖춘 늘린 인과 겹말기 쌓기.
    늘림 비율이 층마다 두 배가 된다: 1, 2, 4, 8, 1, 2, 4, 8, …"""

    def __init__(self, in_channels=1, hidden=20, n_layers=8, kernel_size=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** (i % 4)          # 돌기: 1,2,4,8
            c_in = in_channels if i == 0 else hidden
            layers.append(CausalConv1d(c_in, hidden, kernel_size, dilation))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(hidden, 10, kernel_size=1))  # 점마다 내놓기
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── 자료 만들기 ───────────────────────────────────────────────────────────
def generate_time_series(batch_size, n_steps):
    """여러 걸음 앞을 헤아리는 단순한 목표를 만든다."""
    freq1, freq2 = np.random.uniform(0.1, 0.5, (2, batch_size, 1))
    offsets1, offsets2 = np.random.uniform(0, 2 * np.pi, (2, batch_size, 1))
    time = np.linspace(0, 1, n_steps + 10).reshape(1, -1)
    series = 0.5 * np.sin((time - offsets1) * (n_steps * freq1))
    series += 0.2 * np.sin((time - offsets2) * (n_steps * freq2))
    series += 0.1 * (np.random.randn(batch_size, n_steps + 10) - 0.5)
    return series.astype(np.float32)


np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps)
X_train = torch.tensor(series[:7000, :n_steps]).unsqueeze(1)      # (B,1,T)
Y_train = torch.tensor(series[:7000, 1:n_steps + 1]).unsqueeze(1)
X_valid = torch.tensor(series[7000:9000, :n_steps]).unsqueeze(1)
Y_valid = torch.tensor(series[7000:9000, 1:n_steps + 1]).unsqueeze(1)
X_test  = torch.tensor(series[9000:, :n_steps]).unsqueeze(1)
Y_test  = torch.tensor(series[9000:, 1:n_steps + 1]).unsqueeze(1)

print(f"X_train shape: {X_train.shape}  Y_train shape: {Y_train.shape}")

# ─── 익히기 ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDilatedStack(in_channels=1, hidden=20, n_layers=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}\n")

epochs = 20
batch_size = 128
train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = X_train[idx].to(device), Y_train[idx].to(device)
        pred = model(xb)
        # 마지막 때 걸음 10개만 견준다(모델이 채널 10개를 내놓는다)
        loss = loss_fn(pred[:, :, -10:], yb[:, :, -10:].expand_as(pred[:, :, -10:]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    train_losses.append(epoch_loss / n_batches)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_valid.to(device))
        val_loss = loss_fn(val_pred[:, :, -10:], Y_valid[:, :, -10:].to(device).expand_as(val_pred[:, :, -10:]))
        val_losses.append(val_loss.item())

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}")

# ─── 그림 ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="valid")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Simple Dilated Causal Conv — Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("wavenet_basic_training.png", dpi=150)
plt.show()
print("Done.")


if __name__ == "__main__":
    pass```

## 논의

이 짜기는 갈래 2개(`CausalConv1d`, `SimpleDilatedStack`)를 뜻매김하며 이들이 함께 온전한 자기 되돌이 모델 얼개를 이룬다. 갈래마다 뚜렷이 구분되는 부품을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 자동 미분에 쓰는 셈 그래프를 뜻매김한다.

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
