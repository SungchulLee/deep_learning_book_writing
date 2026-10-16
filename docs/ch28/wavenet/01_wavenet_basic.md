# WaveNet 기초: 인과적 확장 컨볼루션

WaveNet의 핵심 요소 중 하나는 인과적 확장 컨볼루션(dilated causal convolution)이다. 과거만 사용하는 컨볼루션을 쌓고 dilation을 키워 긴 문맥을 처리한다. [원 WaveNet](https://arxiv.org/abs/1609.03499)은 오디오의 조건부 분포를 모델링하지만 이 절의 예제는 인공 시계열의 다음 값을 MSE로 예측하는 단순 회귀 모델이다.

## 1. 구현

```python
"""WaveNet 기본."""
# ---
# title: "WaveNet: 기본 늘린 인과 겹말기"
# description: "차례 나타내기를 위한 늘린 인과 겹말기의 PyTorch 짜기"
# ---
#
# WaveNet(van den Oord 외, 2016)은 늘린 인과 겹말기를 쌓아
# 받는 자리를 지수로 키우면서도
# 파라메터가 지수로 늘지 않게 한다.
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
    늘림 비율이 레이어마다 두 배가 된다: 1, 2, 4, 8, 1, 2, 4, 8, …"""

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
    pass
```

CausalConv1d는 왼쪽에만 (k−1)d개를 패딩하고 스트라이드 1 컨볼루션을 적용해 길이를 유지한다. 출력 위치 t는 현재 입력 x_t와 과거에 의존하고 미래 x_(t+1)에는 의존하지 않는다. 같은 위치 출력을 다음 값 x_(t+1)의 타깃과 맞추므로 이 예제의 자기회귀 정렬은 성립한다. 현재 토큰 자체를 타깃으로 쓰려면 별도 이동이 필요하다.

SimpleDilatedStack의 기본 은닉 채널은 20, 커널은 2, 컨볼루션 레이어는 8개다. dilation은 1,2,4,8을 두 번 반복한다. 마지막 1×1 컨볼루션이 10채널 출력을 만들며 입력 `(B,1,T)`에서 출력은 `(B,10,T)`다. 기본 학습 파라미터는 6,010개다.

**기존 출력 기록:**

```
X_train shape: torch.Size([7000, 1, 50])  Y_train shape: torch.Size([7000, 1, 50])

Model parameters: 6,010
Device: cpu

Epoch   1  train_loss=0.1601  val_loss=0.1527
Epoch   5  train_loss=0.0189  val_loss=0.0188
Epoch  10  train_loss=0.0177  val_loss=0.0177
Epoch  15  train_loss=0.0172  val_loss=0.0175
Epoch  20  train_loss=0.0171  val_loss=0.0171
Done.
```

기존 기록은 보존했으며 이 편집에서 학습이나 그림 생성을 실행하지 않았다. 입력·타깃 모양, 로스의 의미와 수치 재현을 구분해 읽는다.

## 2. 리셉티브 필드와 길이

스트라이드 1에서 전체 리셉티브 필드는 R=1+Σ_l(k_l−1)d_l이다. 이 코드에서는 R=1+2(1+2+4+8)=31개 입력 위치다. 단순히 레이어 수가 8이므로 dilation이 끝까지 2⁷까지 증가한다고 계산하면 안 된다. 시퀀스 초반에는 리셉티브 필드 일부가 0 패딩에 해당한다.

마지막 10개 시간 위치 t=40,…,49의 예측만 학습해 각각 실제 다음 값 41,…,50과 비교한다. 이 위치들은 모두 31개 실제 과거·현재 관측을 사용할 수 있다. 전체 길이 50을 모두 로스에 넣는 실험과는 다르다.

## 3. 10채널은 10개 미래 시점이 아니다

Y는 `(B,1,50)`이고 마지막 시간 10개를 잘라 expand_as로 출력 10채널에 반복한다. 따라서 각 시간 t에서 **10채널 모두 같은 다음 값 하나**를 예측한다. 채널별로 x_(t+1),…,x_(t+10)을 예측하는 다중 지평 모델이 아니다. MSE는 배치·채널·선택한 시간의 평균이며 확률 분포의 분산·범주형 logits도 정의하지 않는다.

실제 10개 지평 예측을 만들려면 채널 h의 타깃을 x_(t+h)로 따로 정하고 유효 시간 범위를 맞춰야 한다. 이 절은 코드 변경 없이 현재 로스의 의미를 설명한다.

## 4. 자료와 평가

인공 자료는 각 시퀀스마다 다른 주파수·위상의 사인 성분 둘과 잡음을 결합한다. 잡음은 0.1(randn−0.5)이므로 평균이 −0.05이며 평균 0 가우시안 잡음이 아니다. frequency 변수의 역할도 정규화 시간과 n_steps 배율을 포함한 식으로 해석해야 한다.

독립적으로 만든 10,000개 시퀀스 중 7,000개는 학습, 2,000개는 검증, 1,000개는 테스트로 둔다. 테스트 텐서는 정의하지만 이 코드에서 평가하지 않는다. 학습 로그는 배치 평균들의 단순 평균이라 마지막 배치 크기의 영향을 받고, 검증은 전체 검증 텐서의 평균 MSE다. NumPy 시드는 고정하지만 PyTorch 초기화·배치 순서 시드는 고정하지 않는다.

코드는 최상위에서 자료 생성·학습·그림 저장을 수행하므로 import만 해도 실행되는 부분이 있다. 생성 함수나 오디오 분포·양자화 처리는 없으며 원 WaveNet의 전체 구현과 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치 크기 4,길이 50에서 텐서 모양과 리셉티브 필드를 계산하라.

</div>

??? success "연습문제 1 풀이"

    입력은 `(4,1,50)`이고 첫 인과 컨볼루션 뒤 `(4,20,50)`이다. 나머지 인과 컨볼루션·ReLU도 길이를 유지하며 최종 1×1 뒤 `(4,10,50)`이다. 풀링·Conv2d·평탄화는 없다. dilation 합이 30이므로 커널 2의 리셉티브 필드는 31이며 마지막 10개 시간 출력은 `(4,10,10)`으로 로스에 들어간다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
10개 출력 채널이 실제 10개 미래 지평이 되려면 타깃을 어떻게 정의해야 하는가?

</div>

??? success "연습문제 2 풀이"

    현재 타깃을 반복하는 대신 채널 h에 x_(t+h)를 대응시키는 별도 타깃 텐서를 구성해야 한다. 길이·패딩·관측 종료를 고려해 모든 h의 타깃이 존재하는 t만 평가한다. 현재 입력의 t까지가 예측 시점이며 다른 미래를 입력으로 넣으면 안 된다. 모양이 `(B,10,T)`라는 이유만으로 이미 10지평 모델인 것은 아니다. 이 Conv1d 예제에 RGB Conv2d 입력을 바로 주는 기존 질문은 적용되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
1차원 표준 컨볼루션과 depthwise separable convolution의 비용을 비교하라.

</div>

??? success "연습문제 3 풀이"

    바이어스를 제외하면 표준 Conv1d는 C_in C_out k개이고 depthwise와 pointwise는 C_in k+C_in C_out개다. 같은 출력 길이에서 웨이트·곱셈 누적 횟수의 비는 1/C_out+1/k다. 2차원 커널의 k² 식을 적용하지 않는다. 분리 구조의 표현 제약·실행 커널 때문에 파라미터 감소가 같은 정확도나 같은 속도 비율을 보장하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
깊이 변경은 어느 클래스에서 설정하며 리셉티브 필드는 어떻게 달라지는가?

</div>

??? success "연습문제 4 풀이"

    SimpleDilatedStack에 이미 n_layers가 있다. CausalConv1d는 한 레이어를 감싸므로 기존 클래스에 없는 num_layers를 전달하는 것은 맞지 않는다. 커널 2에서 n_layers=2,4,8이면 dilation 합이 각각 3,15,30이라 리셉티브 필드는 4,16,31이다. 4레이어마다 dilation이 다시 1로 돌아가므로 무조건 2^L로 계산하지 않는다. 아래 ModuleList는 미정 YourBlock을 사용하는 등록 골격이며 현재 Sequential을 반드시 대체해야 하는 것은 아니다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

인과적 확장 컨볼루션은 미래 입력을 보지 않으면서 문맥 범위를 넓힌다. 이 예제의 리셉티브 필드는 31이고 출력 10채널은 같은 다음 값 타깃을 반복해 학습한다. 원 WaveNet의 확률적 오디오 생성과 이 회귀 실험의 범위를 구분한다.
