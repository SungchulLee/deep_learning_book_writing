"""
================================================================================
make_figures.py - 3.3 다층 퍼셉트론의 그림 세 장을 만든다
================================================================================

만드는 그림:
    activations.svg        ReLU, 시그모이드, tanh와 그 도함수. 요점은 점선의 높이다
    backprop_flow.svg      순전파가 도함수를 붙들고, 역전파가 그것을 곱해 내려온다
    mlp_architecture.svg   784 -> 128 -> 10 사슬. 띠의 색은 실제로 잰 활성값이다

마지막 그림은 03_mlp.md의 모델을 그대로 학습시킨 뒤(ToTensor만, 묶음 100,
Adam 1e-3, 5 에포크, 씨앗 42) 시험 집합의 첫 이미지를 통과시켜 단계마다의
값을 그대로 칠한 것이다. 그래서 이 스크립트를 다시 돌리면 그림의 숫자도
다시 계산된다.

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
    MNIST_ROOT=~/data python make_figures.py      # 내려받아 둔 자료를 쓸 때

소요 시간: CPU에서 4~5분 (대부분 학습. 앞의 두 장은 즉시 나온다)
================================================================================
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA_ROOT = os.path.expanduser(os.environ.get("MNIST_ROOT", "./data"))

# 본문 그림과 같은 회색 계열에 붉은색 강조 하나
EDGE, FILL, WHITE = "#9a9a9a", "#f2f2f2", "#ffffff"
TEXT, MUTE, ACCENT, HILITE = "#333333", "#696969", "#dc143c", "#fff3c4"

plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False   # ASCII hyphen, not U+2212


def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.3, ls="-", zorder=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=zorder))


# ================================================================================
# 1부: 03_mlp.md의 모델을 학습시켜 한 이미지의 활성값을 얻는다
# ================================================================================
class MNISTClassifier(nn.Module):
    """03_mlp.md의 모델 그대로."""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


def train_and_capture(index=0):
    """모델을 학습시키고 시험 이미지 하나의 단계별 값을 돌려준다."""
    torch.manual_seed(42)
    np.random.seed(42)

    tf = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(DATA_ROOT, train=True,
                                           transform=tf, download=True)
    test_set = torchvision.datasets.MNIST(DATA_ROOT, train=False,
                                          transform=tf, download=True)
    loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

    model = MNISTClassifier(784, 128, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for images, labels in loader:
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  epoch {epoch + 1}/5", flush=True)

    model.eval()
    X = torch.stack([test_set[i][0] for i in range(len(test_set))])
    y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    with torch.no_grad():
        accuracy = (model(X).argmax(1) == y).float().mean().item() * 100

        image = test_set[index][0]
        flat = image.reshape(1, -1)
        z = model.fc1(flat)[0]                    # ReLU 앞, 음수가 섞여 있다
        h = model.relu(z.unsqueeze(0))[0]         # ReLU 뒤, 음수가 0이 되었다
        logit = model.fc2(h.unsqueeze(0))[0]
        prob = torch.softmax(logit, dim=0)

        # 시험 집합 전체의 0 비율 (본문이 인용하는 값)
        H = model.relu(model.fc1(X.reshape(len(X), -1)))
        zero_rate = (H == 0).float().mean().item() * 100

    pred = int(logit.argmax())
    print(f"  test accuracy {accuracy:.2f}%   image {index}: "
          f"true {test_set[index][1]}, pred {pred}, p {prob[pred]:.4f}")
    print(f"  ReLU clips {int((z < 0).sum())} of 128 on this image; "
          f"{zero_rate:.1f}% zeros across the test set")

    return dict(image=image.numpy()[0], flat=flat.numpy()[0], z=z.numpy(),
                h=h.numpy(), logit=logit.numpy(), prob=prob.numpy(), pred=pred)


# ================================================================================
# 2부: 사슬 그림 — 띠의 색이 곧 실제 활성값이다
# ================================================================================
def draw_architecture(act):
    image, flat, z, h = act["image"], act["flat"], act["z"], act["h"]
    logit, prob, pred = act["logit"], act["prob"], act["pred"]

    # 붉은색 = 음수, 흰색 = 0, 짙은 색 = 양수
    div = LinearSegmentedColormap.from_list("div", [ACCENT, WHITE, TEXT])
    gray = LinearSegmentedColormap.from_list("gray", [WHITE, TEXT])

    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(-2.45, 2.30)
    ax.axis("off")

    bar_w = 0.30

    def strip(x, height, values, cmap, vmin, vmax, label, top=None):
        """값 하나하나를 칸으로 칠한 기둥."""
        ax.imshow(np.asarray(values).reshape(-1, 1),
                  extent=[x - bar_w / 2, x + bar_w / 2, -height / 2, height / 2],
                  aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", zorder=2)
        ax.add_patch(Rectangle((x - bar_w / 2, -height / 2), bar_w, height,
                               facecolor="none", edgecolor=EDGE, lw=1.2, zorder=3))
        ax.text(x, -height / 2 - 0.20, label, ha="center", va="top",
                fontsize=10.5, color=TEXT)
        if top:
            ax.text(x, height / 2 + 0.14, top, ha="center", va="bottom",
                    fontsize=10, color=TEXT)

    # 실제 입력 이미지
    ax.imshow(image, extent=[0.28, 1.12, -0.42, 0.42], aspect="auto",
              cmap="gray_r", vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((0.28, -0.42), 0.84, 0.84, facecolor="none",
                           edgecolor=EDGE, lw=1.2, zorder=3))
    ax.text(0.70, -0.62, "28 x 28", ha="center", va="top", fontsize=10.5, color=TEXT)

    arrow(ax, 1.25, 0, 2.05, 0)
    ax.text(1.65, 0.12, "flatten", ha="center", va="bottom", fontsize=9.5, color=MUTE)

    # 펼친 이미지. 가로줄이 비치는 것은 행을 이어 붙였기 때문이다
    strip(2.35, 2.40, flat, gray, 0, 1, "784", top="$x_0$")

    arrow(ax, 2.58, 0, 3.92, 0)
    ax.text(3.25, 0.30, "$A_0$, $b_0$", ha="center", va="bottom", fontsize=11, color=TEXT)
    ax.text(3.25, -0.16, "784 x 128 + 128", ha="center", va="top", fontsize=9, color=MUTE)
    ax.text(3.25, -0.42, "= 100,480", ha="center", va="top", fontsize=9, color=MUTE)

    # 색은 90분위에서 포화시킨다. 큰 값 몇 개에 나머지가 묻히면 ReLU의
    # 효과가 눈에 보이지 않기 때문이다. 부호와 대략의 크기를 나타낼 뿐
    # 정확한 크기를 읽는 그림이 아니다
    zlim = float(np.percentile(np.abs(z), 90))
    strip(4.15, 1.20, z, div, -zlim, zlim, "128")

    arrow(ax, 4.38, 0, 5.32, 0, color=ACCENT, lw=1.5)
    ax.text(4.85, 0.30, r"$\varphi$ = ReLU", ha="center", va="bottom",
            fontsize=11, color=ACCENT)
    ax.text(4.85, -0.16, f"clips {int((z < 0).sum())} of 128\nto zero",
            ha="center", va="top", fontsize=8.5, color=ACCENT, linespacing=1.35)

    strip(5.55, 1.20, h, div, -zlim, zlim, "128")

    arrow(ax, 5.78, 0, 7.12, 0)
    ax.text(6.45, 0.30, "$A_1$, $b_1$", ha="center", va="bottom", fontsize=11, color=TEXT)
    ax.text(6.45, -0.16, "128 x 10 + 10", ha="center", va="top", fontsize=9, color=MUTE)
    ax.text(6.45, -0.42, "= 1,290", ha="center", va="top", fontsize=9, color=MUTE)

    llim = float(np.percentile(np.abs(logit), 90))
    strip(7.35, 0.50, logit, div, -llim, llim, "10", top="logit")

    arrow(ax, 7.58, 0, 8.72, 0)
    ax.text(8.15, 0.14, "softmax", ha="center", va="bottom", fontsize=9.5, color=MUTE)

    strip(8.95, 0.50, prob, gray, 0, 1, "10", top="prob")

    arrow(ax, 9.18, 0, 10.32, 0)
    ax.text(9.75, 0.14, "argmax", ha="center", va="bottom", fontsize=9.5, color=MUTE)

    ax.add_patch(Rectangle((10.45, -0.30), 0.60, 0.60, facecolor=HILITE,
                           edgecolor=EDGE, lw=1.2, zorder=2))
    ax.text(10.75, 0, str(pred), ha="center", va="center", fontsize=13, color=TEXT)
    ax.text(10.75, -0.42, f"p = {prob[pred]:.4f}", ha="center", va="top",
            fontsize=9, color=MUTE)

    ax.text(5.65, -1.28, "strip colour:   crimson < 0    white = 0    dark > 0",
            ha="center", va="top", fontsize=8.5, color=MUTE)

    # Model 이라는 이름이 어디까지를 가리키는지
    yb, xl, xr, xm = -1.86, 2.35, 8.95, 5.65
    ax.plot([xl, xr], [yb, yb], color=EDGE, lw=1.1)
    ax.plot([xl, xl], [yb, yb + 0.14], color=EDGE, lw=1.1)
    ax.plot([xr, xr], [yb, yb + 0.14], color=EDGE, lw=1.1)
    ax.plot([xm, xm], [yb, yb - 0.14], color=EDGE, lw=1.1)
    ax.text(xm, yb - 0.26, "Model", ha="center", va="top", fontsize=11.5, color=TEXT)

    ax.text(4.85, 1.72,
            "training finds $A_0$, $b_0$, $A_1$, $b_1$  -  101,770 numbers in all",
            ha="center", va="center", fontsize=10.5, color=TEXT)

    fig.tight_layout()
    fig.savefig(HERE / "mlp_architecture.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# ================================================================================
# 3부: 역전파 그림 — 두 줄의 방향이 반대라는 것이 요점이다
# ================================================================================
def draw_backprop_flow():
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-2.35, 2.35)
    ax.axis("off")

    y_f, y_b = 0.95, -1.05

    for x, lab in [(1.15, "$x_0$"), (5.05, "$x_1$"), (8.95, "$x_2$")]:
        ax.add_patch(Rectangle((x - 0.36, y_f - 0.36), 0.72, 0.72,
                               facecolor=WHITE, edgecolor=EDGE, lw=1.2, zorder=3))
        ax.text(x, y_f, lab, ha="center", va="center", fontsize=12, color=TEXT)

    for x, lab in [(3.10, "$f$"), (7.00, "$g$")]:
        ax.add_patch(FancyBboxPatch((x - 0.40, y_f - 0.32), 0.80, 0.64,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=FILL, edgecolor=EDGE, lw=1.2, zorder=3))
        ax.text(x, y_f, lab, ha="center", va="center", fontsize=12, color=TEXT)

    for x0, x1 in [(1.51, 2.70), (3.50, 4.69), (5.41, 6.60), (7.40, 8.59)]:
        arrow(ax, x0, y_f, x1, y_f)

    ax.text(0.10, y_f + 1.05, "forward", ha="left", va="center",
            fontsize=12.5, color=TEXT)
    arrow(ax, 1.15, y_f + 1.05, 8.95, y_f + 1.05, lw=1.1)

    # 지나가면서 마디마다 붙들어 두는 것
    for x, expr in [(3.10, "$f'(x_0)$"), (7.00, "$g'(x_1)$")]:
        arrow(ax, x, y_f - 0.36, x, y_f - 0.80, color=ACCENT, lw=1.0, ls=(0, (3, 2)))
        ax.add_patch(FancyBboxPatch((x - 0.70, y_f - 1.34), 1.40, 0.48,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=HILITE, edgecolor=ACCENT, lw=1.0, zorder=3))
        ax.text(x, y_f - 1.10, "keeps " + expr, ha="center", va="center",
                fontsize=10, color=ACCENT, zorder=4)

    ax.text(0.10, y_b - 1.00, "backward", ha="left", va="center",
            fontsize=12.5, color=ACCENT)
    arrow(ax, 9.60, y_b - 1.00, 1.15, y_b - 1.00, color=ACCENT, lw=1.1)

    for x1, x0 in [(8.59, 7.40), (6.60, 5.41), (4.69, 3.50), (2.70, 1.51)]:
        arrow(ax, x1, y_b, x0, y_b, color=ACCENT, lw=1.3)

    # 내려오면서 도함수가 하나씩 곱해진다
    ax.text(8.95, y_b + 0.30, "$1$", ha="center", va="bottom",
            fontsize=11.5, color=ACCENT)
    ax.text(5.05, y_b + 0.30, "$g'(x_1)$", ha="center", va="bottom",
            fontsize=11.5, color=ACCENT)
    ax.text(1.15, y_b + 0.30, "$g'(x_1)\\, f'(x_0)$", ha="center", va="bottom",
            fontsize=11.5, color=ACCENT)

    ax.text(5.60, y_b - 0.42, "multiply one more derivative at every step down",
            ha="center", va="top", fontsize=9.5, color=ACCENT)

    ax.add_patch(FancyBboxPatch((9.35, 1.55), 1.80, 0.78,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor=WHITE, edgecolor=EDGE, lw=1.0, zorder=3))
    ax.text(10.25, 2.12, "$h(x) = g(f(x))$", ha="center", va="center",
            fontsize=10.5, color=TEXT, zorder=4)
    ax.text(10.25, 1.76, "$h'(x) = g'(f(x))\\, f'(x)$", ha="center", va="center",
            fontsize=10.5, color=TEXT, zorder=4)

    fig.tight_layout()
    fig.savefig(HERE / "backprop_flow.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# ================================================================================
# 4부: 활성화 함수 세 개와 그 도함수
# ================================================================================
def draw_activations():
    """1절의 논증을 그림으로. 요점은 함수가 아니라 도함수의 크기다."""
    x = np.linspace(-5, 5, 1001)
    sigmoid = 1 / (1 + np.exp(-x))

    # shade: 양끝이 모두 평평해지는 함수에만. ReLU의 꺼진 반쪽은 포화가
    # 아니라 설계이므로 같은 표시를 하면 뜻이 뒤집힌다
    panels = [
        ("ReLU", np.maximum(0, x), (x > 0).astype(float), None, False),
        ("sigmoid", sigmoid, sigmoid * (1 - sigmoid), 0.25, True),
        ("tanh", np.tanh(x), 1 - np.tanh(x) ** 2, 1.0, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), sharex=True)

    for ax, (name, f, df, dmax, shade) in zip(axes, panels):
        ax.axhline(0, color=EDGE, lw=0.8, zorder=1)
        ax.axvline(0, color=EDGE, lw=0.8, zorder=1)

        # 도함수가 거의 0인 구간 - 여기서 학습 신호가 죽는다
        flat = df < 0.01
        if shade and flat.any():
            for lo, hi in _runs(x, flat):
                ax.axvspan(lo, hi, color=ACCENT, alpha=0.06, lw=0, zorder=0)

        ax.plot(x, f, color=TEXT, lw=2.0, zorder=3, label=name)
        ax.plot(x, df, color=ACCENT, lw=1.6, ls=(0, (4, 2)), zorder=3,
                label="derivative")

        if dmax is not None:
            ax.axhline(dmax, color=ACCENT, lw=0.8, ls=":", zorder=2)
            txt = "max 1/4" if dmax == 0.25 else "max 1"
            ax.text(4.8, dmax + 0.06, txt, ha="right", va="bottom",
                    fontsize=8.5, color=ACCENT)

        ax.set_title(name, fontsize=12, color=TEXT, pad=8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1.25, 1.65)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.tick_params(labelsize=8.5, colors=MUTE, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(EDGE)

    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False,
                   labelcolor=[TEXT, ACCENT])
    axes[0].text(-4.8, -1.12, "derivative is exactly 1\nwhere it is on",
                 ha="left", va="bottom", fontsize=8, color=ACCENT, linespacing=1.3)
    for ax in axes[1:]:
        ax.text(-4.8, -1.12, "shaded: derivative under 0.01", ha="left",
                va="bottom", fontsize=8, color=ACCENT)

    fig.tight_layout()
    fig.savefig(HERE / "activations.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def _runs(x, mask):
    """mask가 참인 구간들의 (시작, 끝)."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = x[i]
        elif not m and start is not None:
            out.append((start, x[i])); start = None
    if start is not None:
        out.append((start, x[-1]))
    return out


# ================================================================================
if __name__ == "__main__":
    draw_activations()
    print("wrote activations.svg")

    draw_backprop_flow()
    print("wrote backprop_flow.svg")

    print("training the page's model (this is most of the runtime)...")
    activations = train_and_capture(index=0)

    draw_architecture(activations)
    print("wrote mlp_architecture.svg")
