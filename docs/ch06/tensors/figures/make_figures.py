"""7.1 텐서 만들기 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch06/tensors/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: tensor 라는 말의 세 뜻 =================================
#
# 1846 해밀턴 — 크기 하나. 수 하나다.
# 1898 포크트 — 탄성체 한 점의 응력. 좌표를 돌리면 정해진 규칙으로 바뀐다.
# 오늘 PyTorch — 모양과 자료형을 가진 다차원 배열. 변환 규칙은 어디에도 없다.
#
# 그림에 적는 수는 지어내지 않는다. 가운데 판의 화살표는 응력의 **꼴**만
# 보이는 것이라 그림 안에 그렇게 적는다.

def tensor_word():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.6, 3.9))

    # --- 1846 해밀턴: 크기 하나 -------------------------------------------
    ax1.annotate("", xy=(0.78, 0.74), xytext=(0.16, 0.26),
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=2.2))
    ax1.plot([0.16, 0.78], [0.26, 0.26], color=INK, linewidth=0.7, alpha=0.35)
    ax1.plot([0.78, 0.78], [0.26, 0.74], color=INK, linewidth=0.7, alpha=0.35)
    ax1.annotate("", xy=(0.74, 0.80), xytext=(0.12, 0.32),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT, linewidth=1.4))
    ax1.text(0.33, 0.62, "tensor", fontsize=11, color=ACCENT, weight="bold",
             rotation=37.7, ha="center")
    ax1.text(0.5, 0.11, "the magnitude. one number.", ha="center",
             fontsize=8.6, color=INK)
    ax1.set_title("1846   Hamilton", fontsize=10, color=INK, weight="bold")

    # --- 1898 포크트: 한 점의 응력 ----------------------------------------
    ax2.add_patch(plt.Rectangle((0.30, 0.34), 0.40, 0.36, facecolor="none",
                                edgecolor=INK, linewidth=1.4))
    for (x, y, dx, dy) in ((0.50, 0.78, 0.0, -0.09), (0.50, 0.26, 0.0, 0.09),
                           (0.19, 0.52, 0.09, 0.0), (0.81, 0.52, -0.09, 0.0)):
        ax2.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                     arrowprops=dict(arrowstyle="->", color=ACCENT,
                                     linewidth=1.6))
    for (x, y, dx) in ((0.36, 0.745, 0.13), (0.64, 0.295, -0.13)):
        ax2.annotate("", xy=(x + dx, y), xytext=(x, y),
                     arrowprops=dict(arrowstyle="->", color=MUTED,
                                     linewidth=1.3))
    ax2.text(0.5, 0.16, "stress at a point.\n"
                        "turn the axes and it changes\nby a fixed rule.",
             ha="center", va="top", fontsize=8.6, color=INK)
    ax2.text(0.97, 0.90, "schematic", ha="right", fontsize=6.4, color=INK,
             alpha=0.55, style="italic")
    ax2.set_title("1898   Voigt", fontsize=10, color=INK, weight="bold")

    # --- 오늘: 다차원 배열 -------------------------------------------------
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for r in range(2):
        for c in range(3):
            ax3.add_patch(plt.Rectangle((0.22 + c * 0.19, 0.60 - r * 0.19),
                                        0.17, 0.17, facecolor=ACCENT,
                                        alpha=0.13, edgecolor=ACCENT,
                                        linewidth=0.9))
            ax3.text(0.305 + c * 0.19, 0.685 - r * 0.19, f"{vals[r, c]:.0f}",
                     ha="center", va="center", fontsize=9, color=INK)
    ax3.text(0.5, 0.33, "shape (2, 3)   dtype float32", ha="center",
             fontsize=8.6, color=INK, family="monospace")
    ax3.text(0.5, 0.16, "no transformation law\nanywhere in the library.",
             ha="center", va="top", fontsize=8.6, color=ACCENT)
    ax3.set_title("today   torch.Tensor", fontsize=10, color=INK,
                  weight="bold")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.text(0.5, -0.03,
             "three different things wear the same word. "
             "torch.tensor(42) is the 1846 one.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "tensor_word.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote tensor_word.svg")


# === 그림으로 보는 AI: 반열린 구간을 고른 까닭과 그 약속이 깨지는 자리 ==========
#
# 왼쪽은 EWD831 의 네 규약을 그대로 그린다 — 2,3,...,12 를 적는 네 가지 방법.
# 오른쪽은 **실제로 세어** 그린다. torch.arange 를 실수 조합 2,301 개와 정수
# 조합 9,481 개에 돌려 개수가 정확한 답과 어긋나는 자리를 찍는다.

def arange_promise():
    import torch
    from fractions import Fraction

    # --- 실수 조합을 전부 돌려 어긋나는 자리를 모은다 ----------------------
    xs, ys = [], []
    n_float = 0
    for i in range(1, 60):
        for j in range(1, 40):
            stop, step = i / 10.0, j / 100.0
            n_float += 1
            exact = int(-(-Fraction(i, 10) // Fraction(j, 100)))
            if len(torch.arange(0.0, stop, step)) != exact:
                xs.append(stop)
                ys.append(step)
    n_int = sum(1 for n in range(1, 500) for s in range(1, 20))
    bad_int = sum(1 for n in range(1, 500) for s in range(1, 20)
                  if len(torch.arange(0, n, s)) != -(-n // s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: EWD831 의 네 규약 -------------------------------------------
    rows = [("a)", "2 <= i < 13", True), ("b)", "1 <  i <= 12", False),
            ("c)", "2 <= i <= 12", False), ("d)", "1 <  i < 13", False)]
    for k, (tag, expr, win) in enumerate(rows):
        y = 0.80 - k * 0.165
        col = ACCENT if win else INK
        ax1.text(0.08, y, tag, fontsize=9.5, color=col,
                 weight="bold" if win else "normal")
        ax1.text(0.22, y, expr, fontsize=11, color=col, family="monospace",
                 weight="bold" if win else "normal")
        if win:
            ax1.add_patch(plt.Rectangle((0.04, y - 0.052), 0.92, 0.105,
                                        facecolor=ACCENT, alpha=0.12,
                                        edgecolor=ACCENT, linewidth=0.9))
            ax1.text(0.98, y, "13 - 2 = 11 = the length", fontsize=7.8,
                     color=ACCENT, ha="right", style="italic")
    ax1.text(0.5, 0.95, "four ways to write  2, 3, ... , 12", ha="center",
             fontsize=9, color=INK, style="italic")
    ax1.text(0.5, 0.10,
             "Dijkstra picks a).  the point of it is that\n"
             "the gap between the bounds IS the count.",
             ha="center", va="center", fontsize=8.6, color=INK)
    ax1.set_title("EWD831,  11 August 1982", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.04)
    ax1.axis("off")

    # --- 오른쪽: 그 약속이 깨지는 자리 --------------------------------------
    ax2.scatter(xs, ys, s=46, color=ACCENT, marker="x", linewidths=1.6,
                zorder=3)
    ax2.set_xlabel("stop", fontsize=9)
    ax2.set_ylabel("step", fontsize=9)
    ax2.set_title("where torch.arange miscounts", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13)
    ax2.set_ylim(top=max(ys) * 1.38)   # 통계 상자가 점을 가리지 않게
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)
    ax2.text(0.98, 0.95,
             f"{len(xs)} of {n_float:,} float pairs\n"
             f"{bad_int} of {n_int:,} integer pairs",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.6,
             color=INK,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor=INK, alpha=0.9))

    fig.text(0.5, -0.03,
             "rare enough to miss, common enough to hit. "
             "with integer steps it never happens.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "arange_promise.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote arange_promise.svg   float {len(xs)}/{n_float}   "
          f"int {bad_int}/{n_int}")


# === 그림으로 보는 AI: 로스 스케일링이 막는 것 ================================
#
# 왼쪽은 세 자료형의 표현 범위다 — bfloat16 의 범위가 float32 와 **같다**는 것이
# 눈에 보여야 한다. 오른쪽은 작은 신경망의 실제 그래디언트를 재어, float16 의
# 최소 정상수 아래로 떨어지는 비율이 스케일에 따라 어떻게 변하는지 보인다.
# 수는 전부 여기서 돌려 얻는다.

def loss_scaling():
    import torch

    f16 = torch.finfo(torch.float16)
    bf16 = torch.finfo(torch.bfloat16)
    f32 = torch.finfo(torch.float32)

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 10))
    x = torch.randn(512, 256)
    y = torch.randint(0, 10, (512,))
    torch.nn.functional.cross_entropy(net(x), y).backward()
    g = torch.cat([p.grad.flatten() for p in net.parameters()])
    g = g[g != 0].abs()
    lg = np.log10(g.numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # --- 왼쪽: 세 자료형의 범위 --------------------------------------------
    fmts = [("float32", f32.tiny, f32.max, INK),
            ("bfloat16", bf16.tiny, bf16.max, MUTED),
            ("float16", f16.tiny, f16.max, ACCENT)]
    for k, (name, lo, hi, col) in enumerate(fmts):
        y0 = 2 - k
        ax1.plot([np.log10(lo), np.log10(hi)], [y0, y0], color=col,
                 linewidth=9, solid_capstyle="butt", alpha=0.85)
        ax1.text(0, y0 + 0.30, name, ha="center", fontsize=9.5, color=col,
                 weight="bold")
        ax1.text(np.log10(lo), y0 - 0.34, f"1e{int(np.log10(lo))}",
                 ha="center", fontsize=7.6, color=INK, alpha=0.8)
        ax1.text(np.log10(hi), y0 - 0.34, f"1e{int(np.log10(hi))}",
                 ha="center", fontsize=7.6, color=INK, alpha=0.8)
    ax1.set_xlim(-44, 44)
    ax1.set_ylim(-0.9, 2.75)
    ax1.set_yticks([])
    ax1.set_xlabel("magnitude  (log10)", fontsize=9)
    ax1.set_title("bfloat16 has float32's range", fontsize=10, color=INK,
                  weight="bold")
    ax1.tick_params(labelsize=8, colors=INK)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.axhline(-0.55, color=INK, linewidth=0.6, alpha=0.3)

    # --- 오른쪽: 실제 그래디언트가 어디에 있는가 ----------------------------
    floor = np.log10(f16.tiny)
    ax2.hist(lg, bins=70, color=INK, alpha=0.30, label="gradients as they are")
    ax2.hist(lg + np.log10(8), bins=70, color=ACCENT, alpha=0.45,
             label="after scaling by 8")
    ax2.axvline(floor, color=ACCENT, linestyle="--", linewidth=1.6)
    ax2.text(floor - 0.12, ax2.get_ylim()[1] * 0.80,
             "float16 smallest normal  ", fontsize=8.2, color=ACCENT,
             va="top", ha="right")

    lost = [(s, (g * s < f16.tiny).float().mean().item()) for s in (1, 8, 1024)]
    txt = "\n".join(f"scale {s:>5} ->  {p*100:5.2f}% lost" for s, p in lost)
    ax2.text(0.015, 0.60, txt, transform=ax2.transAxes, fontsize=8.6,
             color=INK, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor=INK, alpha=0.92))
    ax2.set_xlabel("gradient magnitude  (log10)", fontsize=9)
    ax2.set_ylabel("count", fontsize=9)
    ax2.set_title("what falls off the bottom", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "one small network, one backward pass. scaling does not make "
             "gradients bigger - it moves them into range.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "loss_scaling.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote loss_scaling.svg   lost at 1x {lost[0][1]*100:.2f}%   "
          f"at 8x {lost[1][1]*100:.2f}%")


# === 그림으로 보는 AI: 공식이 없어서 반복해야 한다 ============================
#
# 고윳값은 특성다항식의 근인데, 5차부터는 근호로 적는 일반 공식이 없다
# (아벨-루피니). 그래서 eig 는 **식을 세워 푸는 것이 아니라 반복해서 다가간다.**
# 오른쪽은 x^5 - x - 1 (갈루아 군이 S5 라 근호로 안 풀리는 고전적 보기) 의
# 동반행렬을 만들어 torch 로 직접 근을 구하고 잔차를 잰다.

def no_formula():
    import torch

    # --- x^5 - x - 1 의 동반행렬 -------------------------------------------
    coef = [1, 0, 0, 0, -1, -1]
    C = torch.zeros(5, 5)
    C[0, :] = torch.tensor([-c for c in coef[1:]]) / coef[0]
    C[1:, :-1] = torch.eye(4)
    ev = torch.linalg.eigvals(C)
    res = max(abs(complex(z) ** 5 - complex(z) - 1) for z in ev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # --- 왼쪽: 어디까지 공식이 있는가 ---------------------------------------
    items = [("2", "the quadratic formula", True),
             ("3", "Cardano, 1545", True),
             ("4", "Ferrari, 1540s", True),
             ("5", "no formula exists", False),
             ("6 and up", "no formula exists", False)]
    for k, (deg, note, has) in enumerate(items):
        y = 0.86 - k * 0.165
        col = INK if has else ACCENT
        ax1.text(0.17, y, f"degree {deg}", fontsize=10, color=col,
                 ha="right", weight="bold" if not has else "normal")
        ax1.text(0.24, y, "v" if has else "x", fontsize=11, color=col,
                 family="monospace", weight="bold")
        ax1.text(0.34, y, note, fontsize=9, color=col,
                 style="italic" if not has else "normal")
    ax1.plot([0.03, 0.99], [0.86 - 2.5 * 0.165, 0.86 - 2.5 * 0.165],
             color=ACCENT, linewidth=1.2, linestyle="--")
    ax1.text(0.5, 0.055,
             "Ruffini 1799 (with gaps), Abel 1824 (complete),\n"
             "Galois 1832: which equations are solvable at all.",
             ha="center", fontsize=8.4, color=INK)
    ax1.set_title("how far the formulas go", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # --- 오른쪽: 그래도 답은 나온다 ----------------------------------------
    th = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(th), np.sin(th), color=INK, linewidth=0.6, alpha=0.25)
    ax2.axhline(0, color=INK, linewidth=0.6, alpha=0.3)
    ax2.axvline(0, color=INK, linewidth=0.6, alpha=0.3)
    ax2.scatter([z.real.item() for z in ev], [z.imag.item() for z in ev],
                s=85, color=ACCENT, zorder=4, edgecolors="white",
                linewidths=1.2)
    ax2.set_xlabel("real", fontsize=9)
    ax2.set_ylabel("imaginary", fontsize=9)
    ax2.set_title("torch.linalg.eigvals finds them anyway", fontsize=10,
                  color=INK, weight="bold")
    ax2.text(0.5, -0.30,
             f"roots of  x^5 - x - 1\n"
             f"largest residual {res:.1e}  -  not zero",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8.6,
             color=ACCENT)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.12)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "no_formula.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote no_formula.svg   residual {res:.3e}")


# === 그림으로 보는 AI: 서로 모르는 두 라이브러리가 같은 메모리를 보는 법 =======
#
# 왼쪽은 PEP 3118 이 표준으로 정한 것 — 메모리 덩어리를 가리키는 포인터만이
# 아니라 자료형·모양·스트라이드까지 함께 알려 준다. 그래서 numpy 와 torch 가
# 서로를 몰라도 같은 덩어리를 읽을 수 있다.
# 오른쪽은 **직접 재어** 그린다. 공유는 크기와 무관하고 복사는 크기에 비례한다.

def buffer_protocol():
    import time

    import numpy as np
    import torch

    def bench(fn, reps):
        for _ in range(20):
            fn()                                  # 워밍업 — 첫 호출은 느리다
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e6

    ns = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    share, copy = [], []
    for n in ns:
        a = np.random.rand(n)
        share.append(bench(lambda: torch.from_numpy(a), 200))
        copy.append(bench(lambda: torch.tensor(a), 30))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: 무엇을 표준으로 정했는가 -------------------------------------
    ax1.add_patch(plt.Rectangle((0.28, 0.34), 0.44, 0.20, facecolor=ACCENT,
                                alpha=0.16, edgecolor=ACCENT, linewidth=1.2))
    ax1.text(0.50, 0.44, "one block of memory", ha="center", va="center",
             fontsize=9.5, color=INK)

    for x, name in ((0.14, "numpy"), (0.86, "torch")):
        ax1.add_patch(plt.Rectangle((x - 0.12, 0.72), 0.24, 0.14,
                                    facecolor="white", edgecolor=INK,
                                    linewidth=1.1))
        ax1.text(x, 0.79, name, ha="center", va="center", fontsize=9.5,
                 color=INK, family="monospace")
        ax1.annotate("", xy=(0.5 - np.sign(0.5 - x) * 0.21, 0.545),
                     xytext=(x, 0.715),
                     arrowprops=dict(arrowstyle="->", color=INK,
                                     linewidth=1.3))

    ax1.text(0.5, 0.235, "the protocol also carries", ha="center",
             fontsize=8.2, color=INK, alpha=0.8)
    ax1.text(0.5, 0.135, "format   shape   strides", ha="center",
             fontsize=10, color=ACCENT, family="monospace")
    ax1.text(0.5, 0.035, "without those it is just a pointer and a length",
             ha="center", fontsize=7.8, color=INK, alpha=0.7, style="italic")
    ax1.set_title("PEP 3118,  2006", fontsize=10, color=INK, weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # --- 오른쪽: 공유는 크기를 타지 않는다 ---------------------------------
    ax2.loglog(ns, copy, "s--", color=INK, linewidth=1.8, ms=5.5,
               label="torch.tensor(a)   copies")
    ax2.loglog(ns, share, "o-", color=ACCENT, linewidth=2.0, ms=5.5,
               label="torch.from_numpy(a)   shares")
    ax2.annotate(f"{copy[-1]/share[-1]:,.0f}x apart",
                 xy=(ns[-1], copy[-1]), xytext=(-96, -16),
                 textcoords="offset points", fontsize=8.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("number of elements", fontsize=9)
    ax2.set_ylabel("microseconds per call", fontsize=9)
    ax2.set_title("one of these reads the data", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "measured on this machine, warmed up first. the flat line is the "
             "whole point - sharing never touches the data.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "buffer_protocol.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote buffer_protocol.svg   share {share[-1]:.2f}us  "
          f"copy {copy[-1]:.0f}us")


if __name__ == "__main__":
    tensor_word()
    arange_promise()
    loss_scaling()
    no_formula()
    buffer_protocol()
