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


# === 그림으로 보는 AI: 섞는 것이 실제로 사 주는 것 ============================
#
# 흔한 설명은 "섞어야 그래디언트가 불편(unbiased)이 된다" 인데, 재어 보면
# **에포크 평균은 섞든 안 섞든 같다** — 한 에포크면 어느 쪽이든 모든 표본을
# 꼭 한 번씩 보기 때문이다. 섞어서 달라지는 것은 **한 걸음 한 걸음**이다.
# 두 가지를 다 재어 나란히 그린다.

def shuffling():
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    n, d, C, B = 2000, 20, 5, 50
    X = torch.randn(n, d)
    y = torch.arange(n) % C
    X += F.one_hot(y, C).float() @ torch.randn(C, d) * 1.5
    order = torch.argsort(y)                    # 레이블 순으로 정렬해 둔다
    Xs, ys = X[order], y[order]
    W = torch.zeros(d, C, requires_grad=True)

    def grad(xb, yb):
        if W.grad is not None:
            W.grad = None
        F.cross_entropy(xb @ W, yb).backward()
        return W.grad.flatten().clone()

    full = grad(Xs, ys)

    def sweep(Xd, yd):
        gs = [grad(Xd[i:i + B], yd[i:i + B]) for i in range(0, n, B)]
        cos = [F.cosine_similarity(g, full, dim=0).item() for g in gs]
        err = (torch.stack(gs).mean(0) - full).norm() / full.norm()
        return np.array(cos), err.item()

    p = torch.randperm(n)
    cos_no, err_no = sweep(Xs, ys)
    cos_sh, err_sh = sweep(Xs[p], ys[p])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # --- 왼쪽: 한 걸음 한 걸음은 전혀 다르다 -------------------------------
    k = np.arange(1, len(cos_no) + 1)
    ax1.plot(k, cos_sh, "o-", color=ACCENT, ms=3.4, linewidth=1.3,
             label=f"shuffle=True   mean {cos_sh.mean():.3f}")
    ax1.plot(k, cos_no, "s-", color=INK, ms=3.4, linewidth=1.3, alpha=0.75,
             label=f"sorted by label   mean {cos_no.mean():.3f}")
    ax1.axhline(1.0, color=INK, linestyle=":", linewidth=1.0, alpha=0.5)
    ax1.text(len(cos_no) * 0.5, 1.012, "the full gradient", fontsize=7.8,
             color=INK, alpha=0.7, ha="center")
    ax1.set_xlabel("batch within one epoch", fontsize=9)
    ax1.set_ylabel("cosine with the full gradient", fontsize=9)
    ax1.set_title("every single step", fontsize=10, color=INK, weight="bold")
    ax1.set_ylim(0, 1.09)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 그런데 에포크 평균은 같다 ---------------------------------
    ax2.bar([0, 1], [err_no, err_sh], color=[INK, ACCENT], width=0.5,
            alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["sorted", "shuffled"], fontsize=9)
    ax2.set_ylabel("relative error of the epoch mean", fontsize=9)
    ax2.set_title("but averaged over the epoch", fontsize=10, color=INK,
                  weight="bold")
    for x, v in ((0, err_no), (1, err_sh)):
        ax2.text(x, v * 1.5, f"{v:.0e}", ha="center", fontsize=9, color=INK)
    ax2.set_ylim(err_no * 0.02, err_no * 60)
    ax2.text(0.5, err_no * 14, "the same either way", ha="center",
             fontsize=8.6, color=ACCENT, style="italic")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "shuffling does not buy an unbiased epoch - it buys a usable "
             "step.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "shuffling.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote shuffling.svg   cos sorted {cos_no.mean():.3f} "
          f"shuffled {cos_sh.mean():.3f}   err {err_no:.1e} / {err_sh:.1e}")


# === 그림으로 보는 AI: 네 번째 관계 ===========================================
#
# IEEE 754 는 두 수 사이에 **네** 가지 관계를 둔다 — 작다, 같다, 크다, 그리고
# **차례 없음(unordered)**. nan 이 끼면 늘 넷째다. 삼분법을 가정하고 쓴 알고리즘
# (정렬·이분 탐색·중복 제거) 은 그 자리에서 조용히 어긋난다.
# 오른쪽의 정렬 결과는 그릴 때마다 실제로 돌려 얻는다.

def fourth_relation():
    nan = float("nan")
    a = [3.0, nan, 1.0, 2.0]
    b = [1.0, nan, 3.0, 2.0]
    sa, sb = sorted(a), sorted(b)

    def fmt(xs):
        return "[" + ", ".join("nan" if x != x else f"{x:.0f}" for x in xs) + "]"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: 관계가 셋이 아니라 넷이다 -----------------------------------
    rels = [("a < b", INK), ("a = b", INK), ("a > b", INK),
            ("unordered", ACCENT)]
    for k, (name, col) in enumerate(rels):
        x = 0.5
        y = 0.80 - k * 0.165
        ax1.add_patch(plt.Rectangle((0.16, y - 0.056), 0.68, 0.112,
                                    facecolor=col, alpha=0.12 if k < 3 else 0.2,
                                    edgecolor=col, linewidth=1.0))
        ax1.text(x, y, name, ha="center", va="center", fontsize=10.5,
                 color=col, family="monospace",
                 weight="bold" if k == 3 else "normal")
    ax1.text(0.5, 0.95, "IEEE 754 allows four, not three", ha="center",
             fontsize=9.2, color=INK, style="italic")
    ax1.text(0.5, 0.10,
             "any comparison with nan lands in the fourth,\n"
             "including nan against itself.",
             ha="center", va="center", fontsize=8.6, color=ACCENT)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.04)
    ax1.axis("off")

    # --- 오른쪽: 삼분법을 믿은 정렬이 하는 일 -------------------------------
    rows = [("sorted(", fmt(a), ")", fmt(sa), sa == sorted(x for x in sa)),
            ("sorted(", fmt(b), ")", fmt(sb), None)]
    ax2.text(0.5, 0.92, "Python's sort assumes three", ha="center",
             fontsize=9.2, color=INK, style="italic")
    for k, (pre, src, post, out, _) in enumerate(rows):
        y = 0.66 - k * 0.30
        ax2.text(0.04, y, f"{pre}{src}{post}", fontsize=10, color=INK,
                 family="monospace", va="center")
        ax2.text(0.04, y - 0.125, f"   -> {out}", fontsize=10, color=ACCENT,
                 family="monospace", va="center")
    ax2.text(0.5, 0.07,
             "same four values, different order in,\n"
             "and the first result is not sorted at all.",
             ha="center", va="center", fontsize=8.6, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.04)
    ax2.axis("off")

    fig.text(0.5, -0.02,
             "no exception, no warning. the list just comes back wrong.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fourth_relation.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fourth_relation.svg   {fmt(a)} -> {fmt(sa)}")


# === 그림으로 보는 AI: 싼 그래디언트와 비싼 야코비안 ==========================
#
# 바우어-슈트라센 1983 은 **비스칼라 복잡도** L 에 대해 L(f, 모든 편미분) <= 3L(f)
# 를 보였다. 재는 잣대가 벽시계가 아니므로 왼쪽 판에 그렇게 적어 둔다 — 여기서
# 보이려는 것은 상수 3 자체가 아니라 **파라메터를 늘려도 비가 늘지 않는다**는 것.
# 오른쪽은 출력이 벡터가 되면 줄 수만큼 든다는 것. 둘 다 직접 잰다.

def cheap_gradient():
    import time

    import torch

    def bench(fn, reps=120, rounds=7):
        """최소값을 쓴다 — 벽시계는 위로만 튄다."""
        for _ in range(20):
            fn()
        best = None
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            t = (time.perf_counter() - t0) / reps * 1e3
            best = t if best is None else min(best, t)
        return best

    torch.manual_seed(0)
    pars, ratio = [], []
    for h in (64, 256, 1024, 4096):
        net = torch.nn.Sequential(torch.nn.Linear(256, h), torch.nn.Tanh(),
                                  torch.nn.Linear(h, 1))
        x = torch.randn(512, 256)

        def ev():                      # 그래프를 만들지 않고 값만
            with torch.no_grad():
                net(x).sum()

        def gr():                      # 값과 **모든** 편미분
            net.zero_grad(set_to_none=True)
            net(x).sum().backward()

        pars.append(sum(p.numel() for p in net.parameters()))
        ratio.append(bench(gr) / bench(ev))

    ms, jac = [], []
    for m in (1, 2, 4, 8, 16, 32):
        nt = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Tanh(),
                                 torch.nn.Linear(128, m))
        xv = torch.randn(1, 64)
        ms.append(m)
        jac.append(bench(
            lambda: torch.autograd.functional.jacobian(lambda z: nt(z), xv),
            reps=15, rounds=3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    # --- 왼쪽: 비가 늘지 않는다 --------------------------------------------
    ax1.semilogx(pars, ratio, "o-", color=ACCENT, linewidth=2.0, ms=6)
    ax1.axhline(3, color=INK, linestyle="--", linewidth=1.2, alpha=0.65)
    ax1.text(pars[0], 3.08, "  the constant 3 in the theorem", fontsize=8,
             color=INK, alpha=0.8)
    ax1.set_xlabel("number of parameters", fontsize=9)
    ax1.set_ylabel("cost of all gradients / cost of the value",
                   fontsize=9)
    ax1.set_title("scalar output: the ratio does not grow", fontsize=10,
                  color=INK, weight="bold")
    ax1.set_ylim(0, max(4.2, max(ratio) * 1.25))
    ax1.grid(alpha=0.13, which="both")
    ax1.text(0.5, 0.055,
             "wall clock is not the theorem's measure -\n"
             "the flatness is the point, not the number",
             transform=ax1.transAxes, ha="center", fontsize=7.8,
             color=INK, alpha=0.75, style="italic")

    # --- 오른쪽: 줄마다 한 번씩 -------------------------------------------
    ax2.plot(ms, jac, "o-", color=ACCENT, linewidth=2.0, ms=6,
             label="measured")
    # m 배의 기준선은 고정 비용까지 m 번 세므로 쓰지 않는다. 직선을 맞춘다.
    slope, intercept = np.polyfit(ms, jac, 1)
    ax2.plot(ms, [slope * m + intercept for m in ms], ":", color=INK,
             linewidth=1.4,
             label=f"straight line, {slope*1000:.0f} us per row")
    ax2.set_xlabel("size of the output", fontsize=9)
    ax2.set_ylabel("time for the full jacobian (ms)", fontsize=9)
    ax2.set_title("vector output: you pay per row", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.13)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "one backward pass hands back every partial derivative at once - "
             "but only when the output is one number.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "cheap_gradient.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cheap_gradient.svg   ratios "
          f"{[round(r,2) for r in ratio]}   jac {jac[0]:.3f}->{jac[-1]:.3f}ms")


# === 그림으로 보는 AI: 줄 먼저냐 칸 먼저냐 ====================================
#
# 왼쪽은 같은 행렬을 메모리에 담는 두 차례 — 줄 먼저(C·NumPy·PyTorch)와
# 칸 먼저(포트란·BLAS·LAPACK). 고른 쪽이 무엇이 연속인지를 정한다.
# 오른쪽은 **직접 잰다**: 같은 자료 같은 셈인데 방향만 다를 때의 값, 그리고
# 연속으로 만드는 값. 수는 전부 여기서 돌려 얻는다.

def memory_order():
    import time

    import torch

    def bench(fn, reps=40, rounds=5):
        for _ in range(10):
            fn()
        best = None
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            t = (time.perf_counter() - t0) / reps * 1e3
            best = t if best is None else min(best, t)
        return best

    torch.manual_seed(0)
    n = 4096
    a = torch.randn(n, n)
    b = a.T
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr(), \
        "전치가 저장소를 공유해야 이 그림의 이야기가 성립한다"
    assert not b.is_contiguous()

    t_row = bench(lambda: a.sum(dim=1))
    t_col = bench(lambda: a.sum(dim=0))
    t_con = bench(lambda: b.contiguous(), reps=10, rounds=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- 왼쪽: 같은 행렬, 두 가지 차례 -------------------------------------
    vals = [[1, 2, 3], [4, 5, 6]]
    row_major = [1, 2, 3, 4, 5, 6]
    col_major = [1, 4, 2, 5, 3, 6]

    for r in range(2):
        for c in range(3):
            ax1.add_patch(plt.Rectangle((0.06 + c * 0.11, 0.70 - r * 0.13),
                                        0.10, 0.12, facecolor=ACCENT,
                                        alpha=0.14, edgecolor=ACCENT,
                                        linewidth=0.9))
            ax1.text(0.11 + c * 0.11, 0.76 - r * 0.13, str(vals[r][c]),
                     ha="center", va="center", fontsize=10, color=INK)
    ax1.text(0.225, 0.90, "the matrix", ha="center", fontsize=9, color=INK,
             style="italic")

    for k, (name, seq, col, y) in enumerate(
            (("row-major   C, NumPy, PyTorch", row_major, ACCENT, 0.36),
             ("column-major   Fortran, BLAS, LAPACK", col_major, MUTED, 0.11))):
        ax1.text(0.03, y + 0.155, name, fontsize=8.6, color=col,
                 weight="bold")
        for i, v in enumerate(seq):
            ax1.add_patch(plt.Rectangle((0.03 + i * 0.095, y), 0.088, 0.105,
                                        facecolor=col, alpha=0.14,
                                        edgecolor=col, linewidth=0.9))
            ax1.text(0.074 + i * 0.095, y + 0.052, str(v), ha="center",
                     va="center", fontsize=10, color=INK)
        ax1.annotate("", xy=(0.62, y + 0.052), xytext=(0.03, y + 0.052),
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.0))
    ax1.text(0.70, 0.41, "one block of memory,\nread in two orders",
             fontsize=8.4, color=INK, alpha=0.85, va="center")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.03, 1.0)
    ax1.axis("off")
    ax1.set_title("the same six numbers", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 고른 차례가 무엇을 값으로 매기는가 ------------------------
    labels = ["sum(dim=1)\nalong memory", "sum(dim=0)\nacross it",
              ".contiguous()\non the transpose"]
    vals_t = [t_row, t_col, t_con]
    ax2.bar(range(3), vals_t, color=[ACCENT, INK, INK], width=0.55,
            alpha=0.85)
    for i, v in enumerate(vals_t):
        ax2.text(i, v + max(vals_t) * 0.025, f"{v:.2f} ms", ha="center",
                 fontsize=9, color=INK)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("milliseconds", fontsize=9)
    ax2.set_ylim(0, max(vals_t) * 1.2)
    ax2.set_title(f"4096 x 4096, same data", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             f"striding the wrong way costs {t_col/t_row:.1f}x. making it "
             f"contiguous costs {t_con/t_row:.0f} sums.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "memory_order.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote memory_order.svg   row {t_row:.3f} col {t_col:.3f} "
          f"contig {t_con:.3f}")


# === 그림으로 보는 AI: 섀넌이 포갠 두 세계와 그 값 ============================
#
# 왼쪽은 섀넌이 1937 년에 포갠 것 — 스위치 두 개를 잇따라 놓은 회로와 불의
# 두 값 대수가 같은 것이라는 관찰. 꼴만 옮긴 것이라 그림 안에 적는다.
# 오른쪽은 그 포갬이 오늘 물리는 자리. 파이썬이 실제로 어떻게 묶는지는
# 그리는 코드가 ast 로 직접 확인한 뒤에만 그린다.

def shannon_precedence():
    import ast

    # 그림이 주장하는 것을 먼저 확인한다
    bad = ast.parse("a == 1 & b == 1", mode="eval").body
    assert isinstance(bad, ast.Compare) and len(bad.ops) == 2, \
        "체인 비교로 묶이지 않으면 이 그림의 이야기가 성립하지 않는다"
    good = ast.parse("(a == 1) & (b == 1)", mode="eval").body
    assert isinstance(good, ast.BinOp) and isinstance(good.op, ast.BitAnd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 회로와 대수가 같다 ------------------------------------------
    y = 0.70
    ax1.plot([0.08, 0.26], [y, y], color=INK, linewidth=1.6)
    ax1.plot([0.26, 0.36], [y, y + 0.09], color=ACCENT, linewidth=2.0)
    ax1.plot([0.36, 0.44], [y, y], color=INK, linewidth=1.6)
    ax1.plot([0.44, 0.54], [y, y + 0.09], color=ACCENT, linewidth=2.0)
    ax1.plot([0.54, 0.72], [y, y], color=INK, linewidth=1.6)
    for x, lab in ((0.31, "A"), (0.49, "B")):
        ax1.text(x, y - 0.09, lab, ha="center", fontsize=10, color=ACCENT,
                 weight="bold")
    ax1.text(0.40, y + 0.20, "two switches in series", ha="center",
             fontsize=8.6, color=INK)

    ax1.text(0.40, 0.42, "A  AND  B", ha="center", fontsize=12, color=ACCENT,
             family="monospace", weight="bold")
    rows = [("0", "0", "0"), ("0", "1", "0"), ("1", "0", "0"), ("1", "1", "1")]
    for i, (p, q, r) in enumerate(rows):
        ax1.text(0.40, 0.30 - i * 0.062, f"{p}   {q}   |   {r}", ha="center",
                 fontsize=8.6, color=INK, family="monospace")
    ax1.text(0.40, 0.035, "schematic", ha="center", fontsize=6.4, color=INK,
             alpha=0.55, style="italic")
    ax1.set_title("Shannon, MIT, 1937", fontsize=10, color=INK, weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.02, 1.0)
    ax1.axis("off")

    # --- 오른쪽: 그 유산이 물리는 자리 -------------------------------------
    ax2.text(0.5, 0.93, "what you wrote", ha="center", fontsize=9,
             color=INK, style="italic")
    ax2.text(0.5, 0.82, "a == 1 & b == 1", ha="center", fontsize=13,
             color=INK, family="monospace")

    ax2.annotate("", xy=(0.5, 0.70), xytext=(0.5, 0.77),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    ax2.text(0.5, 0.60, "a == (1 & b) == 1", ha="center", fontsize=13,
             color=ACCENT, family="monospace", weight="bold")
    ax2.text(0.5, 0.50, "a chained comparison -\n& binds tighter than ==",
             ha="center", va="top", fontsize=8.4, color=ACCENT)

    ax2.plot([0.08, 0.92], [0.34, 0.34], color=INK, linewidth=0.7,
             alpha=0.35)
    ax2.text(0.5, 0.26, "what you meant", ha="center", fontsize=9,
             color=INK, style="italic")
    ax2.text(0.5, 0.15, "(a == 1) & (b == 1)", ha="center", fontsize=13,
             color=INK, family="monospace")
    ax2.text(0.5, 0.04, "the error blames the tensor, not the brackets",
             ha="center", fontsize=8.4, color=INK, alpha=0.8)
    ax2.set_title("the inheritance, today", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    fig.text(0.5, -0.03,
             "logic and bit-twiddling became one set of operators, and they "
             "still bind like arithmetic.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "shannon_precedence.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote shannon_precedence.svg   (ast confirms the chained compare)")


# === 그림으로 보는 AI: 더하는 차례가 답을 바꾼다 ==============================
#
# 왼쪽은 카핸이 1965 년에 적은 보정의 꼴 — 반올림으로 잃은 아랫자리를 따로
# 들고 다니다가 다음 번에 돌려준다. 꼴만 옮긴 것이라 그림 안에 적는다.
# 오른쪽은 **직접 잰다**: 같은 수 천만 개를 차례만 바꿔 더해 본다.
#
# 처음 세운 가설("float32 는 2^24 에서 멈춘다")은 재어 보니 **재현되지 않았다.**
# torch 의 합은 차례대로 쌓지 않고 쌍으로 묶어 더하기 때문이다. 그래서 그림은
# 멈춤이 아니라 **차례에 따라 답이 흔들리는 것**을 보인다.

def summation_order():
    import torch

    torch.manual_seed(0)

    # 큰 수 하나에 작은 수 천만 개
    v = torch.cat([torch.tensor([1e8]), torch.ones(10_000_000)]).float()
    exact = 1e8 + 10_000_000
    got = v.sum().item()
    assert got != exact, "이 그림은 float32 합이 참값과 다르다는 것을 보인다"

    # 같은 수, 차례만 바꿔 여러 번
    x = (torch.randn(4_000_000) * torch.exp(torch.randn(4_000_000) * 6)).float()
    ref = x.double().sum().item()
    sums = [x.sum().item()]
    for k in range(19):
        g = torch.Generator().manual_seed(k)
        sums.append(x[torch.randperm(len(x), generator=g)].sum().item())
    rel = [(s - ref) / abs(ref) for s in sums]
    assert len(set(sums)) > 1, "차례를 바꿔도 답이 같으면 이 그림의 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # --- 왼쪽: 카핸의 보정 -------------------------------------------------
    ax1.text(0.5, 0.92, "add one number", ha="center", fontsize=9,
             color=INK, style="italic")
    ax1.add_patch(plt.Rectangle((0.10, 0.63), 0.80, 0.17, facecolor=ACCENT,
                                alpha=0.13, edgecolor=ACCENT, linewidth=1.0))
    ax1.text(0.5, 0.715, "total  +  x", ha="center", va="center",
             fontsize=12, color=INK, family="monospace")

    ax1.annotate("", xy=(0.30, 0.50), xytext=(0.42, 0.60),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax1.annotate("", xy=(0.70, 0.50), xytext=(0.58, 0.60),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    ax1.text(0.26, 0.43, "what fits", ha="center", fontsize=8.6, color=INK)
    ax1.text(0.74, 0.43, "what fell off\nthe bottom", ha="center",
             fontsize=8.6, color=ACCENT)

    ax1.add_patch(plt.Rectangle((0.52, 0.20), 0.40, 0.15, facecolor=ACCENT,
                                alpha=0.20, edgecolor=ACCENT, linewidth=1.1))
    ax1.text(0.72, 0.275, "keep it", ha="center", va="center", fontsize=10,
             color=ACCENT, weight="bold")
    ax1.annotate("", xy=(0.88, 0.63), xytext=(0.92, 0.35),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3,
                                 connectionstyle="arc3,rad=-0.4"))
    ax1.text(0.5, 0.07, "give it back on the next addition", ha="center",
             fontsize=8.4, color=INK, alpha=0.85)
    ax1.text(0.97, 0.955, "schematic", ha="right", fontsize=6.4, color=INK,
             alpha=0.55, style="italic")
    ax1.set_title("Kahan, one page, 1965", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # --- 오른쪽: 같은 수, 스무 가지 차례 ------------------------------------
    ax2.axhline(0, color=INK, linestyle="--", linewidth=1.2)
    ax2.text(len(rel) * 0.5, 0, "  float64", fontsize=8.4, color=INK,
             va="bottom", ha="center")
    ax2.plot(range(1, len(rel) + 1), [r * 1e7 for r in rel], "o",
             color=ACCENT, ms=6)
    ax2.set_xlabel("the same 4,000,000 numbers, in 20 orders", fontsize=9)
    ax2.set_ylabel("error vs float64   (parts per 10 million)", fontsize=9)
    ax2.set_title("float32 gives a different answer each time", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"and 1e8 plus ten million ones comes back {got:,.0f}, "
             f"not {exact:,.0f}.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "summation_order.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote summation_order.svg   big-sum {got:,.0f} vs {exact:,.0f}   "
          f"spread {max(rel)-min(rel):.2e}")


# === 그림으로 보는 AI: 알고리즘은 그대로인데 증명이 바뀐 내력 =================
#
# 애덤은 수렴 증명과 함께 나왔고, 2018 년에 그 증명이 틀렸음이 밝혀졌으며,
# 고치려고 내놓은 AMSGrad 의 증명도 뒤에 문제가 지적되었다. **알고리즘은 한
# 번도 바뀌지 않았다** — 바뀐 것은 그것을 두고 할 수 있는 말이다.
#
# 반례를 이 자리에서 재현하지는 못했다(온라인 후회에 관한 결과라 마지막
# 반복값만 보아서는 드러나지 않는다). 그래서 그림은 수를 주장하지 않고
# 내력만 그린다.

def adam_proof():
    fig, ax = plt.subplots(figsize=(9.8, 3.6))

    steps = [
        ("Adam", "published with\na convergence proof", INK),
        ("2018", "the proof assumes a property\nAdam does not have", ACCENT),
        ("AMSGrad", "proposed as the fix,\nwith its own proof", INK),
        ("later", "that proof is\nquestioned too", ACCENT),
    ]
    xs = [0.5, 3.4, 6.3, 9.2]
    ax.annotate("", xy=(10.0, 1.0), xytext=(0.0, 1.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))
    for x, (head, body, col) in zip(xs, steps):
        ax.plot([x], [1.0], marker="o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.text(x, 1.26, head, ha="center", fontsize=10.5, color=col,
                weight="bold")
        ax.text(x, 0.82, body, ha="center", va="top", fontsize=8.4,
                color=INK, alpha=0.9)

    ax.add_patch(plt.Rectangle((-0.35, 0.18), 10.7, 0.30, facecolor=ACCENT,
                               alpha=0.12, edgecolor=ACCENT, linewidth=1.0))
    ax.text(5.0, 0.33, "the algorithm never changed",
            ha="center", va="center", fontsize=10, color=ACCENT,
            weight="bold")

    ax.set_xlim(-0.7, 10.7)
    ax.set_ylim(0.05, 1.55)
    ax.axis("off")
    fig.text(0.5, -0.02,
             "what changed is what could be claimed about it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "adam_proof.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote adam_proof.svg")


# === 그림으로 보는 AI: 체크포인트는 자료가 아니라 프로그램이다 ================
#
# torch.save 가 만든 .pt 안의 data.pkl 을 pickletools 로 뜯어 보면 **명령 목록**
# 이다. GLOBAL 은 "이 이름을 찾아라", REDUCE 는 "그것을 불러라" 이고, 둘이 있는
# 한 읽는 일과 실행하는 일이 갈라지지 않는다. 수는 전부 여기서 직접 센다.

def checkpoint_is_a_program():
    import collections
    import io
    import pickletools
    import re
    import tempfile
    import zipfile
    from pathlib import Path as _P

    import torch

    torch.manual_seed(0)
    layer = torch.nn.Linear(3, 2)
    n_numbers = sum(p.numel() for p in layer.parameters())
    with tempfile.TemporaryDirectory() as d:
        f = _P(d) / "sd.pt"
        torch.save(layer.state_dict(), f)
        z = zipfile.ZipFile(f)
        raw = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])

    out = io.StringIO()
    pickletools.dis(raw, out)
    lines = [l.rstrip() for l in out.getvalue().splitlines()]
    ops = [m.group(1) for l in lines
           if (m := re.match(r"\s*\d+:\s+\S+\s+([A-Z_0-9]+)", l))]
    cnt = collections.Counter(ops)
    call_ops = {k: cnt[k] for k in ("GLOBAL", "STACK_GLOBAL", "REDUCE",
                                    "BUILD", "INST", "OBJ") if cnt.get(k)}
    assert call_ops, "이름을 찾아 부르는 명령이 없으면 이 그림의 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # --- 왼쪽: 실제 명령 목록 ----------------------------------------------
    show = lines[:9]
    for i, l in enumerate(show):
        hot = any(k in l for k in ("GLOBAL", "REDUCE", "BUILD"))
        ax1.text(0.02, 0.86 - i * 0.096, l[:52], fontsize=8.6,
                 family="monospace", color=ACCENT if hot else INK,
                 weight="bold" if hot else "normal", va="center")
        if hot:
            ax1.add_patch(plt.Rectangle((0.0, 0.86 - i * 0.096 - 0.040),
                                        1.0, 0.080, facecolor=ACCENT,
                                        alpha=0.11, edgecolor="none"))
    ax1.text(0.02, 0.02, "...", fontsize=9, family="monospace", color=INK,
             alpha=0.6)
    ax1.set_title("inside the .pt file", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.02, 0.96)
    ax1.axis("off")

    # --- 오른쪽: 세어 보면 --------------------------------------------------
    ax2.text(0.5, 0.88, f"{n_numbers} numbers saved", ha="center",
             fontsize=10, color=INK)
    ax2.annotate("", xy=(0.5, 0.70), xytext=(0.5, 0.80),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax2.text(0.5, 0.60, f"{len(ops)}", ha="center", fontsize=30,
             color=ACCENT, weight="bold")
    ax2.text(0.5, 0.47, "instructions", ha="center", fontsize=10,
             color=ACCENT)

    y = 0.33
    ax2.text(0.5, y, "   ".join(f"{k} x{v}" for k, v in call_ops.items()),
             ha="center", fontsize=9, family="monospace", color=INK)
    ax2.text(0.5, 0.21, "GLOBAL: find this name\nREDUCE: call it",
             ha="center", va="top", fontsize=8.6, color=ACCENT)
    ax2.text(0.5, 0.03,
             "you cannot check it without running it",
             ha="center", fontsize=8.8, color=INK, style="italic")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("counted, not asserted", fontsize=10, color=INK,
                  weight="bold")

    fig.tight_layout()
    fig.savefig(HERE / "checkpoint_program.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote checkpoint_program.svg   {n_numbers} numbers -> "
          f"{len(ops)} ops, {call_ops}")


# === 그림으로 보는 AI: 암달의 1967년 상한 =====================================
#
# 가속기가 아무리 빨라져도 **못 줄이는 몫**이 속도를 가둔다. 여기서 못 줄이는
# 몫은 자료를 올리고 내리는 시간이다. 왼쪽은 그 꼴, 오른쪽은 이 기계에서 잰
# 상한. 가속기가 없으면 CPU 로 재고 그림 안에 그렇게 적는다.

def amdahl_ceiling():
    import time

    import torch

    dev = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    def sync():
        if dev == "mps":
            torch.mps.synchronize()
        elif dev == "cuda":
            torch.cuda.synchronize()

    def bench(fn, reps=50, rounds=5):
        for _ in range(10):
            fn()
        sync()
        best = None
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            sync()
            t = (time.perf_counter() - t0) / reps * 1e3
            best = t if best is None else min(best, t)
        return best

    torch.manual_seed(0)
    n = 1024
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    host = torch.randn(n, n)

    t_mm = bench(lambda: a @ b)
    t_up = bench(lambda: host.to(dev), reps=20)
    t_dn = bench(lambda: a.to("cpu"), reps=20)
    serial = t_up + t_dn
    assert serial > 0 and t_mm > 0

    ks = [1, 2, 4, 8, 16, 32, 64]
    ceil = [(k * t_mm + serial) / serial for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 줄어드는 몫과 안 줄어드는 몫 --------------------------------
    for i, (lab, par) in enumerate((("on the CPU", 1.00),
                                    ("4x faster", 0.25),
                                    ("infinitely fast", 0.0))):
        y = 0.72 - i * 0.26
        ax1.add_patch(plt.Rectangle((0.06, y), 0.52 * par, 0.15,
                                    facecolor=INK, alpha=0.30,
                                    edgecolor=INK, linewidth=0.8))
        ax1.add_patch(plt.Rectangle((0.06 + 0.52 * par, y), 0.26, 0.15,
                                    facecolor=ACCENT, alpha=0.55,
                                    edgecolor=ACCENT, linewidth=0.8))
        ax1.text(0.04, y + 0.075, lab, ha="right", va="center", fontsize=8.6,
                 color=INK)
    ax1.text(0.32, 0.93, "compute", ha="center", fontsize=8.6, color=INK)
    ax1.text(0.84, 0.93, "moving the data", ha="center", fontsize=8.6,
             color=ACCENT, weight="bold")
    ax1.text(0.5, 0.10, "the amber block never shrinks.\n"
                        "it is what sets the ceiling.",
             ha="center", va="center", fontsize=8.8, color=INK)
    ax1.set_xlim(-0.22, 1.12)
    ax1.set_ylim(0.02, 1.02)
    ax1.axis("off")
    ax1.set_title("Amdahl, 1967", fontsize=10, color=INK, weight="bold")

    # --- 오른쪽: 이 기계에서 잰 상한 ---------------------------------------
    ax2.semilogx(ks, ceil, "o-", color=ACCENT, linewidth=2.0, ms=6,
                 base=2)
    for k, c in zip(ks, ceil):
        if k in (1, 8, 64):
            ax2.annotate(f"{c:.1f}x", xy=(k, c), xytext=(0, 11),
                         textcoords="offset points", ha="center",
                         fontsize=8.6, color=INK)
    ax2.set_xlabel(f"matmuls done per round trip  ({n} x {n})", fontsize=9)
    ax2.set_ylabel("best speedup still possible", fontsize=9)
    ax2.set_title(f"measured here, on {dev}", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_ylim(0, max(ceil) * 1.16)      # 맨 위 라벨이 잘리지 않게
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"one round trip costs {serial:.3f} ms, one matmul "
             f"{t_mm:.3f} ms. send once and keep it there.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "amdahl_ceiling.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote amdahl_ceiling.svg   dev={dev} mm={t_mm:.3f} "
          f"serial={serial:.3f} ceil1={ceil[0]:.2f} ceil64={ceil[-1]:.1f}")


# === 그림으로 보는 AI: 체크포인트를 몇 군데 둘 것인가 ==========================
#
# 체크포인트를 잘게 쪼갤수록 좋을 것 같지만 **아니다.** 구간을 늘리면 구간
# 경계 자체를 저장해야 해서 어느 지점을 지나면 자리도 시간도 다시 나빠진다.
# 그 최적점이 블록 수의 제곱근 언저리라는 것이 그리방크가 보인 것이다.
# 수는 전부 여기서 직접 잰다.

def checkpoint_tradeoff():
    import math
    import time

    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint_sequential

    torch.manual_seed(0)
    N, W, B = 32, 512, 64
    net = nn.Sequential(*[nn.Sequential(nn.Linear(W, W), nn.ReLU())
                          for _ in range(N)])
    x = torch.randn(B, W, requires_grad=True)

    def once(segs):
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
                lambda t: (saved.append(t.numel() * t.element_size()), t)[1],
                lambda t: t):
            y = net(x) if segs == 0 else checkpoint_sequential(
                net, segs, x, use_reentrant=False)
            y.sum().backward()
        net.zero_grad(set_to_none=True)
        x.grad = None
        return sum(saved) / 1024

    def timed(segs, reps=12, rounds=5):
        """최소값을 쓴다 — 벽시계는 위로만 튄다."""
        for _ in range(3):
            once(segs)
        best = None
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                once(segs)
            t = (time.perf_counter() - t0) / reps * 1e3
            best = t if best is None else min(best, t)
        return best

    base_kb, base_ms = once(0), timed(0)
    segs = [2, 4, 8, 16, 32]
    mem = [once(s) / base_kb for s in segs]
    tim = [timed(s) / base_ms for s in segs]
    best = segs[min(range(len(segs)), key=lambda i: mem[i])]
    assert min(mem) < 0.5, "체크포인트가 자리를 줄이지 못하면 이 그림의 이야기가 없다"
    assert mem[-1] > min(mem), "구간을 끝까지 쪼개면 자리가 도로 늘어야 한다"

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.semilogx(segs, mem, "o-", base=2, color=ACCENT, linewidth=2.1, ms=7,
                label="activations kept")
    ax.semilogx(segs, tim, "s--", base=2, color=INK, linewidth=1.9, ms=6,
                label="time taken")
    ax.axhline(1.0, color=INK, linestyle=":", linewidth=1.0, alpha=0.5)
    ax.text(segs[0], 1.03, "  no checkpointing", fontsize=8.2, color=INK,
            alpha=0.75)

    root = math.sqrt(N)
    ax.axvline(root, color=ACCENT, linestyle=":", linewidth=1.3, alpha=0.8)
    ax.annotate(f"sqrt({N}) = {root:.1f}", xy=(root, 1.55),
                xytext=(10, 0), textcoords="offset points", fontsize=8.6,
                color=ACCENT)

    i = segs.index(best)
    ax.annotate(f"least memory\n{segs[i]} segments",
                xy=(segs[i], mem[i]), xytext=(-6, 34),
                textcoords="offset points", ha="center", fontsize=8.4,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.set_xticks(segs)
    ax.set_xticklabels([str(s) for s in segs])
    ax.set_xlabel(f"segments the {N} blocks are cut into", fontsize=9.5)
    ax.set_ylabel("relative to no checkpointing", fontsize=9.5)
    ax.set_title("cutting finer stops helping", fontsize=10.5, color=INK,
                 weight="bold", pad=8)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             f"at {segs[-1]} segments every boundary must be stored, so memory "
             f"climbs back from {min(mem):.2f}x to {mem[-1]:.2f}x.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "checkpoint_tradeoff.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote checkpoint_tradeoff.svg   mem {[round(m,2) for m in mem]}  "
          f"time {[round(t,2) for t in tim]}  best={best}")


# === 그림으로 보는 AI: 겹침을 다루는 세 가지 길 ===============================
#
# 포트란은 **금지**하고(언어 규칙), C 는 **허락**한 뒤 restrict 로 약속받고,
# PyTorch 는 허락한 채 **버전을 세어 잡는다.** 왼쪽은 그 셋, 오른쪽은 실제로
# 돌려 본 버전 세기. 같은 add_ 인데 저장 뒤에 했느냐가 갈림이다.

def aliasing_three_ways():
    import torch

    def trial(modify_after_save):
        x = torch.ones(3, requires_grad=True)
        y = x * 2
        if not modify_after_save:
            y.add_(1)
        z = (y * y).sum()              # 여기서 y 가 저장된다
        v_save = y._version
        if modify_after_save:
            y.add_(1)
        try:
            z.backward()
            return v_save, y._version, True
        except RuntimeError:
            return v_save, y._version, False

    before = trial(False)
    after = trial(True)
    assert before[2] and not after[2], \
        "저장 전/후가 갈리지 않으면 이 그림의 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- 왼쪽: 세 가지 길 ---------------------------------------------------
    ways = [
        ("Fortran", "the language forbids it", "arguments never alias", INK),
        ("C", "allowed by default", "restrict is a promise you make", INK),
        ("PyTorch", "allowed, and watched", "a counter catches the one\n"
                                            "case that would corrupt a gradient",
         ACCENT),
    ]
    for k, (name, rule, note, col) in enumerate(ways):
        y = 0.80 - k * 0.30
        ax1.add_patch(plt.Rectangle((0.03, y - 0.115), 0.94, 0.235,
                                    facecolor=col,
                                    alpha=0.17 if col is ACCENT else 0.07,
                                    edgecolor=col, linewidth=1.0))
        ax1.text(0.09, y + 0.055, name, fontsize=11, color=col,
                 weight="bold", va="center")
        ax1.text(0.09, y - 0.045, rule, fontsize=9, color=INK, va="center")
        ax1.text(0.95, y, note, fontsize=8.2, color=col, ha="right",
                 va="center", style="italic")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.08, 1.0)
    ax1.axis("off")
    ax1.set_title("three answers to the same hazard", fontsize=10,
                  color=INK, weight="bold")

    # --- 오른쪽: 실제로 잰 버전 세기 ---------------------------------------
    ax2.text(0.5, 0.93, "the same  y.add_(1)", ha="center", fontsize=10.5,
             color=INK, family="monospace")
    rows = [("changed before saving", before, "runs"),
            ("changed after saving", after, "raises")]
    for k, (lab, (v0, v1, ok), word) in enumerate(rows):
        y = 0.63 - k * 0.33
        col = INK if ok else ACCENT
        ax2.add_patch(plt.Rectangle((0.03, y - 0.105), 0.94, 0.225,
                                    facecolor=col, alpha=0.10,
                                    edgecolor=col, linewidth=1.0))
        ax2.text(0.08, y + 0.055, lab, fontsize=9.2, color=INK, va="center")
        ax2.text(0.08, y - 0.045, f"version {v0} at save  ->  {v1} at backward",
                 fontsize=8.6, color=INK, family="monospace", va="center")
        ax2.text(0.93, y, word, fontsize=11, color=col, ha="right",
                 va="center", weight="bold")
    ax2.text(0.5, 0.045, "nothing is forbidden. the difference is\n"
                         "whether the number moved.",
             ha="center", va="center", fontsize=8.6, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("measured here", fontsize=10, color=INK, weight="bold")

    fig.tight_layout()
    fig.savefig(HERE / "aliasing_three_ways.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote aliasing_three_ways.svg   before={before} after={after}")


# === 그림으로 보는 AI: empty 는 비어 있지 않다 ================================
#
# 왼쪽은 놓아준 값이 그대로 돌아오는지를 크기마다 여러 번 세어 본 것이다.
# **언제나 돌아오지는 않는다** — 할당기 상태에 달려 있어 같은 기계에서도 갈린다.
# 그래서 단언하지 않고 센 값을 그대로 적는다. 오른쪽은 왜 안 채워 두는가, 곧
# 채우는 값이 크기에 비례한다는 것이고 이쪽은 언제나 참이라 단언한다.

def empty_is_not_empty():
    import gc
    import time

    import torch

    MARK = 12345.0
    sizes = [4096, 65536, 1_000_000]
    trials = 6

    # 표시한 값을 넣고 놓아준 뒤 같은 크기로 empty 를 받는다. **되돌아오는지는
    # 할당기 상태에 달려 있고 부르는 쪽이 고를 수 없다** — 같은 기계에서도
    # 크기에 따라, 앞서 무엇을 했느냐에 따라 갈린다. 그래서 되돌아왔다고
    # 단언하지 않고 몇 번 되돌아왔는지를 세어 그대로 적는다.
    reuse = []
    for n in sizes:
        hit = 0
        for _ in range(trials):
            old = torch.full((n,), MARK)
            ptr = old.data_ptr()
            del old
            gc.collect()
            e = torch.empty(n)
            if e.data_ptr() == ptr and int((e == MARK).sum()) == n:
                hit += 1
            del e
        reuse.append(hit)

    def bench(fn, reps=200):
        for _ in range(20):
            fn()
        best = None
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            t = (time.perf_counter() - t0) / reps * 1e6
            best = t if best is None else min(best, t)
        return best

    Ns = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    t_e = [bench(lambda N=N: torch.empty(N)) for N in Ns]
    t_z = [bench(lambda N=N: torch.zeros(N)) for N in Ns]
    assert t_z[-1] > t_e[-1] * 10, "zeros 가 크기를 타야 이 그림의 이야기가 산다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 되돌아오기도 하고 아니기도 하다 -----------------------------
    ax1.text(0.5, 0.92, f"fill with {MARK:.0f}, free, then torch.empty",
             ha="center", fontsize=9.3, color=INK, style="italic")
    for i, (n, hit) in enumerate(zip(sizes, reuse)):
        y = 0.68 - i * 0.185
        col = ACCENT if hit else INK
        ax1.add_patch(plt.Rectangle((0.05, y - 0.058), 0.90, 0.130,
                                    facecolor=col,
                                    alpha=0.15 if hit else 0.07,
                                    edgecolor=col, linewidth=0.9))
        ax1.text(0.10, y + 0.007, f"n = {n:,}", fontsize=9,
                 family="monospace", color=INK, va="center")
        ax1.text(0.91, y + 0.007,
                 f"{hit} of {trials} came back full of {MARK:.0f}",
                 fontsize=8.5, color=col, ha="right", va="center",
                 weight="bold" if hit else "normal")
    ax1.text(0.5, 0.10,
             "whether the old block comes back is not\n"
             "something the caller gets to choose.",
             ha="center", va="center", fontsize=8.7, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.02, 1.0)
    ax1.axis("off")
    ax1.set_title("what came back, this run", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 채우는 값은 크기를 탄다 -----------------------------------
    ax2.loglog(Ns, t_z, "s--", color=INK, linewidth=1.9, ms=6,
               label="torch.zeros(N)")
    ax2.loglog(Ns, t_e, "o-", color=ACCENT, linewidth=2.1, ms=6,
               label="torch.empty(N)")
    ax2.annotate(f"{t_z[-1] / t_e[-1]:,.0f}x apart",
                 xy=(Ns[-1], t_z[-1]), xytext=(-104, -6),
                 textcoords="offset points", fontsize=8.8, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("number of elements", fontsize=9)
    ax2.set_ylabel("microseconds per call", fontsize=9)
    ax2.set_title("why it is left unwritten", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.4, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "skipping the write is the point. what is left in its place "
             "is not decided by you.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "empty_is_not_empty.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote empty_is_not_empty.svg   reuse {reuse} of {trials} "
          f"at {sizes}   ratio {t_z[-1] / t_e[-1]:.0f}x")


# === 그림으로 보는 AI: 뷰냐 사본이냐는 고른 것이 아니다 ========================
#
# 뷰는 (시작점, 모양, 스트라이드) 셋으로만 적힌다. 그래서 **등차수열로 고르는
# 것**은 뷰가 되고 아무 자리나 찍어 고르는 것은 될 수 없다. 규칙이 아니라
# 표현력의 한계다. 어느 쪽이 뷰인지는 여기서 직접 확인해 적는다.

def view_or_copy():
    import torch

    a = torch.arange(10.)

    def shares(x, y):
        return (x.untyped_storage().data_ptr()
                == y.untyped_storage().data_ptr())

    same_values = (a[::2].tolist() == a[[0, 2, 4, 6, 8]].tolist())
    assert same_values, "두 표현이 같은 값을 줘야 이 그림의 이야기가 산다"
    assert shares(a, a[::2]) and not shares(a, a[[0, 2, 4, 6, 8]]), \
        "한쪽만 뷰여야 한다"

    rows = []
    for name, fn in (("a[2:7]", lambda: a[2:7]),
                     ("a[::3]", lambda: a[::3]),
                     ("a[[3,1,4]]", lambda: a[[3, 1, 4]]),
                     ("a[a > 5]", lambda: a[a > 5])):
        r = fn()
        rows.append((name, shares(a, r), r.stride()[0], r.storage_offset()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 같은 값, 다른 기제 ------------------------------------------
    ax1.text(0.5, 0.93, "the same five numbers", ha="center", fontsize=9.5,
             color=INK, style="italic")
    for k, (expr, isview, note) in enumerate(
            (("a[::2]", True, "start 0, stride 2 - writable as a rule"),
             ("a[[0,2,4,6,8]]", False, "a rule WOULD reach these.\nthe form copies anyway"))):
        y = 0.68 - k * 0.26
        col = ACCENT if isview else INK
        ax1.add_patch(plt.Rectangle((0.04, y - 0.085), 0.92, 0.185,
                                    facecolor=col, alpha=0.14,
                                    edgecolor=col, linewidth=1.0))
        ax1.text(0.09, y + 0.035, expr, fontsize=11, family="monospace",
                 color=INK, va="center")
        ax1.text(0.93, y + 0.035, "view" if isview else "copy", fontsize=11,
                 color=col, ha="right", va="center", weight="bold")
        ax1.text(0.09, y - 0.050, note, fontsize=8.0, color=col,
                 va="center", linespacing=1.4)

    ax1.text(0.5, 0.15, "the category is set by the form you write,\n"
                        "not by whether the positions happen to be regular",
             ha="center", va="center", fontsize=8.6, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.06, 1.0)
    ax1.axis("off")
    ax1.set_title("what a view can say", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 확인한 표 --------------------------------------------------
    ax2.text(0.5, 0.92, "checked, not assumed", ha="center", fontsize=9.3,
             color=INK, style="italic")
    for k, (name, isview, st, off) in enumerate(rows):
        y = 0.72 - k * 0.165
        col = ACCENT if isview else INK
        ax2.add_patch(plt.Rectangle((0.04, y - 0.055), 0.92, 0.120,
                                    facecolor=col,
                                    alpha=0.14 if isview else 0.06,
                                    edgecolor=col, linewidth=0.9))
        ax2.text(0.09, y + 0.005, name, fontsize=9.6, family="monospace",
                 color=INK, va="center")
        ax2.text(0.60, y + 0.005, f"stride {st}", fontsize=8.4,
                 family="monospace", color=INK, va="center", alpha=0.85)
        ax2.text(0.93, y + 0.005, "view" if isview else "copy",
                 fontsize=9.6, color=col, ha="right", va="center",
                 weight="bold")
    ax2.text(0.5, 0.10, "reading hands back a tensor, so it needs a layout.\n"
                        "writing hands back nothing, so it can scatter.",
             ha="center", va="center", fontsize=8.6, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.02, 1.0)
    ax2.axis("off")
    ax2.set_title("which ones are views", fontsize=10, color=INK,
                  weight="bold")

    fig.tight_layout()
    fig.savefig(HERE / "view_or_copy.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote view_or_copy.svg   {[(n, v) for n, v, _, _ in rows]}")


if __name__ == "__main__":
    view_or_copy()
    empty_is_not_empty()
    aliasing_three_ways()
    checkpoint_tradeoff()
    amdahl_ceiling()
    checkpoint_is_a_program()
    adam_proof()
    summation_order()
    shannon_precedence()
    memory_order()
    tensor_word()
    arange_promise()
    loss_scaling()
    no_formula()
    buffer_protocol()
    shuffling()
    fourth_relation()
    cheap_gradient()
