"""39.6 실제로 살필 것 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

작은 MDP 를 값 반복으로 **정확히** 푼다. 무작위가 없으므로 다시 돌려도 같다.

실행: python docs/ch38/practical/figures/make_figures.py
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

N, GAMMA = 6, 0.95          # 상태 0..N-1, 목표는 N-1 에서 끝난다


def _solve(shape):
    """값 반복. shape(s, s2) 가 더해지는 셰이핑 리워드."""
    V = np.zeros(N)
    for _ in range(20000):
        Vn = V.copy()
        for s in range(N - 1):
            qs = []
            for a in (0, 1):
                s2 = min(max(s + (1 if a else -1), 0), N - 1)
                r = 1.0 if s2 == N - 1 else 0.0
                nxt = 0.0 if s2 == N - 1 else GAMMA * V[s2]
                qs.append(r + shape(s, s2) + nxt)
            Vn[s] = max(qs)
        if np.abs(Vn - V).max() < 1e-13:
            break
        V = Vn
    pi = []
    for s in range(N - 1):
        qs = []
        for a in (0, 1):
            s2 = min(max(s + (1 if a else -1), 0), N - 1)
            r = 1.0 if s2 == N - 1 else 0.0
            nxt = 0.0 if s2 == N - 1 else GAMMA * V[s2]
            qs.append(r + shape(s, s2) + nxt)
        pi.append(int(np.argmax(qs)))
    return pi, V


# === 그림으로 보는 AI: 꼴이 정해져 있는 까닭 ==================================
#
# 응·하라다·러셀 1999 — 최적 폴리시를 지키는 셰이핑은 포텐셜 꼴뿐이다.
# "앞으로 가면 상을 준다"는 그럴듯한 셰이핑이 어디서부터 폴리시를 바꾸는지,
# 그리고 포텐셜 꼴은 바꾸지 않는다는 것을 값 반복으로 정확히 푼다.

def shaping_must_be_potential():
    base, V0 = _solve(lambda s, s2: 0.0)

    PHI = lambda s: 0.3 * s
    pot, Vp = _solve(lambda s, s2: GAMMA * PHI(s2) - PHI(s))
    assert pot == base, "포텐셜 꼴이 폴리시를 바꾸면 정리가 깨진 것이다"

    cs = np.linspace(0.0, 1.0, 41)
    changed, v0 = [], []
    for c in cs:
        pi, V = _solve(lambda s, s2, c=c: c if s2 > s else 0.0)
        changed.append(int(pi != base))
        v0.append(float(V[0]))
    first = float(cs[int(np.argmax(changed))]) if any(changed) else None
    assert first is not None, "어디선가는 폴리시가 바뀌어야 이 그림이 성립한다"

    pi_bad, _ = _solve(lambda s, s2: 0.5 if s2 > s else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- 왼쪽: 복도 위의 두 폴리시 ----------------------------------------
    for row, (pi, lab, col) in enumerate((
            (base, "true reward, and with potential shaping", INK),
            (pi_bad, "with  +c for stepping forward  (c = 0.5)", ACCENT))):
        y = 0.68 - row * 0.34
        for s in range(N):
            is_goal = (s == N - 1)
            ax1.add_patch(plt.Circle((0.08 + s * 0.155, y), 0.040,
                                     facecolor=ACCENT if is_goal else "white",
                                     alpha=0.30 if is_goal else 1.0,
                                     edgecolor=INK, linewidth=1.0, zorder=3))
            if is_goal:
                ax1.text(0.08 + s * 0.155, y - 0.085, "goal", fontsize=7.6,
                         color=INK, ha="center")
        for s in range(N - 1):
            d = 1 if pi[s] else -1
            x0 = 0.08 + s * 0.155 + d * 0.045
            ax1.annotate("", xy=(x0 + d * 0.062, y + 0.052),
                         xytext=(x0, y + 0.052),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax1.text(0.04, y + 0.155, lab, fontsize=8.6, color=col,
                 weight="bold")
    ax1.text(0.5, 0.055, "with the forward bonus the agent turns back at the\n"
                         "last step - stepping to and fro pays better than finishing.",
             ha="center", va="bottom", fontsize=8.3, color=INK, alpha=0.9)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")
    ax1.set_title("same corridor, two shaped rewards", fontsize=10,
                  color=INK, weight="bold")

    # --- 오른쪽: 어디서부터 깨지는가 ---------------------------------------
    ax2.plot(cs, v0, "-", color=INK, linewidth=2.0, label="value of state 0")
    ax2.axvline(first, color=ACCENT, linestyle="--", linewidth=1.6)
    ax2.annotate(f"policy changes\nfrom c = {first:.2f}", xy=(first, v0[-1] * 0.5),
                 xytext=(14, 0), textcoords="offset points", fontsize=8.6,
                 color=ACCENT)
    ax2.axhline(V0[0], color=MUTED, linestyle=":", linewidth=1.4)
    ax2.text(0.02, V0[0] + v0[-1] * 0.03, f"true value {V0[0]:.3f}",
             fontsize=8.3, color=MUTED)
    ax2.set_xlabel("size of the forward bonus, c", fontsize=9.5)
    ax2.set_ylabel("value the agent believes state 0 has", fontsize=9.5)
    ax2.set_title("the bonus inflates what it thinks it is worth",
                  fontsize=10, color=INK, weight="bold")
    ax2.grid(alpha=0.13)
    ax2.tick_params(labelsize=8.5, colors=INK)
    for sp in ax2.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"potential shaping leaves the policy untouched at every scale "
             f"tried; the forward bonus breaks it from c = {first:.2f}.",
             ha="center", fontsize=8.7, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "shaping_potential.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote shaping_potential.svg   base {base}  bad {pi_bad}  "
          f"breaks at c={first:.2f}   V0 {V0[0]:.3f} -> {v0[-1]:.3f}")


if __name__ == "__main__":
    shaping_must_be_potential()
