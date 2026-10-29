"""37.2 밸류 함수 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

**다 그리는 데 100 초쯤 걸린다** — model_free() 가 씨앗 여덟 개로 Q 학습을
백만 걸음까지 돌리기 때문이다. 멈춘 것이 아니다.

실행: python docs/ch36/value_functions/figures/make_figures.py
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

NS, NA, GAMMA = 60, 4, 0.9


def _mdp(seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((NS, NA, NS))
    P /= P.sum(2, keepdims=True)
    R = rng.standard_normal((NS, NA))
    return P, R, rng


# === 그림으로 보는 AI: 되짚는 식 하나가 왜 수렴하는가 =========================
#
# 벨먼 최적 연산자 T 는 sup-노름에서 gamma-축약이다. 두 가지를 재어 그린다.
# (1) 아무 두 밸류 함수에 T 를 먹였을 때의 축약 비율 — gamma 를 넘지 않는다.
# (2) v_k 가 v* 로 가는 속도 — 결국 gamma^k 꼴로 준다.

def contraction():
    P, R, rng = _mdp()

    def T(v):
        return (R + GAMMA * P @ v).max(1)

    ratios = []
    for _ in range(400):
        v = rng.standard_normal(NS) * 5
        w = rng.standard_normal(NS) * 5
        ratios.append(float(np.abs(T(v) - T(w)).max() / np.abs(v - w).max()))
    assert max(ratios) <= GAMMA + 1e-12, f"축약이 깨졌다: {max(ratios)}"

    vstar = np.zeros(NS)
    for _ in range(600):
        vstar = T(vstar)
    v, errs = np.zeros(NS), []
    for _ in range(45):
        errs.append(float(np.abs(v - vstar).max()))
        v = T(v)
    errs = np.array(errs)
    rate = (errs[40] / errs[10]) ** (1 / 30)
    assert errs[-1] < errs[0] / 10, "줄어들지 않는다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # --- 왼쪽: 한 번 먹일 때의 축약 비율 -----------------------------------
    ax1.hist(ratios, bins=35, color=INK, alpha=0.35, edgecolor=INK,
             linewidth=0.6)
    ax1.axvline(GAMMA, color=ACCENT, linewidth=2.2)
    ax1.text(GAMMA - 0.02, ax1.get_ylim()[1] * 0.92,
             f"gamma = {GAMMA}\nthe guarantee", fontsize=8.8, color=ACCENT,
             ha="right", va="top")
    ax1.text(max(ratios) + 0.02, ax1.get_ylim()[1] * 0.45,
             f"worst actually seen\n{max(ratios):.3f}", fontsize=8.4,
             color=INK, ha="left", va="center")
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("how much one application of T shrinks the gap",
                   fontsize=9)
    ax1.set_ylabel("out of 400 random pairs", fontsize=9)
    ax1.set_title("T never expands", fontsize=10, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="y")

    # --- 오른쪽: 그래서 v_k 가 v* 로 간다 ----------------------------------
    ks = np.arange(len(errs))
    ax2.semilogy(ks, errs, "o-", color=ACCENT, linewidth=2.0, ms=4,
                 label="distance to the fixed point")
    ax2.semilogy(ks, errs[0] * GAMMA ** ks, "--", color=INK, linewidth=1.6,
                 label=f"gamma^k")
    ax2.set_xlabel("sweeps of the Bellman operator", fontsize=9)
    ax2.set_ylabel("max error over states", fontsize=9)
    ax2.set_title(f"the late rate settles at {rate:.3f}", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.4, framealpha=0.92)
    ax2.grid(alpha=0.13, which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"a random MDP with {NS} states and {NA} actions. random pairs "
             "shrink faster than promised - the iteration itself settles "
             "onto exactly gamma.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "contraction.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote contraction.svg   worst ratio {max(ratios):.4f}   "
          f"late rate {rate:.4f}")


# === 그림으로 보는 AI: 모형을 보지 않고 가장 좋은 폴리시에 닿는다 =============
#
# 값 되풀이는 P 와 R 을 모두 들여다본다. Q 학습은 표본만 받는다 — P 도 R 도
# 한 번 보지 않는다. 그런데도 같은 폴리시에 닿는다. 그것이 왓킨스가 주장한
# 바로 그것이라, 표본 수를 늘려 가며 몇 상태가 일치하는지 센다.

def model_free():
    rng = np.random.default_rng(0)
    nS, nA, g = 25, 4, 0.9
    P = rng.random((nS, nA, nS))
    P /= P.sum(2, keepdims=True)
    R = rng.standard_normal((nS, nA))

    v = np.zeros(nS)
    for _ in range(5000):
        v = (R + g * P @ v).max(1)
    Qstar = R + g * P @ v
    pi_star = Qstar.argmax(1)

    def q_learning(n_steps, seed=1):
        rr = np.random.default_rng(seed)
        Q = np.zeros((nS, nA))
        s = 0
        for t in range(n_steps):
            a = rr.integers(nA) if rr.random() < 0.3 else int(Q[s].argmax())
            s2 = int(rr.choice(nS, p=P[s, a]))     # 환경이 돌려주는 표본
            r = R[s, a] + 0.01 * rr.standard_normal()
            Q[s, a] += (0.5 / (1 + t / 20000)) * (r + g * Q[s2].max()
                                                  - Q[s, a])
            s = s2
        return Q

    # 한 씨앗만 쓰면 곡선이 들쭉날쭉해 잘못 읽히기 쉽다. 여러 씨앗을 평균한다.
    budgets = [10_000, 30_000, 100_000, 300_000, 1_000_000]
    seeds = range(8)
    match, lo, hi, errs = [], [], [], []
    for n in budgets:
        ms, es = [], []
        for sd in seeds:
            Q = q_learning(n, seed=sd)
            ms.append(int((Q.argmax(1) == pi_star).sum()))
            es.append(float(np.abs(Q - Qstar).max()))
        match.append(float(np.mean(ms))); lo.append(min(ms)); hi.append(max(ms))
        errs.append(float(np.mean(es)))
    assert match[-1] > match[0], f"늘어나지 않는다: {match}"

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.fill_between(budgets, lo, hi, color=ACCENT, alpha=0.16)
    ax.semilogx(budgets, match, "o-", color=ACCENT, linewidth=2.2, ms=7)
    ax.axhline(nS, color=INK, linestyle="--", linewidth=1.3)
    ax.text(budgets[0], nS + 0.25, "the model-based answer", fontsize=8.6,
            color=INK)
    for b, m, e in zip(budgets, match, errs):
        ax.annotate(f"max |Q-Q*|\n{e:.2f}", xy=(b, m), xytext=(0, -34),
                    textcoords="offset points", fontsize=7.4, color=INK,
                    ha="center", alpha=0.85)
    ax.set_xlabel("transitions sampled (P and R never inspected)",
                  fontsize=9.5)
    ax.set_ylabel(f"states where the greedy action matches", fontsize=9.5)
    ax.set_title("closing on the model-based policy, without the model",
                 fontsize=10.5, color=INK, weight="bold")
    ax.set_ylim(min(lo) - 1.5, nS + 1.2)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{nS} states, {nA} actions, {len(list(seeds))} seeds. band is "
             "the spread; the values are still off when the policy lands.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "model_free.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote model_free.svg   match {[round(m,2) for m in match]}   "
          f"err {errs[0]:.2f} -> {errs[-1]:.2f}")


if __name__ == "__main__":
    contraction()
    model_free()
