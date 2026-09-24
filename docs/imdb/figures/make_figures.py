"""
================================================================================
make_figures.py - 6장의 그림을 만든다
================================================================================

만드는 그림:
    lstm_curves.svg     4걸음 LSTM을 30 에포크까지 늘려 씨 다섯 벌의 곡선을
                        따로 그린다. 3걸음과 5걸음의 값을 가로선으로 둔다

왜 띠가 아니라 낱낱의 곡선인가:
    [4.2절](../../ch04/02_depth.md)의 깊이 곡선은 씨 다섯의 최소~최대를 띠로
    둘렀다. 여기서는 그렇게 하면 안 된다. 이 그림의 핵심이 **한 씨가 갑자기
    무너지는 사건**인데, 띠로 그리면 그것이 그저 넓은 띠로 뭉개져 보이고
    무너짐과 "씨마다 높이가 다름"을 구별할 수 없다.

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
    LOG=/path/to/imdb_step4_e30.log python make_figures.py
================================================================================
"""

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "path"

LOG = os.environ.get("LOG", "imdb_step4_e30.log")

# 같은 규약으로 잰 다른 걸음들 (본문 표와 같은 값)
RUNG3 = 87.03          # 박아 넣기 + 평균
RUNG5 = 86.11          # 셀프 어텐션
RUNG2 = 78.80          # 선형 학습
PROTOCOL_EPOCHS = 5    # 이 장이 표준으로 삼은 예산


def load_curves(path):
    """로그에서 씨앗별 {에포크: 정확도}를 읽는다."""
    curves = defaultdict(dict)
    pat = re.compile(r"씨앗 (\d+)\s+에포크\s+(\d+)/\d+\s+([\d.]+)%")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                curves[int(m.group(1))][int(m.group(2))] = float(m.group(3))
    return curves


# === 그림: 씨 다섯 벌의 학습 곡선 ===========================================
def fig_lstm_curves():
    curves = load_curves(LOG)
    if not curves:
        raise SystemExit(f"no curve data in {LOG}")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(9, 4.4))

    # 견줄 값을 먼저 깔아 둔다
    ax.axhline(RUNG3, color="#000000", lw=1.2, ls="-", zorder=1)
    ax.axhline(RUNG5, color="#666666", lw=1.0, ls="--", zorder=1)
    ax.axhline(RUNG2, color="#999999", lw=0.9, ls=":", zorder=1)
    ax.axvline(PROTOCOL_EPOCHS, color="#bbbbbb", lw=0.9, ls="-", zorder=1)

    for i, seed in enumerate(sorted(curves)):
        eps = sorted(curves[seed])
        ax.plot(eps, [curves[seed][e] for e in eps],
                color=colors[i % len(colors)], lw=1.4, marker="o", ms=2.5,
                label=f"seed {seed}", zorder=3)

    xmax = max(e for s in curves.values() for e in s)
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_ylim(60, 89)
    ax.set_xticks(range(5, xmax + 1, 5))          # 정수 에포크만
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")

    # 가로선 이름표. 3걸음과 5걸음은 0.9밖에 안 떨어져 있어 서로 겹치므로,
    # 하나는 선 위, 하나는 선 아래에 두고 좌우로도 갈라 놓는다
    ax.text(0.8, RUNG3 + 0.3, "rung 3: embedding + mean  87.03",
            ha="left", va="bottom", fontsize=8, color="#000000")
    ax.text(xmax + 0.3, RUNG5 - 0.3, "rung 5: self-attention  86.11",
            ha="right", va="top", fontsize=8, color="#666666")
    # 이 선 언저리는 곡선이 무너지며 오르내리는 자리다. 뒤쪽 아래가 비어 있다
    ax.text(xmax * 0.78, RUNG2 - 0.5, "rung 2: linear  78.80",
            ha="center", va="top", fontsize=8, color="#999999")
    ax.text(PROTOCOL_EPOCHS + 0.3, 61.0, "chapter budget: 5 epochs",
            ha="left", va="bottom", fontsize=8, color="#888888")

    ax.legend(loc="lower right", fontsize=8, framealpha=0.0, ncol=5,
              bbox_to_anchor=(1.0, -0.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)

    fig.tight_layout()
    fig.savefig("lstm_curves.svg", transparent=True)
    plt.close(fig)
    print("  lstm_curves.svg")

    # 그림과 본문의 수가 어긋나지 않도록, 그림을 만든 값을 함께 찍는다
    print("\n  seed   peak (epoch)   last (epoch)   worst-after-peak")
    for seed in sorted(curves):
        eps = sorted(curves[seed])
        vals = [curves[seed][e] for e in eps]
        pk = max(vals); pe = eps[vals.index(pk)]
        after = vals[vals.index(pk):]
        print(f"   {seed}     {pk:6.2f} ({pe:2d})   {vals[-1]:6.2f} ({eps[-1]:2d})   {min(after):6.2f}")


if __name__ == "__main__":
    fig_lstm_curves()
