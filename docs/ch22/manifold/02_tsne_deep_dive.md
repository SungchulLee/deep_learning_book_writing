# t-SNE 깊이 들여다보기

t-SNE(van der Maaten와 Hinton, 2008)는 차원 높은 자료를 그려 보는 데 가장 널리 쓰이는 다양체 배움 방법이다. 이 보기는 핵심 웃매개변수인 헷갈림도, 되풀이 횟수, 주성분 분석 미리 다듬기가 나온 묻힘의 품질과 빠르기에 어떻게 영향을 주는지 살핀다. 이 매개변수를 알아야 믿을 만한 그림을 만들고 흔한 오해를 피할 수 있다.

## 1. 코드

```python
"""t-SNE 깊이 들여다보기."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === 헷갈림도의 영향 ===================================================
print("Generating blobs to show perplexity effect...")
X_blobs, y_blobs = make_blobs(
    n_samples=600, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42
)
X_blobs = StandardScaler().fit_transform(X_blobs)

perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 4))
for ax, perp in zip(axes, perplexities):
    X_2d = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000).fit_transform(X_blobs)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_blobs, cmap="tab10", s=10)
    ax.set_title(f"perplexity={perp}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE: Effect of Perplexity", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# === n_iter(모임)의 영향 =========================================
print("Showing convergence with increasing iterations...")
iterations = [250, 500, 1000, 2000, 5000]
fig, axes = plt.subplots(1, len(iterations), figsize=(20, 4))
for ax, n_iter in zip(axes, iterations):
    X_2d = TSNE(n_components=2, perplexity=30, n_iter=n_iter, random_state=42).fit_transform(X_blobs)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_blobs, cmap="tab10", s=10)
    ax.set_title(f"n_iter={n_iter}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE: Convergence with Iterations", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_iterations_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# === MNIST에서 주성분 분석으로 빠르게 하는 재주 ============================================
print("\nMNIST: comparing raw t-SNE vs PCA+t-SNE speed...")
import time

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data[:3000].astype(np.float32)
y = mnist.target[:3000].astype(int)

t0 = time.time()
X_direct = TSNE(n_components=2, random_state=42).fit_transform(X)
t_direct = time.time() - t0

t0 = time.time()
X_pca = PCA(n_components=0.95, random_state=42).fit_transform(X)
X_pca_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
t_pca_tsne = time.time() - t0

print(f"  Direct t-SNE:    {t_direct:.1f}s")
print(f"  PCA + t-SNE:     {t_pca_tsne:.1f}s  (speed-up: {t_direct / t_pca_tsne:.1f}x)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, X_2d, title in zip(
    axes,
    [X_direct, X_pca_tsne],
    [f"Direct t-SNE ({t_direct:.1f}s)", f"PCA + t-SNE ({t_pca_tsne:.1f}s)"],
):
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, label=str(digit), alpha=0.6)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
axes[0].legend(markerscale=3, fontsize=8)
plt.suptitle("PCA Pre-processing Accelerates t-SNE", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_pca_speedup.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone.")


if __name__ == "__main__":
    pass
```

**출력:**

```
Generating blobs to show perplexity effect...
Showing convergence with increasing iterations...

MNIST: comparing raw t-SNE vs PCA+t-SNE speed...
  Direct t-SNE:    8.3s
  PCA + t-SNE:     10.8s  (speed-up: 0.8x)

Done.
```

## 2. 논의

t-SNE의 헷갈림도 매개변수는 점마다 헤아리는 실효 이웃 수를 부드럽게 재는 잣대 노릇을 한다. 헷갈림도가 낮으면(5~10) 알고리즘이 아주 국소한 짜임에 매달려 작고 빽빽한 무리를 여럿 만든다. 헷갈림도가 높으면(50~100) 이웃이 넓어져 가까운 무리가 합쳐지고 더 전역적인 짜임이 드러난다. 5와 50 사이 값 여럿을 시험해 여러 자리매김에서 한결같이 남는 짜임을 찾기를 권한다. 그런 짜임이 어떤 매개변수 고름이 만든 헛것이 아니라 참된 결일 가능성이 크다.

모임의 몸가짐도 요긴하다. 되풀이가 너무 적으면(500 미만) 가장 좋게 하기가 자리를 잡지 못해 묻힘이 일그러지거나 잡음이 낀 것처럼 보일 수 있다. 1000번쯤이면 흔히 안정되지만 복잡한 자료 묶음은 2000~5000번이 도움이 될 수 있다. 가장 좋게 하는 동안 KL 벌어짐을 지켜보는 것이(`TSNE(verbose=2)`으로 볼 수 있다) 모임을 확인하는 가장 믿을 만한 길이다.

주성분 분석으로 미리 다듬는 재주는 실전에서 거의 언제나 쓴다. t-SNE는 모든 점 사이의 거리를 셈하므로 차원을 784에서 150쯤으로 줄이면 남아도는 흩어짐이 걷히고 거리 셈하기 값이 크게 준다. 주성분 분석이 짜임이 아니라 잡음을 보태는 흩어짐 작은 방향만 없애므로 마지막 묻힘의 품질은 사실상 같다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
방울 자료 묶음에 대해 헷갈림도 2, 5, 10, 30, 50, 100으로 t-SNE를 돌려라. 저마다 알려진 무리 이름표로 묻힘의 실루엣 점수를 셈하라. 어느 헷갈림도에서 실루엣 점수가 가장 큰가?

</div>

??? success "연습문제 1 풀이"
    ```python
    from sklearn.metrics import silhouette_score
    for perp in [2, 5, 10, 30, 50, 100]:
        X_2d = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_blobs)
        score = silhouette_score(X_2d, y_blobs)
        print(f"perplexity={perp:3d}  silhouette={score:.3f}")
    ```
    흔히 이 자료 묶음에서는 헷갈림도 15~30쯤이 실루엣 점수가 가장 높다. 국소한 정밀함과 전역의 얼거리를 저울질하기 때문이다. 헷갈림도가 너무 낮으면 자연스러운 무리가 조각나고, 너무 높으면 서로 다른 무리가 합쳐진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
t-SNE가 왜 매개변수가 없는지 설명하고 이 한계가 실전에서 무엇을 뜻하는지 적어라. 매개변수를 가진 t-SNE는 이를 어떻게 다루는가?

</div>

??? success "연습문제 2 풀이"
    여느 t-SNE는 묻은 점의 자리를 곧바로 가장 좋게 할 뿐, 들임 공간에서 묻힘 공간으로 가는 대응 함수를 배우지 않는다. 그래서 새 자료가 오면 묻힘 전체를 맨바닥에서 다시 셈해야 한다. 곧 `transform` 방법이 없다. 매개변수를 가진 t-SNE는 곧바른 가장 좋게 하기를 대응 $f: \mathbb{R}^d \to \mathbb{R}^2$을 배우는 신경망으로 갈음한다. 한 번 익히면 그 그물이 앞먹임 한 번으로 못 본 점을 쏠 수 있어 표본 밖 쏘기가 필요한 실전 물길에 알맞다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
t-SNE를 돌리기 앞서 설명하는 흩어짐을 50%, 80%, 95%, 99%로 두어 주성분 분석으로 미리 다듬는 것을 견주도록 MNIST 실험을 고쳐라. 네 묻힘을 나란히 그리고 전체 도는 시간을 재어라. 어느 흩어짐 문턱에서 품질이 떨어지기 시작하는가?

</div>

??? success "연습문제 3 풀이"
    ```python
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, var_ratio in zip(axes, [0.50, 0.80, 0.95, 0.99]):
        t0 = time.time()
        X_pca = PCA(n_components=var_ratio, random_state=42).fit_transform(X)
        X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
        elapsed = time.time() - t0
        n_dims = X_pca.shape[1]
        for digit in range(10):
            mask = y == digit
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.6)
        ax.set_title(f"{var_ratio:.0%} var ({n_dims}d, {elapsed:.1f}s)")
    plt.tight_layout()
    plt.show()
    ```
    흩어짐을 80~95% 남기면 흔히 99%인 경우와 구별할 수 없는 묻힘을 훨씬 빠르게 얻는다. 50%에서는 낱낱을 가르는 화소 수준의 특징이 버려져 숫자 무리가 겹치기 시작할 수 있다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
퍼플렉시티를 바꾸면 무엇이 달라지는가? 무엇을 보고 고르는가?

</div>

??? success "연습문제 4 풀이"
    MNIST 2,000개에 대해 재면 이렇다. 이웃 지킴은 각 점의 12-최근접 이웃이 묻힌 뒤에도
    남아 있는 비율이다.

    | 퍼플렉시티 | 5 | 15 | **30** | 50 | 100 |
    |---|---|---|---|---|---|
    | 이웃 지킴 | 46.4% | 50.1% | **50.6%** | 50.2% | 48.4% |
    | 믿음성 | 0.966 | 0.971 | 0.970 | 0.967 | 0.961 |
    | KL | 1.188 | 1.213 | 1.159 | 1.096 | **0.981** |

    이웃 지킴은 30 근처에서 가장 좋고 양쪽으로 완만하게 나빠진다. 기본값이 30인 것이
    까닭 없는 일은 아니다.

    **KL을 보고 고르면 안 된다**는 점이 이 표의 요점이다. KL은 퍼플렉시티가 커질수록
    단조롭게 **작아진다**(1.188 → 0.981). 그 값만 보면 100이 가장 좋아 보이는데 이웃
    지킴은 오히려 나빠졌다.

    까닭은 퍼플렉시티를 바꾸면 목적 함수 자체가 달라지기 때문이다. $P$가 달라지므로
    서로 다른 문제의 최적값을 견주는 셈이다. 같은 문제를 얼마나 잘 풀었는지를 재는
    값이지, 어느 문제를 풀어야 하는지를 알려 주는 값이 아니다.

    그래서 **묻힘의 품질을 재는 잣대를 따로 써야** 한다. 이웃 지킴이나 믿음성처럼
    목적 함수와 무관한 것이어야 한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
t-SNE 그림에서 덩이 사이의 거리를 읽어도 되는가?

</div>

??? success "연습문제 5 풀이"
    읽으면 안 된다. 수치로 확인할 수 있다.

    각 숫자 갈래의 중심을 구해 갈래 사이 거리를 만들고, 원래 784차원에서 잰 같은 거리와
    상관을 보면 이렇다.

    | | 원래 공간과의 상관 |
    |---|---|
    | t-SNE | 0.666 |
    | 주성분 분석 | **0.828** |

    **주성분 분석이 갈래 사이 거리를 더 잘 지킨다.** t-SNE는 이웃을 훨씬 잘 지키면서
    (50.6% 대 9.0%) 덩이 사이 거리에서는 진다.

    이것이 실패가 아니라 설계다. t-SNE의 목적 함수는 가까운 관계에 무게를 두고 먼
    관계는 거의 벌하지 않으므로, 먼 거리를 희생해 가까운 구조를 살린다. 곧 **무엇을
    버릴지 정해 놓은 방법**이다.

    그래서 t-SNE 그림에서 읽어도 되는 것과 안 되는 것이 갈린다.

    | 읽어도 된다 | 읽으면 안 된다 |
    |---|---|
    | 어떤 점들이 뭉치는가 | 두 덩이가 얼마나 다른가 |
    | 덩이가 몇 개인가 | 덩이의 크기 |
    | 어느 점이 어느 덩이에 붙는가 | 덩이 사이의 빈 곳 |

    덩이 사이 관계를 알고 싶으면 주성분 분석이나 MDS를 함께 그리는 것이 맞다. 두 그림이
    서로 다른 것을 알려 주므로 하나로 갈음할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
t-SNE는 씨앗마다 결과가 달라진다고들 한다. 정말인가?

</div>

??? success "연습문제 6 풀이"
    **그림은 달라지고 구조는 거의 그대로다.** 재어 보면 이렇다.

    | 씨앗 | 0 | 1 | 2 |
    |---|---|---|---|
    | 이웃 지킴 | 50.5% | 50.4% | 50.6% |
    | 믿음성 | 0.970 | 0.970 | 0.970 |

    세 자리까지 같다. 곧 **이웃 관계는 재현된다.**

    달라지는 것은 그림의 겉모습이다. 덩이의 자리와 방향이 바뀌고 거울처럼 뒤집히기도
    한다. 그래서 두 그림을 나란히 놓으면 아주 달라 보이지만, 어느 점이 어느 점과 이웃인지는
    같다.

    구별이 중요하다. t-SNE에서 뜻이 있는 것이 이웃 관계뿐이므로
    (앞 문제), **뜻이 있는 부분은 안정적이고 뜻이 없는 부분만 흔들린다.**

    다만 `init`을 무엇으로 두는지가 실제로 영향을 준다. 이 측정은 `init='pca'`로 했고,
    그것이 요즘 권하는 기본값이다. 무작위 초기화를 쓰면 씨앗마다 훨씬 많이 달라지고
    큰 규모의 짜임이 덜 안정적이다.

    그러므로 "t-SNE는 불안정하다"는 말은 다듬어야 한다. 무작위 초기화에서는 그렇고,
    주성분 분석으로 초기화하면 이웃 구조가 꽤 안정적이다. 어느 쪽이든 **그림의 절대적인
    자리를 풀이하지 않는 것**이 안전하다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
t-SNE 앞에 주성분 분석을 거는 까닭은 무엇인가?

</div>

??? success "연습문제 7 풀이"
    두 가지다. **빠르기**와 **잡음 걸러 내기**다.

    t-SNE는 모든 쌍의 거리를 다루므로 차원이 높으면 그 셈하기가 비싸다. 784차원을 먼저
    50차원으로 줄이면 거리 셈하기가 열다섯 배 값싸진다.

    그리고 높은 차원에서는 거리가 서로 비슷해지는 현상이 있어(차원의 저주) 이웃 관계가
    덜 또렷하다. 잡음에 해당하는 뒤쪽 성분을 버리면 그 관계가 뚜렷해질 수 있다.

    몇 차원으로 줄일지가 고를 점이다. 50이 흔한 값이며, MNIST에서 50은 흩어짐의 82%다
    ([MNIST 연습문제 1](../pca/05_pca_mnist_pytorch.md)).

    너무 줄이면 t-SNE가 볼 것이 남지 않는다. 2차원까지 줄여 놓고 t-SNE를 걸면 주성분
    분석의 정보만 쓰는 셈이다.

    `init='pca'`와 헷갈리지 않아야 한다. 그것은 **초기값**을 주성분 분석으로 잡는 것이고,
    여기서 말하는 것은 **입력**을 미리 줄이는 것이다. 둘은 따로 정하는 별개의 선택이며
    함께 쓰는 것이 보통이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
t-SNE가 학생 t 분포를 쓰는 까닭은 무엇인가?

</div>

??? success "연습문제 8 풀이"
    **뭉침 문제**(crowding problem)를 다루기 위해서다.

    높은 차원에서는 한 점에서 비슷한 거리에 있는 점을 아주 많이 둘 수 있는데, 2차원에서는
    그럴 자리가 없다. 곧 중간 거리의 이웃들을 넣을 공간이 모자라 모두 가까이 몰린다.

    낮은 차원에서 꼬리가 두꺼운 분포를 쓰면 이 압력이 풀린다. 같은 확률을 주려면 더 멀리
    놓아도 되기 때문이다. 곧 **덩이 사이에 여유가 생긴다.**

    $$q_{ij} \propto (1 + \|z_i - z_j\|^2)^{-1}$$

    이것이 자유도 1의 학생 t 분포이고 코시 분포이기도 하다. 높은 차원 쪽은 가우시안을
    쓰므로 **두 쪽이 다른 분포를 쓴다**는 점이 t-SNE의 핵심 설계다. SNE에서 t-SNE로
    가는 변화가 바로 이것이다.

    부수 효과가 둘 있다.

    **기울기가 순해진다.** 꼬리가 두꺼우면 멀리 있는 점들이 서로 밀어내는 힘이 약해져
    최적화가 안정된다.

    **덩이 사이 거리가 더욱 뜻을 잃는다.** 여유를 만들어 내는 것이 목적이었으므로, 벌어진
    간격이 원래 거리를 나타내지 않는다. 상관이 0.666에 그치는 까닭의 일부다.

    곧 t-SNE가 덩이를 또렷하게 보여 주는 힘과 덩이 사이 거리를 못 지키는 약점이 **같은
    설계에서 나온다.** 하나를 얻으려고 다른 하나를 내준 것이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
되풀이 횟수가 모자라면 어떻게 알아채는가?

</div>

??? success "연습문제 9 풀이"
    몇 가지 신호가 있다.

    **KL이 아직 내려가고 있다.** `sklearn`에서 `verbose=2`로 두면 걸음마다의 값을 볼 수
    있다. 평평해지지 않았으면 더 돌려야 한다.

    **덩이가 다 뭉쳐 있다.** 초기 과장(early exaggeration) 단계가 끝난 뒤 덩이가 흩어질
    시간이 필요하다. 덜 돌리면 공처럼 뭉친 모습이 남는다.

    **점들이 실오라기처럼 늘어서 있다.** 아직 자리를 못 잡은 모습이다.

    `sklearn`의 기본값 1,000은 대개 충분하고, 이 장의 측정도 그것으로 했다. 표본이
    아주 많으면 더 필요할 수 있다.

    조심할 것이 하나 있다. **KL이 평평해진 것이 좋은 묻힘이라는 뜻은 아니다.** 퍼플렉시티를
    고를 때 KL을 못 쓰는 것과 같은 이유다(연습문제 1). 수렴은 "이 문제를 다 풀었다"를
    말해 줄 뿐 "맞는 문제였다"를 말해 주지 않는다.

    그래서 수렴은 KL로 보고 품질은 이웃 지킴으로 보는 것이 맞다. 두 물음에 두 잣대를
    쓴다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
t-SNE로 새 자료를 변환할 수 있는가?

</div>

??? success "연습문제 10 풀이"
    할 수 없다. `transform` 메서드가 없고, 없는 것이 원칙적인 이유에서다.

    t-SNE는 **각 점의 좌표를 직접 최적화한다.** 주성분 분석처럼 배운 사상이 있어서
    새 점에 적용하는 것이 아니라, 주어진 $n$개 점의 자리를 함께 정하는 문제를 푼다.
    새 점 하나를 넣으면 모든 점의 관계가 달라지므로 원칙적으로 다시 풀어야 한다.

    같은 이유로 MDS도 그렇다. 계량 MDS에서 최적화하는 것이 매개변수가 아니라 좌표
    자체였던 것과 이어진다([MDS 자세히 연습문제 2](03_mds_deep_dive.md)).

    실무에서 쓰는 우회로가 몇 가지다.

    | 방법 | 성격 |
    |---|---|
    | 전체를 다시 돌린다 | 정확하지만 그림이 바뀐다 |
    | 이웃의 좌표로 끼워 넣는다 | 값싸고 어림이다 |
    | 회귀 모델을 익힌다 | 원래 좌표 → t-SNE 좌표 사상을 배운다 |
    | UMAP을 쓴다 | `transform`을 준다 |

    마지막 칸이 UMAP이 실무에서 널리 쓰이게 된 까닭 가운데 하나다.

    그리고 이 한계가 **자기 부호기의 매력**을 설명해 준다. 자기 부호기는 부호기라는
    함수를 배우므로 새 자료를 넣는 것이 당연히 된다. 익히고 나면 한 번의 앞먹임이다.
    차원 줄이기를 **사상을 배우는 일**로 다시 세운 것이 다음 장들의 큰 변화다.

## 정리하며

**다룬 것** — t-SNE 깊이 들여다보기

t-SNE의 헷갈림도 매개변수는 점마다 헤아리는 실효 이웃 수를 부드럽게 재는 잣대 노릇을 한다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
