# 다양체 방법 견줌

다양체 배움 알고리즘은 차원 높은 자료 속에 숨은 낮은 차원 짜임을 드러낸다. 이 보기는 널리 쓰이는 네 방법, 곧 t-SNE, MDS, Isomap, LLE를 만든 자료(스위스 롤)와 실제 자료(MNIST) 모두에서 견준다. 어느 쪽이 어디에 센지 알면 그려 보기와 차원 줄이기에 알맞은 연장을 고를 수 있다.

## 1. 코드

```python
"""다양체 방법 견줌."""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# === 만든 자료: 스위스 롤 =============================================
X_swiss, t_swiss = make_swiss_roll(n_samples=1500, noise=0.2, random_state=42)

methods = {
    "MDS": MDS(n_components=2, random_state=42, normalized_stress="auto"),
    "Isomap": Isomap(n_components=2, n_neighbors=10),
    "LLE": LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42),
    "t-SNE": TSNE(n_components=2, random_state=42),
}

fig, axes = plt.subplots(1, len(methods), figsize=(16, 4))
for ax, (name, model) in zip(axes, methods.items()):
    t0 = time.time()
    X_2d = model.fit_transform(X_swiss)
    elapsed = time.time() - t0
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=t_swiss, cmap="hot", s=5)
    ax.set_title(f"{name} ({elapsed:.1f}s)")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.grid(True, alpha=0.3)
fig.suptitle("Manifold Learning on Swiss Roll", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("manifold_swiss_roll_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# === 실제 자료: MNIST =======================================================
print("\nLoading MNIST (subset of 2 000 samples for speed)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist, y_mnist = mnist.data[:2000].astype(np.float32), mnist.target[:2000].astype(int)
X_mnist = StandardScaler().fit_transform(X_mnist)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_mnist)
print(f"PCA reduced {X_mnist.shape[1]} -> {X_pca.shape[1]} dimensions (95% var).\n")


def plot_digits(X_2d, labels, title=""):
    """숫자 이름표로 색을 입힌 흩뿌림 그림."""
    plt.figure(figsize=(8, 6))
    for digit in range(10):
        mask = labels == digit
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, label=str(digit), alpha=0.6)
    plt.legend(markerscale=3, fontsize=8)
    plt.title(title, fontsize=13)
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.grid(True, alpha=0.3)


results = {}
for name, model in methods.items():
    pipe = Pipeline([("pca", PCA(n_components=0.95, random_state=42)), ("manifold", model)])
    t0 = time.time()
    X_2d = pipe.fit_transform(X_mnist)
    elapsed = time.time() - t0
    results[name] = (X_2d, elapsed)
    print(f"PCA + {name}: {elapsed:.1f}s")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (X_2d, elapsed)) in zip(axes, results.items()):
    for digit in range(10):
        mask = y_mnist == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, label=str(digit), alpha=0.6)
    ax.set_title(f"PCA + {name} ({elapsed:.1f}s)")
    ax.grid(True, alpha=0.3)
axes[0].legend(markerscale=3, fontsize=7)
plt.suptitle("Manifold Learning on MNIST (2 000 samples)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("manifold_mnist_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# === t-SNE 깊이 보기: 숫자 부분 배치 ==========================================
print("\nt-SNE close-up on digits {2, 3, 5}...")
mask_subset = np.isin(y_mnist, [2, 3, 5])
X_sub, y_sub = X_mnist[mask_subset], y_mnist[mask_subset]
X_sub_2d = TSNE(n_components=2, random_state=42).fit_transform(X_sub)
plot_digits(X_sub_2d, y_sub, title="t-SNE on digits {2, 3, 5}")
plt.savefig("tsne_digit_subset.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Three PNG files saved.")


if __name__ == "__main__":
    pass
```

**출력:**

```
Loading MNIST (subset of 2 000 samples for speed)...
PCA reduced 784 -> 233 dimensions (95% var).

PCA + MDS: 54.2s
PCA + Isomap: 3.2s
PCA + LLE: 1.1s
PCA + t-SNE: 4.6s

t-SNE close-up on digits {2, 3, 5}...

Done. Three PNG files saved.
```

## 2. 논의

다양체 배움 방법마다 다른 기하 직관을 담고 있다. MDS는 둘씩의 거리를 전역으로 지켜, 유클리드가 아닌 잣대로 주성분 분석을 자연스럽게 넓힌 것이 된다. Isomap은 유클리드 거리를 가장 가까운 이웃 그래프에서 셈한 측지 거리로 갈음해 이 생각을 넓히며, 그래서 스위스 롤 같은 굽은 다양체를 "펼" 수 있다. LLE는 국소 선형 이웃에 힘을 쏟아 점마다 이웃의 무게 붙은 아우름으로 다시 세운 뒤 그 무게를 지키는 낮은 차원 배치를 찾는다.

t-SNE는 둘씩의 거리를 조건부 확률로 바꾸고 높은 차원과 낮은 차원 확률 분포 사이 KL 벌어짐을 가장 작게 한다는 점에서 남다르다. 그래서 국소 무리 짜임을 지키는 데 뛰어나며, MNIST 그림에서 숫자 갈래가 그토록 또렷이 갈리는 까닭이다. 다만 t-SNE는 매개변수가 없어(다시 돌리지 않고는 새 점을 쏠 수 없다) 내놓은 그림에서 무리의 크기와 무리 사이 거리는 오해를 부를 수 있다.

이 견줌에서 얻은 실전 관찰 하나는 주성분 분석으로 미리 다듬으면 엄청나게 빨라진다는 것이다. 다양체 방법을 쓰기 앞서 MNIST를 784에서 150쯤 차원으로 줄이면(흩어짐의 95%를 남긴 채) 도는 시간이 한 자릿수만큼 줄고 묻힘 품질에는 거의 영향이 없다. 주성분 분석 뒤에 비선형 방법을 잇는 이 물길은 자료를 살피는 일에서 표준 흐름이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Isomap과 LLE에 대해 `n_samples=5000`, `n_neighbors=20`으로 스위스 롤 견줌을 돌려라. 방법마다 걸린 시간을 재고 표본 수가 늘 때 어느 알고리즘이 가장 무난하게 버티는지 알려라.

</div>

??? success "연습문제 1 풀이"
    표본 수와 이웃 수를 늘린 뒤 맞추기마다 시간을 재어라:
    ```python
    X_swiss, t_swiss = make_swiss_roll(n_samples=5000, noise=0.2, random_state=42)
    for name, model in methods.items():
        if hasattr(model, 'n_neighbors'):
            model.n_neighbors = 20
        t0 = time.time()
        model.fit_transform(X_swiss)
        print(f"{name}: {time.time() - t0:.1f}s")
    ```
    MDS와 t-SNE가 가장 비싸다. 둘씩의 거리 행렬 전체를 다루기 때문이다($O(n^2)$ 공간). Isomap과 LLE는 성긴 이웃 그래프 덕분에 규모를 더 잘 키우지만 Isomap은 최단 길 셈하기가 더 든다. 넷 가운데 표본이 많아지면 흔히 LLE가 가장 빠르다. 성긴 고윳값 문제를 풀기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
t-SNE의 무리 크기를 갈래마다의 참 흩어짐으로 읽으면 안 되는 까닭을 설명하라. t-SNE 값 함수의 어떤 성질이 이를 부르는가?

</div>

??? success "연습문제 2 풀이"
    t-SNE는 낮은 차원 공간에서는 스튜던트 t 분포를, 높은 차원 공간에서는 정규 분포를 쓴다. 스튜던트 t 분포의 두꺼운 꼬리 덕분에 높은 차원에서 중간이거나 큰 거리를 낮은 차원의 넓은 거리 범위로 나타낼 수 있다. 그래서 속 흩어짐이 큰 무리와 작은 무리가 묻힘에서 비슷한 크기로 보일 수 있다. KL 벌어짐 목표는 높은 차원에서 가까운 점을 멀리 놓는 것(붐빔)에는 벌을 주지만, 먼 점을 가까이 놓거나 들쭉날쭉한 거리에 놓는 것에는 세게 벌을 주지 않는다. 그러므로 t-SNE 무리가 차지한 넓이는 본디 공간에서 그 갈래가 얼마나 퍼져 있는지에 대해 알려 주는 바가 거의 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MNIST 물길을 t-SNE 대신 UMAP(`umap-learn`을 깔아라)을 쓰도록 고쳐라. 2차원 흩뿌림 그림을 눈으로 견주고 빠르기 차이를 재어라. 실전에서 UMAP이 t-SNE보다 나은 점은 무엇인가?

</div>

??? success "연습문제 3 풀이"
    ```python
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    pipe = Pipeline([("pca", PCA(n_components=0.95, random_state=42)),
                     ("umap", reducer)])
    t0 = time.time()
    X_umap = pipe.fit_transform(X_mnist)
    print(f"PCA + UMAP: {time.time() - t0:.1f}s")
    ```
    UMAP은 흔히 t-SNE보다 5~10배 빠르고 국소 짜임과 전역 짜임을 모두 더 잘 지키는 묻힘을 낸다. t-SNE와 달리 UMAP은 (`umap.parametric_umap`으로) 매개변수를 가진 바꾸개로 쓸 수 있어 알고리즘 전체를 다시 돌리지 않고도 새 점을 쏠 수 있다. 그래서 표본 밖 쏘기가 필요한 실전 물길에 훨씬 알맞다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
네 방법을 같은 자료에 걸어 이웃 지킴과 걸린 시간을 견주어라.

</div>

??? success "연습문제 4 풀이"
    MNIST 2,000개를 2차원으로 줄여 재면 이렇다. 이웃 지킴은 12-최근접 이웃이 남아
    있는 비율이고, 믿음성은 순위 왜곡까지 보는 잣대다.

    | 방법 | 걸린 시간 | 이웃 지킴 | 믿음성 |
    |---|---|---|---|
    | 주성분 분석 | **2.8초** | 9.0% | 0.741 |
    | MDS (계량, SMACOF) | 20.7초 | 12.9% | 0.792 |
    | Isomap | 7.2초 | 12.5% | 0.768 |
    | t-SNE | 7.3초 | **50.6%** | **0.970** |

    t-SNE가 이웃 지킴에서 압도적이다. 주성분 분석의 다섯 배가 넘는다.

    당연한 결과이기도 하다. t-SNE의 목적 함수가 바로 이웃 관계를 지키는 것이므로,
    그것을 재는 잣대에서 이기는 것이 마땅하다. **잣대가 방법의 목표와 같으면 그 방법이
    이긴다.**

    그래서 이 표만 보고 "t-SNE가 가장 좋다"고 결론 내리면 안 된다. 다른 잣대로 재면
    순위가 바뀐다. 갈래 사이 거리를 지키는 것으로 재면 주성분 분석이 0.828 대 0.666으로
    이긴다([t-SNE 연습문제 2](02_tsne_deep_dive.md)).

    MDS가 주성분 분석보다 이웃 지킴이 나은 것이 눈에 걸릴 수 있다. 유클리드 거리에서
    고전 MDS와 주성분 분석은 같다고 했는데([MDS 연습문제 1](mds.md)) 값이 다르다.

    표의 MDS가 **고전 MDS가 아니기** 때문이다. `sklearn.manifold.MDS`는 고전 MDS를
    짜 두지 않았고 SMACOF로 스트레스를 직접 줄이는 계량 MDS를 쓴다. 다른 목적 함수를
    다른 방법으로 푸니 다른 답이 나오는 것이 맞다.

    고전 MDS를 견주고 싶으면 주성분 분석 줄을 보면 된다. 그 둘이 같다는 것을 이미
    소수점 열넷째 자리까지 확인했다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
어느 방법을 고를지 어떻게 정하는가?

</div>

??? success "연습문제 5 풀이"
    목적이 정한다. 잣대가 하나로 정해지지 않기 때문이다.

    | 목적 | 고를 것 |
    |---|---|
    | 그려 보기, 덩이 찾기 | t-SNE 또는 UMAP |
    | 특징으로 써서 다음 모델에 넣기 | 주성분 분석 |
    | 새 자료도 변환해야 한다 | 주성분 분석, UMAP, 자기 부호기 |
    | 거리만 있고 좌표가 없다 | MDS |
    | 전체 짜임과 거리 관계 | 주성분 분석 또는 MDS |
    | 되돌리기, 표본 만들기 | 자기 부호기 계열 |

    두 번째 줄이 흔히 잘못되는 자리다. t-SNE 좌표를 분류기의 입력으로 쓰는 것은 대개
    나쁜 생각이다. 2차원뿐이고, 새 자료에 적용할 수 없으며
    ([t-SNE 연습문제 7](02_tsne_deep_dive.md)), 거리가 왜곡되어 있다.

    **그려 보기 위한 방법과 특징을 만들기 위한 방법이 다르다**는 것이 이 장의 큰 교훈
    가운데 하나다. 한 그림에서 둘을 다 얻으려 하면 양쪽에서 나쁜 것을 얻는다.

    자주 쓰는 조합이 있다. 주성분 분석으로 50차원쯤 줄여 특징으로 쓰고, 그 위에 t-SNE를
    걸어 그림을 따로 만든다. 각자 잘하는 일을 시키는 것이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
이웃 지킴과 믿음성은 무엇이 다른가?

</div>

??? success "연습문제 6 풀이"
    둘 다 이웃 관계를 재지만 보는 각도가 다르다.

    **이웃 지킴**은 집합으로 본다. 원래 $k$-최근접 이웃과 묻힌 뒤의 $k$-최근접 이웃이
    얼마나 겹치는지 센다. 순서는 안 본다.

    **믿음성**(trustworthiness)은 순위를 본다. 묻힌 뒤에 이웃이 되었지만 원래는 멀었던
    점을, 원래 얼마나 멀었는지에 비례해 벌한다. 곧 **가짜 이웃**을 잡아낸다.

    쌍이 되는 잣대가 하나 더 있다. **이어짐**(continuity)은 원래 이웃이었는데 묻힌 뒤
    멀어진 것을 벌한다. 곧 잃어버린 이웃을 잡는다.

    | 잣대 | 무엇을 벌하는가 |
    |---|---|
    | 믿음성 | 없던 이웃을 만든 것 |
    | 이어짐 | 있던 이웃을 잃은 것 |

    둘을 함께 보아야 온전하다. 한쪽만 보면 속을 수 있다. 모든 점을 한군데 뭉쳐 놓으면
    이어짐이 완벽해지고, 모두 멀리 흩어 놓으면 믿음성이 높아진다.

    그림을 읽는 쪽에서는 믿음성이 더 중요하다. **그림에서 가까이 보이는 것이 정말
    가까운가**를 재기 때문이다. 없는 덩이를 보게 되는 일을 막아 준다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
차원 줄이기 그림에서 보이는 덩이가 진짜인지 어떻게 확인하는가?

</div>

??? success "연습문제 7 풀이"
    한 그림만으로는 확인할 수 없다. 여러 방향에서 따져야 한다.

    **다른 방법으로도 나오는가.** t-SNE에서 보인 덩이가 주성분 분석이나 UMAP에서도
    보이면 믿을 만하다. 한 방법에서만 보이면 그 방법의 산물일 수 있다.

    **웃매개변수를 바꾸어도 남는가.** 퍼플렉시티를 5에서 100까지 바꾸며 보라. 덩이가
    생기고 사라지면 진짜가 아니다. t-SNE는 퍼플렉시티가 작으면 **없는 덩이를 만들어 내는**
    경향이 알려져 있다.

    **씨앗을 바꾸어도 남는가.** 이웃 구조 자체는 꽤 안정적이므로
    ([t-SNE 연습문제 3](02_tsne_deep_dive.md)) 덩이가 씨앗마다 달라지면 의심해야 한다.

    **원래 공간에서 확인되는가.** 이것이 가장 결정적이다. 그림에서 나뉜 두 무리를 표지로
    받아 **원래 784차원에서** 갈라지는지 보라. 군집 타당도 잣대를 쓰거나, 두 무리를
    가르는 분류기를 익혀 보면 된다.

    마지막 것을 빠뜨리면 안 된다. 차원 줄이기 그림은 **가설을 만드는 도구**이고 확인은
    원래 공간에서 해야 한다. 그림에서 본 것을 결론으로 적는 것이 이 분야에서 가장 흔한
    잘못이다.

    무작위 자료로 검사해 보는 것도 배울 만하다. 아무 짜임 없는 가우시안 잡음에 t-SNE를
    걸어도 덩이처럼 보이는 것이 나온다. 그것을 한 번 보아 두면 그림을 훨씬 조심해서
    읽게 된다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
Isomap은 무엇이 다른가? 왜 MDS보다 나을 수 있는가?

</div>

??? success "연습문제 8 풀이"
    거리를 재는 방식이 다르다. 직선 거리 대신 **다양체를 따라간 거리**를 쓴다.

    이웃 그래프를 만들고 그 위의 최단 경로로 거리를 정한 뒤, 그 거리 행렬에 고전 MDS를
    적용한다. 곧 Isomap = 측지 거리 + MDS다.

    이것이 이로운 경우가 또렷하다. 두루마리처럼 말린 자료에서 직선 거리는 말린 층을
    건너뛰므로 가깝다고 판단하는데, 실제로는 표면을 따라 한참 가야 한다. 측지 거리가
    그것을 바로잡는다.

    MNIST에서는 이득이 작게 나왔다(12.5% 대 계량 MDS의 12.9%). 손글씨 자료가 깔끔한
    두루마리 같은 모양이 아니고, 다양체가 여러 조각으로 갈라져 있기 때문이다.

    Isomap의 약점이 거기 있다. **이웃 그래프에 크게 딸린다.**

    - 이웃 수가 너무 크면 층을 건너뛰는 **짧은 지름길**이 생겨 측지 거리가 망가진다
    - 너무 작으면 그래프가 여러 조각으로 끊어져 거리가 무한이 된다

    자료가 정말 하나의 매끄러운 다양체 위에 촘촘히 놓여 있을 때 잘 듣는 방법이며,
    그 조건이 실제 자료에서 자주 어긋난다. 그래서 오늘날 그려 보기에는 t-SNE와 UMAP이
    더 널리 쓰인다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
표본이 60,000개로 늘면 이 견줌이 어떻게 달라지는가?

</div>

??? success "연습문제 9 풀이"
    **쓸 수 있는 방법이 걸러진다.** 2,000개에서는 넷 다 몇 초에서 몇십 초였지만,
    커지면 사정이 다르다.

    | 방법 | 표본 수에 따른 비용 | 60,000개에서 |
    |---|---|---|
    | 주성분 분석 | 선형 | 100초. 된다 |
    | MDS | $O(n^2)$ 기억, $O(n^3)$ 셈 | 거리 행렬이 29 GB. 안 된다 |
    | Isomap | 최단 경로까지 더 비싸다 | 안 된다 |
    | t-SNE | 나무 어림으로 $O(n \log n)$ | 오래 걸리지만 된다 |

    t-SNE가 큰 자료에서도 쓰이는 것은 Barnes–Hut 어림 덕이다. 멀리 있는 점들을 뭉쳐서
    다루므로 모든 쌍을 셈하지 않는다. `sklearn`의 기본 `method='barnes_hut'`이 그것이다.

    실무의 순서가 이렇게 정해진다. 주성분 분석으로 먼저 50차원쯤 줄이고, 그 위에 t-SNE나
    UMAP을 건다. 주성분 분석은 값싸고 t-SNE의 입력을 줄여 주므로 둘이 잘 맞는다
    ([t-SNE 연습문제 4](02_tsne_deep_dive.md)).

    표본을 덜어 쓰는 것도 정당한 선택이다. 그려 보기가 목적이면 60,000개를 다 찍어도
    점이 겹쳐 보이지 않으므로, 5,000개만 골라도 그림의 뜻은 같다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
이 장의 방법들과 자기 부호기는 무엇이 근본적으로 다른가?

</div>

??? success "연습문제 10 풀이"
    **사상을 배우는가**가 갈림길이다.

    | | 사상을 배우는가 | 새 자료 | 되돌리기 |
    |---|---|---|---|
    | 주성분 분석 | 배운다 (선형) | 된다 | 된다 |
    | 알맹이 주성분 분석 | 배운다 (암묵적) | 된다 | 어림만 |
    | MDS, t-SNE | **안 배운다** | 안 된다 | 안 된다 |
    | 자기 부호기 | 배운다 (비선형) | 된다 | 된다 |

    셋째 줄이 이 장의 두 방법이고, 그들은 주어진 점들의 **좌표를 직접 최적화**한다. 그래서
    새 점을 넣을 수 없고 되돌릴 수도 없다.

    표의 첫 줄과 넷째 줄을 견주면 다음 장이 왜 필요한지 보인다.

    - 주성분 분석: 사상을 배우지만 **선형**이라 굽음을 못 따라간다
    - 자기 부호기: 사상을 배우면서 **비선형**이다

    곧 자기 부호기는 주성분 분석의 좋은 성질(사상을 배운다, 되돌린다, 커진다)을 지키면서
    선형이라는 제약만 푼 것이다. 그리고 실제로 그 제약을 풀어 얻는 것이 크다. 같은
    병목 16에서 어긋남이 0.02686에서 0.00926으로 준다
    ([25장](../../ch25/architecture/01_ae_fully_connected.md)).

    알맹이 주성분 분석도 굽음을 다루지만 앞그림 문제가 있고 표본 수에 $O(n^2)$이다
    ([알맹이 주성분 분석 연습문제 6](../pca/kernel_pca.md)). 자기 부호기는 둘 다 낫다.

    이것이 **주성분 분석 → 자기 부호기 → 변분 자기 부호기 → 적대적 생성망** 사다리의
    첫 칸에서 두 번째 칸으로 가는 걸음이다. 그리고 그 걸음마다 무엇을 얻고 무엇을 잃는지
    수로 재어 두는 것이 이 책의 방식이다.

## 정리하며

**다룬 것** — 다양체 방법 견줌

다양체 배움 방법마다 다른 기하 직관을 담고 있다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
