# t-SNE 자세히 살펴보기

t-SNE는 고차원 데이터의 이웃 관계를 2·3차원에서 탐색하는 시각화 방법이다. 이 예제는 퍼플렉서티(perplexity), 반복 횟수와 PCA 전처리의 영향을 비교한다. 그림의 군집 면적·개수·군집 간 거리는 원래 데이터의 사실로 바로 해석하지 않는다.

## 1. 예제 코드와 출력

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

```
Generating blobs to show perplexity effect...
Showing convergence with increasing iterations...

MNIST: comparing raw t-SNE vs PCA+t-SNE speed...
  Direct t-SNE:    8.3s
  PCA + t-SNE:     10.8s  (speed-up: 0.8x)

Done.
```

## 2. 이웃 확률과 퍼플렉서티

원래 공간에서 각 점의 이웃 확률 p_{j|i}는 거리의 가우시안 함수로 계산하고, 점별 폭을 정해 목표 퍼플렉서티에 맞춘다.

$$\operatorname{Perp}(P_i)=2^{-\sum_jp_{j|i}\log_2p_{j|i}}.$$

이는 분포의 유효 이웃 수이지 정확히 그 개수만 이웃으로 사용한다는 뜻은 아니다. 퍼플렉서티는 양수이고 표본 수보다 작아야 하며, 달성 가능한 범위에는 중복점과 거리 분포도 영향을 준다. 작은 값은 좁은 이웃, 큰 값은 넓은 이웃에 더 무게를 두지만 군집 개수나 분리 정도가 단조롭게 바뀌는 법칙은 아니다.

고차원 조건부 확률을 대칭 결합해 P를 만들고 저차원에서는 다음 친화도를 정규화한다.

$$q_{ij}=\frac{(1+\|z_i-z_j\|^2)^{-1}}{\sum_{a\ne b}(1+\|z_a-z_b\|^2)^{-1}},\qquad
L=D_{KL}(P\|Q).$$

이 자유도 1의 Student t 커널은 저차원의 crowding 문제를 완화한다. 두꺼운 꼬리와 정규화가 인력·척력의 균형에 영향을 주며, 모든 군집을 정확히 분리하거나 거리 전체를 보존하는 보장은 아니다. [원 논문](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)을 참고한다.

## 3. 구현의 범위와 버전 조건

합성 데이터는 표본 600개, 중심 4개와 서로 다른 군집 표준편차를 사용하고 피처별로 표준화한다. 퍼플렉서티 5·15·30·50·100과 여러 반복 수를 각각 다시 적합한다. 같은 초기 시드라도 목적이 바뀌면 다른 배치를 얻을 수 있다.

코드의 `n_iter`는 이전 scikit-learn 인수다. 공식 인터페이스에서는 1.5에 `max_iter`로 이름이 바뀌었으며, 보존된 코드를 읽을 때 설치 버전과 호환성을 확인해야 한다. 인수와 초기 과장·중지 조건은 [TSNE 문서](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)에 설명되어 있다. 현재 API에 그대로 통하는 코드라고 가정하지 않는다.

## 4. 최적화 수렴과 그림 품질

초기 과장(early exaggeration) 단계는 보통의 최종 KL 최적화 단계와 다르다. 전체 예산이 그 단계 정도에 그치면 정상 단계가 충분하지 않을 수 있다. 초기 과장 계수·러닝레이트·반복 수와 종료 조건을 함께 확인한다.

같은 P에서 KL 추이와 최종값은 최적화 진행을 살피는 자료다. 그러나 KL이 평평해져도 좋은 국소해나 의미 있는 그림이라는 증거는 아니다. 구름·공·실 모양 하나만으로 반복 부족을 진단할 수도 없다. 퍼플렉서티나 전처리를 바꾸면 P 자체가 달라져 KL 값끼리 단순한 품질 순위를 매길 수 없다. KL이 퍼플렉서티에 따라 항상 감소하는 것도 아니다.

## 5. PCA 전처리와 초기화의 차이

PCA 전처리는 t-SNE가 받는 **입력 피처**을 줄이고, `init='pca'`는 저차원 좌표의 **초기값**을 정한다. 둘은 별도의 선택이다. PCA로 작은 분산 방향을 버리면 거리 계산과 이웃 탐색 비용을 줄일 수 있으나 버린 방향이 잡음뿐이라고 보장할 수 없다.

이 예제의 MNIST 비교는 앞 3,000개 픽셀 데이터를 사용하며 피처별 표준화를 하지 않는다. 직접 t-SNE와 95% PCA 후 t-SNE의 경과 시간을 한 번씩 재고, 후자의 시간에는 PCA 적합도 포함한다. 서로 다른 설정의 품질 지표나 여러 시드의 안정성을 계산하지는 않는다. 고정된 속도 배율과 품질 동등성을 이 출력에서 일반화하지 않는다.

## 6. 군집과 새 데이터 해석

점별 폭의 적응과 이웃 중심 목적 때문에 군집 면적·간격이 원래 분산·거리와 달라진다. 보이는 군집 수나 특정 점의 소속도 설정·초기값에 민감할 수 있다. 여러 시드와 전처리에서 이웃 구조를 비교하고 원래 공간과 보류 표본에서 확인한다. PCA 초기화도 안정성을 보장하는 정리는 아니다.

scikit-learn의 일반 `TSNE`는 새 표본의 `transform`을 제공하지 않는다. 기존 좌표를 고정하고 새 점을 최적화하는 확장, 좌표 회귀나 Parametric t-SNE 같은 대안은 가능하다. 전체 재적합도 비볼록 문제의 전역 해를 보장하지 않으므로 단순히 “정확한” 대안이라고 부르지 않는다. 각 확장은 기존 t-SNE와 다른 가정·평가가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
퍼플렉서티별 실루엣 점수의 의미와 아래 코드가 증명하지 못하는 것을 설명하라.

</div>

??? success "연습문제 1 풀이"

    주어진 합성 레이블과 임베딩 좌표의 응집·분리 정도를 측정한다. 최고 값은 실제 결과를 확인해야 하며 15~30이라고 미리 정할 수 없다. 높은 투영 실루엣은 원래 공간의 거리 보존이나 새 표본의 군집을 증명하지 않는다. 같은 표본에서 시각화와 설정 선택을 한 효과도 고려한다.

    ```python
    from sklearn.metrics import silhouette_score
    for perp in [2, 5, 10, 30, 50, 100]:
        X_2d = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_blobs)
        score = silhouette_score(X_2d, y_blobs)
        print(f"perplexity={perp:3d}  silhouette={score:.3f}")
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
t-SNE를 비파라메터 방법이라고 할 때 파라메터가 전혀 없다는 뜻인가?

</div>

??? success "연습문제 2 풀이"

    아니다. 퍼플렉서티·러닝레이트 등 하이퍼파라미터와 표본별 좌표가 있다. 여기서 비파라메터라는 말은 입력에서 좌표로 가는 고정 크기 함수 모델을 일반적으로 적합하지 않는다는 의미다. Parametric t-SNE는 뉴럴 넷트웍 사상을 학습해 새 입력을 순전파할 수 있지만 그 일반화와 근사 품질은 별도로 검증한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
PCA 분산 임계값 50·80·95·99% 비교에서 시간과 품질을 어떻게 판단하는가?

</div>

??? success "연습문제 3 풀이"

    아래 코드는 PCA와 t-SNE를 합친 경과 시간과 그림을 제공한다. 어떤 임계값에서 품질이 떨어지는지는 이웃·신뢰도 등 목적 지표와 여러 시드로 확인해야 한다. 작은 분산의 유용한 구조를 버릴 수 있어 80~95%면 항상 같다고 할 수 없다. 실제 선택 차원도 함께 기록한다.

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
퍼플렉서티를 고를 때 서로 다른 실행의 최종 KL만 비교하면 안 되는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    퍼플렉서티가 바뀌면 목표 고차원 분포 P가 바뀐다. 따라서 KL은 서로 다른 목적 문제의 값이다. 같은 P 안에서 최적화 진행을 평가하는 용도와 설정 간 품질 선택은 다르다. 이웃 지표도 이웃 크기·입력 척도를 정하고 한 종류만으로 최선이라 판단하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
t-SNE 군집 사이 거리와 개수를 어떻게 해석해야 하는가?

</div>

??? success "연습문제 5 풀이"

    군집 간 간격을 원래 공간 거리나 집단 차이의 비율로 읽을 수 없다. 보이는 군집 개수도 퍼플렉서티·시드·데이터 밀도에 영향을 받는다. 근처 관계를 탐색하는 가설로 보고 원래 공간과 외부 근거로 확인한다. 특정 거리 상관 수치를 코드가 측정했다고 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
시드를 바꾸어 비슷한 품질 점수를 얻으면 점별 이웃도 같다는 뜻인가?

</div>

??? success "연습문제 6 풀이"

    같은 평균 지표는 서로 다른 점의 오차를 상쇄할 수 있어 이웃 집합의 일치를 뜻하지 않는다. 표본 대응을 유지해 이웃 집합과 부분집단별 변화를 비교한다. 회전·반사·이동처럼 관계를 보존하는 차이와 이웃 자체의 변화를 구분한다. PCA 초기화도 완전한 재현성을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
PCA 입력 전처리와 PCA 초기화를 구분하라.

</div>

??? success "연습문제 7 풀이"

    전처리는 원래 피처를 줄여 t-SNE의 이웃 확률을 계산할 공간을 바꾼다. PCA 초기화는 주어진 확률 문제의 저차원 좌표 시작점을 정한다. 전처리에는 계산 절감과 정보 로스가 함께 있고 초기화는 최적화 경로에 영향을 준다. 둘을 함께 써도 원래 데이터 구조 보존을 별도로 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
저차원 Student t 커널의 두꺼운 꼬리가 주는 효과를 설명하라.

</div>

??? success "연습문제 8 풀이"

    중간·먼 관계를 저차원에서 더 넓게 배치할 여지를 주어 crowding을 완화한다. q_ij는 전체 쌍으로 정규화되므로 각 쌍의 영향이 독립적이지 않다. 꼬리가 두꺼워 모든 그래디언트가 순해지거나 최적화가 항상 안정된다고 일반화하지 않는다. 군집 간 간격의 원래 단위 해석도 유지되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
반복 수가 충분한지와 시각화 품질이 적절한지를 각각 어떻게 확인하는가?

</div>

??? success "연습문제 9 풀이"

    같은 P와 최적화 단계에서 KL·그래디언트와 중지 조건을 살펴 진행을 확인한다. 초기 과장 종료 뒤의 시간을 충분히 둔다. 별도로 이웃 지표, 설정·시드 안정성과 원래 공간 관계를 평가한다. 로스가 평평하다는 사실은 전역 최적해나 의미 있는 구조를 증명하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
일반 scikit-learn TSNE로 새 데이터를 transform할 수 있는지와 대안을 설명하라.

</div>

??? success "연습문제 10 풀이"

    일반 인터페이스에는 transform이 없다. 전체 재적합, 기존 좌표 고정의 표본 외 최적화, 좌표 회귀와 파라메터 사상 학습 등이 대안이다. 각각 기존 배치 변화와 새로운 분포에 대한 오차를 검증해야 한다. 새 점 변환이 원리적으로 불가능한 것은 아니며 재적합도 유일·정확한 해를 보장하지 않는다.



## 정리하며

t-SNE는 퍼플렉서티로 정한 확률적 이웃 관계를 저차원에 배치한다. 설정별 KL은 같은 문제의 품질 지표가 아니며, PCA 전처리와 초기화도 구분해야 한다. 군집 면적·개수·간격을 사실로 단정하지 말고 입력 공간의 관계와 여러 설정에서 검증한다.
