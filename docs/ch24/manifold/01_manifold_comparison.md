# 매니폴드 학습 방법 비교

이 예제는 스위스 롤과 MNIST 일부 표본에 MDS, Isomap, LLE와 t-SNE를 적용해 시각화한다. 네 방법은 보존하려는 관계와 최적화 방식이 다르므로 한 그림이나 실행 시간만으로 전체 순위를 정할 수 없다. 전처리와 입력 거리, 이웃 설정을 함께 읽는 것이 목적이다.

## 1. 예제 코드와 출력

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

## 2. 네 방법이 보존하는 관계

| 방법 | 주요 목표와 조건 |
|---|---|
| 계량 MDS | 원래 비유사도와 임베딩 거리의 stress를 줄임 |
| Isomap | 이웃 그래프 최단 경로로 측지 거리를 근사한 뒤 고전 MDS 적용 |
| LLE | 원래 공간의 국소 선형 재구성 가중치를 저차원 좌표에서도 유지 |
| t-SNE | 고차원과 저차원의 확률적 이웃 관계 사이 KL 발산을 줄임 |

여기 `MDS`는 고전 MDS의 그람 행렬 분해와 다르다. 유클리드 거리의 **고전** MDS와 PCA 점수가 같다는 결과를 이 계량 MDS에 그대로 적용하지 않는다. Isomap은 연결된 이웃 그래프가 실제 매니폴드 이웃을 잘 반영해야 하며, LLE도 이웃의 국소 기하와 수치 안정성에 영향을 받는다.

## 3. 비교 데이터와 전처리

스위스 롤은 표본 1,500개와 잡음 0.2를 사용하고 색은 롤을 따라가는 위치다. 색이 연속적으로 이어지는지와 근처 점이 잘 보존되는지를 함께 살핀다. Isomap과 LLE의 기본 이웃 수는 10이다.

MNIST는 불러온 데이터의 앞 2,000개를 사용하며 무작위 대표 표본이라고 보장하지 않는다. 각 픽셀을 `StandardScaler`로 표준화한 뒤 95% 분산 PCA를 적용한다. 앞 PCA 예제의 비표준화 픽셀과는 다른 거리·분산 기준이므로 “95%면 약 150차원” 같은 수를 그대로 옮길 수 없다.

코드는 PCA를 한 번 계산해 차원을 출력한 뒤, 방법별 Pipeline에서 다시 PCA를 적합한다. 시간은 각 Pipeline의 PCA와 매니폴드 변환을 함께 포함한다. 마지막 숫자 2·3·5 부분집합의 t-SNE는 표준화 데이터에 직접 적용하므로 앞의 PCA 결합 결과와 입력도 다르다.

## 4. 시각화가 보여 주는 것

t-SNE의 국소 폭은 점별로 적응하며 KL 목적은 이웃 관계를 강조한다. 이 과정에서 원래 밀도가 다른 집단의 면적이 비슷해질 수 있고 군집 간 거리가 원래 거리와 다를 수 있다. 가까운 모든 쌍에 벌점이 없는 것도 아니며 정규화된 저차원 분포를 통해 관계가 연결된다. [t-SNE 원 논문](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)의 시각화 목적과 전체 거리 보존을 구분한다.

시드·이웃 수·퍼플렉서티와 전처리를 바꿔 그림이 얼마나 유지되는지 확인한다. 안정적이라고 참 군집이 증명되는 것도 아니고, 설정에 민감하다고 원래 구조가 전혀 없다고 단정할 수도 없다. 투영에서 발견한 가설은 원래 데이터의 관계와 독립 표본에서 검증해야 한다.

## 5. 평가 지표와 시간

이웃 겹침은 원래와 임베딩의 k-최근접 이웃 집합 교집합 비율이다. 신뢰도(trustworthiness)는 임베딩에서 새로 이웃이 된 점을 원래 순위에 따라 벌점화하고, 연속성(continuity)은 원래 이웃을 놓친 정도를 다룬다. 둘은 순위에 기반하므로 모든 좌표를 함께 확대해도 같고, 점을 모두 겹쳐 동점으로 만들면 완벽한 점수가 자동으로 나오는 것도 아니다.

코드는 실행 시간을 출력하지만 이웃 겹침·신뢰도·분류 정확도를 계산하지 않는다. 따라서 특정 이웃 보존율과 방법 간 우열을 여기서 측정한 결과로 제시할 수 없다. 시간은 장치·스레드·설치 버전·초기값과 반복 수를 함께 기록하고 워밍업·반복 측정으로 비교한다.

## 6. 규모와 새 데이터 변환

전체 MDS는 쌍별 행렬 때문에 메모리 부담이 크다. Isomap도 이웃 그래프를 만들고 끝나는 것이 아니라 전체 최단 경로와 고유분해 단계가 필요하다. LLE는 국소 연산과 희소 고유분해를 사용하지만 이웃 탐색·해법에 따라 비용이 달라진다. 근사 t-SNE는 전체 쌍을 명시하는 exact 방식과 계산 특성이 다르다. 고정 표본 수에서 어느 방법이 반드시 가장 빠르다고 할 수 없다.

PCA 전처리는 입력 차원을 줄여 이후 계산을 가볍게 할 수 있지만 이웃과 거리도 바꾸므로 품질 영향과 적합 비용을 포함해 비교한다. 표본을 줄일 때는 희귀 집단과 대표성 손실을 확인한다. 많은 점을 그리기 어렵다는 이유만으로 일부 표본의 그림이 전체와 같다고 가정하지 않는다.

이 예제의 MDS와 scikit-learn t-SNE는 저장된 일반 `transform`을 제공하지 않는다. Isomap·LLE·PCA와 일반 UMAP에는 새 표본 변환 경로가 있다. UMAP의 `transform`은 신경망을 사용하는 Parametric UMAP만의 기능이 아니다. [UMAP의 새 데이터 변환 문서](https://umap-learn.readthedocs.io/en/latest/transform.html)를 참고한다. 각 방법의 표본 외 확장 조건과 새 데이터 분포 변화를 따로 평가한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표본 5,000개·이웃 수 20의 비교에서 아래 코드가 바꾸는 설정과 시간 해석의 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"

    hasattr로 n_neighbors가 있는 Isomap·LLE만 이웃 수를 바꾸지만 반복문은 네 방법 모두 실행한다. 표본 수도 기존 1500에서 5000으로 바뀌므로 두 효과가 섞인다. 반복·장치·버전을 통일하고 메모리도 측정해야 하며 희소 이웃 그래프만으로 Isomap 전체 비용이 선형이라고 할 수 없다.

    ```python
    X_swiss, t_swiss = make_swiss_roll(n_samples=5000, noise=0.2, random_state=42)
    for name, model in methods.items():
        if hasattr(model, 'n_neighbors'):
            model.n_neighbors = 20
        t0 = time.time()
        model.fit_transform(X_swiss)
        print(f"{name}: {time.time() - t0:.1f}s")
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
t-SNE의 군집 면적을 원래 분산으로 읽으면 안 되는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    점별로 적응한 고차원 이웃 폭과 KL 목표가 거리·밀도를 비선형으로 바꾼다. 다른 밀도의 집단이 비슷한 면적으로 보일 수 있다. 저차원 분포의 정규화 때문에 쌍들이 연결되므로 먼 점을 가까이 놓아도 아무 벌점이 없다는 설명도 부정확하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 UMAP 대체 코드와 새 데이터 변환을 평가할 때 무엇을 확인해야 하는가?

</div>

??? success "연습문제 3 풀이"

    같은 표준화 입력과 PCA, 시드·측정 범위를 사용하고 품질과 시간을 함께 비교한다. umap-learn 의존성이 필요하며 첫 실행의 준비 비용과 반복 실행도 구분한다. 일반 UMAP도 transform을 제공하므로 Parametric UMAP이 필수는 아니다. 기존 좌표를 그대로 쓰는지와 새 표본의 분포 차이를 평가한다.

    ```python
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    pipe = Pipeline([("pca", PCA(n_components=0.95, random_state=42)),
                     ("umap", reducer)])
    t0 = time.time()
    X_umap = pipe.fit_transform(X_mnist)
    print(f"PCA + UMAP: {time.time() - t0:.1f}s")
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
네 방법의 이웃 보존과 시간을 비교하는 측정 계획을 세워라.

</div>

??? success "연습문제 4 풀이"

    동일한 원래 입력 척도에서 자기 자신을 제외한 k-이웃을 정의하고 이웃 교집합, 신뢰도와 연속성을 계산한다. 거리 보존도 별도로 본다. 같은 목표 차원·전처리와 명시한 방법별 설정, 여러 시드와 반복 실행 시간을 기록한다. 이 페이지의 본 코드는 이런 품질 지표를 계산하지 않으므로 고정 표를 실측 결과로 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
시각화와 예측용 특징 선택의 목적 차이를 설명하라.

</div>

??? success "연습문제 5 풀이"

    시각화는 사람이 읽을 2·3차원 관계를, 예측용 특징은 독립 표본의 성능과 안정적인 변환을 중시한다. t-SNE의 분리된 그림이 좋은 분류 특징을 보장하지 않는다. 새 데이터 변환, 재구성 필요성과 품질 기준에 따라 PCA·커널·매니폴드·오토인코더를 비교한다. 디코더가 있다는 사실만으로 적절한 생성 모델이 되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
이웃 겹침, 신뢰도와 연속성의 차이를 설명하라.

</div>

??? success "연습문제 6 풀이"

    이웃 겹침은 k-이웃 집합의 교집합 비율이다. 신뢰도는 새로 생긴 이웃의 원래 순위를, 연속성은 잃어버린 이웃의 임베딩 순위를 반영한다. 좌표의 절대 크기만 키우거나 줄여서는 순위가 바뀌지 않는다. 동점과 표본 수, 유효한 k 범위도 정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
그림에서 발견한 군집을 검증하는 방법과 순환 검증의 위험을 설명하라.

</div>

??? success "연습문제 7 풀이"

    전처리·시드·하이퍼파라미터 변화와 원래 공간의 이웃·거리, 독립 표본의 재현성을 확인한다. 같은 투영에서 정한 군집을 같은 데이터로 분류해 높은 학습 정확도를 얻는 것은 독립 증거가 아니다. 외부 레이블·지식이나 사전에 정한 기준과 보류 데이터를 사용할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
Isomap에서 이웃 수가 너무 크거나 작을 때 생기는 문제를 설명하라.

</div>

??? success "연습문제 8 풀이"

    너무 크면 매니폴드의 서로 먼 부분 사이에 지름길이 생겨 측지 거리 근사가 잘못된다. 너무 작으면 그래프가 끊어져 경로가 무한이거나 구현의 연결 보정에 의존한다. 잡음·밀도와 그래프 연결성을 살피고 무조건 MDS보다 낫다고 가정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
표본 60,000개에서 알고리즘의 비용을 판단할 때 확인할 항목을 설명하라.

</div>

??? success "연습문제 9 풀이"

    float64 전체 쌍 행렬 하나만 28.8 GB이며 작업 공간이 추가된다. 계량 MDS 반복과 고전 MDS 고유분해의 비용은 다르다. Isomap의 전체 경로, t-SNE 근사 방식, 이웃 탐색과 LLE 해법도 확인한다. 표본 수만으로 불가능 여부나 고정 실행 시간을 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
표본 좌표를 직접 적합하는 방법과 입력→잠재 좌표 사상을 학습하는 방법을 비교하라.

</div>

??? success "연습문제 10 풀이"

    여기의 MDS·t-SNE는 기존 표본 좌표를 직접 적합하고 기본 transform이 없다. PCA는 선형 사상을, 커널 PCA는 커널 기반 변환을, 오토인코더는 인코더·디코더를 적합한다. Isomap·LLE에도 표본 외 확장이 있다. 오토인코더는 유연하지만 정확한 역변환·좋은 일반화나 항상 더 나은 성능을 보장하지 않는다.



## 정리하며

매니폴드 학습 방법은 서로 다른 관계를 보존하며 전처리와 설정에 따라 결과가 달라진다. 같은 입력 조건에서 국소·전역 지표와 계산 비용을 나누어 비교하고, 시각화에서 얻은 가설을 원래 공간과 독립 데이터에서 확인한다. 새 데이터 변환과 역변환의 지원 범위도 별도로 판단한다.
