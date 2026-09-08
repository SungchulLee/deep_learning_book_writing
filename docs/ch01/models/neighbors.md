# K-최근접 이웃

K-최근접 이웃(KNN)은 가장 가까운 학습 예제들을 바탕으로 예측하는 비모수 방법이다. 중요한 기준선 역할을 하며, 딥러닝에서 더 정교한 모델이 필요한 이유가 되는 편향-분산 절충을 잘 보여준다.

---

## 1. 정의

질의점 $\mathbf{x}$에 대해 KNN은 선택한 거리 척도 아래에서 가장 가까운 학습 점 $k$개의 집합 $\mathcal{N}_k(\mathbf{x})$를 찾는다. 분류에서는 다수결로 예측한다.

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}[y_i = c]
$$

회귀에서는 이웃 목표값의 (가중) 평균으로 예측한다.

$$
\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \, y_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}
$$

---

## 2. 설명

KNN에는 학습 단계가 없다. 모든 데이터를 저장했다가 예측 시점에 거리를 계산한다. 그래서 **게으른 학습기(lazy learner)** 라고 부른다. 주요 고려사항은 다음과 같다.

- **특징 스케일 조정이 필수**: 거리 척도는 특징의 크기에 민감하다. KNN을 적용하기 전에는 항상 표준화해야 한다.
- **$k$ 선택**: $k$가 작으면 편향은 낮지만 분산이 크다(잡음에 민감). $k$가 크면 편향은 크지만 분산이 작다(경계가 지나치게 매끄러워짐). 교차 검증으로 $k$를 고른다.
- **차원의 저주**: 고차원에서는 점들 사이의 거리가 거의 균일해져 최근접 이웃 질의가 정보를 주지 못한다. 이는 신경망으로 저차원 표현을 학습해야 하는 핵심 동기이다.
- **계산 비용**: 완전 탐색은 질의당 $O(nd)$가 든다. 트리 구조(KD 트리, Ball 트리)는 저차원에서 이를 $O(d \log n)$으로 줄이지만 고차원에서는 성능이 떨어진다.

---

## 3. 예제

```python
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# 데이터를 생성하고 스케일을 조정한다
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 교차 검증으로 k를 선택한다
best_k, best_score = 1, 0.0
for k in range(1, 21):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                            X_train_s, y_train, cv=5).mean()
    if score > best_score:
        best_k, best_score = k, score
print(f"Best k={best_k}, CV accuracy={best_score:.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(X_train_s, y_train)
print(f"Test accuracy: {knn.score(X_test_s, y_test):.4f}")

# PyTorch에서의 KNN(학습된 임베딩을 위한 직접 구현)
X_tr = torch.tensor(X_train_s, dtype=torch.float32)
X_te = torch.tensor(X_test_s, dtype=torch.float32)
y_tr = torch.tensor(y_train)

dists = torch.cdist(X_te[:5], X_tr)  # 쌍별 거리
_, idx = dists.topk(best_k, largest=False)
neighbor_labels = y_tr[idx]
print(f"Neighbor labels for first 5 queries:\n{neighbor_labels}")
```

---

## 연습문제

**연습문제 1.**
$k = 3$인 최근접 이웃의 레이블이 $\{1, 0, 1\}$인 질의점 $\mathbf{x}$에 대해 KNN은 무엇을 예측하는가? 거리가 $\{0.5, 0.3, 0.8\}$일 때 거리 가중 투표를 사용하면 어떻게 되는가?

??? success "연습문제 1 풀이"
    **다수결**: 클래스 1이 2표, 클래스 0이 1표이다. 예측: 클래스 1. **거리 가중**: 가중치는 거리에 반비례한다. $w_i = 1/d_i$. 클래스 1: $1/0.5 + 1/0.8 = 2.0 + 1.25 = 3.25$. 클래스 0: $1/0.3 = 3.33$. 예측: 클래스 0. 가장 가까운 이웃(거리 0.3)이 클래스 0에 속하고 불균형하게 큰 가중치를 받기 때문에 거리 가중이 예측을 바꾸었다.

---

**연습문제 2.**
완전 탐색 KNN의 질의당 시간 복잡도가 $O(nd)$임을 증명하라. 여기서 $n$은 학습 점의 개수, $d$는 차원이다. $d$가 클 때 KD 트리가 도움이 되지 않는 이유는 무엇인가?

??? success "연습문제 2 풀이"
    질의마다 $n$개의 학습 점 모두에 대한 거리를 계산한다. $\mathbb{R}^d$에서 거리 계산 하나당 $O(d)$번의 연산이 든다($d$개의 제곱 차이의 합). 총 $O(nd)$이다. KD 트리는 좌표축을 따라 공간을 분할하고 탐색 중 가지를 쳐내어 저차원에서 $O(d \log n)$을 달성한다. 그러나 $d$가 크면 트리가 효과적으로 가지를 쳐낼 수 없다. 축에 평행한 어떤 분할 안에서 "가장 가까운" 점이 다른 차원에서는 멀 수 있어 대부분의 가지를 탐색해야 하기 때문이다. 경험적으로 KD 트리는 $d \gtrsim 20$이면 완전 탐색 수준으로 성능이 떨어진다.

---

**연습문제 3.**
KNN에는 특징 스케일 조정이 필수인데 결정 트리에는 그렇지 않은 이유를 설명하라. 스케일이 조정되지 않은 특징이 잘못된 KNN 예측을 낳는 구체적인 예를 들라.

??? success "연습문제 3 풀이"
    KNN은 특징의 크기에 민감한 거리 척도(예: 유클리드)에 의존한다. 특징 1의 범위가 $[0, 1000]$이고 특징 2가 $[0, 1]$이면 거리는 특징 1에 지배되어 특징 2가 사실상 무시된다. 결정 트리는 각 특징을 독립적으로 분할하므로 특징의 스케일이 분할 임계값 선택에 영향을 주지 않는다. 예: 나이(0–100)와 혈압(60–200)으로 환자를 분류하는 경우. 스케일을 조정하지 않으면 혈압의 10단위 차이가 나이 10년 차이보다 거리에 더 크게 기여하는데, 나이가 더 예측력이 높을 수도 있다.

---

**연습문제 4.**
차원의 저주는 고차원에서 최근접 이웃까지의 거리와 최원접 이웃까지의 거리의 비가 1에 가까워진다는 것이다. 이것이 KNN에 주는 함의를 설명하고, 딥러닝이 이를 어떻게 다루는지 설명하라.

??? success "연습문제 4 풀이"
    거리가 거의 균일해지면 "가장 가깝다"는 개념이 의미를 잃는다. $k$개의 최근접 이웃이 무작위 점보다 거의 가깝지 않게 된다. KNN의 예측은 거의 무작위 투표 수준으로 떨어진다. 형식적으로, $[0,1]^d$에 균등 분포한 점들에 대해 $\lim_{d \to \infty} \frac{d_{\min}}{d_{\max}} = 1$이다. 딥러닝은 의미 있는 거리가 보존되는 저차원 임베딩 $\phi(\mathbf{x}) \in \mathbb{R}^{d'}$($d' \ll d$)을 학습하여 이 문제를 다룬다. 학습된 임베딩(예: 신경망의 마지막 직전 층 출력) 위에서의 KNN이 잘 동작하는 이유는, 임베딩 공간이 의미적으로 유사한 점들을 가깝게 배치하도록 구조화되어 있기 때문이다.

---

**연습문제 5.**
학습 점 $n$개, 특징 $d$개, 이웃 $k$개인 KNN의 저장 공간과 예측 시간 복잡도를 유도하라. 이를 매개변수가 $P$개이고 추론 비용이 고정된 신경망과 비교하라.

??? success "연습문제 5 풀이"
    **KNN**: 저장 공간은 $O(nd)$(모든 학습 점). 질의당 예측: 완전 탐색 거리 계산에 $O(nd)$, 상위 $k$개를 찾는 부분 정렬에 $O(n \log k)$. $k \ll n$이면 총 예측 비용은 $O(nd + n\log k) = O(nd)$이다. **신경망**: 저장 공간은 $O(P)$(매개변수만, 학습 데이터는 필요 없음). 예측: $O(P)$(층마다 행렬 곱 한 번). $P$는 고정이고 큰 데이터셋에서는 보통 $P \ll nd$이므로 신경망이 추론에서 훨씬 효율적이다. 대가는 신경망이 비싼 학습을 요구하는 반면 KNN은 학습 비용이 0이라는 점이다.

## 정리하며

이 마당은 정의、설명、예제을 차례로 짚었다.
