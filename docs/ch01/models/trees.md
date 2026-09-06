# 결정 트리

결정 트리는 축에 평행한 분할로 특징 공간을 재귀적으로 나누어, 특징 스케일 조정이 필요 없는 해석 가능한 모델을 만든다. 앙상블 방법(랜덤 포레스트, 그래디언트 부스팅)의 구성 요소이며, ReLU 신경망의 조각별 선형 동작을 이해하는 데도 도움이 된다.

## 정의

결정 트리는 분할 후 불순도를 최소화하는 특징 $j$와 임계값 $t$를 탐욕적으로 선택하여 입력 공간을 나눈다. 분류에서 표준적인 두 가지 불순도 측도는 다음과 같다.

$$
\text{지니: } G(p) = 1 - \sum_{k=1}^{K} p_k^2 \qquad \text{엔트로피: } H(p) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

여기서 $p_k$는 노드에서 클래스 $k$가 차지하는 비율이다. 각 잎은 (분류에서는) 다수 클래스를, (회귀에서는) 목표값의 평균을 출력한다.

## 설명

트리는 탐욕적이다. 각 노드에서 이후의 분할을 고려하지 않고 국소적으로 최적인 분할을 고른다. 덕분에 학습이 빠르지만 만들어진 트리가 전역적으로 최적이 아닐 수 있다.

**과적합 제어** 가 핵심 과제이다.

- `max_depth`: 트리 깊이를 제한한다. 가장 중요한 정칙화 매개변수이다.
- `min_samples_leaf`: 각 잎이 충분한 표본을 갖도록 보장한다.
- **비용-복잡도 가지치기**(`ccp_alpha`): 복잡도에 비해 불순도 감소가 적은 부분 트리를 제거한다.

**장점**: 해석 가능하고, 스케일 조정이 필요 없으며, 혼합된 특징 유형을 다룰 수 있고, 추론이 빠르다(표본당 $O(\log n)$).

**단점**: 분산이 크고(데이터가 조금만 달라져도 다른 트리가 나온다), 축에 평행한 분할로는 대각선 경계를 효율적으로 포착할 수 없으며, 단일 트리는 과적합되기 쉽다.

**딥러닝과의 연결**: 은닉층이 하나인 ReLU 신경망은 결정 트리와 비슷한 조각별 선형 함수를 구현한다. 다만 신경망은 (축에 평행하지 않은) 비스듬한 분할을 학습하고 영역 간에 매개변수를 공유하므로 훨씬 큰 표현 용량을 가진다.

## 예제

```python
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 교차 검증으로 max_depth를 선택한다
best_depth, best_score = 1, 0.0
for d in range(1, 15):
    score = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42),
                            X_tr, y_tr, cv=5).mean()
    if score > best_score:
        best_depth, best_score = d, score
print(f"Best depth={best_depth}, CV accuracy={best_score:.4f}")

tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
tree.fit(X_tr, y_tr)
print(f"Test accuracy: {tree.score(X_te, y_te):.4f}")
print(f"Nodes: {tree.tree_.node_count}, Leaves: {tree.get_n_leaves()}")

# 특징 중요도(지니 기반)
importances = tree.feature_importances_
top3 = np.argsort(importances)[-3:][::-1]
for i in top3:
    print(f"  Feature {i}: importance={importances[i]:.4f}")

# 비교: 같은 데이터에 대한 간단한 신경망
X_t = torch.tensor(X_tr, dtype=torch.float32)
y_t = torch.tensor(y_tr, dtype=torch.long)
net = torch.nn.Sequential(
    torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(200):
    loss = torch.nn.functional.cross_entropy(net(X_t), y_t)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    nn_acc = (net(X_te_t).argmax(1).numpy() == y_te).mean()
print(f"Neural net accuracy: {nn_acc:.4f}")
```

## 연습문제

**연습문제 1.**
클래스 A 표본 60개와 클래스 B 표본 40개를 가진 노드의 지니 불순도를 계산하라. 이 노드는 순수한가?

??? success "연습문제 1 풀이"
    $p_A = 60/100 = 0.6$, $p_B = 40/100 = 0.4$. $G = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$. 이 노드는 순수하지 않다. 순수한 노드는 $G = 0$이다(모든 표본이 한 클래스에 속한다). 두 클래스에서 지니의 최댓값은 $0.5$이므로($p_A = p_B = 0.5$일 때) $0.48$은 거의 최대에 가까운 불순도를 나타낸다.

---

**연습문제 2.**
클래스가 $K$개일 때 지니 불순도 $G(p) = 1 - \sum_k p_k^2$이 분포가 균등할 때 최댓값 $1 - 1/K$를 가짐을 증명하라.

??? success "연습문제 2 풀이"
    $\sum_k p_k = 1$, $p_k \geq 0$ 제약 아래 $G = 1 - \sum_k p_k^2$을 최대화한다. 이는 $\sum_k p_k^2$을 최소화하는 것과 같다. 코시-슈바르츠 부등식(또는 라그랑주 승수법)에 의해 $\sum p_k^2$은 모든 $p_k$가 같을 때, 즉 $p_k = 1/K$일 때 최소가 된다. 그러면 $\sum p_k^2 = K \cdot (1/K)^2 = 1/K$이므로 $G_{\max} = 1 - 1/K$이다. $K = 2$이면 $G_{\max} = 0.5$, $K = 10$이면 $G_{\max} = 0.9$이다. $\square$

---

**연습문제 3.**
깊이 $d$인 결정 트리는 $O(d)$ 시간에 예측한다. $n$개 표본으로 학습된 균형 트리에서 $d$를 $n$으로 표현하면? 매개변수가 $P$개인 신경망의 추론 비용과 비교하라.

??? success "연습문제 3 풀이"
    $n$개 표본에 대한 균형 이진 트리의 깊이는 $d = O(\log_2 n)$이다. 추론은 뿌리에서 잎까지 하나의 경로를 따르므로 $O(\log n)$번의 비교가 든다. 매개변수가 $P$개인 신경망은 순전파에 $O(P)$가 든다. 적당한 크기의 신경망($P = 10^6$)과 큰 데이터셋($n = 10^6$)이라면 트리 추론은 $O(20)$인 반면 신경망 추론은 $O(10^6)$이다. 트리는 추론에서 압도적으로 빠르며, 지연 시간이 중요한 응용(광고 순위 매기기, 사기 탐지)에서 여전히 널리 쓰이는 이유이다.

---

**연습문제 4.**
단일 결정 트리의 분산이 큰 이유를 설명하라. 랜덤 포레스트와 그래디언트 부스팅 트리는 각각 이 문제를 어떻게 다루는가?

??? success "연습문제 4 풀이"
    단일 트리는 학습 데이터에 탐욕적으로 적합되므로, 작은 교란(표본 몇 개 제거, 분할 임계값 변경)만으로도 완전히 다른 트리 구조가 나올 수 있다. 이러한 불안정성이 곧 높은 분산이다. **랜덤 포레스트** 는 부트스트랩 표본과 무작위 특징 부분집합으로 학습한 독립적인 트리 $B$개를 평균 내어 분산을 줄인다. (트리들이 무상관이라면) 평균화는 분산을 대략 $B$배 줄인다. **그래디언트 부스팅 트리** 는 다른 방식으로 분산을 줄인다. 얕은 트리(낮은 깊이 = 높은 편향, 낮은 분산)를 순차적으로 적합시키며 각 트리가 앙상블의 잔차 오차를 보정한다. 축소 계수(학습률)가 과적합을 막는다.

---

**연습문제 5.**
ReLU 신경망과 결정 트리의 관계를 설명하라. 뉴런이 $n$개인 은닉층 하나를 가진 ReLU 신경망이 깊이 $\log n$인 결정 트리보다 더 일반적인 함수 부류를 표현할 수 있는 이유는 무엇인가?

??? success "연습문제 5 풀이"
    둘 다 조각별 선형 함수를 구현한다. 결정 트리는 입력 공간을 축에 평행한 직사각형 영역으로 나누고, ReLU 신경망은 활성화 패턴으로 정의되는 볼록 다면체로 나눈다. 핵심 차이는 결정 트리의 경계가 축에 평행한(특징 축에 수직인) 반면, ReLU 신경망은 임의의 $\mathbf{w}$에 대해 $\mathbf{w}^\top\mathbf{x} + b = 0$인 비스듬한 경계를 만든다는 점이다. 즉 트리에서 축에 평행한 분할이 $O(2^d)$번 필요한 대각선 결정 경계를 뉴런 하나로 구현할 수 있다. 또한 ReLU 신경망은 영역 간에 매개변수를 공유하므로(같은 가중치 행렬이 어디서나 적용된다) 암묵적인 정칙화 효과가 있고 매끄러운 함수를 더 효율적으로 표현할 수 있다.
