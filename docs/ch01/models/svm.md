# 서포트 벡터 머신

서포트 벡터 머신(SVM)은 클래스 사이의 마진을 최대화하는 분리 초평면을 찾는다. SVM의 목적함수를 이해하면 마진, 정칙화, 커널 방법의 역할이 분명해지며, 이는 딥러닝의 손실 함수와 특징 표현 설계에 직접 연결된다.

---

## 1. 정의

SVM은 다음 제약 최적화 문제를 푼다.

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \;\; \xi_i \geq 0
$$

마진은 $2 / \|\mathbf{w}\|$이다. 매개변수 $C$는 마진 폭과 오분류 사이의 절충을 조절한다. 마진 위나 안쪽에 있는 점들이 **서포트 벡터** 이며, 결정 경계는 오직 이들만으로 결정된다.

---

## 2. 설명

**커널 기법**: SVM은 커널 함수 $K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$를 통해 입력을 더 높은 차원 공간으로 보냄으로써, $\phi$를 명시적으로 계산하지 않고도 비선형 경계를 학습할 수 있다. RBF 커널 $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$는 무한 차원 공간으로 대응시킨다.

**주요 하이퍼파라미터**:

- $C$: $C$가 작으면 위반을 더 허용하며 마진이 넓어진다(일반화에 유리). $C$가 크면 엄격한 분류를 강제한다(과적합 위험).
- $\gamma$ (RBF 커널): 각 서포트 벡터의 영향 반경을 조절한다. $\gamma$가 크면 복잡한 경계가 만들어진다.

**딥러닝과의 연결**: SVM에서 쓰는 힌지 손실 $\max(0, 1 - y \cdot f(x))$는 ReLU 활성화와 밀접하게 관련된다. 신경망은 SVM이 커널로 주어진다고 가정하는 특징 사상 $\phi$를 학습하는 것으로 볼 수 있다.

**한계**: 학습 복잡도가 $O(n^2)$에서 $O(n^3)$이므로, 신경망이 강점을 보이는 대규모 데이터셋에서는 SVM이 비현실적이다.

---

## 3. 예제

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터(SVM에는 스케일 조정이 필요하다)
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

# RBF 커널을 쓰는 sklearn SVM
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_tr_s, y_tr)
print(f"SVM accuracy: {svm.score(X_te_s, y_te):.4f}")
print(f"Support vectors: {svm.n_support_}")

# PyTorch: 힌지 손실을 이용한 선형 SVM
X_t = torch.tensor(X_tr_s, dtype=torch.float32)
y_t = torch.tensor(2 * y_tr - 1, dtype=torch.float32)  # {0,1} -> {-1,+1}

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

for _ in range(500):
    out = model(X_t).squeeze()
    loss = torch.clamp(1 - y_t * out, min=0).mean()  # 힌지 손실
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32)
    preds = (model(X_te_t).squeeze() > 0).long().numpy()
    acc = (preds == y_te).mean()
    print(f"PyTorch SVM accuracy: {acc:.4f}")
```

**출력:**

```
SVM accuracy: 0.8700
Support vectors: [82 91]
PyTorch SVM accuracy: 0.8600
```

---

## 연습문제

**연습문제 1.**
$\|\mathbf{w}\| = 2$인 선형 SVM의 마진 폭을 계산하라. $\mathbf{w}$를 3배로 늘리면 마진과 결정 경계는 어떻게 되는가?

??? success "연습문제 1 풀이"
    마진 폭은 $2/\|\mathbf{w}\| = 2/2 = 1$이다. $\mathbf{w}$를 $3\mathbf{w}$로($b$도 $3b$로) 늘리면 마진은 $2/\|3\mathbf{w}\| = 2/6 = 1/3$이 된다. 그러나 결정 경계 $\mathbf{w}^\top\mathbf{x} + b = 0$은 변하지 않는다. $3\mathbf{w}^\top\mathbf{x} + 3b = 0 \iff \mathbf{w}^\top\mathbf{x} + b = 0$이기 때문이다. 마진이 줄어든 것처럼 보이는 것은 매개변수화에 따른 현상일 뿐이다. SVM 목적함수는 $\|\mathbf{w}\|$를 최소화하여 정규화하며 이로써 표준적인 스케일이 선택된다.

---

**연습문제 2.**
힌지 손실 $\max(0, 1 - y \cdot f(x))$가 $f$에 대해 볼록임을 보여라. $yf(x)$가 어떤 값일 때 손실이 0에서 양수로 바뀌는가?

??? success "연습문제 2 풀이"
    $z = yf(x)$라 하자. 힌지 손실은 $h(z) = \max(0, 1-z)$이다. 이는 두 볼록 함수($0$과 $1-z$)의 점별 최댓값이므로 볼록이다. 전환은 $z = 1$, 즉 $yf(x) = 1$에서 일어난다. $yf(x) \geq 1$이면 손실이 0이다(충분한 마진으로 올바르게 분류됨). $yf(x) < 1$이면 손실이 $1 - yf(x)$이다(오분류되었거나 마진 안쪽에 있음). 힌지 손실은 $z = 1$에서 미분 불가능하지만 준경사가 존재하므로 최적화가 가능하다.

---

**연습문제 3.**
커널 기법을 설명하라. 왜 $\phi$를 명시적으로 계산하지 않고도 $K(\mathbf{x}, \mathbf{x}') = \langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle$를 계산할 수 있는가? $\mathbf{x} \in \mathbb{R}^2$에 대해 다항 커널 $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top\mathbf{x}' + 1)^2$이 특정한 특징 사상에 대응함을 보여라.

??? success "연습문제 3 풀이"
    커널 기법이 통하는 이유는 SVM의 쌍대 정식화가 입력을 오직 내적 $\langle\mathbf{x}_i, \mathbf{x}_j\rangle$을 통해서만 사용하기 때문이다. 이를 $K(\mathbf{x}_i, \mathbf{x}_j)$로 바꾸는 것은 $\phi$를 명시적으로 계산하지 않고 특징 공간에서 작업하는 것과 동등하다. $\mathbf{x} = (x_1, x_2)$에 대해 $K(\mathbf{x}, \mathbf{x}') = (x_1 x_1' + x_2 x_2' + 1)^2 = x_1^2 x_1'^2 + x_2^2 x_2'^2 + 1 + 2x_1 x_1' x_2 x_2' + 2x_1 x_1' + 2x_2 x_2'$이다. 이는 $\phi(\mathbf{x}) = (x_1^2, x_2^2, 1, \sqrt{2}x_1 x_2, \sqrt{2}x_1, \sqrt{2}x_2)$일 때의 $\langle\phi(\mathbf{x}), \phi(\mathbf{x}')\rangle$과 같다. 커널은 2차원 입력으로부터 6차원 내적을 계산한다.

---

**연습문제 4.**
어떤 데이터셋에 표본이 $n = 10{,}000$개 있다. SVM의 학습 시간 복잡도($O(n^2)$에서 $O(n^3)$)와 SGD로 $E$ 에폭 학습하는 2층 신경망의 복잡도($O(E \cdot n \cdot P)$, $P$는 매개변수 개수)를 비교하라. $n$이 어느 정도일 때 신경망이 더 효율적이 되는가?

??? success "연습문제 4 풀이"
    SVM 학습(SMO나 QP 사용): $O(n^2)$에서 $O(n^3)$이므로 $10^8$에서 $10^{12}$번의 연산이다. 매개변수 $P = 10{,}000$개, $E = 100$ 에폭인 신경망: $100 \times 10{,}000 \times 10{,}000 = 10^{10}$번의 연산이다. 이 설정에서 $n = 10{,}000$일 때 SVM과 신경망은 비슷하다. $n$이 더 커지면 SVM의 이차/삼차 증가가 지배한다. $n = 10^6$에서 SVM은 $10^{12}$–$10^{18}$이 드는 반면 신경망은 $100 \times 10^6 \times 10^4 = 10^{12}$이다. 신경망은 에폭당 비용이 $O(n)$이어서 전체가 $O(En)$이므로 큰 $n$에서 더 효율적이다.

---

**연습문제 5.**
이진 분류에서 힌지 손실(SVM)과 교차 엔트로피 손실(로지스틱 회귀/신경망)을 비교하라. 둘 다 0-1 손실의 볼록 상계임을 보이고, 딥러닝이 대개 힌지 손실 대신 교차 엔트로피를 쓰는 이유를 설명하라.

??? success "연습문제 5 풀이"
    0-1 손실은 $\mathbf{1}[yf(x) \leq 0]$이다. 힌지: $yf(x) \leq 0$일 때 힌지 $\geq 1$이고 지시함수는 1이므로 $\max(0, 1 - yf(x)) \geq \mathbf{1}[yf(x) \leq 0]$이다. 교차 엔트로피: 로지스틱 손실은 $yf(x) = 0$에서 $\geq \ln 2 > 0$이고 단조 감소하므로 $\log(1 + e^{-yf(x)}) \geq \mathbf{1}[yf(x) \leq 0]$이다. 둘 다 볼록이다. 딥러닝이 교차 엔트로피를 선호하는 이유는 다음과 같다. (1) 어디서나 매끄럽다(미분 가능한 반면 힌지는 미분 불가능한 꺾임이 있다). (2) 시그모이드를 통해 잘 보정된 확률 출력을 준다. (3) 교차 엔트로피의 경사 $p - 1$은 이미 잘 분류된 예제에 대해서도 더 강한 신호를 주어 확신을 높이도록 유도하는 반면, 힌지의 경사는 $yf > 1$에서 0이다.

## 정리하며

이 마당은 정의、설명、예제을 차례로 짚었다.
