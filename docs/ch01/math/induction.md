# 수학적 귀납법

수학적 귀납법은 모든 자연수에 대한 명제를 증명하는 표준적인 기법이다. 딥러닝에서는 신경망 깊이, 재귀적 구조, $n$번 반복 후의 수렴에 관한 증명에 귀납 논증이 등장한다.

---

## 1. 정의

수학적 귀납법은 성질 $P(n)$이 모든 정수 $n \geq n_0$에 대해 성립함을 두 단계로 증명한다.

1. **기저 단계**: $P(n_0)$을 증명한다
2. **귀납 단계**: 모든 $k \geq n_0$에 대해 $P(k) \implies P(k+1)$을 증명한다

이 원리는 자연수의 정렬성에 기대고 있다. 반례의 집합이 공집합이 아니라면 최소 원소를 가질 텐데, 기저 단계와 귀납 단계가 함께 그것을 배제한다.

---

## 2. 설명

귀납 단계는 $P(k+1)$을 홀로 증명하는 것이 아니다. *만약* $P(k)$가 성립한다면 $P(k+1)$이 따라온다는 함의를 증명한다. 이것이 기저 단계와 결합되면 모든 $n \geq n_0$을 덮는 함의의 사슬이 만들어진다.

표준적인 예로, 처음 $n$개 정수의 합은 다음을 만족한다.

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

**기저 단계**: $P(1)$: $1 = \frac{1 \cdot 2}{2}$. 참이다.

**귀납 단계**: $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$를 가정한다. 그러면

$$
\sum_{i=1}^{k+1} i = \frac{k(k+1)}{2} + (k+1) = \frac{(k+1)(k+2)}{2}
$$

딥러닝에서 귀납법은 $L$개 층을 가진 신경망이 $L$번의 아핀-비선형 변환을 합성함을 증명하거나, 특정 조건 아래 경사 하강법이 매 단계마다 손실을 줄임을 증명하는 데 쓰인다.

---

## 3. 예제

```python
import torch

# 귀납법 방식으로 합 공식을 확인한다
def sum_formula(n):
    return n * (n + 1) // 2

# 기저 단계
assert sum_formula(1) == 1, "Base case failed"

# 귀납 단계: 여러 k에 대해 P(k) + (k+1) = P(k+1)을 확인한다
for k in range(1, 100):
    lhs = sum_formula(k) + (k + 1)
    rhs = sum_formula(k + 1)
    assert lhs == rhs, f"Inductive step failed at k={k}"
print("Induction verified for n = 1..100")

# 딥러닝과의 연결: L개의 선형 층을 합성한 것이
# 하나의 선형 층과 동등함을 확인한다(깊이에 대한 귀납)
torch.manual_seed(42)
d = 4
x = torch.randn(d)

L = 5
matrices = [torch.randn(d, d) for _ in range(L)]

# 층을 차례로 적용한다
result = x.clone()
for W in matrices:
    result = W @ result

# 하나의 행렬로 합성한다
composed = torch.eye(d)
for W in matrices:
    composed = W @ composed

print(f"Sequential result: {result[:3]}")
print(f"Composed result:   {(composed @ x)[:3]}")
print(f"Match: {torch.allclose(result, composed @ x, atol=1e-4)}")
```

**출력:**

```
Induction verified for n = 1..100
Sequential result: tensor([ 0.0679, -0.8439, -1.6294])
Composed result:   tensor([ 0.0679, -0.8439, -1.6294])
Match: True
```

---

## 연습문제

**연습문제 1.**
모든 $L$개 층의 너비가 $d$인(입력과 출력의 차원도 $d$) 완전연결 신경망의 매개변수 개수가 (가중치와 편향을 모두 세어) $L \cdot d(d + 1)$임을 수학적 귀납법으로 증명하라.

??? success "연습문제 1 풀이"
    각 층은 $d \times d$ 가중치 행렬과 $d$차원 편향 벡터를 가지므로 층당 $d^2 + d = d(d+1)$개의 매개변수를 가진다. **기저 단계** ($L = 1$): 한 층은 $d(d+1)$개의 매개변수를 가진다. $\checkmark$ **귀납 단계**: $k$개 층의 신경망이 $k \cdot d(d+1)$개의 매개변수를 가진다고 가정한다. 층을 하나 더하면 $d(d+1)$개가 추가되어 총 $(k+1) \cdot d(d+1)$개가 된다. 귀납법에 의해 이 공식은 모든 $L \geq 1$에 대해 성립한다. $\square$

---

**연습문제 2.**
활성화 함수가 없는 신경망($\mathbf{z}_l = \mathbf{W}_l \mathbf{z}_{l-1}$)에서 $L$개 층에 연쇄 법칙을 적용하면 경사 $\frac{\partial \ell}{\partial \mathbf{W}_1} = \left(\prod_{i=2}^{L} \mathbf{W}_i^\top\right) \frac{\partial \ell}{\partial \mathbf{z}_L}$이 나옴을 귀납법으로 증명하라.

??? success "연습문제 2 풀이"
    **기저 단계** ($L = 1$): $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{z}_0$이므로 $\frac{\partial \ell}{\partial \mathbf{W}_1} = \frac{\partial \ell}{\partial \mathbf{z}_1} \mathbf{z}_0^\top$이다. 곱 $\prod_{i=2}^{1}$은 공곱(항등원)이므로 일관된다. **귀납 단계**: $k$층 신경망에 대해 $\frac{\partial \ell}{\partial \mathbf{z}_1} = \left(\prod_{i=2}^{k} \mathbf{W}_i^\top\right) \frac{\partial \ell}{\partial \mathbf{z}_k}$를 가정한다. $\mathbf{z}_{k+1} = \mathbf{W}_{k+1}\mathbf{z}_k$인 $k+1$번째 층을 더하면 연쇄 법칙에 의해 $\frac{\partial \ell}{\partial \mathbf{z}_k} = \mathbf{W}_{k+1}^\top \frac{\partial \ell}{\partial \mathbf{z}_{k+1}}$이다. 대입하면 $\frac{\partial \ell}{\partial \mathbf{z}_1} = \left(\prod_{i=2}^{k} \mathbf{W}_i^\top\right) \mathbf{W}_{k+1}^\top \frac{\partial \ell}{\partial \mathbf{z}_{k+1}} = \left(\prod_{i=2}^{k+1} \mathbf{W}_i^\top\right) \frac{\partial \ell}{\partial \mathbf{z}_{k+1}}$이다. $\square$

---

**연습문제 3.**
$\sum_{i=0}^{n} 2^i = 2^{n+1} - 1$을 귀납법으로 증명하라. 그리고 이 공식이 빔 탐색에서 힙 기반 우선순위 큐에 쓰이는 이진 트리 자료구조와 어떤 관련이 있는지 설명하라.

??? success "연습문제 3 풀이"
    **기저 단계** ($n = 0$): $2^0 = 1 = 2^1 - 1$. $\checkmark$ **귀납 단계**: $\sum_{i=0}^{k} 2^i = 2^{k+1} - 1$을 가정한다. 그러면 $\sum_{i=0}^{k+1} 2^i = 2^{k+1} - 1 + 2^{k+1} = 2 \cdot 2^{k+1} - 1 = 2^{k+2} - 1$이다. $\square$ 이 공식은 높이 $n$인 완전 이진 트리의 전체 노드 수를 준다. 빔 탐색에서 힙 기반 우선순위 큐는 상위 $k$개 후보를 유지하는데, 트리 크기를 알면 힙 배열에 필요한 메모리 할당량이 결정된다.

---

**연습문제 4.**
모든 정수 $n \geq 2$가 소인수분해를 가짐을 강한 귀납법으로 증명하라. 여기서 보통의 귀납법이 아니라 강한 귀납법이 필요한 이유는 무엇인가?

??? success "연습문제 4 풀이"
    **기저 단계** ($n = 2$): 2는 소수이므로 그 분해는 $\{2\}$ 자신이다. $\checkmark$ **강한 귀납 단계**: $2 \leq m < n$인 모든 정수가 소인수분해를 가진다고 가정한다. $n$이 소수이면 그 자신이 분해이다. $n$이 합성수이면 $2 \leq a, b < n$인 $n = a \cdot b$로 쓸 수 있다. 강한 귀납 가정에 의해 $a$와 $b$ 모두 소인수분해를 가진다. 이들을 이어 붙이면 $n$의 소인수분해가 된다. $\square$ 강한 귀납법이 필요한 이유는 $n = a \cdot b$로 분해할 때 $n - 1$보다 훨씬 작을 수 있는 $a$와 $b$에 대한 가정이 필요하기 때문이다. 보통의 귀납법은 $n - 1$에 대한 가정만 제공한다.

---

**연습문제 5.**
너비 $d$인 은닉층을 $L$개 가진 ReLU 신경망이 $\mathbb{R}^d$를 최대 $(2^d)^L$개의 선형 영역으로 분할함을 귀납법으로 증명하라.

??? success "연습문제 5 풀이"
    **기저 단계** ($L = 1$): ReLU 뉴런 $d$개를 가진 은닉층 하나는 $\mathbb{R}^d$에 초평면 $d$개를 만든다. 각 뉴런은 활성이거나 비활성이므로 서로 다른 활성화 패턴(선형 영역)은 최대 $2^d$개이다. $\checkmark$ **귀납 단계**: $k$개의 은닉층을 가진 신경망이 최대 $(2^d)^k$개의 선형 영역을 가진다고 가정한다. ReLU 뉴런 $d$개를 가진 $k+1$번째 은닉층을 더하면 기존의 각 선형 영역이 최대 $2^d$개의 부분 영역으로 세분된다(기존 영역 안에서 새 층은 아핀이고, 새 ReLU 뉴런 $d$개가 최대 $2^d$개의 활성화 패턴을 만들기 때문이다). 총합: $(2^d)^k \cdot 2^d = (2^d)^{k+1}$. $\square$ 참고: 이는 느슨한 상계이다. 더 엄밀한 상계는 모든 활성화 패턴이 실현 가능하지는 않다는 점을 반영한다.

## 정리하며

이 마당은 정의、설명、예제을 차례로 짚었다.
