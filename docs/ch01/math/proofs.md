# 증명 기법

증명 기법은 알고리즘의 정확성, 최적화 방법의 수렴, 일반화 오차의 상계를 확립하는 논리적 도구를 제공한다. 딥러닝의 모든 이론적 결과는 이 기법들 중 하나 이상에 기대고 있다.

## 정의

수학적 증명은 공리와 이미 확립된 결과에서 출발하여 어떤 명제가 참임을 논리적으로 엄밀하게 논증하는 것이다. 주요 증명 전략은 다음과 같다.

$$
\begin{array}{ll}
\text{직접 증명} & \text{전제를 가정하고 결론을 유도한다} \\
\text{모순법} & \text{부정을 가정하고 불가능성을 유도한다} \\
\text{귀납법} & \text{기저 단계 + 귀납 단계} \\
\text{구성적 증명} & \text{구체적인 예를 제시한다} \\
\text{대우 증명} & P \Rightarrow Q \text{ 대신 } \neg Q \Rightarrow \neg P \text{를 증명한다}
\end{array}
$$

## 설명

각 기법은 서로 다른 상황에 적합하다.

- **직접 증명**: 기본이 되는 접근이다. 가설이 주어지면 정의와 알려진 결과를 적용해 결론에 도달한다. 예: MSE 손실의 경사가 잔차의 선형 함수임을 증명하는 것.
- **모순법**: 불가능성 결과에 유용하다. 주장이 거짓이라고 가정하고 논리적 충돌을 유도한다. 예: 특정 복잡도 가정 아래 어떤 결정적 알고리즘도 비볼록 함수를 다항 시간에 최소화할 수 없음을 증명하는 것.
- **귀납법**: 자연수로 색인된 명제에 필수적이다. 예: 역전파가 $L$개 층을 통해 경사를 올바르게 계산함을 $L$에 대한 귀납법으로 증명하는 것.
- **구성적 증명**: 명시적인 예를 만들어 존재성을 증명한다. 예: 임의의 연속 함수를 근사하는 신경망을 구성하는 것(보편 근사 정리의 증명).
- **대우 증명**: $P \Rightarrow Q$를 증명하는 대신 논리적으로 동치인 $\neg Q \Rightarrow \neg P$를 증명한다. 결론의 부정이 더 강한 출발점을 줄 때 유용하다.

## 예제

```python
import torch

# 직접 증명의 검증: L2 손실의 경사는 2*(pred - target)/n
torch.manual_seed(0)
pred = torch.randn(5, requires_grad=True)
target = torch.randn(5)

loss = ((pred - target) ** 2).mean()
loss.backward()

# 해석적 경사
analytical_grad = 2 * (pred.detach() - target) / pred.numel()
print(f"Autograd gradient:    {pred.grad.tolist()}")
print(f"Analytical gradient:  {analytical_grad.tolist()}")
print(f"Match: {torch.allclose(pred.grad, analytical_grad)}")

# 구성적 증명: 입력을 출력으로 보내는 신경망을 만든다
# f(x) = 2x + 1을 구현하는 가중치를 구성한다
W = torch.tensor([[2.0]])
b = torch.tensor([1.0])
x = torch.tensor([3.0])
y = W @ x + b
print(f"\nConstructed linear map: f({x.item()}) = {y.item()}")
```

## 연습문제

**연습문제 1.**
MSE 손실 $\ell = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$의 스칼라 예측 $\hat{y}_j$에 대한 경사가 $\frac{2}{n}(\hat{y}_j - y_j)$임을 직접 증명하라.

??? success "연습문제 1 풀이"
    $\frac{\partial \ell}{\partial \hat{y}_j} = \frac{\partial}{\partial \hat{y}_j}\left[\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2\right]$이다. $\hat{y}_j$에 의존하는 항은 $i = j$인 항뿐이므로 $= \frac{1}{n} \cdot 2(\hat{y}_j - y_j) \cdot 1 = \frac{2}{n}(\hat{y}_j - y_j)$이다. 이것은 직접 증명이다. $\ell$의 정의에서 출발하여 부정을 가정하지 않고 미분 규칙을 적용해 결론에 도달했다. $\square$

---

**연습문제 2.**
임의의 연속 함수 $f: [0,1] \to \mathbb{R}$와 임의의 $\epsilon > 0$에 대해 $f$를 $\epsilon$ 이내로 근사하는 ReLU 신경망이 존재함을 구성적 증명으로 보여라(구성의 개요를 제시하라).

??? success "연습문제 2 풀이"
    $[0, 1]$을 $N$개의 같은 구간으로 분할하되, $f$가 각 구간에서 $\epsilon$보다 작게 변하도록 $N$을 충분히 크게 잡는다(균등 연속성에 의해 가능하다). 각 구간 $[k/N, (k+1)/N]$에서 $f$를 $(k/N, f(k/N))$과 $((k+1)/N, f((k+1)/N))$을 잇는 선분으로 근사한다. 이 조각별 선형 함수는 은닉 뉴런 $N$개를 가진 ReLU 신경망으로 정확히 표현할 수 있다. $g(x) = f(0) + \sum_{k=1}^{N}(s_k - s_{k-1})\text{ReLU}(x - k/N)$이며 $s_k$는 각 선분의 기울기이다. 구성에 의해 $|f(x) - g(x)| < \epsilon$이므로 존재성이 구성적으로 확립된다. $\square$

---

**연습문제 3.**
대우로 증명하라. 경사 하강법이 $\nabla \ell(\theta^*) \neq 0$인 점 $\theta^*$로 수렴한다면 학습률이 너무 컸다. (형식화: $L$이 $\nabla \ell$의 립시츠 상수일 때 $\eta < 1/L$이면 임의의 고정점은 $\nabla \ell(\theta^*) = 0$을 만족한다.)

??? success "연습문제 3 풀이"
    "$\eta < 1/L$이면 임의의 고정점에서 $\nabla \ell(\theta^*) = 0$이다"의 대우는 "고정점에서 $\nabla \ell(\theta^*) \neq 0$이면 $\eta \geq 1/L$이다"이다. 원래 명제의 증명: 경사 하강법의 고정점은 $\theta^* = \theta^* - \eta \nabla \ell(\theta^*)$를 만족하므로 $\eta \nabla \ell(\theta^*) = 0$이다. 가정에 의해 $\eta > 0$이므로 $\nabla \ell(\theta^*) = 0$이다. 이는 $\eta < 1/L$뿐 아니라 임의의 $\eta > 0$에 대해 성립한다. $\eta < 1/L$ 조건은 애초에 고정점으로의 수렴을 보장하는 데 필요하지만(하강 보조정리), 일단 수렴하면 고정점 방정식이 경사가 0임을 강제한다. $\square$

---

**연습문제 4.**
$f$가 차수 $d$인 상수가 아닌 다항식이면 $f$의 실근이 최대 $d$개임을 모순법으로 증명하라.

??? success "연습문제 4 풀이"
    모순을 위해 $f$가 서로 다른 실근 $r_1, \ldots, r_{d+1}$을 $d + 1$개 가진다고 가정하자. 그러면 어떤 다항식 $g$에 대해 $f(x) = (x - r_1)(x - r_2)\cdots(x - r_{d+1}) \cdot g(x)$이다. 좌변의 차수는 $d$이지만 우변의 차수는 적어도 $d + 1$이다. 다항식의 차수는 유일하므로 이는 모순이다. 따라서 $f$의 근은 최대 $d$개이다. $\square$ 이는 신경망과도 관련이 있다. 단일 ReLU 뉴런은 "근"(전환점)을 최대 1개 가지며, 이들의 합성은 꺾인 점의 개수가 유계인 조각별 선형 함수를 만든다.

---

**연습문제 5.**
$L$개 층을 통한 역전파가 역방향 계산에서 정확히 $L$번의 행렬 곱(층당 1회)을 필요로 함을 귀납법으로 증명하라.

??? success "연습문제 5 풀이"
    각 $f_l$이 미분 가능한 층인 신경망 $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$을 생각하자. **기저 단계** ($L = 1$): $\frac{\partial \ell}{\partial \theta_1}$을 계산하려면 야코비안 $\frac{\partial f_1}{\partial \theta_1}$과의 행렬 곱 1회가 필요하다. **귀납 단계**: $(k+1)$층 신경망에서 연쇄 법칙은 $\frac{\partial \ell}{\partial \theta_j} = \frac{\partial \ell}{\partial \mathbf{z}_{k+1}} \cdot \frac{\partial \mathbf{z}_{k+1}}{\partial \mathbf{z}_k} \cdot \frac{\partial \mathbf{z}_k}{\partial \theta_j}$을 준다. $\frac{\partial \ell}{\partial \mathbf{z}_{k+1}}$에서 $\frac{\partial \ell}{\partial \mathbf{z}_k}$를 계산하려면 (층 $k+1$의 야코비안과) 행렬 곱 1회가 필요하다. 나머지 $k$번의 행렬 곱은 귀납 가정에 의해 층 $1$부터 $k$까지를 처리한다. 총합: $k + 1$번의 곱셈. $\square$
