# 로그

로그는 딥러닝의 중심에 있다. 자연로그는 교차 엔트로피 손실을 정의하고, 로그 확률은 수치 계산을 안정시키며, 로그 스케일은 학습률 스케줄과 정보 이론적 양에 나타난다.

## 정의

밑이 $b$인 $x$의 로그는 $b$를 몇 제곱해야 $x$가 되는지를 나타내는 지수이다.

$$
\log_b x = y \iff b^y = x
$$

자연로그 $\ln x = \log_e x$는 지수함수의 역함수이고 최대가능도 추정에서 자연스럽게 등장하므로 딥러닝에서 가장 중요한 형태이다.

## 설명

딥러닝 전반에서 쓰이는 주요 성질은 다음과 같다.

$$
\begin{array}{ll}
\ln(xy) = \ln x + \ln y \\
\ln(x/y) = \ln x - \ln y \\
\ln(x^n) = n \ln x \\
\log_b x = \frac{\ln x}{\ln b} & \text{(밑 변환)}
\end{array}
$$

딥러닝에서 로그가 중요한 이유는 다음과 같다.

- **교차 엔트로피 손실**: 음의 로그가능도 $-\ln p(y \mid x)$가 표준적인 분류 손실이다. 로그는 확률의 곱을 합으로 바꾸어 수치적으로 안정하고 미분하기 쉽게 만든다.
- **log-sum-exp 기법**: $\ln \sum_i e^{x_i}$를 $\max(x) + \ln \sum_i e^{x_i - \max(x)}$로 계산하면 오버플로를 막는다.
- **정보 이론**: 엔트로피 $H = -\sum p_i \ln p_i$와 KL 발산이 로그로 정의된다.
- **로그 스케일 조정**: 학습률은 흔히 로그 스케일로 감쇠한다(예: $10^{-2}$에서 $10^{-5}$까지).

## 예제

```python
import torch
import numpy as np

# 교차 엔트로피 손실은 내부적으로 로그를 사용한다
logits = torch.tensor([2.0, 1.0, 0.1])
target = 0  # 정답 클래스
probs = torch.softmax(logits, dim=0)
loss = -torch.log(probs[target])
print(f"Softmax probs: {probs.tolist()}")
print(f"Cross-entropy loss: {loss.item():.4f}")

# 수치적 안정성을 위한 log-sum-exp 기법
x = torch.tensor([1000.0, 1001.0, 1002.0])
# 소박한 계산: 오버플로 발생
# 안정한 방법: 먼저 최댓값을 뺀다
max_x = x.max()
stable = max_x + torch.log(torch.exp(x - max_x).sum())
print(f"Log-sum-exp (stable): {stable.item():.4f}")

# PyTorch 내장 함수
builtin = torch.logsumexp(x, dim=0)
print(f"Log-sum-exp (builtin): {builtin.item():.4f}")

# 밑 변환
val = 1024.0
print(f"log2({val:.0f}) = {np.log2(val):.2f}")
print(f"ln({val:.0f}) = {np.log(val):.4f}")
```

## 연습문제

**연습문제 1.**
$p_c = e^{z_c} / \sum_j e^{z_j}$일 때, 교차 엔트로피 손실 $\ell = -\ln p_c$의 정답 클래스 로짓 $z_c$에 대한 경사를 유도하라.

??? success "연습문제 1 풀이"
    $\ell = -\ln p_c = -\ln\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right) = -z_c + \ln\left(\sum_j e^{z_j}\right)$이다. 미분하면 $\frac{\partial \ell}{\partial z_c} = -1 + \frac{e^{z_c}}{\sum_j e^{z_j}} = -1 + p_c = p_c - 1$이다. 이 우아한 결과는 정답 클래스에 대한 경사가 단순히 예측 확률에서 1을 뺀 값임을 보여준다.

---

**연습문제 2.**
log-sum-exp 기법이 값을 정확히 보존함을 보여라. 즉 임의의 상수 $m$에 대해 $\ln \sum_i e^{x_i} = m + \ln \sum_i e^{x_i - m}$임을 보여라. $m = \max_i x_i$로 두면 오버플로가 방지되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    $m + \ln \sum_i e^{x_i - m} = m + \ln\left(e^{-m} \sum_i e^{x_i}\right) = m + \ln(e^{-m}) + \ln \sum_i e^{x_i} = m - m + \ln \sum_i e^{x_i} = \ln \sum_i e^{x_i}$. $\square$ $m = \max_i x_i$이면 모든 지수 $x_i - m \leq 0$이므로 $e^{x_i - m} \leq 1$이다. 이는 오버플로($e^{x_i}$가 부동소수점 범위, float32에서는 대략 $e^{88}$을 넘을 때 발생)를 막는다. 합은 최소 1이므로($e^{x_{\max} - m} = e^0 = 1$이기 때문) $\ln(\cdot)$이 잘 정의되고 음이 아니다.

---

**연습문제 3.**
코사인 학습률 스케줄이 $T$ 단계에 걸쳐 $\eta_0$에서 $\eta_{\min}$까지 감쇠한다. $\eta_0 = 10^{-2}$이고 $\eta_{\min} = 10^{-5}$일 때 비 $\eta_0 / \eta_{\min}$을 데시벨(dB)로 계산하라. 여기서 $\text{dB} = 10 \log_{10}(\text{ratio})$이다.

??? success "연습문제 3 풀이"
    $\eta_0 / \eta_{\min} = 10^{-2} / 10^{-5} = 10^3 = 1000$. 데시벨로는 $10 \log_{10}(1000) = 10 \times 3 = 30\;\text{dB}$이다. 로그 스케일이 1000배 범위를 다루기 쉬운 수로 압축해 주며, 학습률 그래프가 보통 로그 축을 쓰는 이유가 바로 이것이다.

---

**연습문제 4.**
엔트로피 $H(p) = -\sum_i p_i \ln p_i$가 $p$가 $K$개 클래스에 대한 균등 분포일 때 최대가 되며 최댓값이 $\ln K$임을 증명하라.

??? success "연습문제 4 풀이"
    $\sum_i p_i = 1$, $p_i \geq 0$ 제약 아래 $H(p) = -\sum_{i=1}^{K} p_i \ln p_i$를 최대화한다. 라그랑주 승수법을 쓰면 $\frac{\partial}{\partial p_i}\left[-\sum_j p_j \ln p_j - \lambda\left(\sum_j p_j - 1\right)\right] = -\ln p_i - 1 - \lambda = 0$이다. 이로부터 모든 $i$에 대해 $p_i = e^{-1-\lambda}$이며 이는 상수이다. 제약 $\sum_i p_i = 1$로부터 $p_i = 1/K$를 얻는다. 대입하면 $H(1/K, \ldots, 1/K) = -\sum_{i=1}^{K} \frac{1}{K} \ln \frac{1}{K} = -K \cdot \frac{1}{K} \cdot (-\ln K) = \ln K$이다. $\square$

---

**연습문제 5.**
KL 발산은 $D_{\text{KL}}(p \| q) = \sum_i p_i \ln(p_i / q_i)$이다. $\ln$의 오목성에 옌센 부등식을 적용하여 $D_{\text{KL}}(p \| q) \geq 0$임을 보여라.

??? success "연습문제 5 풀이"
    $D_{\text{KL}}(p \| q) = \sum_i p_i \ln\frac{p_i}{q_i} = -\sum_i p_i \ln\frac{q_i}{p_i}$이다. 오목함수 $\ln$에 옌센 부등식을 적용하면 $-\sum_i p_i \ln\frac{q_i}{p_i} \geq -\ln\left(\sum_i p_i \cdot \frac{q_i}{p_i}\right) = -\ln\left(\sum_i q_i\right) = -\ln(1) = 0$이다. 등호는 모든 $i$에 대해 $q_i / p_i$가 상수일 때, 즉 (둘 다 합이 1이므로) $p = q$일 때에만 성립한다. $\square$
