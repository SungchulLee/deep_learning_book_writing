# 해밀턴 움직임
해밀턴 움직임은 해밀턴 몬테카를로의 밑바탕이 되는 고전 역학의 수학 얼개이다. 이 절에서는 물리의 바탕을 펼친다. 곧 역학의 해밀턴 꼴, 심플렉틱 짜임, 지킴 법칙, 그리고 이 성질들이 어떻게 HMC를 가능하게 하는지를 다룬다.

---

## 뉴턴 역학에서 해밀턴 역학으로

### 라그랑주 꼴

고전 역학은 뉴턴의 둘째 법칙 $\mathbf{F} = m\mathbf{a}$에서 비롯했지만 **라그랑주 꼴**이 더 우아한 길을 준다. 넓힌 좌표 $\mathbf{q}$과 속도 $\dot{\mathbf{q}}$을 갖는 체계에 대해 라그랑주 함수를 다음과 같이 정한다:

$$
L(\mathbf{q}, \dot{\mathbf{q}}) = T(\dot{\mathbf{q}}) - U(\mathbf{q})
$$

여기서 $T$은 운동 에너지이고 $U$은 퍼텐셜 에너지이다. 운동 방정식은 **최소 작용 원리**에서 따라 나온다. 곧 체계가 지나는 길이 작용 적분 $S = \int L \, dt$을 극값으로 만든다.

오일러-라그랑주 방정식은 다음과 같다:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0
$$

### 르장드르 바꿈

**해밀턴 꼴**은 르장드르 바꿈으로 속도를 운동량으로 바꾼다. **켤레 운동량**을 다음과 같이 정한다:

$$
p_i = \frac{\partial L}{\partial \dot{q}_i}
$$

운동 에너지가 $T = \frac{1}{2}m|\dot{\mathbf{q}}|^2$인 알갱이에서는 낯익은 운동량 $\mathbf{p} = m\dot{\mathbf{q}}$이 나온다.

**해밀턴 함수**는 르장드르 바꿈으로 얻는다:

$$
H(\mathbf{q}, \mathbf{p}) = \mathbf{p} \cdot \dot{\mathbf{q}} - L(\mathbf{q}, \dot{\mathbf{q}})
$$

여기서 $\dot{\mathbf{q}}$은 $\mathbf{p} = \partial L / \partial \dot{\mathbf{q}}$을 써서 $\mathbf{p}$으로 나타낸다.

**핵심 결과**: 지킴 체계(시간에 안 달린 $L$)에서 해밀턴 함수는 전체 에너지와 같다:

$$
H = T + U
$$

### 해밀턴 방정식

해밀턴 꼴의 운동 방정식은 다음과 같다:

$$
\frac{d\mathbf{q}}{dt} = \frac{\partial H}{\partial \mathbf{p}}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{q}}
$$

이는 변수 $2d$개(자리와 운동량)에 대한 **일차** 상미분방정식이며, 뉴턴 역학의 변수 $d$개에 대한 **이차** 상미분방정식과 견주어진다. 한 식에는 $+$, 다른 식에는 $-$이 붙는 이 대칭 짜임이 해밀턴 체계의 표지이다.

**보기**: 퍼텐셜 속의 알갱이가 $H = \frac{|\mathbf{p}|^2}{2m} + U(\mathbf{q})$을 가질 때:

$$
\frac{d\mathbf{q}}{dt} = \frac{\mathbf{p}}{m}, \quad \frac{d\mathbf{p}}{dt} = -\nabla U(\mathbf{q})
$$

첫 식은 속도가 운동량을 질량으로 나눈 것임을 말하고, 둘째 식은 뉴턴 법칙 $\dot{\mathbf{p}} = \mathbf{F}$이다.

---

## 심플렉틱 짜임

### 해밀턴 벡터장

위상 공간에는 **심플렉틱 형식**이 주는 본디의 기하 짜임이 있다:

$$
\omega = \sum_{i=1}^{d} dp_i \wedge dq_i
$$

이 2형식은 위상 공간에서 "방향을 가진 넓이"를 잰다. 심플렉틱 형식은 **닫혀 있고**($d\omega = 0$) **찌부러지지 않는다**(0이 아닌 어떤 접벡터 $\mathbf{v}$에 대해서도 $\omega(\mathbf{v}, \mathbf{w}) \neq 0$인 $\mathbf{w}$이 있다).

해밀턴 함수 $H$이 주어지면 **해밀턴 벡터장** $X_H$을 다음으로 넌지시 정한다:

$$
\omega(X_H, \cdot) = dH
$$

좌표로 쓰면 다음이 된다:

$$
X_H = \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{\partial}{\partial \mathbf{q}} - \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{\partial}{\partial \mathbf{p}}
$$

$X_H$의 적분 곡선이 해밀턴 방정식의 풀이이다.

**다르부 정리**: 차원이 같은 심플렉틱 다양체는 모두 그 자리에서 같다. 표준 $(\mathbf{q}, \mathbf{p})$ 좌표가 어디서나 있는 까닭이 이것이다.

### 심플렉틱 사상

미분동형사상 $\phi: (\mathbf{q}, \mathbf{p}) \mapsto (\mathbf{Q}, \mathbf{P})$이 심플렉틱 형식을 지키면 **심플렉틱**(또는 **정준**)이라고 한다:

$$
\phi^* \omega = \omega
$$

같은 말로 행렬 꼴에서 $\mathbf{J} = \frac{\partial(\mathbf{Q}, \mathbf{P})}{\partial(\mathbf{q}, \mathbf{p})}$이 야코비 행렬이면 다음이 성립한다:

$$
\mathbf{J}^T \mathbf{\Omega} \mathbf{J} = \mathbf{\Omega}, \quad \text{where } \mathbf{\Omega} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}
$$

**심플렉틱 사상의 성질**:

1. **부피 지킴**: $|\det \mathbf{J}| = 1$($\det(\mathbf{J}^T \mathbf{\Omega} \mathbf{J}) = \det \mathbf{\Omega}$에서 따라 나온다)
2. **합성**: $\phi_1$과 $\phi_2$이 심플렉틱이면 $\phi_1 \circ \phi_2$도 그렇다
3. **역사상**: $\phi$이 심플렉틱이면 $\phi^{-1}$도 그렇다
4. **해밀턴 흐름은 심플렉틱이다**: 아무 해밀턴 체계의 시간 $t$ 흐름 사상 $\phi_t$은 심플렉틱이다

### 낳는 함수

심플렉틱 사상은 **낳는 함수**로 특징지을 수 있다. 사상 $(\mathbf{q}, \mathbf{p}) \mapsto (\mathbf{Q}, \mathbf{P})$에 대해 다음을 만족하는 $S(\mathbf{q}, \mathbf{Q})$이 있으면

$$
\mathbf{p} = \frac{\partial S}{\partial \mathbf{q}}, \quad \mathbf{P} = -\frac{\partial S}{\partial \mathbf{Q}}
$$

그 사상은 심플렉틱이다. 이는 몇몇 앞선 HMC 갈래의 바탕이 된다.

---

## 지킴 법칙

### 에너지 지킴

**정리**: 해밀턴 방정식의 풀이를 따라가면 해밀턴 함수는 상수이다.

**증명**:

$$
\frac{dH}{dt} = \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{d\mathbf{q}}{dt} + \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{d\mathbf{p}}{dt} = \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{\partial H}{\partial \mathbf{p}} - \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{\partial H}{\partial \mathbf{q}} = 0
$$

해밀턴 방정식의 반대칭 짜임 덕분에 엇갈린 항이 정확히 지워진다.

**HMC에서의 중요함**: 에너지 지킴은 쏠림 없이 긴 자취를 가능하게 하는 핵심 성질이다. 이 덕분에 시작점에서 멀리 떨어진 제안도 받아들임 확률이 높게 남는다.

### 위상 공간 부피 지킴(리우빌 정리)

**정리**(리우빌): 해밀턴 흐름은 위상 공간의 부피를 지킨다.

위상 공간의 구역 $\Omega$이 해밀턴 방정식을 따라 흘러간다고 하자. 시간 $t$에서의 상을 $\Omega_t$이라 하면 다음이 성립한다:

$$
\text{Vol}(\Omega_t) = \text{Vol}(\Omega_0) \quad \text{for all } t
$$

**증명**: 위상 공간의 속도장은 $\mathbf{v} = (\dot{\mathbf{q}}, \dot{\mathbf{p}})$이다. 그 발산은 다음과 같다:

$$
\nabla \cdot \mathbf{v} = \sum_i \left( \frac{\partial \dot{q}_i}{\partial q_i} + \frac{\partial \dot{p}_i}{\partial p_i} \right) = \sum_i \left( \frac{\partial^2 H}{\partial q_i \partial p_i} - \frac{\partial^2 H}{\partial p_i \partial q_i} \right) = 0
$$

이음 방정식에 따라 발산이 0이면 부피가 지켜진다.

**HMC에서의 중요함**: 부피가 지켜지므로 메트로폴리스-헤이스팅스 받아들임 비에 야코비 바로잡기가 필요 없다.

### 푸앵카레 되돌아옴

**정리**(푸앵카레): 묶인 에너지 면 위의 해밀턴 체계에서는 거의 모든 자취가 시작점에 원하는 만큼 가깝게 되돌아온다.

이는 부피 지킴에서 따라 나온다. 구역 $\Omega$이 언제까지나 서로 겹치지 않는 구역으로만 흘러간다면 전체 부피가 끝없이 커져 리우빌 정리에 어긋난다.

**뜻하는 바**: 해밀턴 움직임은 본디 모이는 것이 아니라 **되돌아오는** 것이다. HMC가 최적화가 아니라 표집을 하는 까닭이 이것이다.

---

## 시간을 되돌릴 수 있음

### 정의

$R^2 = \text{identity}$을 뜻하는 대합 $R$이 있어 다음이 성립하면 그 움직임 체계는 **시간을 되돌릴 수 있다**고 한다:

$$
\phi_{-t} = R \circ \phi_t \circ R
$$

여기서 $\phi_t$은 시간 $t$의 흐름이다.

### 해밀턴 체계에서 되돌릴 수 있음

해밀턴 체계에서는 운동량 뒤집기 사상 $R: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, -\mathbf{v})$이 되돌릴 수 있음을 준다.

**되는 까닭**: $\mathbf{v} \mapsto -\mathbf{v}$ 아래에서:

- 운동 에너지 $K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^T\mathbf{M}^{-1}\mathbf{v}$은 바뀌지 않는다($\mathbf{v}$에 대해 짝함수이다)
- 해밀턴 방정식은 $\frac{d\mathbf{x}}{d(-t)} = -\mathbf{M}^{-1}\mathbf{v}$이 되고, $\mathbf{v}$을 뒤집으면 이는 $\frac{d\mathbf{x}}{dt}$과 같다

**HMC에서의 중요함**: 시간을 되돌릴 수 있으므로 제안이 알맞은 뜻에서 대칭이 되어 메트로폴리스-헤이스팅스 받아들임 잣대가 단순해진다.

---

## 뇌터 정리와 대칭

### 진술

**뇌터 정리**: 해밀턴 함수의 이어진 대칭마다 지켜지는 양이 하나씩 딸린다.

| 대칭 | 지켜지는 양 |
|----------|-------------------|
| 시간 옮김 | 에너지 |
| 공간 옮김 | 선운동량 |
| 돌림 | 각운동량 |

### 표집에 뜻하는 바

기본 HMC에서 우리가 쓰는 대칭은 시간 옮김(에너지 지킴) 하나뿐이다. 그러나 과녁의 **돌림 대칭**은 자취가 봉우리를 놓치게 할 수 있고(지켜지는 각운동량이 궤도를 옭아맨다), (자리바꿈 불변 같은) **띄엄띄엄한 대칭**은 표집 효율에 영향을 줄 수 있다.

---

## 표집을 위한 해밀턴 함수

### 표준 꼴

$\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$에서 표집하려면 다음을 쓴다:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v})
$$

여기서 각 기호는 다음과 같다.

- **자리** $\mathbf{x}$이 $\mathbf{q}$을 대신한다(표집할 변수)
- **운동량** $\mathbf{v}$이 $\mathbf{p}$을 대신한다(도움 변수)
- **퍼텐셜 에너지**: $U(\mathbf{x}) = -\log \tilde{\pi}(\mathbf{x})$
- **운동 에너지**: $K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}$

이에 딸린 해밀턴 방정식은 다음과 같다:

$$
\frac{d\mathbf{x}}{dt} = \frac{\partial H}{\partial \mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\frac{\partial H}{\partial \mathbf{x}} = -\nabla U(\mathbf{x}) = \nabla \log \pi(\mathbf{x})
$$

운동량 방정식은 **점수 함수가 힘 노릇을 함**을 보여 준다.

### 나뉘는 해밀턴 함수

엇갈린 항 없이 $H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v})$이면 그 해밀턴 함수는 **나뉜다**고 한다. 표준 HMC의 해밀턴 함수는 나뉜다.

**나뉨이 왜 중요한가**: 나뉘는 해밀턴 함수에서는 해밀턴 방정식이 서로 풀린다. 곧 $\frac{d\mathbf{x}}{dt}$은 $\mathbf{v}$에만, $\frac{d\mathbf{v}}{dt}$은 $\mathbf{x}$에만 달려 있다. 이 짜임 덕분에 개구리뜀 적분기 같은 **연산자 쪼개기** 방법을 쓸 수 있으며, 여기서는 작은 걸음마다 정확히 풀고 그 합성이 온전한 움직임을 어림한다.

### 나뉘지 않는 해밀턴 함수

몇몇 앞선 HMC 갈래는 나뉘지 않는 해밀턴 함수를 쓴다:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}(\mathbf{x})^{-1} \mathbf{v} + \frac{1}{2}\log|\mathbf{M}(\mathbf{x})|
$$

여기서는 질량 행렬이 자리에 달려 있다. 이것이 **리만 HMC**이며 더 정교한 적분기가 필요하다.

---

## 해밀턴 움직임이 왜 표집을 가능하게 하나

### 물리 직관

퍼텐셜 우물 속의 알갱이를 보자:

1. **기울기 내리기**는 알갱이를 바닥까지 굴린 뒤 멈춘다
2. **해밀턴 움직임**은 알갱이가 떨어지는 동안 퍼텐셜 에너지를 운동 에너지로 바꾼다
3. 알갱이는 가장 낮은 곳을 **지나쳐** 반대편으로 오른다
4. 끝없이 흔들리며 우물을 살펴본다

핵심 통찰은 **에너지 지킴이 봉우리로 모이는 것을 막는다**는 것이다.

### 수학적 설명

최적화(기울기 내리기)에서는 에너지가 줄어든다:

$$
\frac{dU}{dt} = \nabla U \cdot \frac{d\mathbf{x}}{dt} = -|\nabla U|^2 \leq 0
$$

해밀턴 움직임에서는 전체 에너지가 상수이지만 $U$과 $K$이 서로 오간다:

$$
\frac{dU}{dt} = -\frac{dK}{dt}
$$

표집에서 중요한 것은 점마다 머문 시간이 아니라 그 움직임의 **멈춘 분포**이다. 에너지 지킴과 부피 지킴이 볼츠만 분포 $\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H)$이 멈춘 분포임을 보장한다.

### 에르고드성과 섞임

표집이 되려면 그 움직임이 **에르고드적**(시간 평균이 앙상블 평균과 같음)이고 **섞여야**(첫 조건이 "잊혀야") 한다.

순수한 해밀턴 움직임은 흔히 에르고드적이지 않다. 자취가 붙박인 에너지 면 위에 머물기 때문이다. HMC는 되풀이마다 **운동량을 다시 표집하여** 에르고드성을 얻는다. 곧 무작위 운동량이 새 운동 에너지를 불어넣어 다른 에너지 면을 살펴볼 수 있게 한다.

---

## 통계 역학과의 이음

### 볼츠만 분포

통계 역학에서 온도 $T$의 열 평형에 있는 체계의 상태 분포는 다음과 같다:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp\left(-\frac{H(\mathbf{x}, \mathbf{v})}{k_B T}\right)
$$

여기서 $k_B$은 볼츠만 상수이다. 표집에서는 $k_B T = 1$으로 둔다.

### 나눔 함수와 자유 에너지

**나눔 함수**는 $Z = \int \exp(-H(\mathbf{x}, \mathbf{v})) \, d\mathbf{x} \, d\mathbf{v}$이고 **자유 에너지**는 $F = -\log Z$이다. $Z$을 셈하는 일은 대개 다룰 수 없으며, 그래서 곧바로 적분하는 대신 MCMC로 표집한다.

### 작은 정준 앙상블과 정준 앙상블

- **작은 정준**(에너지 일정): 상태가 에너지 면 $H = E$ 위에 고르게 흩어진다
- **정준**(온도 일정): 상태가 $\exp(-H)$을 따라 흩어진다

HMC는 정준 앙상블에서 굴러간다. 운동량을 다시 표집하는 걸음이 온도를 지킨다.

---

## 요약

| 개념 | 설명 | HMC에서 하는 일 |
|---------|-------------|-------------|
| 해밀턴 함수 | 전체 에너지 $H = U + K$ | 움직임을 정한다 |
| 위상 공간 | $(\mathbf{x}, \mathbf{v})$의 공간 | 넓힌 상태 공간 |
| 해밀턴 방정식 | $\dot{\mathbf{x}} = \partial_\mathbf{v} H$, $\dot{\mathbf{v}} = -\partial_\mathbf{x} H$ | 흘러감을 다스린다 |
| 에너지 지킴 | $dH/dt = 0$ | 높은 받아들임 비율 |
| 부피 지킴 | $\det \mathbf{J} = 1$ | 야코비 바로잡기가 필요 없음 |
| 시간을 되돌릴 수 있음 | $\phi_{-t} = R \circ \phi_t \circ R$ | 자세한 균형 |
| 나뉨 | $H = U(\mathbf{x}) + K(\mathbf{v})$ | 개구리뜀을 가능하게 함 |

HMC의 힘은 이 고전 역학의 원리에서 나온다. 곧 에너지 지킴이 봉우리로 모이는 것을 막고, 부피 지킴이 야코비 바로잡기를 없애며, 심플렉틱 짜임이 효율적인 수치 적분을 가능하게 한다.

---

## 참고 문헌

1. Arnold, V. I. (1989). *Mathematical Methods of Classical Mechanics*. Springer.
2. Goldstein, H., Poole, C., & Safko, J. (2002). *Classical Mechanics* (3rd ed.). Addison Wesley.
3. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
4. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.

## 연습문제

1. **해밀턴 방정식 확인하기**. $H = \frac{p^2}{2m} + \frac{1}{2}kx^2$(조화 떨개)에 대해 해밀턴 방정식을 이끌어 내고 풀어라. 그 풀이가 되풀이되며 $x = 0$으로 모이지 않음을 보여라.

2. **심플렉틱함 확인하기**. 사상 $(q, p) \mapsto (q + \epsilon p, p)$이 심플렉틱임을 보여라. $(q, p) \mapsto (q, p - \epsilon \nabla U(q))$도 심플렉틱임을 보여라.

3. **에너지 면의 기하**. $\mathbf{M} = \mathbf{I}$인 2차원 가우스 과녁 $\pi(x) \propto \exp(-\frac{1}{2}x^T \Sigma^{-1} x)$에 대해 4차원 위상 공간의 에너지 면을 설명하여라. 그 꼴은 무엇인가?

4. **나뉘지 않는 해밀턴 함수**. 질량이 자리에 달린 $H(x, v) = U(x) + \frac{v^2}{2m(x)}$에 대해 해밀턴 방정식을 이끌어 내어라. 이것이 나뉘는 경우보다 왜 더 복잡한가?

---
