# 위상 공간
위상 공간은 해밀턴 몬테카를로가 굴러가는 수학의 마당이다. 이 절에서는 위상 공간의 기하, 표집을 위해 넓힌 상태 공간, 효율적인 살펴보기를 가능하게 하는 기하 짜임을 자세히 다룬다.

---

## 정의와 그 까닭

### 상태 공간을 왜 넓히나?

과녁 분포 $\pi(\mathbf{x})$에서 표집하려고 HMC는 도움 **운동량 변수** $\mathbf{v}$을 들여와 넓힌 공간 $(\mathbf{x}, \mathbf{v})$에서 일한다. 변수가 $d$개에서 $2d$개로 늘어나니 문제가 복잡해지는 듯하지만, 이 넓힘은 결정적인 이로움을 준다:

1. **정해진 움직임**: $(\mathbf{x}, \mathbf{v})$이 주어지면 자취가 온전히 정해진다
2. **기울기 써먹기**: 점수 $\nabla \log \pi(\mathbf{x})$이 한 방향으로 이어지는 움직임을 이끈다
3. **에너지 지킴**: 제안이 거의 에너지가 일정한 면 위에 머문다
4. **효율적인 살펴보기**: 탄도 움직임은 거리 $\propto L$을 지나가고 퍼짐꼴은 $\propto \sqrt{L}$이다

### 위상 공간

**정의**: 자리 공간이 $d$차원인 체계에서 **위상 공간**은 $2d$차원 다양체이다:

$$
\Gamma = \{(\mathbf{x}, \mathbf{v}) : \mathbf{x} \in \mathbb{R}^d, \mathbf{v} \in \mathbb{R}^d\}
$$

위상 공간의 점마다 체계의 온전한 **상태**를 나타낸다. 곧 어디에 있는지($\mathbf{x}$)와 어떻게 움직이는지($\mathbf{v}$)를 함께 담는다.

### 자리 공간과 위상 공간의 견줌

| 성질 | 자리 공간 | 위상 공간 |
|----------|---------------|-------------|
| 차원 | $d$ | $2d$ |
| 상태 | 자리 $\mathbf{x}$ | 자리와 운동량 $(\mathbf{x}, \mathbf{v})$ |
| 움직임 | 없음(멈춰 있음) | 해밀턴 방정식 |
| 과녁 | $\pi(\mathbf{x})$ | $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})$ |

---

## 결합 분포

### 만들기

과녁 $\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$이 주어지면 위상 공간의 결합 분포를 다음과 같이 정한다:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H(\mathbf{x}, \mathbf{v}))
$$

여기서 해밀턴 함수는 다음과 같다:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v}) = -\log \tilde{\pi}(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}
$$

### 인수로 나뉨

해밀턴 함수가 자리 항과 운동량 항으로 나뉘므로:

$$
\pi(\mathbf{x}, \mathbf{v}) = \frac{1}{Z} \exp(-U(\mathbf{x})) \exp(-K(\mathbf{v})) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})
$$

**핵심 성질**:

1. **독립**: 결합 분포 아래에서 자리와 운동량은 독립이다
2. **주변 분포 되찾기**: $\int \pi(\mathbf{x}, \mathbf{v}) \, d\mathbf{v} = \pi(\mathbf{x})$
3. **쉬운 운동량 표집**: $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$은 어렵지 않다

### 고르게 하는 상수

나눔 함수는 인수로 나뉜다:

$$
Z = \int \exp(-H(\mathbf{x}, \mathbf{v})) \, d\mathbf{x} \, d\mathbf{v} = Z_x \cdot Z_v
$$

여기서 $Z_x = \int \exp(-U(\mathbf{x})) \, d\mathbf{x}$은 다룰 수 없고 $Z_v = (2\pi)^{d/2} |\mathbf{M}|^{1/2}$은 알려져 있다.

MCMC에서는 $Z_x$을 셈할 일이 없다. $\pi$의 비만 있으면 된다.

---

## 에너지 면

### 정의

**에너지 면**(또는 **등위 집합**)은 에너지가 붙박인 위상 공간 점 모두의 모임이다:

$$
\Sigma_E = \{(\mathbf{x}, \mathbf{v}) \in \Gamma : H(\mathbf{x}, \mathbf{v}) = E\}
$$

해밀턴 움직임은 에너지를 지키므로 자취는 이 $(2d-1)$차원 면에 갇힌다.

### 에너지 면의 기하

표준 HMC의 해밀턴 함수 $H = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}$에 대해:

**자리 $\mathbf{x}$을 붙박아 두면**: 제약 $K(\mathbf{v}) = E - U(\mathbf{x})$이 운동량 공간에서 타원체를 정한다($E > U(\mathbf{x})$이라 놓을 때):

$$
\frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v} = E - U(\mathbf{x})
$$

**운동량 $\mathbf{v}$을 붙박아 두면**: 제약 $U(\mathbf{x}) = E - K(\mathbf{v})$이 퍼텐셜 에너지의 등위 집합(과녁 밀도의 등고선)을 정한다.

### 에너지와 확률

에너지가 높으면 확률이 낮다:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H(\mathbf{x}, \mathbf{v})) = \exp(-E)
$$

**전형 집합**: 확률의 대부분은 $H$이 가장 작은 곳(봉우리)이 아니라 에너지가 중간인 "껍질" 위에 놓인다. 이것이 차원 높은 곳의 **측도 몰림** 현상이다.

$d$차원 표준 가우스에서 전형 에너지는 (봉우리의 $E = 0$이 아니라) $E \approx d$이다.

---

## 부피 요소와 측도

### 리우빌 측도

위상 공간의 자연스러운 측도는 **리우빌 측도**이다:

$$
d\mu = d\mathbf{x} \, d\mathbf{v} = dx_1 \cdots dx_d \, dv_1 \cdots dv_d
$$

이는 $\mathbb{R}^{2d}$의 표준 르베그 측도이다.

**리우빌 정리**: 해밀턴 흐름은 리우빌 측도를 지킨다. $\phi_t$이 시간 $t$의 흐름 사상이면 잴 수 있는 아무 집합 $A$에 대해 다음이 성립한다:

$$
\mu(\phi_t(A)) = \mu(A)
$$

### 작은 정준 측도

에너지 면 $\Sigma_E$ 위의 자연스러운 측도는 **작은 정준 측도**이다:

$$
d\sigma_E = \frac{d\mu}{|\nabla H|}
$$

여기서 분모는 기울기의 크기(에너지 껍질의 "두께")로 고르게 한다.

해밀턴 흐름은 에너지 면마다 이 측도도 지킨다.

### 정준 측도

**정준 측도**(볼츠만 분포)는 다음과 같다:

$$
d\nu = \exp(-H(\mathbf{x}, \mathbf{v})) \, d\mathbf{x} \, d\mathbf{v}
$$

이것이 우리가 표집하려는 측도이다. 해밀턴 움직임은 $H$을 지키고(에너지 지킴) 부피를 지키므로(리우빌 정리) 이 측도를 지킨다.

---

## 위상 공간의 심플렉틱 기하

### 심플렉틱 형식

위상 공간은 근본적인 기하 짜임인 **심플렉틱 2형식**을 지닌다:

$$
\omega = \sum_{i=1}^{d} dv_i \wedge dx_i
$$

행렬로 쓰면 $\mathbf{z} = (\mathbf{x}, \mathbf{v})^T$일 때 다음과 같다:

$$
\omega(\mathbf{u}, \mathbf{w}) = \mathbf{u}^T \mathbf{J} \mathbf{w}, \quad \text{where } \mathbf{J} = \begin{pmatrix} \mathbf{0} & -\mathbf{I} \\ \mathbf{I} & \mathbf{0} \end{pmatrix}
$$

### 심플렉틱 형식의 성질

1. **반대칭**: $\omega(\mathbf{u}, \mathbf{w}) = -\omega(\mathbf{w}, \mathbf{u})$
2. **찌부러지지 않음**: 모든 $\mathbf{w}$에 대해 $\omega(\mathbf{u}, \mathbf{w}) = 0$이면 $\mathbf{u} = \mathbf{0}$이다
3. **닫혀 있음**: $d\omega = 0$(바깥 미분이 사라진다)

### 심플렉틱 부피

심플렉틱 형식은 부피 형식을 이끌어 낸다:

$$
\omega^d = \omega \wedge \omega \wedge \cdots \wedge \omega = d! \, dx_1 \wedge dv_1 \wedge \cdots \wedge dx_d \wedge dv_d
$$

이는 (상수 배를 빼면) 리우빌 측도이다.

### 다르부 정리

**정리**: 어떤 심플렉틱 다양체든 그 자리에서는 좌표가 $(\mathbf{x}, \mathbf{v})$이고 심플렉틱 형식이 $\omega = \sum dv_i \wedge dx_i$인 표준 위상 공간처럼 보인다.

이 두루 통함 덕분에 과녁 분포가 무엇이든 표준 해밀턴 꼴이 통한다.

---

## 정준 바꿈

### 정의

매끄러운 사상 $\phi: \Gamma \to \Gamma$이 심플렉틱 형식을 지키면 **정준 바꿈**(또는 **심플렉틱 동형사상**)이라 한다:

$$
\phi^* \omega = \omega
$$

같은 말로 $\mathbf{J}_\phi$이 $\phi$의 야코비 행렬일 때 다음이 성립한다:

$$
\mathbf{J}_\phi^T \mathbf{J} \mathbf{J}_\phi = \mathbf{J}
$$

### 보기

**시간 흘러감**: 아무 해밀턴 체계의 흐름 $\phi_t$은 정준이다.

**운동량 크기 바꾸기**: $(\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, c\mathbf{v})$은 $c = \pm 1$일 때만 정준이다.

**선형 정준 바꿈**: $\mathbf{A}^T \mathbf{J} \mathbf{A} = \mathbf{J}$을 만족하는 아무 행렬 $\mathbf{A}$이 정준 바꿈 $\mathbf{z} \mapsto \mathbf{A}\mathbf{z}$을 정한다.

**운동량 뒤집기**: $(\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, -\mathbf{v})$은 정준이다(그리고 제 자신이 역사상이다).

### 정준 바꿈이 HMC에서 왜 중요한가

1. **개구리뜀 걸음**이 정준 바꿈이다
2. 개구리뜀 걸음의 **합성**도 정준이다
3. **부피 지킴**이 저절로 따라 나온다
4. MH 비에 **야코비 바로잡기**가 필요 없다

---

## 위상 공간의 자취

### 해밀턴 흐름

첫 조건 $(\mathbf{x}_0, \mathbf{v}_0)$이 주어지면 해밀턴 방정식이 오직 하나뿐인 자취 $(\mathbf{x}(t), \mathbf{v}(t))$을 정한다.

**해밀턴 자취의 성질**:

1. **정해짐**: 첫 조건이 주어지면 자취는 하나뿐이다
2. **되돌릴 수 있음**: 시간을 거꾸로 돌리고 운동량을 뒤집으면 같은 길을 되짚는다
3. **엇갈리지 않음**: 위상 공간에서 자취는 서로 가로지를 수 없다(풀이가 하나뿐이기 때문)
4. **에너지를 지킴**: 자취가 $\Sigma_{H(\mathbf{x}_0, \mathbf{v}_0)}$ 위에 머문다

### 궤도와 주기

묶인 움직임에서 자취는 **되풀이될**(시간 $T$ 뒤에 시작점으로 정확히 돌아옴) 수도, **거의 되풀이될**(원환면을 빽빽이 채우지만 결코 똑같이 되풀이하지는 않음) 수도, (적분할 수 없는 체계에서) **어지러울**(첫 조건에 예민하게 달림) 수도 있다.

가우스 과녁에서 HMC 자취는 되풀이되거나 거의 되풀이된다(적분할 수 있는 체계이다).

### 위상 그림

**위상 그림**은 위상 공간의 자취를 눈으로 보여 준다. 1차원 체계 $(x, v)$에서:

**조화 떨개**($U(x) = \frac{1}{2}kx^2$): 자취는 원점을 가운데로 하는 타원이다. 에너지가 높을수록 타원이 커진다.

**두 우물**($U(x) = (x^2 - 1)^2$): $x = \pm 1$에 안정된 평형이 둘, $x = 0$에 불안정한 평형이 하나 있다. 에너지가 낮은 자취는 우물 하나를 맴돌고, 에너지가 높은 자취는 둘을 다 넘나든다.

```
        v (momentum)
        ↑
        |     ╭─────╮
        |   ╭─┘     └─╮    Energy contours
        |  ╭┘  mode   └╮
        | ╭┘    •      └╮
    ────┼─┼─────────────┼────→ x (position)
        | ╰╮           ╭╯
        |  ╰╮         ╭╯
        |   ╰─╮     ╭─╯
        |     ╰─────╯
```

자취는 이 등고선(에너지가 일정한 선)을 따라간다. 봉우리는 한가운데 있고, 전형 집합은 에너지가 중간인 고리이다.

---

## 위상 공간의 전형 집합

### 측도의 집중

차원이 높으면 확률은 봉우리가 아니라 얇은 껍질에 몰린다. 결합 분포 $\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H)$에 대해:

**봉우리**: $\pi$이 가장 큰 곳은 $H$이 가장 작은 곳이며 보통 $\mathbf{x} = \mathbf{x}^*$(과녁의 봉우리)이고 $\mathbf{v} = \mathbf{0}$이다.

**전형 집합**: 확률의 대부분은 $H \approx \mathbb{E}[H] = \mathbb{E}[U] + \mathbb{E}[K]$인 곳에 놓인다.

$\mathbf{M} = \mathbf{I}$인 $d$차원 표준 가우스 과녁에서:

- $\mathbb{E}[U] = \frac{d}{2}$, $\mathbb{E}[K] = \frac{d}{2}$
- 전형 에너지: $H \approx d$
- 전형 집합은 $H = d$ 둘레에 두께가 $O(\sqrt{d})$인 에너지 껍질이다

### 표집에 뜻하는 바

**봉우리에서 시작하면 나쁘다**: $(\mathbf{x}^*, \mathbf{0})$의 표본은 전형에서 벗어나게 에너지가 낮다. 해밀턴 움직임이 살펴보기는 하겠지만 자취가 이 낮은 에너지 면에 머문다.

**운동량을 다시 표집하는 것이 꼭 필요하다**: $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$을 새로 뽑으면 운동 에너지를 그 주변 분포에서 표집하게 되어 전형 에너지 면을 살펴볼 수 있다.

---

## 쏘아 내림과 주변으로 만들기

### 위상 공간에서 자리 공간으로

HMC가 $\pi(\mathbf{x}, \mathbf{v})$에서 표본 $\{(\mathbf{x}^{(t)}, \mathbf{v}^{(t)})\}$을 만들고 나면 자리 표본 $\{\mathbf{x}^{(t)}\}$만 뽑아낸다.

**정리**: $(\mathbf{x}, \mathbf{v}) \sim \pi(\mathbf{x}, \mathbf{v})$이면 $\mathbf{x} \sim \pi(\mathbf{x})$이다.

**증명**: 만든 방식대로 $\pi(\mathbf{v}) = \mathcal{N}(\mathbf{0}, \mathbf{M})$일 때 $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \pi(\mathbf{v})$이다. $\mathbf{v}$을 적분해 없애면:

$$
\int \pi(\mathbf{x}, \mathbf{v}) \, d\mathbf{v} = \pi(\mathbf{x}) \int \pi(\mathbf{v}) \, d\mathbf{v} = \pi(\mathbf{x})
$$

### 운동량 버리기

운동량 표본은 버린다. 과녁 $\pi(\mathbf{x})$에 대한 정보를 담고 있지 않기 때문이다. 운동량이 하는 일은 순전히 도움 노릇이다. 곧 정해진 움직임을 가능하게 하고, 걸음 사이에 기울기 정보를 나르며, 한 방향으로 이어지는 살펴보기를 가능하게 한다.

---

## 유클리드가 아닌 과녁의 위상 공간

### 다양체 위의 자리

$\mathbf{x}$이 다양체 $\mathcal{M}$ 위에 있으면(이를테면 구면, 원환면, 양의 정부호 행렬) 위상 공간은 **여접다발** $T^*\mathcal{M}$이 된다.

점 $\mathbf{x} \in \mathcal{M}$마다 운동량 $\mathbf{v}$은 여접공간 $T^*_\mathbf{x}\mathcal{M}$에 있다.

### 리만 짜임

리만 HMC에서는 질량 행렬 $\mathbf{M}(\mathbf{x})$이 자리에 달려 있어 다양체 위의 거리 재는 법을 정한다. 운동 에너지는 다음이 된다:

$$
K(\mathbf{x}, \mathbf{v}) = \frac{1}{2}\mathbf{v}^T \mathbf{M}(\mathbf{x})^{-1} \mathbf{v}
$$

위상 공간 짜임이 더 복잡해진다. 곧 해밀턴 함수가 더 이상 나뉘지 않고, 표준 개구리뜀을 쓸 수 없으며, 넌지시 푸는 적분기나 넓힌 적분기가 필요하다.

---

## 요약

| 개념 | 정의 | 중요함 |
|---------|------------|------------|
| 위상 공간 | $\Gamma = \{(\mathbf{x}, \mathbf{v})\}$ | HMC 움직임의 마당 |
| 결합 분포 | $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v})$ | 인수로 나뉘고 주변 분포가 과녁이다 |
| 에너지 면 | $\Sigma_E = \{H = E\}$ | 자취가 여기에 갇힌다 |
| 심플렉틱 형식 | $\omega = \sum dv_i \wedge dx_i$ | 해밀턴 흐름이 지킨다 |
| 정준 바꿈 | $\omega$을 지킨다 | 부피를 지키고 야코비가 없다 |
| 전형 집합 | $H \approx \mathbb{E}[H]$ | 표본이 몰리는 곳 |

위상 공간의 기하는 HMC가 왜 되는지를 이해하는 수학의 바탕을 준다. 곧 심플렉틱 짜임이 부피 지킴을 보장하고, 에너지 면이 자취를 옭아매며, 인수로 나뉜 결합 분포 덕분에 주변으로 만들어 과녁을 쉽게 되찾을 수 있다.

---

## 참고 문헌

1. Arnold, V. I. (1989). *Mathematical Methods of Classical Mechanics*. Springer.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
4. da Silva, A. C. (2001). *Lectures on Symplectic Geometry*. Springer.

## 연습문제

1. **에너지 면 그려 보기**. $M = 1$인 1차원 가우스 과녁 $\pi(x) \propto \exp(-x^2/2)$에 대해 $(x, v)$ 위상 평면의 에너지 등고선을 그려라. 그 꼴은 무엇인가? 봉우리는 어디에 있는가? 전형 집합은 어디에 있는가?

2. **심플렉틱함 확인하기**. 사상 $(x, v) \mapsto (x \cos\theta + v\sin\theta, -x\sin\theta + v\cos\theta)$이 정준임을 보여라. 어떤 해밀턴 함수가 이 흐름을 낳는가?

3. **전형 집합 셈하기**. $\mathbf{M} = \mathbf{I}$인 $d$차원 표준 가우스 과녁에 대해 $\mathbb{E}[H]$과 $\text{Var}[H]$을 셈하여라. 전형 집합의 두께는 $d$에 따라 어떻게 커지는가?

4. **인수로 나뉘지 않는 결합 분포**. $\mathbf{M}$이 $\mathbf{x}$에 달린 $\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-U(\mathbf{x}) - \frac{1}{2}\mathbf{v}^T\mathbf{M}(\mathbf{x})^{-1}\mathbf{v})$을 썼다고 하자. 결합 분포가 여전히 인수로 나뉘는가? $\mathbf{x}$에 대한 주변 분포는 무엇인가?

---
