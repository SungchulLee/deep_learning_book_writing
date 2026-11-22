# 신뢰 영역 폴리시 최적화(TRPO)

**신뢰 영역 폴리시 최적화**(TRPO; 슐먼 외, 2015)는 폴리시 경사 방법의 근본 어려움, 곧 갱신 스텝을 얼마나 크게 잡아야 하는가를 다룬다. 스텝이 너무 크면 폴리시가 파국적으로 무너지고, 너무 작으면 샘플과 계산을 낭비한다. TRPO 는 옛 폴리시와 새 폴리시 사이의 쿨백-라이블러(KL) 발산 — 두 확률 분포가 얼마나 다른지 재는 양 — 이 정해 둔 값을 넘지 않는 범위, 곧 **신뢰 영역** 안에서만 갱신하도록 제약 최적화 문제를 세워 이를 푼다.

---

## 1. 동기

일반 폴리시 경사 갱신 $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ 는 폴리시가 얼마나 바뀌는지에 아무 보장이 없다. 파라미터 $\theta$ 의 작은 변화가, 특히 민감한 영역에서는 폴리시 $\pi_\theta$ 의 큰 변화를 낳을 수 있다.

TRPO 는 갱신마다 폴리시 분포가 얼마나 바뀔 수 있는지를 제한한다. 출발점은 그렇게 하면 성능이 단조롭게 개선된다는 이론이다 — 다만 아래 그림 상자가 보이듯 실제 알고리즘은 그 이론을 그대로 풀지 않는다.

---

## 2. 이론적 기초

### 대리 목표

TRPO 는 옛 폴리시 근처에서 실제 성능을 근사하는 대리 목표를 최대화한다.

$$L_{\pi_{\theta_\text{옛}}}(\theta) = \mathbb{E}_{s \sim d^{\pi_{\theta_\text{옛}}}, a \sim \pi_{\theta_\text{옛}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{옛}}(a|s)} A^{\pi_{\theta_\text{옛}}}(s, a)\right]$$

중요도 샘플링을 쓰는 이 목표는 옛 폴리시 $\pi_{\theta_\text{옛}}$의 데이터로 새 폴리시 $\pi_\theta$ 를 평가할 수 있게 한다. **중요도 샘플링**은 다른 분포에서 뽑은 표본에 확률의 비를 곱해 원하는 분포의 기댓값을 추정하는 기법이다.

### 단조로운 개선 정리

카케이드와 랭퍼드(2002)의 결과를 슐먼 외(2015)가 일반 확률 폴리시로 넓혀, 실제 성능이 다음을 만족함을 보였다. 오른쪽 전체가 하한이므로, 이 하한을 키우면 실제 성능도 그만큼은 오른다.

$$J(\pi_\theta) \geq L_{\pi_{\theta_\text{옛}}}(\theta) - C \cdot D_\text{KL}^{\max}(\pi_{\theta_\text{옛}} \| \pi_\theta)$$

여기서 $C = \frac{4\gamma \epsilon}{(1-\gamma)^2}$이고 $\epsilon = \max_{s,a} |A^{\pi}(s, a)|$이다.

### TRPO 최적화 문제

TRPO는 벌점을 하드 제약으로 대체한다.

$$\max_\theta \quad L_{\pi_{\theta_\text{옛}}}(\theta)$$

$$\text{단} \quad \bar{D}_\text{KL}(\pi_{\theta_\text{옛}} \| \pi_\theta) \leq \delta$$

여기서 $\bar{D}_\text{KL}$ 은 상태에 걸친 평균 쿨백-라이블러 발산이고 $\delta$ 는 신뢰 영역 반지름(흔히 $0.01$)이다.

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 정리에서 알고리즘으로 가는 길에 난 두 자리

위 두 절을 나란히 읽으면 조용히 바뀐 데가 둘이다. 정리는 **벌점**으로 적혀 있는데 알고리즘은 **하드 제약**을 쓰고, 정리는 **모든 상태에서의 최대** KL을 말하는데 알고리즘은 **평균** KL을 쓴다. 눈여겨볼 것은 이 바꿔 쓰기를 뒷사람이 지적한 것이 아니라 **TRPO 논문이 스스로 적어 두었다**는 점이다.

첫째에 대해 — "In practice, if we used the penalty coefficient $C$ recommended by the theory above, the step sizes would be **very small**." 그래서 계수를 줄이고 싶은데 "it is hard to robustly choose the penalty coefficient, so we use a hard constraint instead of a penalty, with parameter $\delta$." 이론이 준 $C$ 를 그대로 쓰면 걸음이 너무 작아 쓸 수 없더라는 것이다.

둘째에 대해 — 상태마다 제약을 하나씩 두는 문제는 "impractical to solve due to the large number of constraints"이고, 그래서 평균을 쓰는 것을 논문은 **"a heuristic approximation"** 이라고 부른다. 어림이라고 제 입으로 적은 것이다.

![왼쪽은 정리가 요구하는 것과 알고리즘이 실제로 하는 것을 화살표로 짝지은 그림이다. KL 에 대한 벌점이 델타를 쓰는 하드 제약으로, 모든 상태에서의 KL 이 평균 KL 로 바뀌며 각각 그 까닭이 아래에 적혀 있다. 오른쪽은 가로축이 상태 수인 로그 그래프로, 가장 큰 KL 을 평균 KL 로 나눈 값이 상태 열 개에서 1.5배, 백 개에서 4.0배, 천 개에서 5.6배, 만 개에서 7.7배로 꾸준히 커진다.](figures/theory_to_algorithm.svg)

둘째 바꿔 쓰기가 얼마나 느슨한지는 재어 볼 수 있다. 상태마다 폴리시를 한 스텝 바꿔 놓고 **가장 큰 KL**을 **평균 KL**로 나눠 보면, 상태가 열 개일 때 1.5배인 것이 만 개가 되면 **7.7배**가 된다. 평균 KL을 흔히 쓰는 $\delta = 0.01$ 에 맞춰 놓아도 어떤 상태에서는 그보다 **여섯 배 넘게** 큰 변화가 일어난다 — 상태 천 개에서 시드 열 개를 돌려 중앙값 6.1배였다.

**그리고 벌어지는 방향이 하필 나쁜 쪽이다.** 상태가 많을수록 차이가 커지는데, 신뢰 영역이 필요한 문제가 바로 상태가 많은 문제다. 작은 문제에서 "평균이나 최대나 비슷하다"고 확인하고 큰 문제로 옮기면, 확인한 성질이 함께 따라오지 않는다.

**그렇다고 TRPO가 틀린 것은 아니다.** 논문은 최대 KL 버전을 따로 돌려 보고 "the average KL divergence constraint has a similar effect as the theoretically justified maximum KL divergence"라고 적어 두었다 — 다만 그 확인은 카트폴에서만 할 수 있었다고 함께 밝힌다. 요점은 이것이다. 아래 3절부터 보게 될 켤레 그래디언트도 백트래킹 선 탐색도 **정리가 보증한 그 문제를 푸는 것이 아니라, 두 번 갈아 끼운 뒤의 문제를 푼다.** 단조로운 개선이 수학으로 보장된다고 읽으면 안 되는 까닭이 그것이다.

_확인한 곳 — J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel, "Trust Region Policy Optimization", ICML 2015 (arXiv:1502.05477). 벌점 계수를 쓰면 걸음이 너무 작다는 말과 하드 제약으로 바꾼다는 말은 4절과 5절의 요약, 평균 KL을 "heuristic approximation"이라 부른 것은 4절, 최대 KL 버전과 비슷하더라는 실험은 8절이다. 단조로운 개선 경계 자체는 S. Kakade, J. Langford, "Approximately Optimal Approximate Reinforcement Learning" (2002) 에서 온다. 오른쪽 그림의 수는 그림을 그리는 코드가 시드를 박고 직접 재어 얻는다._

</div>

---

## 3. 알고리즘

### 켤레 그래디언트 방법

TRPO 는 대리 목표를 일차, KL 제약을 이차로 근사한 뒤 다음 갱신을 계산한다. 방향 $\hat F^{-1}\hat g$ 는 켤레 그래디언트 알고리즘 — 선형 연립방정식 $\hat F x = \hat g$ 를 행렬-벡터 곱만으로 되풀이해 푸는 방법 — 으로 근사한다.

$$\theta_{k+1} = \theta_k + \alpha \hat{F}^{-1} \hat{g}$$

여기서:

- $\hat{g} = \nabla_\theta L(\theta)|_{\theta_k}$ 는 폴리시 경사다
- $\hat{F}$ 는 피셔 정보 행렬(쿨백-라이블러 발산의 헤세 행렬)이다
- $\alpha$ 는 쿨백-라이블러 제약을 채우도록 백트래킹 선 탐색으로 정한다

### 피셔-벡터 곱

$\hat{F}^{-1} g$ 를 직접 계산하는 일은 큰 네트워크에서 다룰 수 없다 — 파라미터가 $n$ 개면 $\hat F$ 가 $n\times n$ 행렬이다. 그 대신 켤레 그래디언트 방법은 피셔-벡터 곱 $\hat{F}v$만을 필요로 하며 이는 다음으로 값싸게 계산한다.

$$\hat{F}v = \nabla_\theta \left[(\nabla_\theta \bar{D}_\text{KL})^\top v\right]$$

이는 역전파 두 번이면 되므로 행렬을 명시적으로 만들지 않아도 된다.

### 백트래킹 선 탐색

탐색 방향 $s = \hat{F}^{-1}g$ 를 계산한 뒤 TRPO는 스텝 크기를 정한다.

1. 가장 큰 스텝을 계산한다: $\alpha_\text{최대} = \sqrt{\frac{2\delta}{s^\top \hat{F} s}}$
2. $k = 0, 1, 2, \ldots$에 대해 $\alpha = \alpha_\text{최대} \cdot \beta^k$ 를 시도한다
3. (a) 쿨백-라이블러 제약과 (b) 대리 목표의 개선을 함께 채우는 첫 $\alpha$ 를 받아들인다

이차 근사로 정한 $\alpha_\text{최대}$ 는 실제 KL 을 넘길 수 있으므로, 선 탐색이 실제 KL 을 다시 재어 확인한다.

---

## 4. TRPO 의사코드

```
되풀이마다:
    π_θ으로 자취를 모은다
    이점을 셈한다(GAE)

    폴리시 그래디언트 g = ∇_θ L(θ)을 셈한다
    켤레 그래디언트로 찾는 쪽 s = F⁻¹g을 셈한다
    가장 큰 스텝 크기를 셈한다: α_최대 = sqrt(2δ / s^T F s)

    되짚어 가는 줄 찾기:
        for k = 0, 1, 2, ..., K:
            θ_새 = θ + α_최대 · β^k · s
            if KL(π_옛 || π_새) ≤ δ and L(θ_새) > L(θ):
                θ_새를 받아들인다
                멈춘다

    값 함수를 고친다(따로 둔 가장 좋게 하는 개)
```

---

## 5. 성질

### 장점
- 이론의 벌점 형태를 정확히 풀면 단조로운 개선이 보장된다(실제 알고리즘은 근사다)
- 파라미터 선택에 강건하다
- 여러 둘레에 걸쳐 트레인이 강건하다
- 러닝레이트를 튜닝할 필요가 없다

### 단점
- 구현이 복잡하다(켤레 그래디언트, 선 탐색)
- 되풀이마다 계산이 비싸다
- 아주 큰 네트워크에서는 잘 확장되지 않는다
- 이차 계산의 오버헤드가 있다

---

## 6. 파라미터

| 파라미터 | 흔한 값 | 뜻 |
|-----------|--------------|-------------|
| $\delta$ | 0.01 | 쿨백-라이블러 제약(신뢰 영역 크기) |
| 켤레 그래디언트 반복 | 10 | 켤레 그래디언트 스텝 |
| 켤레 그래디언트 감쇠 | 0.1 | 피셔 행렬 감쇠(damping) |
| 선 탐색 스텝 | 10 | 백트래킹 시도 |
| 백트래킹 비율 $\beta$ | 0.5 | 스텝 크기 감쇠 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이 절에서 다룬 방법의 폴리시 경사를 유도하라. 어느 항이 추정해야 하는 것이고 어느 항이 정확히 계산할 수 있는 것인지 분명히 밝혀라.

</div>

??? success "풀이"
    폴리시 경사는 $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ 형태이며, 여기서 $\hat{A}_t$ 는 어드밴티지 추정값이다. 로그 확률 그래디언트 $\nabla_\theta \log \pi_\theta$ 는 자동 미분으로 정확히 계산할 수 있다. 어드밴티지 $\hat{A}_t$ 는 샘플링한 궤적에서 추정해야 하므로 분산이 들어온다. 기댓값은 궤적 배치에 걸친 평균으로 추정한다. TRPO 의 대리 목표 $L(\theta)$ 를 $\theta_\text{old}$ 에서 미분하면 $\nabla_\theta \frac{\pi_\theta}{\pi_{\theta_\text{old}}}\big|_{\theta_\text{old}} = \nabla_\theta\log\pi_\theta$ 이므로 이 폴리시 경사와 같아진다 — 그래서 $\hat g$ 를 대리 목표의 그래디언트로 계산해도 된다. 피셔 행렬 $\hat F$ 도 같은 표본으로 추정한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
KL 제약을 이차로 근사해 $\bar D_\text{KL}(\theta_k \,\|\, \theta_k + \alpha s) \approx \tfrac12 \alpha^2 s^\top \hat F s$ 로 두고, 이것이 $\delta$ 와 같아지는 $\alpha$ 를 구해 3절의 $\alpha_\text{최대}$ 를 유도하라. 근사가 틀리면 무슨 일이 생기며, 선 탐색이 그것을 어떻게 막는가?

</div>

??? success "풀이"
    $\tfrac12\alpha^2 s^\top\hat Fs = \delta$ 를 풀면 $\alpha = \sqrt{2\delta/(s^\top\hat F s)}$ 다. KL 은 $\theta_k$ 에서 값과 그래디언트가 0 이므로 이차 항이 첫 근사가 된다. 그러나 스텝이 크면 삼차 이상의 항이 무시할 수 없어 실제 KL 이 $\delta$ 를 넘을 수 있고, 대리 목표의 일차 근사도 틀려 실제로는 목표가 줄어들 수 있다. 백트래킹 선 탐색은 $\alpha_\text{최대}$ 에서 시작해 $\beta$ 배씩 줄이며 **실제** KL 과 **실제** 대리 목표를 다시 계산해, 둘 다 만족하는 첫 스텝만 받아들인다. 끝내 못 찾으면 그 갱신을 건너뛴다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
3절의 피셔-벡터 곱 $\hat Fv = \nabla_\theta[(\nabla_\theta\bar D_\text{KL})^\top v]$ 가 왜 $\hat F$ 와 $v$ 의 곱인지 설명하라. 이 식을 $\theta = \theta_\text{old}$ 에서 계산해야 하는 까닭은 무엇인가? 6절 표의 「켤레 그래디언트 감쇠 0.1」은 무엇을 하는가?

</div>

??? success "풀이"
    $(\nabla_\theta\bar D_\text{KL})^\top v$ 는 스칼라이고, 그것을 다시 미분하면 $v$ 가 $\theta$ 에 의존하지 않으므로 $(\nabla^2_\theta\bar D_\text{KL})\,v$, 곧 헤세 행렬과 $v$ 의 곱이 된다. 역전파 두 번이면 되고 $n\times n$ 행렬을 만들 필요가 없다. KL 의 헤세 행렬이 피셔 정보 행렬과 같은 것은 $\theta = \theta_\text{old}$ 에서만이다. 그 점에서 KL 은 최솟값 0 을 가지므로 그래디언트가 0 이고, 헤세 행렬이 $\mathbb E[\nabla\log\pi\,\nabla\log\pi^\top]$ 가 된다. 감쇠(damping)는 $\hat F$ 대신 $\hat F + 0.1\,I$ 를 쓰는 것이다. 표본으로 추정한 $\hat F$ 는 특이에 가깝거나 잡음이 커서 켤레 그래디언트가 아주 큰 스텝을 내놓을 수 있는데, $0.1\,I$ 를 더하면 고윳값이 0.1 아래로 내려가지 않아 계산이 안정된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
2절의 벌점 계수 $C = 2\gamma\epsilon/(1-\gamma)^2$ 을 $\gamma = 0.99$, $\epsilon = 1$ 로 계산하라. 대리 목표를 0.1 개선하는 스텝이 하한을 실제로 올리려면 최대 KL 이 얼마 아래여야 하는가? 그림 상자의 「이론이 준 $C$ 를 쓰면 걸음이 너무 작다」는 말과 견주어 보라.

</div>

??? success "풀이"
    $C = 2\cdot0.99/0.01^2 = 19{,}800$ 이다. 하한 $L - C\,D_\text{KL}^{\max}$ 이 옛 폴리시의 값보다 커지려면 $C\,D_\text{KL}^{\max} < 0.1$, 곧 $D_\text{KL}^{\max} < 0.1/19{,}800 \approx 5\times10^{-6}$ 이어야 한다. 흔히 쓰는 $\delta = 0.01$ 보다 2,000 배쯤 작은 신뢰 영역이다. TRPO 논문의 정리는 상수를 $4\gamma\epsilon/(1-\gamma)^2$ 로 적으므로 그 경우 영역은 다시 절반이 된다. 할인이 1 에 가까울수록 $(1-\gamma)^2$ 때문에 $C$ 가 폭발적으로 커져서, 이론을 그대로 따르면 폴리시가 거의 움직이지 못한다. TRPO 가 벌점을 버리고 $\delta$ 를 직접 고르는 하드 제약으로 바꾼 까닭이 이 수에 있다. $\square$

## 정리하며

TRPO 는 KL 발산으로 갱신의 크기를 묶는 신뢰 영역으로, 폴리시 경사에 스텝 크기를 정하는 원리를 준다. 이론은 벌점 형태의 하한으로 단조로운 개선을 보장하지만, 실제 알고리즘은 그것을 평균 KL 의 하드 제약으로 바꾸고 켤레 그래디언트와 선 탐색으로 근사해 푼다. 구현이 더 쉬운 PPO 가 실제로는 TRPO 를 거의 대체했으나, 신뢰 영역이라는 생각은 여전히 그 바탕에 있다.
