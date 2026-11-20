# 34.3.4 PPO 구현

이 절은 "37가지 구현 세부 사항"의 모범 사례(황 외, 2022)을 따르는 완전한 프로덕션 품질의 PPO 구현을 준다. 벡터 둘레, 올바른 어드밴티지 계산, 전체 학습 루프, 그리고 성능에 크게 영향을 주는 중요한 구현 세부 사항을 다룬다.

---

## 1. 중요한 구현 세부 사항

### 1. 벡터 둘레
둘레 사본 $N$개를 병렬로 돌려 다양한 데이터를 값싸게 모은다. 에피소드가 끝나면 둘레마다 자동으로 리셋된다.

### 2. 어드밴티지 정규화
그래디언트 크기를 안정적으로 하려고 (온 배치가 아니라) 미니배치 수준에서 어드밴티지를 정규화한다.

$$\hat{A}_t = \frac{A_t - \mu_\text{작은배치}}{\sigma_\text{작은배치} + \epsilon}$$

### 3. 밸류 함수 자르기(선택 사항)
값이 크게 갱신되는 것을 막으려 값 예측을 자른다.

$$V_\text{자름} = V_\text{옛} + \text{clip}(V_\theta - V_\text{옛}, -\epsilon, \epsilon)$$

$$L_V = \max\left[(V_\theta - V_\text{타깃})^2, (V_\text{자름} - V_\text{타깃})^2\right]$$

### 4. 러닝레이트 어닐링
학습하는 동안 러닝레이트를 0까지 선형으로 줄이면 마지막 성능이 나아진다.

### 5. 전역 그래디언트 클리핑
(파라미터마다가 아니라) 전역 그래디언트 노름을 0.5으로 자른다.

### 6. 직교 초기화
웨이트에 직교 행렬로 초기화한다.

- 은닉 레이어: 이득 $\sqrt{2}$
- 폴리시 헤드: 이득 $0.01$
- 값 헤드: 이득 $1.0$

### 7. 관측 정규화
이동 평균/분산으로 관측을 정규화하고 $[-10, 10]$으로 자른다.

### 8. 리워드 정규화
리워드 스케일이 서로 다른 것을 다루려 (평균이 아니라) 이동 표준편차로 리워드를 정규화한다.

---

## 2. 학습 재귀 구조

```
온 고침 수 = total_timesteps / (n_envs × n_steps)

고침마다:
    # 굴림 마디
    for t = 0, ..., n_steps-1:
        action, logprob, value = agent(obs)
        obs, reward, done = envs.step(action)
        store(obs, action, reward, done, logprob, value)

    # 마지막 값 부트스트랩
    last_value = agent.get_value(last_obs)

    # GAE 셈하기
    advantages, returns = compute_gae(rewards, values, dones, last_value)

    # 배치 납작하게 펴기
    batch = flatten(rollout)  # 형태: (n_envs × n_steps, ...)

    # 가장 좋게 하기 마디
    for epoch = 1, ..., n_epochs:
        indices = random_permutation(batch_size)
        각 작은 배치 번호에 대해:
            ratio = exp(new_logprob - old_logprob)
            clipped_loss = min(ratio × adv, clip(ratio) × adv)
            value_loss = MSE(value, returns)
            entropy = policy.entropy()
            loss = -clip_loss + c_v × value_loss - c_e × entropy
            optimizer.step(loss)

    # 러닝레이트 식히기
```

---

## 3. 성능 기준

올바르게 구현하면 PPO는 다음을 이룬다.

- **CartPole-v1**: 10만 스텝 안에 풂(리워드 약 500)
- **LunarLander-v2**: 약 50만 스텝에 풂(리워드 약 200)
- **아타리(Breakout)**: 약 1000만 프레임에 400점 이상

---

## 4. 흔히 빠지는 구현 함정

1. **작은 배치마다 어드밴티지를 정규화하지 않기**: 학습이 불안정해진다
2. **로그 확률을 잘못 계산**: 연속 액션에서는 액션 차원에 걸쳐 로그 확률을 더해야 한다
3. **옛 로그 확률을 분리(detach)하기를 잊기**: 옛 로그 확률에 그래디언트가 붙으면 안 된다
4. **자동 리셋을 빠뜨리기**: 둘레는 끝날 때 자동으로 리셋되어야 한다
5. **에피소드 경계에서 GAE를 잘못 다루기**: done 플래그로 가려야 한다
6. **러닝레이트를 어닐링하지 않기**: 어닐링하지 않으면 마지막 성능이 나빠진다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이 절에서 다룬 방법의 폴리시 경사를 유도하라. 어느 항이 추정해야 하는 것이고 어느 항이 정확히 계산할 수 있는 것인지 분명히 밝혀라.

</div>

??? success "풀이"
    폴리시 경사는 $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ 형태이며, 여기서 $\hat{A}_t$은 어드밴티지 추정값이다. 로그 확률 그래디언트 $\nabla_\theta \log \pi_\theta$은 자동 미분으로 정확히 계산할 수 있다. 어드밴티지 $\hat{A}_t$은 샘플링한 궤적에서 추정해야 하므로 분산이 들어온다. 기댓값은 궤적 배치에 걸친 평균으로 추정한다. 베이스라인으로 분산을 줄이면 불편성을 지키면서 추정 잡음을 줄인다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
연속 제어 과제에서 이 방법의 샘플 효율을 밸류 기반 방법(예를 들어 DQN)과 비교하여라. 관찰되는 차이의 이론적 까닭을 설명하라.

</div>

??? success "풀이"
    폴리시 기반 방법은 온폴리시 데이터를 쓰므로(궤적마다 한 번씩 쓴다) 밸류 기반 방법보다 샘플 효율이 대체로 낮다. DQN은 경험 리플레이로 데이터를 재사용해서 더 나은 샘플 효율을 얻는다. 그러나 폴리시 방법은 연속 액션을 자연스럽게 다루고(액션 공간에 대한 argmax가 필요 없다), 최적 폴리시가 확률적 폴리시일 때 그리로 수렴하며, 신뢰 영역 아래에서 단조로운 개선을 보장한다. 오프폴리시 Actor-Critic 방법(DDPG, SAC)은 폴리시 최적화와 경험 리플레이를 엮어 이 간극을 메운다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
간단한 연속 제어 과제(예를 들어 Pendulum-v1)에 이 방법을 구현하라. 러닝레이트와 이 방법 고유의 중요한 하이퍼파라미터에 대해 얼마나 민감한지 보고하라.

</div>

??? success "풀이"
    가우스 폴리시를 쓰는 Pendulum-v1에서 전형적인 결과는 이렇다. 러닝레이트 $3 \times 10^{-4}$이면 약 500 에피소드에 수렴하고, $10^{-3}$이면 진동하며, $10^{-5}$이면 너무 더디게 수렴한다. 이 방법 고유의 하이퍼파라미터(예를 들어 PPO의 클리핑 범위, TRPO의 쿨백-라이블러 제약)는 갱신의 공격성과 안정성 사이의 트레이드오프를 조절한다. 너무 공격적이면 성능이 무너지고 너무 보수적이면 샘플을 낭비한다. 최적 지점은 이 둘의 균형을 맞추는 곳이며 흔히 좁은 범위에서 격자 탐색으로 찾는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
액션 공간이 단체(합이 1인 자본 웨이트)이고 리워드가 위험 조정 리턴인 포트폴리오 배분 최적화에 이 방법을 어떻게 쓸 수 있을지 논하라.

</div>

??? success "풀이"
    액션 공간은 $(n-1)$차원 단체 $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$이다. 폴리시는 디리클레 분포나 소프트맥스로 변환한 가우스 분포를 쓸 수 있다. 리워드는 그 포트폴리오 배분의 샤프 비율이나 미분 샤프 비율이다. 어려움으로는 고차원 액션 공간(자산이 많다), 잦은 리밸런싱에 벌점을 주는 거래 비용, 시장 리턴의 비정상성이 있다. 이 절의 방법은 폴리시를 안정적으로 갱신하는 자체 장치로 이를 다룬다. $\square$

## 정리하며

올바른 PPO 구현은 핵심인 잘라 낸 목표 말고도 많은 세부 사항을 꼼꼼히 살펴야 한다. 벡터 둘레, GAE, 적절한 정규화, 러닝레이트 스케줄의 결합이 강건하고 효율적인 학습 시스템을 만든다.
