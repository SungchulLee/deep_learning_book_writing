# 모멘텀을 쓰는 SGD

모멘텀은 지난 그래디언트의 지수 감쇠 이동 평균을 쌓아, 그래디언트가 한결같은 방향으로는 수렴을 빠르게 하고 곡률이 큰 방향의 진동은 누그러뜨린다.

---

## 1. 갱신 규칙

$$v_t = \mu \, v_{t-1} + g_t$$

$$\theta_{t+1} = \theta_t - \eta \, v_t$$

여기서 $\mu \in [0, 1)$은 모멘텀 계수(보통 0.9)이고 $v_t$은 속도(누적된 그래디언트)이다.

---

## 2. 직관

길쭉한 골짜기 모양의 로스 지형을 생각해 보자. 기본 SGD는 좁은 방향으로 진동하면서 긴 방향으로는 더디게 나아간다. 모멘텀은 한결같은(긴) 방향으로 속도를 쌓고 좁은 방향의 진동은 서로 상쇄시킨다. 관성을 지닌 공이 비탈을 굴러 내려가는 것과 같다.

---

## 3. PyTorch 구현

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 4. 모멘텀 계수의 효과

- $\mu = 0$: 기본 SGD가 된다.
- $\mu = 0.9$: 표준적인 선택. 그래디언트가 일정하고 초기 속도가 0일 때 속도는 극한에서 현재 그래디언트의 $1/(1-\mu)=10$배가 된다. 임의의 경사열에 대한 현재 경사 대비 상한은 아니다.
- $\mu = 0.99$: 더 강한 평활화. 그래디언트의 잡음이 아주 클 때 쓸모 있다.

---

## 5. 현재 경사 기여 조절(dampening)

PyTorch는 현재 그래디언트의 몫을 줄이는 dampening을 지원한다.

$$v_t = \mu \, v_{t-1} + (1 - d) \, g_t$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, dampening=0.1)
```

---

## 6. 핵심 정리

- 모멘텀은 그래디언트의 이력을 쌓아 수렴을 빠르게 한다.
- 표준 모멘텀 계수는 0.9이다.
- 모멘텀을 쓰는 SGD는 여러 컴퓨터 비전 과제의 기본 옵티마이저이며, 잘 조율하면 최종 일반화에서 적응형 방법과 맞먹거나 그보다 나을 때가 많다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
모멘텀의 속도를 과거 경사의 가중합으로 나타내라. 수렴 가속이 항상 보장되는가?

</div>

??? success "연습문제 1 풀이"
    $v_0=0$이면 $v_t=\sum_{i=1}^t\mu^{t-i}g_i$이다. 일관된 경사가 누적되고 반대 방향 성분이 상쇄될 수 있다. 특정 이차 문제와 조율 조건에서는 가속을 분석할 수 있지만 임의의 비볼록 로스에서 일정한 속도 향상을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
모든 단계에서 경사가 같은 벡터 g일 때 속도와 극한 갱신을 구하라.

</div>

??? success "연습문제 2 풀이"
    $v_t=(1-\mu^t)g/(1-\mu)$이고 극한 갱신은 $-\eta g/(1-\mu)$이다. 현재 경사가 변하면 속도는 과거의 합이므로 단일 실효 러닝레이트로 모든 좌표와 시점을 설명할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
고전 모멘텀과 네스테로프의 비교에서 O(1/t²) 주장의 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"
    고전 모멘텀은 현재점의 경사를 이용하고 네스테로프는 적절히 재표현한 미리 보기의 경사를 이용한다. 매끄러운 볼록 함수의 가속 이론은 러닝레이트와 시간에 따른 모멘텀 규칙을 포함한다. 고정 momentum=0.9를 설정했다고 모든 목적함수에 O(1/t²)가 보장되지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
모멘텀 SGD를 바닥부터 구현하고 `torch.optim.SGD(momentum=0.9)`과 맞는지 확인하라.

</div>

??? success "연습문제 4 풀이"
    ```python
    v = {p: torch.zeros_like(p) for p in model.parameters()}
    for p in model.parameters():
        v[p] = 0.9 * v[p] + p.grad
        p.data -= lr * v[p]
    ```

## 정리하며

모멘텀은 과거 경사의 지수 감쇠 합을 갱신에 사용한다. 일정 경사에서의 극한 배율과 임의 학습의 실제 스트라이드를 구분하고, 모멘텀·러닝레이트·dampening을 함께 맞춘다. PyTorch는 첫 버퍼를 첫 경사로 초기화하므로 dampening을 포함한 구현 비교에는 초기 단계도 확인한다.
