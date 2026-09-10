# RMSprop

RMSprop(제곱평균제곱근 전파)은 Adagrad의 학습률 감쇠 문제를 바로잡으려고 제프리 힌턴이 제안했다. 기울기 제곱의 누적합 대신 지수 감쇠 평균을 쓴다.

---

## 1. 갱신 규칙

$$s_t = \alpha \, s_{t-1} + (1 - \alpha) \, g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \, g_t$$

여기서 $\alpha$(보통 0.99)은 기울기 제곱의 이동 평균에 대한 감쇠율이다.

---

## 2. PyTorch 구현

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, eps=1e-8)

# 모멘텀과 함께
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, momentum=0.9)
```

---

## 3. Adagrad와의 비교

핵심 차이는 기울기 이력을 보는 창의 크기이다.

- **Adagrad**: $s_t = \sum_{i=1}^t g_i^2$ (이력 전체, 단조 증가)
- **RMSprop**: $s_t = \alpha s_{t-1} + (1-\alpha) g_t^2$ (지수 가중, 유계)

RMSprop의 분모는 유계이므로 학습률이 0으로 줄지 않아 오랜 학습 동안에도 계속 배울 수 있다.

---

## 4. 중심화한 기울기를 쓸 때

중심화한 RMSprop은 제곱하기 전에 평균 기울기를 빼서 이차 모멘트가 아니라 분산을 쓴다.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, centered=True)
```

$$\bar{g}_t = \alpha \, \bar{g}_{t-1} + (1 - \alpha) \, g_t$$

$$s_t = \alpha \, s_{t-1} + (1 - \alpha) \, g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t - \bar{g}_t^2} + \epsilon} \, g_t$$

---

## 5. 언제 쓰는가

RMSprop은 Adam 이전의 기본 적응형 최적화기였다. Adam이 이따금 불안정할 수 있는 순환 신경망과 강화 학습에서는 여전히 쓸모 있다.

---

## 6. 핵심 정리

- RMSprop은 기울기 제곱에 지수 감쇠를 써서 Adagrad의 학습률 붕괴를 바로잡는다.
- 표준 기본값은 $\alpha = 0.99$, $\eta = 0.001$이다.
- 대체로 Adam에 자리를 내주었지만 RNN과 강화 학습에서는 여전히 쓸모 있다.

---

## 연습문제

**연습문제 1.**
RMSprop의 갱신 규칙을 유도하고 학습률을 어떻게 맞추는지 설명하라.

??? success "연습문제 1 풀이"
    RMSprop은 $v_t = \beta v_{t-1} + (1-\beta)g_t^2$, $\theta_{t+1} = \theta_t - \eta g_t / \sqrt{v_t + \epsilon}$이다. 기울기가 큰 매개변수는 실효 학습률이 작아지고($v$이 크면 $\eta/\sqrt{v}$이 작다) 그 반대도 마찬가지이다. 이렇게 손실 곡면의 국소적인 기하에 맞춘다.

---

**연습문제 2.**
RMSprop이 기울기 제곱의 지수 가중 이동 평균임을 보여라.

??? success "연습문제 2 풀이"
    펼치면 $v_t = (1-\beta)\sum_{i=0}^{t-1} \beta^i g_{t-i}^2$이다. 최근 기울기는 가중치 $(1-\beta)$을 갖고 오래된 것은 $\beta^i$으로 줄어든다. 실효 창의 크기는 $1/(1-\beta)$이며, $\beta=0.99$이면 최근 100개의 기울기가 지배한다.

---

**연습문제 3.**
RMSprop이 Adagrad를 개선하려고 나온 이유를 설명하라.

??? success "연습문제 3 풀이"
    Adagrad는 지난 기울기 제곱을 모두 누적하여 $v_t = \sum_{i=1}^t g_i^2$을 쓰는데, 이 값이 단조 증가하여 결국 학습률이 사라질 만큼 작아진다. RMSprop은 지수 감쇠를 써서 $v_t$을 유계로 유지하므로 학습 내내 계속 배울 수 있다.

---

**연습문제 4.**
RMSprop과 Adam을 비교하라. Adam은 RMSprop에 무엇을 더하는가?

??? success "연습문제 4 풀이"
    Adam = RMSprop + 모멘텀 + 편향 보정이다. RMSprop은 학습률을 맞추지만 방향에는 날 기울기를 쓴다. Adam은 갱신을 더 매끄럽게 하려고 모멘텀 항(기울기의 지수 평균)을, 초반 추정을 정확히 하려고 편향 보정을 더한다.

## 정리하며

이 마당은 갱신 규칙、PyTorch 구현、Adagrad와의 비교、중심화한 기울기를 쓸 때을 차례로 짚었다.
