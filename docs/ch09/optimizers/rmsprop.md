# RMSprop

RMSprop(제곱평균제곱근 전파)은 Adagrad의 러닝레이트 감쇠 문제를 바로잡으려고 제프리 힌턴이 제안했다. 그래디언트 제곱의 누적합 대신 지수 감쇠 평균을 쓴다.

---

## 1. 갱신 규칙

$$s_t = \alpha \, s_{t-1} + (1 - \alpha) \, g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \, g_t$$

여기서 $\alpha$(보통 0.99)은 그래디언트 제곱의 이동 평균에 대한 감쇠율이다.

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

핵심 차이는 그래디언트 이력을 보는 창의 크기이다.

- **Adagrad**: $s_t = \sum_{i=1}^t g_i^2$ (이력 전체, 단조 증가)
- **RMSprop**: $s_t = \alpha s_{t-1} + (1-\alpha) g_t^2$ (지수 가중, 유계)

지수 평균은 모든 과거를 같은 비중으로 더하지 않아 오래된 큰 경사의 영향을 줄인다. 다만 경사가 제한 없이 커지면 상태도 유계가 아닐 수 있고, 전역 러닝레이트 스케줄에 따라 실제 배율은 감소할 수 있다.

---

## 4. 중심화한 그래디언트를 쓸 때

중심화 RMSprop은 경사의 1차·2차 이동 평균을 따로 구한 뒤 2차 평균에서 1차 평균의 제곱을 뺀 값을 분모에 사용한다. 각 단계에서 중심화한 경사를 제곱해 누적하는 식과는 다르다.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, centered=True)
```

$$\bar{g}_t = \alpha \, \bar{g}_{t-1} + (1 - \alpha) \, g_t$$

$$s_t = \alpha \, s_{t-1} + (1 - \alpha) \, g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t - \bar{g}_t^2} + \epsilon} \, g_t$$

---

## 5. 언제 쓰는가

RMSprop은 Adam 이전의 기본 적응형 옵티마이저였다. Adam이 이따금 불안정할 수 있는 순환 뉴럴 넷트웍과 강화 학습에서는 여전히 쓸모 있다.

---

## 6. 핵심 정리

- RMSprop은 그래디언트 제곱에 지수 감쇠를 써서 Adagrad의 러닝레이트 붕괴를 바로잡는다.
- PyTorch RMSprop의 기본 alpha는 0.99, lr는 0.01이다. 이 페이지에서 지정한 0.001 등은 예제 설정이며 라이브러리 기본값과 구분한다.
- 대체로 Adam에 자리를 내주었지만 RNN과 강화 학습에서는 여전히 쓸모 있다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
RMSprop의 갱신 규칙을 유도하고 러닝레이트를 어떻게 맞추는지 설명하라.

</div>

??? success "풀이"
    RMSprop은 $v_t = \beta v_{t-1} + (1-\beta)g_t^2$, $\theta_{t+1} = \theta_t - \eta g_t / (\sqrt{v_t}+\epsilon)$이다. 그래디언트가 큰 파라메터는 실효 러닝레이트가 작아지고($v$이 크면 $\eta/\sqrt{v}$이 작다) 그 반대도 마찬가지이다. 이렇게 로스 곡면의 국소적인 기하에 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
RMSprop이 그래디언트 제곱의 지수 가중 이동 평균임을 보여라.

</div>

??? success "풀이"
    펼치면 $v_t = (1-\beta)\sum_{i=0}^{t-1} \beta^i g_{t-i}^2$이다. 최근 그래디언트는 웨이트 $(1-\beta)$을 갖고 오래된 것은 $\beta^i$으로 줄어든다. 실효 창의 크기는 $1/(1-\beta)$이며, $\beta=0.99$이면 최근 100개의 그래디언트가 지배한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Adagrad의 누적합을 지수 평균으로 바꾸는 효과와 한계를 설명하라.

</div>

??? success "풀이"
    오래된 제곱 경사의 영향이 감쇠해 최근 스케일에 반응할 수 있다. 경사가 유계이면 이동 평균도 유계로 분석할 수 있지만 일반적인 수렴이나 지속 학습을 자동으로 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
RMSprop과 Adam을 비교하라. Adam은 RMSprop에 무엇을 더하는가?

</div>

??? success "풀이"
    Adam = RMSprop + 모멘텀 + 바이어스 보정이다. RMSprop은 러닝레이트를 맞추지만 방향에는 날 그래디언트를 쓴다. Adam은 갱신을 더 매끄럽게 하려고 모멘텀 항(그래디언트의 지수 평균)을, 초반 추정을 정확히 하려고 바이어스 보정을 더한다.

## 정리하며

RMSprop은 경사 제곱의 지수 평균으로 좌표별 배율을 정한다. 중심화와 모멘텀 옵션을 구분하고, 분모가 비중심 2차 적률인지 중심화된 값인지 확인한다. Adagrad와 다른 이력 처리 방식이지 항상 더 좋은 성능이라는 뜻은 아니다.
