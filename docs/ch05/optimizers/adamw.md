# AdamW
## 개요

AdamW는 가중치 감쇠를 기울기 기반 갱신에서 떼어 내어 Adam이 가중치 감쇠와 잘 어울리지 못하는 문제를 바로잡는다. 사소해 보이는 이 변화가 정칙화의 효과에 큰 영향을 주며, 이제는 대부분의 딥러닝 과제에서 권장되는 기본 최적화기이다.

## Adam + L2의 문제

SGD에서는 L2 정칙화와 가중치 감쇠가 같다.

$$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t) = (1 - \eta\lambda)\theta_t - \eta g_t$$

Adam에서는 L2 정칙화가 적응형 배율 조정 *전에* 기울기에 적용된다.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda\theta_t)$$

적응형 분모 $\sqrt{\hat{v}_t}$이 기울기와 함께 정칙화 항의 배율까지 조정하여 사실상 매개변수마다 *다른* 가중치 감쇠를 걸게 되고, 고르게 정칙화한다는 취지가 무너진다.

## AdamW의 갱신 규칙

AdamW는 적응형 갱신 바깥에서 매개변수에 가중치 감쇠를 곧바로 적용한다.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

가중치 감쇠 $\lambda$은 적응형 학습률과 무관하게 모든 매개변수에 고르게 작용한다.

## PyTorch 구현

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                              betas=(0.9, 0.999), weight_decay=0.01)
```

## 대표적인 초매개변수

- $\eta = 0.001$~$0.0001$ (학습률)
- $\lambda = 0.01$~$0.1$ (가중치 감쇠)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$

## 언제 쓰는가

AdamW는 다음에 권장되는 기본 최적화기이다.

- 트랜스포머 모델 (BERT, GPT, 비전 트랜스포머)
- 미리 학습된 모델의 미세 조정
- 적응형 최적화와 가중치 감쇠를 함께 쓰고 싶은 모든 과제

## 핵심 정리

- AdamW는 가중치 감쇠를 적응형 기울기 갱신에서 떼어 낸다.
- 덕분에 매개변수별 학습률 배율과 무관하게 올바르고 고른 정칙화가 이루어진다.
- AdamW는 요즘의 기본 최적화기이며 `Adam(weight_decay=...)`보다 낫다.

## 연습문제

**연습문제 1.**
L2 정칙화와 분리된 가중치 감쇠의 차이, 그리고 AdamW가 둘을 나누는 이유를 설명하라.

??? success "연습문제 1 풀이"
    SGD에서는 L2 정칙화($\nabla L + \lambda w$)와 가중치 감쇠($w \leftarrow (1-\lambda)w$)가 같다. Adam에서는 L2 정칙화가 적응형 학습률로 나뉘어 기울기가 큰 매개변수에 대한 효과가 약해진다. AdamW는 가중치 감쇠를 곧바로 적용하여($w \leftarrow (1-\lambda)w - \eta\hat{m}/\sqrt{\hat{v}}$) 정칙화가 한결같게 한다.

---

**연습문제 2.**
Adam 대신 AdamW를 써야 할 때는 언제인가?

??? success "연습문제 2 풀이"
    적응형 최적화기와 가중치 감쇠를 함께 쓸 때에는 언제나 AdamW를 쓰라. AdamW는 제대로 된 정칙화를 주므로 트랜스포머 기반 모델(BERT, GPT, ViT)의 기본 최적화기이다. `weight_decay`을 쓰는 Adam은 참된 가중치 감쇠가 아니라 L2 정칙화를 구현하므로 최적이 아니다.

---

**연습문제 3.**
AdamW의 갱신 규칙을 구현하고 Adam과 어디가 다른지 보여라.

??? success "연습문제 3 풀이"
    ```python
    # Adam: p.data -= lr * m_hat / (v_hat.sqrt() + eps)  # L2 in grad
    # AdamW: 가중치 감쇠는 따로 적용된다
    for p in model.parameters():
        p.data *= (1 - lr * weight_decay)  # 분리된 가중치 감쇠
        p.data -= lr * m_hat / (v_hat.sqrt() + eps)  # Adam의 걸음
    ```

---

**연습문제 4.**
기울기의 분산이 크거나 작은 매개변수에 대해 Adam과 AdamW의 실효 정칙화 강도를 유도하라.

??? success "연습문제 4 풀이"
    L2를 쓰는 Adam에서 실효 감쇠는 $\lambda / \sqrt{\hat{v}_t}$이며, 분산이 큰(자주 갱신되는) 매개변수에서 작아진다. AdamW에서는 감쇠가 고르게 $\lambda$이다. 즉 Adam은 자주 갱신되는 매개변수를 덜 정칙화하고 드문 매개변수를 지나치게 정칙화하는 반면, AdamW는 모두에 같은 정칙화를 건다.
