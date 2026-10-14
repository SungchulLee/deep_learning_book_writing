# Scheduled Sampling

Scheduled Sampling(Bengio 외, 2015)는 익히는 동안 참값 입력을 모델이 스스로 헤아린 값으로 차츰 바꾸어 Teacher Forcing과 스스로 도는 생성 사이의 틈을 잇는다.

---

## 1. 알고리즘

학습 걸음마다 각 자리 $t$에 대해:

- 확률 $\epsilon_i$(학습 차례 확률)으로 참값 $x_t$을 쓴다
- 확률 $1 - \epsilon_i$으로 모델이 헤아린 값 $\hat{x}_t = \arg\max p_\theta(\cdot \mid x_{<t})$을 쓴다

학습 차례 $\epsilon_i$은 익히는 동안 줄어든다.

$$\epsilon_i = \max(\epsilon_{\min}, k^i) \quad \text{(exponential decay)}$$

$$\epsilon_i = \max(\epsilon_{\min}, \frac{k}{k + \exp(i/k)}) \quad \text{(inverse sigmoid)}$$

---

## 2. 구현

```python
def scheduled_sampling_step(model, x, epsilon):
    batch_size, seq_len = x.shape
    input_tokens = x[:, 0:1]  # 참값 시작 토큰에서 시작한다
    
    for t in range(1, seq_len):
        logits = model(input_tokens)
        predicted = logits[:, -1].argmax(dim=-1, keepdim=True)
        
        # 동전 던지기: 참값을 쓸지 헤아린 값을 쓸지
        use_gt = (torch.rand(batch_size, 1) < epsilon).to(x.device)
        next_token = torch.where(use_gt, x[:, t:t+1], predicted)
        input_tokens = torch.cat([input_tokens, next_token], dim=1)
    
    # 온 차례에 대해 손실을 셈한다
    full_logits = model(input_tokens[:, :-1])
    return F.cross_entropy(full_logits.reshape(-1, vocab_size),
                          x[:, 1:].reshape(-1))
```

---

## 3. 한계

Scheduled Sampling은 치우쳐 있다. 입력 분포가 참값과 모델이 헤아린 값의 섞임이므로 학습 목표가 더는 최대 우도와 맞지 않는다. 이론으로는 한결같지 않은 학습로 이어질 수 있으나 실제로는 흔히 도움이 된다.

---

## 4. 대안

- **교수 밀어 넣기**: 분류기를 써서 Teacher Forcing 모드와 스스로 도는 모드의 은닉 상태 분포를 맞춘다
- **미분할 수 있는 샘플링**: 검벨-소프트맥스나 곧바로 통과 어림개를 써서 샘플링을 미분할 수 있게 한다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
자기회귀 차례 생성에서의 드러남 편향 문제를 설명하라.

</div>

??? success "연습문제 1 풀이"
    Teacher Forcing로 익히는 동안 모델은 걸음마다 늘 참값 토큰을 입력으로 본다. 추론할 때는 제가 헤아린 (틀릴 수도 있는) 값을 조건으로 삼는다. 이 학습과 시험 사이의 어긋남이 **드러남 편향**이다. 곧 모델은 익히는 동안 제 어긋남에서 되돌아오는 법을 배운 적이 없다. 어긋남은 쌓인다. 곧 잘못 헤아린 하나가 낯선 상태로 이끌어 더 많은 어긋남을 낳는다. 적어도 하나쯤 틀릴 확률이 커지는 긴 차례에서 특히 문제가 되며, 어긋남마다 모델을 학습 분포에서 더 멀리 밀어낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Scheduled Sampling 알고리즘을 설명하라. 익히는 동안 차례표는 어떻게 바뀌는가?

</div>

??? success "연습문제 2 풀이"
    Scheduled Sampling은 익히는 동안 Teacher Forcing과 스스로 돌기를 섞는다. 푸는 걸음마다 (바퀴 $i$에서) 확률 $p_i$으로 모델은 참값 토큰을 쓰고 확률 $1-p_i$으로 제가 헤아린 값을 쓴다. 확률 $p_i$은 익히는 동안 **선형 줄이기** $p_i = \max(\epsilon, 1 - i/k)$, **지수 줄이기** $p_i = k^i$, **거꾸로 된 S자** $p_i = k/(k + \exp(i/k))$ 같은 차례표로 줄어든다. 학습 앞머리($p \approx 1$)에는 Teacher Forcing이 안정된 기울기를 준다. 학습 뒷머리($p \ll 1$)에는 모델이 스스로 조건 주기를 익혀 추론을 준비한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Scheduled Sampling의 이론상 한계는 무엇인가? 왜 치우친 어림개로 여기는가?

</div>

??? success "연습문제 3 풀이"
    Scheduled Sampling은 익히는 동안 서로 다른 두 분포(참값과 모델이 헤아린 값)의 표본을 섞는데, 이는 어느 쪽 분포의 우도를 가장 크게 하는 것과도 맞지 않는다. 그래서 나오는 학습 목표는 로그 우도의 올바른 아래 한계가 아니므로 치우친 어림개가 된다. 편향이 생기는 까닭은 (1) 모델이 헤아린 값을 쓸 때 기울기가 이를 붙박이 입력으로 다루고(샘플링 결정을 지나는 기울기가 없다), (2) 섞임 분포가 익히는 동안 바뀌어 멈춰 있지 않은 목표가 되기 때문이다. 이런 이론상 문제에도 Scheduled Sampling은 실제로 생성 품질을 꾸준히 높인다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Scheduled Sampling을 드러남 편향을 다루는 다른 방식(빔 찾기, 강화 학습, 교수 밀어 넣기)과 비교하여라.

</div>

??? success "연습문제 4 풀이"
    | 방법 | 방식 | 좋은 점 | 나쁜 점 |
    |--------|---------|------|------|
    | **Scheduled Sampling** | Teacher Forcing + 스스로 조건 주기 섞기 | 단순하고 잘 듣는다 | 기울기가 치우친다 |
    | **빔 찾기** | 추론할 때 여러 가설을 살핀다 | 학습을 바꾸지 않는다 | 추론 때만 도움이 된다 |
    | **강화 학습(REINFORCE)** | 차례 수준 보상을 가장 좋게 한다 | 치우치지 않고 너그럽다 | 분산이 크고 불안정하다 |
    | **교수 밀어 넣기** | 은닉 상태에 대한 적대적 학습 | 이론으로 뒷받침된다 | 복잡하고 GAN 같은 불안정이 있다 |

    Scheduled Sampling은 단순함과 잘 듣는 정도의 트레이드오프가 가장 좋아 가장 널리 쓰인다.

## 정리하며

이 마당은 알고리즘, 구현, 한계, 대안을 차례로 짚었다.
