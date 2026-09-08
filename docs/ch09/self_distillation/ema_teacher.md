# 자기 증류의 지수 이동 평균 교사

지수 이동 평균(EMA) 교사 얼개는 자기 증류 틀에서 안정된 목표 모형을 만드는 강력한 기법이다. 지수 이동 평균 갱신으로 학생 신경망의 시간 평균 판을 지킴으로써 EMA 교사는 따로 사전 학습할 필요를 없애고 꾸준히 나아지는 지도 신호를 준다.

EMA에 바탕한 방식은 요즘 자기 지도 학습 파이프라인(BYOL, SimCLR v2, DenseCL)에서 근본이 되었고, 드러난 음성 표집이나 기억 은행이 필요 없게 한다. EMA 교사의 안정성과 계산 효율은 이름표 없는 방대한 데이터셋의 큰 규모 사전 학습에 특히 매력적이다.

---

## 1. 핵심 개념

- **시간 평균**: 지수 가중으로 매개변수 갱신을 매끄럽게 한다
- **관성 계수**: 교사가 바뀌는 속도를 다스리는 초매개변수
- **안정성**: EMA가 목표 표현의 흩어짐을 줄인다
- **기울기 떼어 놓기**: 교사의 갱신이 학생의 기울기와 무관하다
- **초기화**: 교사는 학생 신경망의 사본으로 시작한다

---

## 2. 수학적 정식화

### EMA 갱신 규칙

핵심 갱신 얼개는 다음과 같이 돈다.

$$\theta_t^{\text{EMA}} \leftarrow \tau \theta_t^{\text{EMA}} + (1 - \tau) \theta_t^{\text{student}}$$

여기서 각 기호는 다음과 같다.

- $\theta_t^{\text{EMA}}$은 단계 $t$의 교사 매개변수이다
- $\theta_t^{\text{student}}$은 지금의 학생 매개변수이다
- $\tau \in (0, 1)$은 관성 계수이다

### 초기화

처음에 교사 매개변수는 학생 매개변수와 같다.

$$\theta_0^{\text{EMA}} = \theta_0^{\text{student}}$$

### 재귀 펼치기

재귀를 펼치면 교사 매개변수가 지난 학생 매개변수의 가중 평균임이 드러난다.

$$\theta_t^{\text{EMA}} = (1-\tau) \sum_{i=0}^{t} \tau^{t-i} \theta_i^{\text{student}}$$

지난 매개변수 $\theta_i$의 가중치는 $\tau^{t-i}(1-\tau)$이며 기하 분포를 이룬다.

---

## 3. 안정성 분석

### 수렴의 성질

EMA 갱신에는 매력적인 이론적 성질이 있다.

$$\lim_{t \to \infty} \theta_t^{\text{EMA}} = \text{steady state of } \{\theta_t^{\text{student}}\}$$

학생의 학습이 수렴하면 교사도 같은 점으로 수렴하되 흩어짐이 줄어든다.

### 흩어짐 줄이기

학생 매개변수의 분산을 $\sigma^2$이라 하자. 교사 매개변수의 분산은 다음과 같다.

$$\text{Var}(\theta_t^{\text{EMA}}) = \frac{(1-\tau)}{2-\tau} \sigma^2$$

!!! tip "안정성의 이점"
    $\tau = 0.999$이면 교사의 분산이 학생 분산의 약 50%여서 튼튼한 목표 표현을 준다.

---

## 4. 관성 계수 고르기

### tau 값의 맞바꿈

| $\tau$ | 성질 | 쓰임새 |
|---|---|---|
| **0.95** | 교사가 빠르게 좇는다 | 변화가 심하고 갱신이 잦을 때 |
| **0.99** | 표준 선택 | 안정성과 반응성의 균형 |
| **0.999** | 느린 수렴 | 매우 안정된 목표 |
| **0.9999** | 갱신이 아주 적다 | 큰 규모 사전 학습 |

### 움직임에 따라 고르기

학습 안정성 요구에 따라 $\tau$을 고른다.

$$\tau_{\text{recommended}} = 1 - \frac{\alpha}{N}$$

여기서 $\alpha$은 학습률이고 $N$은 데이터셋의 크기이다.

---

## 5. EMA 교사를 쓰는 학습 틀

### 두 가지 구조

```
Input
├─ Student Branch ──> Prediction ──> Loss₁
│                                     │
└─ EMA Teacher Branch ──> Features ──┘
                              │
                         EMA Update
```

### 앞먹임 알고리즘

```
1. Forward through student: z_s = Student(x)
2. Forward through EMA teacher: z_t = EMA_Teacher(x)
3. Compute loss: L = D(z_s, z_t)
4. Backward through student: ∇L_s = ∂L/∂θ_s
5. Update student: θ_s ← θ_s - α∇L_s
6. Update EMA teacher: θ_t ← τθ_t + (1-τ)θ_s
```

---

## 6. 기울기 떼어 놓기

EMA 교사의 매우 중요한 성질은 기울기의 독립이다.

$$\frac{\partial \mathcal{L}}{\partial \theta_t^{\text{EMA}}} = 0$$

교사 매개변수는 기울기를 곧바로 받지 않는다. 이것이 다음을 막는다.

- **무너짐**: 두 신경망이 동시에 뜻없는 해로 빠지는 일
- **불안정**: 교사와 학생 사이의 되먹임 고리
- **국소 최솟값**: 다양한 최적화 궤적으로 벗어난다

!!! note "핵심 이점"
    기울기를 떼어 놓으면 두 신경망이 모두 기울기를 받는 방식에 견주어 규제를 아주 조금만 써도 된다.

---

## 7. 구현의 세부

### 기억 효율

EMA 교사는 매개변수 기억을 두 배로 만든다.

$$M_{\text{total}} = 2M_{\text{model}} + M_{\text{optim}}$$

큰 모형에는 모형 병렬이나 기울기 검문점을 쓴다.

### 계산 비용

되풀이마다의 비용은 다음을 담는다.

1. **학생 앞먹임**: 학생을 지나는 앞먹임 한 번
2. **교사 앞먹임**: 교사를 지나는 앞먹임 한 번
3. **학생 역전파**: 학생을 지나는 역전파
4. **매개변수 갱신**: EMA 갱신에 약 $2\%$의 짐

전체 비용은 모형 하나를 학습할 때의 약 2.1배이다.

### 수치적 안정성

$\tau$이 아주 클 때(0.999 이상) 쌓임 오차가 생길 수 있다.

$$\theta_t^{\text{EMA}} = \alpha \cdot \theta_t^{\text{EMA}} + \beta \cdot \theta_t^{\text{student}}$$

여기서 $\alpha + \beta = 1$이며 늘린 정밀도로 셈한다.

---

## 8. 더 나아간 EMA 변형

### 관성 담금질

초기 수렴을 낫게 하려고 학습 중에 $\tau$을 차츰 올린다.

$$\tau(t) = 1 - (1-\tau_{\text{final}}) \exp(-t/\lambda)$$

### 적응 관성

학습의 움직임에 따라 $\tau$을 조정한다.

$$\tau(t) = \max(\tau_{\text{min}}, 1 - \alpha(t) \cdot \sqrt{\frac{t}{t_{\text{total}}}})$$

### 섞은 갱신

EMA에 이따금씩의 온전한 맞춤을 곁들인다.

$$\theta_t^{\text{EMA}} = \begin{cases}
\tau \theta_t^{\text{EMA}} + (1-\tau) \theta_t^{\text{student}} & \text{if } t \mod k \neq 0 \\
\theta_t^{\text{student}} & \text{그 밖에는}
\end{cases}$$

---

## 9. 이론적 이음

### 폴랴크 평균과의 관계

EMA는 이어서 적용한 폴랴크 평균과 같다.

$$\bar{\theta}_t = \frac{1}{t} \sum_{i=1}^{t} \theta_i$$

알맞은 조건에서 둘은 같은 분포로 수렴한다.

### 정보 이론의 눈

EMA 교사는 잡음 섞인 학생 매개변수 아래에서 기대 손실을 가장 작게 한다.

$$\mathbb{E}[\mathcal{L}(\text{EMA Teacher})] < \mathbb{E}[\mathcal{L}(\text{Student})]$$

---

## 10. 계량 금융에서의 쓰임

EMA에 바탕한 자기 증류는 다음에 특히 값지다.

- **시장 국면 알아채기**: 천천히 갱신되는 교사가 긴 흐름을 잡고 학생은 짧은 변화에 맞춘다
- **위험 모형의 변화**: EMA 교사가 안정된 위험 어림값을 주고 학생은 새 무늬를 배운다
- **포트폴리오 최적화**: 교사가 지난 제약을 주고 학생은 지금의 시장 데이터로 다듬는다

---

## 11. 관련 주제

- 자기 증류 훑어보기 (9.2.0절)
- 지식 증류의 기초 (9.2.1절)
- BYOL 구조
- 관성 대조 (MoCo)

---

## 연습문제

**연습문제 1.**
지수 이동 평균(EMA) 교사 얼개를 설명하라.

??? success "연습문제 1 풀이"
    EMA 교사는 학생의 천천히 갱신되는 사본을 지킨다. $\alpha \approx 0.999$으로 $\theta_{\text{teacher}}^{(t)} = \alpha \theta_{\text{teacher}}^{(t-1)} + (1-\alpha) \theta_{\text{student}}^{(t)}$이다. 교사가 매끄럽게 바뀌는 안정된 목표를 주어 자기 증류의 무너짐 문제를 피한다.

---

**연습문제 2.**
목표를 만드는 데 EMA 교사가 실시간 학생보다 안정적인 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    학생의 매개변수는 배치마다 빠르게 바뀌어 잡음 섞인 목표를 낸다. EMA는 여러 매개변수 상태에 걸쳐 평균 내어 배치마다의 출렁임을 다듬는다. 폴랴크 평균과 비슷하다. EMA 궤적은 흩어짐이 작고 그때그때의 매개변수보다 일반화가 나은 경우가 많다.

---

**연습문제 3.**
파이토치에서 EMA 갱신 단계를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    @torch.no_grad()
    def update_ema(student, teacher, decay=0.999):
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)
    ```

---

**연습문제 4.**
EMA 감쇠율 $\alpha$이 학습에 어떤 영향을 주는가? $\alpha$이 너무 높거나 낮으면 어떻게 되는가?

??? success "연습문제 4 풀이"
    $\alpha$이 너무 높으면(0.9999) 교사가 아주 천천히 갱신되어 낡아지고 철 지난 목표를 준다. $\alpha$이 너무 낮으면(0.9) 교사가 학생을 너무 바짝 좇아 안정된 목표를 준다는 뜻이 사라진다. 흔한 범위는 0.996~0.999이다. 어떤 방법은 코사인 일정을 써서 $\alpha$을 낮게 시작해 학습 중에 올린다.

## 정리하며

이 마당은 핵심 개념、수학적 정식화、안정성 분석、관성 계수 고르기을 차례로 짚었다.
