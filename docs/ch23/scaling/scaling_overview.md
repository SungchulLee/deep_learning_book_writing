# 대규모 언어 모델의 스케일링 법칙

---

## 1. 학습 목표

- 대규모 언어 모델의 성능을 다스리는 실증 스케일링 법칙을 이해한다
- 계산에 가장 알맞은 학습 자리매김을 이끌어 낸다
- 친칠라 규모 키우기로 모델과 데이터의 크기를 정한다
- 매개변수, 데이터, 계산 사이의 트레이드오프를 살핀다

---

## 2. 들어가며

스케일링 법칙은 모델 성능과 세 핵심 요소, 곧 모델 크기(매개변수), 데이터셋 크기(토막), 계산 예산(FLOPs) 사이의 어림할 수 있는 관계를 그린다. 이 실증 법칙 덕분에 연구자는 학습 전에 성능을 어림하고 자원 나눔을 가장 좋게 할 수 있다.

---

## 3. 캐플런 스케일링 법칙(OpenAI, 2020)

### 거듭제곱 법칙 관계

성능(엇갈린 엔트로피 손실 $L$으로 잰다)은 거듭제곱 법칙을 따른다:

**모델 크기의 규모 키우기**:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

**데이터 크기의 규모 키우기**:

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

**계산의 규모 키우기**:

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

여기서 $N_c$, $D_c$, $C_c$은 결정적 상수이고 $N$, $D$, $C$은 저마다 매개변수, 데이터 토막, 계산 FLOPs를 나타낸다.

### 아우른 스케일링 법칙

모든 요소를 한꺼번에 키울 때:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

### 핵심 발견

1. **매끄러운 거듭제곱 법칙**: 규모에 따라 손실이 어림할 수 있게 줄어든다
2. **모델 크기가 좌우한다**: 큰 모델이 표본을 더 아낀다
3. **계산 예산**: 계산이 정해졌으면 데이터를 덜 쓰고 학습된 큰 모델을 낫게 여긴다

---

## 4. 친칠라 스케일링 법칙(DeepMind, 2022)

### 고쳐진 가장 좋은 나눔

Hoffmann 외는 처음 스케일링 법칙이 데이터 요구량을 낮잡았음을 찾아냈다:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

맞춘 매개변수로:

- $E \approx 1.69$(더 줄일 수 없는 손실)
- $A \approx 406.4$, $\alpha \approx 0.34$
- $B \approx 410.7$, $\beta \approx 0.28$

### 계산에 가장 알맞은 학습

계산 예산이 $C \approx 6ND$(앞으로 걸음 + 뒤로 걸음)일 때

**가장 좋은 매개변수**:

$$N_{opt} \propto C^{0.50}$$

**가장 좋은 토막 수**:

$$D_{opt} \propto C^{0.50}$$

**핵심 눈썰미**: 매개변수와 데이터는 계산에 따라 똑같이 커져야 한다.

### 친칠라와 Gopher

| 모델 | 매개변수 | 학습 토막 | 계산 |
|-------|------------|-----------------|---------|
| Gopher | 280B | 300B | ~$5 \times 10^{23}$ |
| 친칠라 | 700억 | 1조 4000억 | ~$5 \times 10^{23}$ |

친칠라는 4배 작고 데이터를 4배 더 써서 대부분의 잣대에서 Gopher를 앞섰다.

---

## 5. 실전 쓰임새

### 필요한 계산 어림하기

```python
def estimate_training_flops(
    num_parameters: int,
    num_tokens: int,
    flops_per_token_per_param: float = 6.0
) -> float:
    """
    전체 익히기 FLOPs를 어림한다.
    
    6이라는 인수는 다음을 헤아린 것이다:
    - 앞먹임: 토막마다 매개변수마다 2 FLOPs
    - 뒤먹임: 토막마다 매개변수마다 4 FLOPs
    
    인수:
        num_parameters: 모델 매개변수 수
        num_tokens: 익히기 토막 수
        flops_per_token_per_param: FLOPs 곱수(붙박이 6)
    
    반환값:
        익히기의 전체 FLOPs
    """
    return flops_per_token_per_param * num_parameters * num_tokens

def chinchilla_optimal_config(
    compute_budget_flops: float,
    flops_per_token_per_param: float = 6.0
) -> dict:
    """
    셈에 가장 알맞은 모델 크기와 자료 크기를 셈한다.
    
    친칠라 잣수 맞추기에 따라, 가장 좋게 나누면 N ≈ D이다.
    
    인수:
        compute_budget_flops: 쓸 수 있는 셈의 FLOPs
        
    반환값:
        가장 좋은 매개변수 수와 토막 수를 담은 사전
    """
    # C = 6 * N * D이고 가장 좋을 때 N ≈ D
    # 따라서 C = 6 * N^2이므로 N = sqrt(C/6)
    
    optimal_n = (compute_budget_flops / flops_per_token_per_param) ** 0.5
    optimal_d = optimal_n  # 고르게 잣수 맞추기
    
    return {
        'optimal_parameters': int(optimal_n),
        'optimal_tokens': int(optimal_d),
        'tokens_per_parameter': optimal_d / optimal_n
    }

# 보기: 10^24 FLOPs 예산
config = chinchilla_optimal_config(1e24)
print(f"Optimal parameters: {config['optimal_parameters'] / 1e9:.1f}B")
print(f"Optimal tokens: {config['optimal_tokens'] / 1e12:.2f}T")
```

**출력:**

```
Optimal parameters: 408.2B
Optimal tokens: 0.41T
```

### 손실 어림하기

```python
import numpy as np

def predict_loss_chinchilla(
    num_parameters: float,
    num_tokens: float,
    E: float = 1.69,
    A: float = 406.4,
    alpha: float = 0.34,
    B: float = 410.7,
    beta: float = 0.28
) -> float:
    """
    친칠라 잣수 맞추기 법칙으로 익히기 손실을 어림한다.
    
    L(N, D) = E + A/N^α + B/D^β
    """
    return E + A / (num_parameters ** alpha) + B / (num_tokens ** beta)

# 여러 자리매김을 견준다
configs = [
    ("7B, 1T tokens", 7e9, 1e12),
    ("13B, 1T tokens", 13e9, 1e12),
    ("7B, 2T tokens", 7e9, 2e12),
    ("70B, 1.4T tokens", 70e9, 1.4e12),
]

for name, n, d in configs:
    loss = predict_loss_chinchilla(n, d)
    print(f"{name}: predicted loss = {loss:.3f}")
```

**출력:**

```
7B, 1T tokens: predicted loss = 2.052
13B, 1T tokens: predicted loss = 2.017
7B, 2T tokens: predicted loss = 2.020
70B, 1.4T tokens: predicted loss = 1.937
```

---

## 6. 친칠라를 넘어: 최근의 발견

### LLaMA의 규모 철학

메타의 LLaMA 모델은 **추론의 효율**을 앞세운다:

- 친칠라 최적보다 데이터를 더 써서 작은 모델을 학습된이다
- 70억 모델을 토막 1조 이상으로 학습된이다(친칠라 최적은 약 2000억)
- 펼친 모델의 추론 값이 더 낫다

### 떠오르는 규모 몸짓

어떤 능력은 **끊긴** 나아짐을 보인다:

```
Performance
    │
    │                    ╭──── Emergent capability
    │                   ╱
    │          ────────╯
    │    ─────╯
    │───╯
    └────────────────────────── Scale (log)
```

여기에는 다음이 든다:

- Chain-of-Thought 따지기
- 맥락 안에서 학습하기
- 코드 만들기

### 데이터의 좋음과 규모 키우기

최근 연구는 데이터의 좋음이 다르게 커짐을 시사한다:

$$L(N, D, Q) = E + \frac{A}{N^\alpha} + \frac{B}{(D \cdot Q)^\beta}$$

여기서 $Q$은 데이터의 좋음(거르기, 겹침 없애기)을 나타낸다.

---

## 7. 스케일링 법칙의 한계

### 스케일링 법칙이 담아내지 못하는 것

1. **능력의 문턱값**: 어떤 능력은 갑자기 떠오른다
2. **일별 성능**: 일마다 커지는 모습이 다르다
3. **구조의 영향**: 법칙은 트랜스포머에 대해 이끌어 낸 것이다
4. **데이터의 분포**: 양을 넘어 좋음과 여러 갈래임이 중요하다
5. **파인튜닝의 흐름**: 법칙은 사전 학습에 초점을 둔다

### 밖으로 늘릴 때의 위험

```python
def scaling_uncertainty(
    predicted_loss: float,
    extrapolation_factor: float,
    uncertainty_per_order: float = 0.05
) -> tuple:
    """
    잣수 맞추기 법칙을 밖으로 늘여 잡을 때의 불확실함을 어림한다.
    
    밖으로 늘여 잡는 거리가 멀수록 불확실함이 커진다.
    """
    log_extrapolation = np.log10(extrapolation_factor)
    relative_uncertainty = uncertainty_per_order * log_extrapolation
    
    lower = predicted_loss * (1 - relative_uncertainty)
    upper = predicted_loss * (1 + relative_uncertainty)
    
    return lower, upper
```

---

## 8. 핵심 식

$$\boxed{L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}}$$

$$\boxed{C_{opt} = 6 \cdot N_{opt} \cdot D_{opt}, \quad N_{opt} \approx D_{opt}}$$

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
친칠라 스케일링 법칙을 말하여라. 이는 모델 크기와 학습 토막 사이의 가장 좋은 계산 나눔을 어떻게 바꾸는가?

</div>

??? success "연습문제 1 풀이"
    친칠라 크기 법칙(호프만 외, 2022)에 따르면 계산을 가장 잘 쓰는 학습에서는 모델 크기 $N$과 학습 토막 수 $D$을 똑같이 키워야 한다. 곧 $N \propto C^{0.5}$이고 $D \propto C^{0.5}$이며 $C$은 계산 예산이다. 그러면 가장 좋은 비교는 대략 $D \approx 20N$이다(매개변수마다 토막 20개). 그전 버릇(GPT-3 따위)은 큰 모델을 덜 익혔다. GPT-3(매개변수 1750억 개)은 토막 3000억 개로 익혔지만, 친칠라에 맞춘 길은 700억 모델을 토막 1조 4000억 개로 익혀 같은 계산으로 더 나은 성능을 이룬다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
대규모 언어 모델의 창발적 능력이란 무엇인가? 이 생각은 왜 논쟁거리인가?

</div>

??? success "연습문제 2 풀이"
    창발적 능력이란 모델 크기가 어떤 문턱을 넘으면 갑자기 나타나고 작은 모델에는 없어 보이는 능력이다. 보기로 여러 자리 계산, Chain-of-Thought 따지기, 단어 풀어내기가 있다. 논쟁은 이렇다. Schaeffer 외(2023)는 "떠오름"이 끊긴 잣대(딱 맞음 정확도)를 쓴 데서 온 찌꺼기라고 주장한다. 매끄러운 잣대(토막 수준 로그 확률)로 재면 성능이 규모에 따라 차츰 어림할 수 있게 나아진다. 논쟁의 핵심은 떠오름이 모델의 성질인지 평가 방법의 성질인지이다. 두 관점 모두 일리가 있다. 곧 질적인 능력의 문턱은 있지만 그것이 기초의 매끄러운 나아짐으로 어림될 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
큰 모델을 익힐 때의 모델 나란히 하기, 데이터 나란히 하기, 물길 나란히 하기의 차이를 밝혀라.

</div>

??? success "연습문제 3 풀이"
    **데이터 나란히 하기**: GPU마다 온전한 모델 복사본을 지니고 서로 다른 데이터셋을 다루며, 기울기를 모두 모으기로 맞춘다. 모델이 GPU 하나에 들어가야 한다는 제한이 있다. **모델/텐서 나란히 하기**: 층을 GPU에 나눠 쪼갠다(보기로 어텐션 머리 쪼개기). 더 큰 모델을 쓸 수 있지만 대역폭이 넓은 이음이 필요하다. **물길 나란히 하기**: 잇닿은 층을 GPU에 나눈다. GPU마다 단계별로 서로 다른 작은 배치를 다룬다. 주고받기가 줄지만 물길 거품(노는 때)이 생긴다. 요즘 체계(Megatron-LM, DeepSpeed)는 셋을 모두 아우른다. 곧 마디 안에서는 텐서 나란히 하기, 마디 사이에는 물길 나란히 하기, 배치 사이에는 데이터 나란히 하기를 쓴다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
계산 예산이 $10^{23}$ 뜨는 계산 횟수로 붙박여 있다면 모델 크기와 학습 데이터에 어떻게 나누어 쓰겠는가? 크기 법칙으로 답을 뒷받침하여라.

</div>

??? success "연습문제 4 풀이"
    $C = 6ND$(트랜스포머 학습의 어림 뜨는 계산 횟수)을 쓰는 친칠라 크기 법칙을 쓴다. $10^{23} = 6ND$이고 가장 좋은 $D \approx 20N$이다. 넣어 보면 $10^{23} = 6N \cdot 20N = 120N^2$이므로 $N \approx \sqrt{10^{23}/120} \approx 2.9 \times 10^{10}$, 곧 매개변수 약 290억 개다. 학습 토막은 $D = 10^{23}/(6 \times 29 \times 10^9) \approx 5750$억 개다. 이렇게 나누면 계산을 가장 잘 쓰는 비교를 따르므로, 같은 계산 예산에서 덜 학습된 더 큰 모델(1750억을 토막 960억 개로)이나 지나치게 학습된 더 작은 모델(70억을 토막 2조 4000억 개로)보다 낫다.

## 정리하며

| 스케일링 법칙 | 핵심 눈썰미 | 가장 좋은 비(N:D) |
|-------------|-------------|---------------------|
| 캐플런(2020) | 큰 모델이 더 효율적 | 약 10:1(매개변수를 앞세움) |
| 친칠라(2022) | 데이터도 똑같이 중요 | 약 1:1(균형) |
| LLaMA(2023) | 추론 값이 중요 | 약 1:20 이상(데이터를 앞세움) |

**참고 문헌**

1. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
2. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *arXiv:2203.15556*.
3. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.
