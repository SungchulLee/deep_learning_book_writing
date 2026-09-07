# 큰 말 모델의 떠오르는 능력
## 학습 목표

- 큰 말 모델에서의 떠오름을 정의하고 매끄러운 규모 키우기와 가른다
- 핵심 떠오르는 능력과 그것이 나타나는 규모를 가려낸다
- 떠오름이 재기의 찌꺼기인지 참된 현상인지를 둘러싼 논쟁을 이해한다
- 능력 어림과 인공지능 안전에 미치는 뜻을 살핀다

## 들어가며

떠오르는 능력이란 크기에 따라 차츰 나아지는 것이 아니라, 큰 말 모델이 어떤 규모 문턱을 넘는 순간 갑자기 나타나는 능력이다. 작은 모델에는 없거나 거의 마구잡이 수준이지만 큰 모델에서는 놀라운 힘으로 나타난다.

## 떠오름 정의하기

### 형식적 정의

다음을 채우면 그 능력을 **떠오른다**고 한다:

1. 작은 모델에는 없다
2. 큰 모델에는 있다
3. 그 바뀜이 **끊긴다**(차츰이 아니다)

수학으로 보면, 규모 $S$의 함수인 성능 잣대 $P$에 대해:

$$\frac{dP}{dS} \approx 0 \text{ for } S < S_{threshold}$$

$$\frac{dP}{dS} \gg 0 \text{ for } S \approx S_{threshold}$$

### 어림할 수 있는 규모 키우기와의 대비

| 몸짓 | 손실의 규모 키우기 | 떠오르는 능력 |
|----------|--------------|----------------------|
| 무늬 | 매끄러운 거듭제곱 법칙 | 계단 함수 |
| 어림할 수 있음 | 높음 | 낮음 |
| 보기 | 헷갈림도 | 생각의 사슬 |

## 기록된 떠오르는 능력

### BIG-Bench 살피기

Wei 외(2022)는 200개가 넘는 일에서 떠오름을 가려냈다:

| 능력 | 떠오르는 규모 | 보기 일 |
|------------|-----------------|---------------|
| 셈하기 | 매개변수 약 100억 | 세 자리 덧셈 |
| 생각의 사슬 | 매개변수 약 1000억 | 여러 걸음 따짐 |
| 낱말 풀어내기 | 매개변수 약 100억 | 글자 바꾸기 풀기 |
| 페르시아어 물음 답하기 | 매개변수 약 1000억 | 말을 넘나드는 옮김 |

### 핵심 떠오르는 능력

**1. 몇 발 맥락 안에서 배우기**
```
Small models: Random guessing regardless of examples
Large models: Learn from examples without gradient updates

Prompt: "positive: great movie → positive
         terrible film → negative  
         amazing story →"
         
GPT-2 (1.5B): Random
GPT-3 (175B): "positive" (correct)
```

**2. 생각의 사슬 따지기**
```
Question: "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. 
           How many does he have now?"

Without CoT (small): "8" (wrong)
With CoT (large): "Roger starts with 5 balls. 
                   2 cans × 3 balls = 6 balls.
                   5 + 6 = 11 balls." (correct)
```

**3. 시킴 따르기**
```
Instruction: "Translate to French without using the letter 'e'"

Small models: Ignore constraints
Large models: Follow complex instructions
```

## 상 바뀜 모델

### 통계 역학과의 비유

떠오름은 물리의 상 바뀜을 닮았다:

$$P(capability) = \frac{1}{1 + e^{-\beta(S - S_c)}}$$

여기서:

- $S$ = 모델의 규모
- $S_c$ = 결정적 규모(문턱값)
- $\beta$ = 넘어감의 날카로움

```python
import numpy as np
import matplotlib.pyplot as plt

def emergence_curve(scale, critical_scale, sharpness=1.0):
    """떠오르는 능력을 S자 상전이로 나타낸다."""
    return 1 / (1 + np.exp(-sharpness * (np.log10(scale) - np.log10(critical_scale))))

scales = np.logspace(8, 12, 100)  # 매개변수 1억에서 1조까지

plt.figure(figsize=(10, 6))
for task, s_c, beta in [
    ("Arithmetic", 1e10, 2),
    ("Chain-of-thought", 1e11, 3),
    ("Complex reasoning", 5e11, 4)
]:
    perf = emergence_curve(scales, s_c, beta)
    plt.semilogx(scales, perf, label=task, linewidth=2)

plt.xlabel("Parameters")
plt.ylabel("Task Performance")
plt.title("Emergent Capabilities vs Scale")
plt.legend()
plt.grid(True, alpha=0.3)
```

## 떠오름 논쟁

### "떠오름은 신기루다"(Schaeffer 외, 2023)

**주장**: 떠오름은 다음에서 온 찌꺼기일 수 있다:

1. **비선형 잣대**: 문턱값 전까지 정확도가 0이다가 튀어 오른다
2. **모자란 해상도**: 시험한 모델 크기가 넉넉하지 않다
3. **일을 띄엄띄엄 나눔**: 되고 안 되고의 두 값이 차츰 나아짐을 가린다

**증거**:
```python
def smooth_capability(scale, alpha=0.3):
    """바탕에 깔린 매끄러운 나아짐."""
    return (scale / 1e12) ** alpha

def discrete_metric(capability, threshold=0.5):
    """두 값 잣대가 겉보기 떠오름을 만든다."""
    return 1.0 if capability > threshold else 0.0

# 바탕 능력은 같고 잣대만 다르다
scales = np.logspace(9, 12, 50)
smooth = [smooth_capability(s) for s in scales]
discrete = [discrete_metric(c) for c in smooth]

# smooth는 차츰 나아짐을 보인다
# discrete는 갑작스러운 "떠오름"을 보인다
```

### 맞선 주장

1. **본디 띄엄띄엄한 잣대도 있다**: 여러 걸음 따짐은 모든 걸음이 맞아야 한다
2. **두루 통함의 무늬**: 성능만이 아니라 새로운 능력의 갈래이다
3. **질의 차이**: 그저 "더 나은" 것이 아니라 "다른" 것이다

## 시사하는 것

### 능력 어림에 대해

**어려움**: 떠오르는 능력은 나타나기 전에 어림하기 어렵다

```python
def capability_forecast_uncertainty(
    current_scale: float,
    target_scale: float,
    known_emergent_thresholds: list
) -> dict:
    """
    능력 내다보기의 불확실함을 어림한다.
    
    떠오름 문턱이 될 만한 곳을 넘을 때 불확실함이 커진다.
    """
    scale_ratio = target_scale / current_scale
    
    # 범위 안에서 떠오름 문턱이 될 만한 곳을 살핀다
    potential_emergences = [
        t for t in known_emergent_thresholds 
        if current_scale < t <= target_scale
    ]
    
    return {
        'scale_increase': scale_ratio,
        'potential_new_capabilities': len(potential_emergences),
        'predictability': 'low' if potential_emergences else 'moderate'
    }
```

### 인공지능 안전에 대해

1. **어림할 수 없는 능력**: 위험한 능력이 갑자기 떠오를 수 있다
2. **값매김의 틈**: 아직 없는 능력은 시험할 수 없다
3. **다스리기의 어려움**: 규모가 커지면 몸짓이 질적으로 바뀔 수 있다

### 익히기 결정에 대해

| 헤아릴 점 | 뜻하는 바 |
|---------------|-------------|
| 떠오름의 흔들림 | 안전 여유를 둔다 |
| 능력 시험 | 가운데 규모에서 시험한다 |
| 셈 나누기 | 본전을 뽑으려면 문턱값에 이르러야 할 수 있다 |

## 떠오름 재기

### 떠오름의 세기 수로 나타내기

```python
def emergence_score(
    performances: list,
    scales: list,
    random_baseline: float = 0.0
) -> float:
    """
    능력이 얼마나 "떠오르는" 것인지 재어 값으로 나타낸다.
    
    점수가 높을수록 더 갑작스러운 옮아감이다.
    
    인수:
        performances: 규모마다의 일 성능
        scales: 모델 규모(매개변수)
        random_baseline: 아무렇게나 찍었을 때의 성능
        
    반환값:
        떠오름 점수(0 = 매끄러움, 1 = 계단 함수)
    """
    # 성능을 고르게 한다
    perf_range = max(performances) - random_baseline
    if perf_range == 0:
        return 0.0
    
    normalized = [(p - random_baseline) / perf_range for p in performances]
    
    # 미분의 흩어짐으로 "계단다움"을 셈한다
    derivatives = np.diff(normalized) / np.diff(np.log10(scales))
    
    # 미분의 흩어짐이 크다 = 갑작스러운 뜀 = 떠오름
    if np.mean(np.abs(derivatives)) == 0:
        return 0.0
    
    return np.std(derivatives) / np.mean(np.abs(derivatives))
```

### 여러 잣대 값매김

참된 떠오름과 잣대의 찌꺼기를 가리려면:

```python
def evaluate_emergence_robustness(
    model_outputs: dict,
    scales: list
) -> dict:
    """
    같은 능력을 여러 잣대로 값매김한다.
    
    모든 잣대가 떠오름을 보이면 → 참일 가능성이 크다
    따로 떨어진 잣대만 떠오름을 보이면 → 헛것일 가능성이 크다
    """
    metrics = {
        'accuracy': lambda x: x['correct'] / x['total'],
        'partial_credit': lambda x: x['partial_score'],
        'log_probability': lambda x: x['target_logprob'],
        'brier_score': lambda x: 1 - x['calibration_error']
    }
    
    emergence_by_metric = {}
    for metric_name, metric_fn in metrics.items():
        perfs = [metric_fn(model_outputs[s]) for s in scales]
        emergence_by_metric[metric_name] = emergence_score(perfs, scales)
    
    return {
        'metrics': emergence_by_metric,
        'robust_emergence': min(emergence_by_metric.values()) > 0.5
    }
```

## 떠오르는 능력 목록

### 떠오름이 확인됨(여러 연구)

| 능력 | 어림 문턱값 | 증거의 세기 |
|---------|----------------------|-------------------|
| 여러 걸음 셈하기 | 100억~1000억 | 셈 |
| 생각의 사슬 | 600억~1000억 | 셈 |
| 코드 만들기 | 100억 이상 | 보통 |
| 낱말 유추 | 100억~500억 | 보통 |

### 논쟁 중 / 잣대에 매임

| 능력 | 비고 |
|---------|-------|
| 참됨 | 더 나은 잣대로 보면 차츰 나아질 수 있다 |
| 상식 | 정의에 매인다 |
| 시킴 따르기 | 시킴의 복잡도에 매인다 |

## 요약

1. **떠오름** = 규모에서 갑자기 나타나는 능력
2. **보기**: 생각의 사슬, 복잡한 셈하기, 시킴 따르기
3. **논쟁**: 어떤 떠오름은 잣대의 찌꺼기일 수 있다
4. **뜻하는 바**: 어림의 어려움, 안전의 걱정거리
5. **좋은 버릇**: 잣대를 여럿 쓰고 가운데 규모에서 시험한다

## 핵심 통찰

$$\boxed{\text{Emergence} = \text{Qualitative change, not just quantitative improvement}}$$

떠오름이 "참된" 것이든 찌꺼기이든, 실전에서 뜻하는 바는 그대로다. 곧 **규모가 커지면 능력이 뜻밖에 바뀔 수 있다**.

## 참고 문헌

1. Wei, J., et al. (2022). Emergent Abilities of Large Language Models. *TMLR*.
2. Schaeffer, R., et al. (2023). Are Emergent Abilities of Large Language Models a Mirage? *NeurIPS*.
3. Ganguli, D., et al. (2022). Predictability and Surprise in Large Generative Models. *FAccT*.

## 연습문제

**연습문제 1.**
친칠라 규모 법칙을 말하여라. 이는 모델 크기와 익힘 토막 사이의 가장 좋은 셈 나눔을 어떻게 바꾸는가?

??? success "연습문제 1 풀이"
    친칠라 크기 법칙(호프만 외, 2022)에 따르면 셈을 가장 잘 쓰는 익힘에서는 모델 크기 $N$과 익힘 토막 수 $D$을 똑같이 키워야 한다. 곧 $N \propto C^{0.5}$이고 $D \propto C^{0.5}$이며 $C$은 셈 예산이다. 그러면 가장 좋은 견줌은 대략 $D \approx 20N$이다(매개변수마다 토막 20개). 그전 버릇(GPT-3 따위)은 큰 모델을 덜 익혔다. GPT-3(매개변수 1750억 개)은 토막 3000억 개로 익혔지만, 친칠라에 맞춘 길은 700억 모델을 토막 1조 4000억 개로 익혀 같은 셈으로 더 나은 됨됨이를 이룬다.

---

**연습문제 2.**
큰 말 모델의 떠오르는 능력이란 무엇인가? 이 생각은 왜 논쟁거리인가?

??? success "연습문제 2 풀이"
    떠오르는 능력이란 모델 크기가 어떤 문턱을 넘으면 갑자기 나타나고 작은 모델에는 없어 보이는 능력이다. 보기로 여러 자리 셈하기, 생각의 사슬 따지기, 낱말 풀어내기가 있다. 논쟁은 이렇다. Schaeffer 외(2023)는 "떠오름"이 끊긴 잣대(딱 맞음 정확도)를 쓴 데서 온 찌꺼기라고 주장한다. 매끄러운 잣대(토막 수준 로그 확률)로 재면 성능이 규모에 따라 차츰 어림할 수 있게 나아진다. 논쟁의 고갱이는 떠오름이 모델의 성질인지 값매김 방법의 성질인지이다. 두 관점 모두 일리가 있다. 곧 질적인 능력의 문턱은 있지만 그것이 밑바탕의 매끄러운 나아짐으로 어림될 수도 있다.

---

**연습문제 3.**
큰 모델을 익힐 때의 모델 나란히 하기, 자료 나란히 하기, 물길 나란히 하기의 차이를 밝혀라.

??? success "연습문제 3 풀이"
    **자료 나란히 하기**: GPU마다 온전한 모델 복사본을 지니고 서로 다른 자료 묶음을 다루며, 기울기를 모두 모으기로 맞춘다. 모델이 GPU 하나에 들어가야 한다는 제한이 있다. **모델/텐서 나란히 하기**: 층을 GPU에 나눠 쪼갠다(보기로 눈길 머리 쪼개기). 더 큰 모델을 쓸 수 있지만 대역폭이 넓은 이음이 필요하다. **물길 나란히 하기**: 잇닿은 층을 GPU에 나눈다. GPU마다 단계별로 서로 다른 작은 묶음을 다룬다. 주고받기가 줄지만 물길 거품(노는 때)이 생긴다. 요즘 체계(Megatron-LM, DeepSpeed)는 셋을 모두 아우른다. 곧 마디 안에서는 텐서 나란히 하기, 마디 사이에는 물길 나란히 하기, 묶음 사이에는 자료 나란히 하기를 쓴다.

---

**연습문제 4.**
셈 예산이 $10^{23}$ 뜨는 셈 횟수로 붙박여 있다면 모델 크기와 익힘 자료에 어떻게 나누어 쓰겠는가? 크기 법칙으로 답을 뒷받침하여라.

??? success "연습문제 4 풀이"
    $C = 6ND$(변환기 익힘의 어림 뜨는 셈 횟수)을 쓰는 친칠라 크기 법칙을 쓴다. $10^{23} = 6ND$이고 가장 좋은 $D \approx 20N$이다. 넣어 보면 $10^{23} = 6N \cdot 20N = 120N^2$이므로 $N \approx \sqrt{10^{23}/120} \approx 2.9 \times 10^{10}$, 곧 매개변수 약 290억 개다. 익힘 토막은 $D = 10^{23}/(6 \times 29 \times 10^9) \approx 5750$억 개다. 이렇게 나누면 셈을 가장 잘 쓰는 견줌을 따르므로, 같은 셈 예산에서 덜 익힌 더 큰 모델(1750억을 토막 960억 개로)이나 지나치게 익힌 더 작은 모델(70억을 토막 2조 4000억 개로)보다 낫다.
