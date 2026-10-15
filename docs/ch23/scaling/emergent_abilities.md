# 대규모 언어 모델의 창발적 능력

창발적 능력(emergent abilities)은 작은 모델의 결과만으로 쉽게 예상하지 못한 성능 변화가 큰 규모에서 관측되는 현상을 가리키는 연구 용어다. 그 정의·측정과 원인에는 논쟁이 있다. 유한한 몇 모델의 점수가 급격히 변했다는 사실을 수학적인 불연속이나 새로운 내부 메커니즘의 증명과 동일시하지 않는다.

## 1. 모델 변화와 측정 방식

[Wei 등(2022)](https://arxiv.org/abs/2206.07682)은 언어 모델의 여러 능력에서 작은 규모의 결과로 예측하기 어려운 변화를 논의했다. [Schaeffer 등(2023)](https://arxiv.org/abs/2304.15004)은 불연속적·비선형 지표와 적은 측정 지점이 일부 급격한 변화를 만들 수 있음을 분석했다. 후자가 모든 능력의 미래 변화가 완전히 예측 가능하다고 증명한 것은 아니다.

모델 크기만이 아니라 자료·학습 계산·토크나이저·후속 학습·프롬프트·시도 횟수·정답 판정이 결과를 바꾼다. ‘계산은 100억, 추론은 1000억에서 출현’ 같은 고정 문턱을 모든 모델 계열에 적용할 수 없다. 작은 모델은 항상 무작위이고 큰 모델은 항상 시범을 따른다는 이분법도 맞지 않는다.

## 2. 기존 개념 예시의 범위

```
Small models: Random guessing regardless of examples
Large models: Learn from examples without gradient updates

Prompt: "positive: great movie → positive
         terrible film → negative  
         amazing story →"
         
GPT-2 (1.5B): Random
GPT-3 (175B): "positive" (correct)
```

이 예시는 ICL의 아이디어를 보여 주는 기존 도식이며 GPT-2·GPT-3의 통제된 실측이 아니다. 가중치 갱신 없이 시범에 조건을 건다는 뜻과 특정 모델이 반드시 오답·정답을 낸다는 주장을 구분한다.

```
Question: "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. 
           How many does he have now?"

Without CoT (small): "8" (wrong)
With CoT (large): "Roger starts with 5 balls. 
                   2 cans × 3 balls = 6 balls.
                   5 + 6 = 11 balls." (correct)
```

공의 수는 5+2×3=11이지만, 이 도식은 모델 크기·CoT·프롬프트를 동시에 바꾸어 비교한다. 여기서 어느 요인이 성능을 바꿨는지 분리할 수 없다. 중간 추론을 출력했다고 모든 단계가 사실상 올바르거나 그 텍스트가 내부 계산의 완전한 설명이라는 보장도 없다.

```
Instruction: "Translate to French without using the letter 'e'"

Small models: Ignore constraints
Large models: Follow complex instructions
```

지시 준수 역시 작은 모델이 언제나 실패하고 큰 모델이 언제나 성공하는 성질이 아니다. 제약의 난이도·표현·검증 기준·후속 학습을 맞추어 평가해야 한다. 이 세 도식을 새로운 실험 증거로 제시하지 않는다.

## 3. 연속 곡선과 이산 성공 지표

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

emergence_curve는 log10(scale)에 대한 매끄러운 sigmoid다. critical_scale에서 0.5가 되고 sharpness는 기울기를 바꾸지만 유한한 값에서 수학적 불연속은 없다. 그래프의 과제 이름·임계 크기는 사용자가 정한 값이며 실제 모델을 학습·평가해 추정한 결과가 아니다. 이 편집에서는 해당 그림을 실행·재생성하지 않는다.

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

smooth_capability와 discrete_metric은 매끄러운 수치에 임계값을 적용하면 계단처럼 보일 수 있다는 구성 예다. 실제 전체 정답률도 여러 부분 성공이 함께 필요하면 짧은 구간에서 빠르게 변할 수 있다. 반대로 확률 지표가 매끄럽다는 이유만으로 실제 과제의 성공 조건을 무의미하다고 할 수는 없다. 두 관점의 결과와 의미를 함께 보고한다.

## 4. 예측 함수와 검증의 차이

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

capability_forecast_uncertainty는 주어진 임계값 목록에서 현재·목표 규모 사이의 원소 수를 센다. 새로운 문턱을 발견하거나 확률을 추정하지 않는다. ‘low’·‘moderate’는 고정 규칙이며 보정된 예측 신뢰도가 아니다. 양수 규모·중복 임계값·역방향 비교의 정책도 필요하다.

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

emergence_score는 log10 규모 간 차분 기울기의 변동을 평균 절댓값으로 나눈 임의 진단량이다. 큰 값이 ‘참된 창발성’을 입증하지 않으며 모델 수·간격·노이즈에 민감하다. 감소·비단조 곡선에서도 커질 수 있다. 최소 두 개 이상의 길이가 맞는 관측, 양수이며 중복 없는 규모와 유효한 baseline을 요구한다.

## 5. 여러 지표의 일치와 한계

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

이 함수의 brier_score는 1−calibration_error로 정의되어 있지만 실제 Brier 점수는 예측 확률과 원핫 정답의 제곱 오차 평균이다. 보정 오차와 같지 않고 보통 낮을수록 좋으므로 다른 지표와 방향도 맞춰야 한다. model_outputs에 필요한 값이 있다고 해서 지표 정의가 자동으로 맞는 것은 아니다.

최소 emergence_score가 0.5보다 크면 robust_emergence라는 규칙도 통계적으로 검증된 판정이 아니다. 지표마다 성공의 의미·baseline·불확실성을 확인해야 하며 여러 임의 점수의 동의가 현상의 원인을 증명하지 않는다.

## 6. 실제로 조사하는 방법

동일 모델 계열·자료·학습 예산 관계와 평가 조건을 기록하고 중간 규모·여러 시드를 포함한다. 정확 일치, 부분 점수, 정답 로그확률·보정과 오류 유형을 함께 살핀다. 가능한 경우 작은 규모에서 만든 예측을 미리 고정하고 더 큰 규모의 독립 결과로 검증한다.

안전과 능력 평가는 변화가 확실히 나타난 뒤에만 시작할 필요는 없다. 작은 모델·부분 성공·도구 사용·다른 프롬프트 조건에서도 실패와 잠재 능력을 조사할 수 있다. ‘아직 없는 능력은 전혀 시험할 수 없다’거나 특정 파라미터 문턱에 도달해야만 자원의 가치가 생긴다고 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
급격한 정확도 변화가 수학적 불연속을 증명하지 못하는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"

    관측 모델 수가 적고 정확 일치 등 지표가 비선형이면 매끄러운 기초 변화도 급격하게 보일 수 있다. 유한 지점만으로 연속성이나 내부 메커니즘을 확정할 수 없다. 중간 규모·부분 점수·로그확률과 반복 불확실성을 함께 확인한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
emergence_curve 그래프가 실제 능력 문턱의 증거가 아닌 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    과제별 critical_scale과 sharpness를 사람이 정한 연속 sigmoid다. 모델의 실제 측정값·적합이나 통계 검정이 없다. log10 규모의 형태와 0.5 지점을 설명할 수는 있지만 모든 모델의 계산·CoT 문턱을 알려 주지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
1−calibration_error를 Brier 점수로 부를 수 없는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    Brier는 정답과 예측 확률의 제곱 오차 평균이며 calibration error와 다른 양이다. 원래 Brier는 낮을수록 좋고 분류·집계 관례를 명시해야 한다. 지표 이름만 바꾸거나 임의의 0.5 기준을 적용해 진짜 창발성을 판정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
도식의 작은 모델 무CoT 오답과 큰 모델 CoT 정답만으로 크기의 인과 효과를 판단할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"

    크기와 프롬프트·추론 출력 조건이 동시에 달라지고 반복 측정도 없다. 동일 자료·프롬프트 조건에서 크기를 바꾸거나 같은 모델에 CoT 유무를 비교해 요인을 나누어야 한다. 단일 정답 시연과 일반적인 성능 분포를 구분한다.



## 정리하며

창발성의 해석에는 모델·평가 지표·측정 해상도를 함께 고려해야 한다. 합성 곡선과 임의 진단 점수를 실험 증거로 취급하지 않고, 여러 지표·중간 규모·독립 예측 검증으로 실제 변화를 조사한다.

**참고 문헌**

1. Wei, J., et al. (2022). Emergent Abilities of Large Language Models. *TMLR*.
2. Schaeffer, R., et al. (2023). Are Emergent Abilities of Large Language Models a Mirage? *NeurIPS*.
3. Ganguli, D., et al. (2022). Predictability and Surprise in Large Generative Models. *FAccT*.
