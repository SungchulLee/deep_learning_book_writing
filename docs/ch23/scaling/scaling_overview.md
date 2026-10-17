# 대규모 언어 모델의 스케일링 법칙

스케일링 법칙은 모델 크기 N, 학습 토큰 수 D, 계산량 C와 최종 로스 사이의 실증 관계다. 비슷한 자료·구조·학습 설정의 실험으로 계수를 맞춰 자원 배분을 추정한다. 자연 법칙이나 모든 능력의 보편적 예측식이 아니며, 적합 범위 밖에서는 불확실성과 조건 변화를 함께 고려해야 한다.

## 1. 로스와 규모의 관계

[Kaplan 등(2020)](https://arxiv.org/abs/2001.08361)은 언어 모델의 교차 엔트로피 로스가 모델·자료·계산의 제한 조건 아래 거듭제곱 형태로 변하는 경향을 연구했다. N_c 같은 계수는 데이터·단위·모델과 적합에 따른 스케일 상수이지 임계 능력의 임계가 아니다.

$$L(N)\approx(N_c/N)^{\alpha_N},\qquad L(D)\approx(D_c/D)^{\alpha_D}.$$

각 식은 다른 자원이 충분한 영역의 근사이며 아무 N·D에 독립적으로 적용할 수 없다. 원 연구의 계산 최적 배분 경향을 N:D=10:1 같은 고정 비율로 요약하지 않는다. 파라미터가 큰 모델의 표본 효율과 정해진 계산에서의 최적 크기는 다른 질문이다.

## 2. Chinchilla와 계산 예산

[Hoffmann 등(2022)](https://arxiv.org/html/2203.15556v1)은 최종 로스를 다음 형태로 적합하는 접근을 포함해 계산 최적 배분을 분석했다.

$$L(N,D)=E+AN^{-\alpha}+BD^{-\beta},\qquad C\approx6ND.$$

6ND는 밀집 트랜스포머 학습의 주요 연산을 거칠게 세는 근사다. 긴 문맥의 어텐션, 재계산·통신·이용률과 하드웨어 비용을 전부 표현하지 않는다. E는 적합한 로스 바닥 항이며 모든 언어·자료에서 같은 보편적인 엔트로피 값이 아니다.

이 식을 제약 아래 미분하면 다음을 얻는다.

$$\alpha AN^{-\alpha}=\beta BD^{-\beta},$$

$$N_* =\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}
\left(\frac C6\right)^{\beta/(\alpha+\beta)},\qquad D_* =\frac C{6N_*}.$$

따라서 N과 D의 지수는 각각 β/(α+β), α/(α+β)다. α=0.34, β=0.28을 그대로 쓰면 약 0.45와 0.55이며 정확히 0.5씩은 아니다. 논문의 여러 추정에서 두 자원을 비슷한 비율로 늘린다는 근사 결론과 특정 적합식의 지수를 구분한다.

Chinchilla의 700억 파라메터·1.4조 토큰은 D/N=20인 역사적 구성이다. 이를 모든 규모·자료의 필수 비율로 고정하지 않는다. 비슷한 계산 예산으로 더 큰 Gopher보다 나은 결과를 보인 연구이지 GPT-3의 총 학습 계산과 동일하다는 뜻도 아니다.

## 3. 계산량 예제의 한계

```python
def estimate_training_flops(
    num_parameters: int,
    num_tokens: int,
    flops_per_token_per_param: float = 6.0
) -> float:
    """
    전체 익히기 FLOPs를 어림한다.
    
    6이라는 인수는 다음을 헤아린 것이다:
    - 피드포워드: 토막마다 파라메터마다 2 FLOPs
    - 뒤먹임: 토막마다 파라메터마다 4 FLOPs
    
    인수:
        num_parameters: 모델 파라메터 수
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
        가장 좋은 파라메터 수와 토막 수를 담은 사전
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

estimate_training_flops는 6ND를 계산한다. 그러나 chinchilla_optimal_config는 로스식이나 D/N 비율 없이 N=D=√(C/6)을 강제로 둔다. 계산 제약을 만족하는 한 점일 뿐 Chinchilla 최적 구성을 유도한 함수가 아니다. C>0과 유효한 계수도 필요하다.

**기존 출력:**

```
Optimal parameters: 408.2B
Optimal tokens: 0.41T
```

이 출력의 토큰/파라메터 비율은 1이다. D=20N을 가정한 계산과 다르다. 예를 들어 같은 C=10²⁴ FLOPs에서 그 비율을 가정하면 N=√(C/120)≈913억, D≈1.826조가 된다. 어느 결과도 해당 자료의 로스 적합·검증 없이 실제 최적 성능을 보장하지 않는다.

## 4. 로스 예측과 계수의 범위

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
    친칠라 잣수 맞추기 법칙으로 익히기 로스를 어림한다.
    
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

함수는 지정된 계수로 대수식을 평가한다. 토큰화·자료 혼합·파라메터 정의·훈련 스케줄이 바뀌면 같은 계수를 그대로 이식할 수 없다. 실제 모델을 학습하거나 검증 로스를 측정하는 함수가 아니다.

**기존 출력:**

```
7B, 1T tokens: predicted loss = 2.052
13B, 1T tokens: predicted loss = 2.017
7B, 2T tokens: predicted loss = 2.020
70B, 1.4T tokens: predicted loss = 1.937
```

기존 숫자는 재실행하지 않고 보존했다. 이 식의 예측이나 출력 표를 모델·데이터 구성 전체의 검증된 성능 순위로 사용하지 않는다. 계산 예산이 서로 다른 점들의 예측 로스와 같은 예산의 최적 배분도 구분해야 한다.

## 5. 학습 최적과 운영 최적

같은 학습 예산의 로스 최소화와 전체 운영 비용 최소화는 다르다. 많은 요청에 답해야 한다면 작은 모델을 더 많은 토큰으로 학습하는 편이 추론 비용을 줄일 수 있다. 단, 지연·메모리·출력 길이·품질·하드웨어와 사용량을 함께 측정해야 한다. 데이터의 양과 중복 없는 고유 정보량도 다르다.

자료 품질을 단일 Q로 두고 DQ를 대입하는 식은 별도 적합 없는 한 설명용 가정이다. 필터링·중복 제거가 항상 같은 배율로 유효 토큰 수를 바꾸는 것은 아니다. 평가와 학습의 분포를 함께 고려한다.

## 6. 외삽과 능력 예측

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

이 도식은 개념적 성능 곡선이며 특정 능력의 측정값이나 보편적 임계를 나타내지 않는다. 평균 로스의 매끄러운 변화와 정확 일치 같은 과제 지표의 급격한 변화는 함께 나타날 수 있다. 실제 능력은 지표·프롬프트·자료·구조에 따라 평가해야 한다.

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

scaling_uncertainty의 범위는 외삽 배수에 임의 계수를 곱한 예시다. 잔차·반복 실험으로 추정한 신뢰구간이나 예측구간이 아니다. 양수 배수를 확인해야 하고 1보다 작은 배수나 아주 큰 배수에서는 범위가 뒤집히거나 음수가 될 수도 있다. 통계적 불확실성은 적합 오차·표본 변동·조건 변화의 검증으로 따로 추정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
로스식 E+AN^−α+BD^−β와 C=6ND에서 최적 배분의 조건을 설명하라.

</div>

??? success "풀이"

    D=C/(6N)을 넣고 N으로 미분하면 αAN^−α=βBD^−β다. 따라서 N은 C의 β/(α+β)제곱, D는 α/(α+β)제곱에 비례한다. 두 지수의 합은 1이며 둘 다 정확히 0.5이려면 α=β가 필요하다. 계수와 자료 범위를 고정한 근사라는 조건도 남는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
chinchilla_optimal_config가 D/N=20을 구현하지 않는 이유를 설명하라.

</div>

??? success "풀이"

    함수는 N과 D를 같은 제곱근 값으로 두므로 비율이 항상 1이다. 계산 제약 하나로 두 자원의 최적값을 정할 수 없으며 로스식이나 별도 비율 가정이 더 필요하다. 함수 이름과 출력이 최적성의 증거는 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
D=20N을 가정할 때 C=10²³ FLOPs의 N·D를 계산하라.

</div>

??? success "풀이"

    C=120N²이므로 N≈2.887×10¹⁰, D≈5.774×10¹¹이다. 각각 약 289억 파라메터·5774억 토큰이다. 이 값은 고정 비율과 근사 FLOPs에 따른 계산이며 모든 실제 자료에서 다른 배분보다 반드시 낫다는 결론은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
로스 예측식·학습 FLOPs·실제 실행 비용과 불확실성 범위를 구분하라.

</div>

??? success "풀이"

    로스식은 해당 실험 조건에서 적합한 통계적 근사이고 6ND는 주요 연산량의 근사다. 실행 시간·비용에는 통신·메모리·이용률과 하드웨어가 추가된다. 임의로 만든 ±범위는 신뢰구간이 아니므로 반복 실험·잔차·외삽 범위와 분포 변화로 검증해야 한다.



## 정리하며

스케일링 법칙은 조건이 맞는 실험 영역에서 자원 배분을 돕는다. 같은 증가 지수와 같은 N·D 값, 학습 계산과 운영 비용을 구분하고 계수·예측·최적성의 근거를 명시해야 한다.

**참고 문헌**

1. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
2. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *arXiv:2203.15556*.
3. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.
