# Chain-of-Thought 프롬프팅

Chain-of-Thought(CoT) 프롬프팅은 중간 풀이 예시나 단계적 문제 해결 지시를 제공해 최종 답을 생성하는 접근이다. [Wei 등(2022)](https://arxiv.org/abs/2201.11903)은 여러 계산·상식·기호 과제에서 그 효과를 연구했다. 관측된 성능 향상과 특정 모델·과제의 조건을 구분하고, 생성된 설명이 실제 내부 판단을 그대로 드러낸다고 가정하지 않는다.

## 1. 시범 없는 단계적 문제 해결

[Kojima 등(2022)](https://arxiv.org/abs/2205.11916)은 과제별 풀이 시범 없이 단계적 해결을 유도하는 지시를 평가했다. 이런 zero-shot CoT와 풀이·정답이 있는 시범을 제공하는 few-shot CoT는 다른 설정이다. 특정 문구가 모든 모델과 과제에서 성능을 개선하는 보편적 규칙은 아니다.

```
Q: A company's stock price was \$150. It dropped 20% on Monday,
   then rose 25% on Tuesday. What is the final price?

A: Let's think step by step.
   Step 1: Starting price = \$150
   Step 2: After 20% drop: \$150 × 0.80 = \$120
   Step 3: After 25% rise: \$120 × 1.25 = \$150
   Final price: \$150
```

이 블록은 문제와 풀이가 함께 작성된 개념 예이지 실제 모델에서 생성한 관측값이 아니다. 주어진 수치의 산술은 150×0.8×1.25=150이다. 하락률과 상승률의 기준값이 다르므로 −20%와 +25%를 단순 합산하지 않는다.

## 2. 풀이 시범 제공

```python
cot_examples = [
    {
        "question": "A bond has face value $1000, coupon 5%, trades at \$950. "
                    "What is the current yield?",
        "reasoning": "Annual coupon = \$1000 × 5% = \$50. "
                     "Current yield = \$50 / \$950 = 5.26%.",
        "answer": "5.26%"
    },
]
```

cot_examples는 문제·풀이·답 문자열을 담은 목록이다. 이를 입력에 넣는 템플릿·토크나이저·모델 호출·채점은 구현하지 않는다. 예시의 계산 50/950≈0.0526은 주어진 연간 쿠폰과 가격의 비율이며 다른 수익률 개념과 혼동하지 않는다. 금융 판단이나 실제 채권 자료의 검증 결과가 아니다.

시범의 정확성·대표성·순서·길이를 검토하고 독립 시험 정답을 예시 선택에 사용하지 않는다. 모델이 단순히 형식이나 바이어스된 단서를 따라 답하는지도 확인해야 한다.

## 3. 기대 효과와 제한

문제를 하위 계산으로 나누고 중간 결과를 활용하는 출력 형식은 일부 여러 단계 과제를 도울 수 있다. 그러나 오류가 뒤 단계로 전파되거나 그럴듯한 설명과 오답이 함께 나올 수도 있다. 긴 출력·표본 집계·검증 호출은 시간과 토큰 비용을 늘린다.

원 논문의 특정 모델 수치를 모든 모델의 고정 향상률로 일반화하지 않는다. 같은 모델·프롬프트·자료·시도 수·답 파싱에서 직접 답하는 기준선과 비교해야 한다. 단순 분류에서는 효과가 작거나 악화될 수 있지만 과제 이름만으로 유무를 확정하지 않는다.

## 4. 잘못된 정확도 비교 피하기

각 단계가 조건부로 확률 p로 성공하고 모든 n단계가 맞아야 한다는 단순 가정이면 전체 성공률은 p^n이다. 단계적 풀이를 출력한다고 이 전체 성공률이 자동으로 p가 되는 것은 아니다. p=0.95, n=5면 p^n≈0.774이며 이는 실제 모델의 오류 독립성이나 자가 수정 능력을 측정한 값이 아니다.

CoT가 단계별 p를 바꾸거나 도구·검증이 오류를 줄일 수는 있지만 별도 증거가 필요하다. ‘직접 답은 p^n, CoT는 p’라는 비교는 단계별 확률과 전체 확률을 혼용한다. 더 긴 설명 자체를 정확도 보장으로 취급하지 않는다.

## 5. 답 집계와 검증

여러 경로를 샘플링해 답을 집계하는 [self-consistency](https://arxiv.org/abs/2203.11171)는 단일 CoT와 다른 추론 예산을 사용한다. 답의 숫자·단위·동의 표현을 정규화하고 표본 수·온도·동점 폴리시를 정해야 한다. 다수결은 상관된 오류를 제거하지 못하고 실제 사실을 검증하는 절차도 아니다.

검증 가능한 계산은 별도의 산술·도구·근거와 대조하고 필요한 정도의 설명과 최종 답을 평가한다. 이 페이지의 예시와 기존 코드는 그대로 두었으며 모델을 호출하거나 새 출력을 생성하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
zero-shot CoT와 few-shot CoT의 입력 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    zero-shot은 해당 과제의 정답 풀이 시범 없이 단계적 해결 지시를 준다. few-shot은 문제·중간 풀이·정답 시범을 함께 제공한다. 둘 다 일반 프롬프트 호출에서는 웨이트를 갱신하지 않으며 실제 효과는 같은 모델과 독립 자료에서 비교해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
단계별 성공률 p와 전체 성공률을 혼용한 비교가 왜 잘못인지 설명하라.

</div>

??? success "연습문제 2 풀이"

    모든 단계가 필요하고 각 단계의 조건부 성공률이 p라는 가정에서는 전체가 p^n이다. CoT를 출력해도 이 식이 자동으로 p로 바뀌지 않는다. 단계 오류·의존성·검증·재시도의 변화를 별도로 측정해야 하며 설명 길이만으로 자가 수정 성공을 가정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정답과 생성된 설명의 충실성을 별도로 살펴야 하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    정답을 내도 설명이 사후 합리화이거나 잘못된 단계를 포함할 수 있고, 자연스러운 설명이 오답을 가릴 수도 있다. 최종 답·단위와 검증 가능한 계산·근거를 확인하되 출력 텍스트를 내부 계산의 완전한 기록이라고 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 자동차 시범과 기차 문제를 각각 계산하고 단위를 비교하라.

</div>

??? success "연습문제 4 풀이"

    자동차의 총 시간은 30/60+45/45=1.5시간이다. 기차의 총 거리는 60×2.5+80×1.5=270마일이다. 속도·거리·시간의 관계를 문제의 목표에 맞춰 적용해야 한다. 아래는 시범 프롬프트이며 이 편집에서 실제 모델의 정답률을 측정하지 않는다.

    ```
    Solve the following math problem step by step.

    Q: A car drives 30 miles at 60 mph, then 45 miles at 45 mph.
    What is the total time?
    A: Let's think step by step.
    Time for first segment: 30/60 = 0.5 hours.
    Time for second segment: 45/45 = 1 hour.
    Total time: 0.5 + 1 = 1.5 hours.

    Q: A train travels at 60 mph for 2.5 hours, then at 80 mph
    for 1.5 hours. What is the total distance?
    A: Let's think step by step.
    ```


## 정리하며

CoT는 단계적 해결을 유도하는 프롬프트 구성이다. 시범·호출·표본 집계와 검증 예산을 구분하고, 전체 정답률과 설명의 타당성을 확인하되 자동적인 정확성·충실성을 가정하지 않는다.

**참고 문헌**

1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs." *NeurIPS*.
2. Kojima, T., et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS*.
