# 프롬프팅 개요

프롬프팅은 지시·맥락·예시와 입력 형식으로 모델의 출력을 조건화하는 방법이다. 일반적인 텍스트 프롬프트를 사용할 때 그 호출 자체가 모델 웨이트를 갱신하지는 않는다. 그러나 파인튜닝과 상호 배타적인 대안은 아니며, 지시 파인튜닝된 모델에도 프롬프트가 필요하다. 좋은 평가 자료와 품질 확인의 필요성도 사라지지 않는다.

## 1. 입력에 포함할 정보

과제의 목표와 출력 형식, 실제 처리할 입력, 근거 자료, 필요하면 시범을 구분해 제공한다. 자료의 기준 시점·단위·출처와 답할 수 없는 경우의 폴리시도 과제에 맞춰 명시할 수 있다. 역할을 부여하는 문구만으로 전문성이나 사실 정확성이 보장되지는 않는다.

```
[System] You are a financial analyst specializing in equity research.
         Always cite specific metrics and provide confidence levels.

[Context] Company: NVIDIA (NVDA)
          Q3 FY2025 Revenue: \$35.1B (+94% YoY)
          Data Center Revenue: \$30.8B (+112% YoY)

[Query] Provide a brief assessment of NVIDIA's growth trajectory.
```

이 블록의 System·Context·Query는 개념적 구분이며 실제 메시지 API를 호출하지 않는다. 포함된 회사 수치는 예시 입력이지 이 편집에서 공시를 검증한 자료가 아니다. 신뢰도를 쓰라고 요구하는 것만으로 보정된 확률을 얻거나, 인용을 요구하는 것만으로 실제 근거가 확보되는 것도 아니다.

외부 문서의 문장을 모델에 대한 명령과 구분하고 필요한 정보의 우선순위를 명확히 해야 한다. 구분자·JSON 등 형식만으로 prompt injection이 완전히 차단되지는 않으며 도구·저장소 권한과 입력 검증도 별도로 설계해야 한다.

## 2. 예시 수와 과제 구성

zero-shot은 해당 과제의 입력·정답 시범 없이 지시하는 설정, few-shot은 시범을 함께 제공하는 설정이다. 한 개면 one-shot이다. 시범은 레이블 의미와 형식을 보여 줄 수 있지만 잘못된 예시·순서·길이·분포에 민감할 수 있다. 테스트 정답을 시범 선택이나 프롬프트 튜닝에 사용하면 평가 누출이다.

프롬프트는 학습 자료가 전혀 필요 없거나 과거 모델은 항상 과제별 파인튜닝만 가능했다는 뜻의 단절이 아니다. 고정 프롬프트·검색·도구·파인튜닝을 함께 비교하고 목표 품질과 비용에 맞게 선택한다.

## 3. 여러 호출과 탐색 전략

| 전략 | 핵심 구성 | 검토할 조건 |
|------|-----------|-------------|
| 단계적 풀이 유도 | 중간 문제 분해·계산과 답을 생성 | 단계의 정확성·추가 비용 |
| Self-consistency | 여러 표본의 최종 답을 모음 | 답 정규화·상관 오류·시도 수 |
| Tree of Thoughts | 후보 단계를 생성·평가·탐색 | 평가자의 오류·탐색 예산 |
| Least-to-most | 작은 하위 문제부터 순차 해결 | 앞 답의 오류 전파 |
| ReAct 계열 | 모델 출력과 실제 도구 실행을 연결 | 도구 결과·권한·종료 조건 |

이름을 프롬프트에 적는 것만으로 여러 호출·도구·탐색이 구현되지는 않는다. [Self-consistency](https://arxiv.org/abs/2203.11171)와 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)는 서로 다른 표본 집계·탐색 구성을 연구했다. 어떤 전략도 항상 더 정확하거나 사실 검증을 대신하지 않는다.

## 4. 성능과 비용의 평가

독립 검증 자료에서 정답·형식 준수·근거·유보·지연과 토큰/호출 비용을 비교한다. 평가 시 모델 버전·시스템 지시·시범·검색 자료·디코딩과 최대 시도 수를 기록한다. 한 프롬프트의 그럴듯한 답만으로 일반적인 성능 향상을 확정하지 않는다.

단계적 풀이를 출력하도록 유도하면 일부 과제에서 도움이 될 수 있지만 출력 텍스트가 내부 계산의 충실한 기록이라는 보장은 없다. 필요한 검증 가능한 계산·짧은 설명·최종 답의 정확성에 맞춰 출력 요구를 정한다. 모델과 과제에 따라 더 긴 설명이 오히려 비용·오류를 늘릴 수도 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
zero-shot·one-shot·few-shot과 웨이트 파인튜닝을 구분하라.

</div>

??? success "연습문제 1 풀이"

    앞의 세 설정은 현재 입력에 과제 시범을 몇 개 넣는지의 차이다. 일반 텍스트 프롬프팅 호출은 웨이트를 갱신하지 않지만 파인튜닝은 학습 로스로 파라메터를 갱신한다. 시범이 없다는 사실만으로 사전 학습 중 유사 과제를 본 적이 없다고 결론 내릴 수 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
역할·인용·신뢰도 지시만으로 전문성·근거·보정이 보장되지 않는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    문구는 출력의 조건이지만 실제 지식과 근거 자료·확률 보정의 검증을 대체하지 않는다. 제공한 출처와 답의 지지 관계, 정답률·불확실성·유보를 독립 자료에서 확인해야 한다. 형식적으로 자신 있는 말이나 인용 모양이 맞는 것과 내용의 타당성은 다르다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
여러 호출 전략을 비교할 때 호출 수와 검증 예산을 맞춰야 하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    여러 표본·탐색·도구 호출은 단일 응답보다 많은 자원을 쓴다. 최종 답의 정확도 외에 총 토큰·시간·실패 처리와 선택 절차를 포함해야 공정한 비교가 된다. 다수결도 공유된 오류를 반복할 수 있고 도구 결과도 검증해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 프롬프트의 목표 답을 모델 실행 없이 계산하고 시범과 평가를 구분하라.

</div>

??? success "연습문제 4 풀이"

    기차의 두 구간 거리는 60×2.5=150마일, 80×1.5=120마일이며 합은 270마일이다. 앞 자동차 예는 시간 계산, 뒤 기차 문제는 거리 계산이라 단위와 목표를 구분해야 한다. 이 값은 산술 정답이지 실제 모델 출력이나 프롬프트의 일반 성능을 측정한 결과가 아니다.

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

프롬프팅은 지시·근거·예시와 출력 조건을 구성하는 방법이다. 실제 호출·탐색·도구 실행과 단순 문구를 구분하고, 독립 자료에서 품질·형식·비용과 실패를 함께 평가한다.

**참고 문헌**

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
2. Liu, P., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP."
