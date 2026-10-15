# ReAct: 추론과 행동의 결합

ReAct는 언어 모델이 작업 진행에 관한 설명과 도구 사용을 번갈아 생성하고, 도구의 결과를 다음 판단에 반영하도록 구성하는 방식이다. 외부 정보를 조회하거나 환경과 상호작용해야 하는 작업에서 활용할 수 있다. 출력 형식을 정했다는 사실만으로 사실성이나 도구 사용의 올바름이 보장되지는 않는다.

## 1. 행동과 관찰의 반복

[Yao 등의 ReAct 논문](https://arxiv.org/abs/2210.03629)은 언어로 된 추론 기록과 작업별 행동을 결합하는 접근을 제시한다. 도구의 관찰 결과로 다음 행동을 수정할 수 있다는 점이 외부 상호작용 없이 답하는 방식과 다르다.

```
Thought: I need to find AAPL's latest earnings data
Action: search("AAPL Q3 2024 earnings results")
Observation: AAPL reported Q3 revenue of \$85.8B...

Thought: Now I need to compare with analyst estimates
Action: search("AAPL Q3 2024 consensus estimate")
Observation: Consensus estimate was \$84.5B...

Thought: Revenue beat by \$1.3B (1.5%). Let me check guidance.
Action: search("AAPL Q4 2024 guidance")
Observation: Apple guided Q4 revenue of \$89-93B...

Thought: I now have enough information to provide analysis.
Answer: AAPL beat Q3 estimates by 1.5%...
```

위 실적 조회 대화는 형식을 보여 주는 설명용 기록이다. 실제 검색 결과나 검증된 금융 분석으로 제시한 것이 아니며, 특히 마지막 전망 수치에는 원문 근거가 연결되어 있지 않다. 실제 보고에서는 해당 회계 기간과 발표일, 공시 원문, 추정치 출처·시점을 확인해야 한다. Observation이라는 표지가 붙었다고 내용이 자동으로 검증되는 것은 아니다.

## 2. 개념적인 제어 루프

```python
def react_loop(llm, question, tools, max_steps=10):
    prompt = f"""Answer the following question using the available tools.

응답을 다음 꼴로 하라:
생각: [너의 따져 봄]
움직임: [연장_이름(인자)]
(살핌을 기다린다)
... 필요한 만큼 되풀이한다 ...
생각: 앎이 넉넉하다.
답: [마지막 답]

물음: {question}
"""
    history = prompt

    for step in range(max_steps):
        response = llm(history)
        history += response

        # 마지막 답이 있는지 살핀다
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
            return {"answer": answer, "steps": step + 1, "trace": history}

        # 움직임을 뽑아 실행한다
        if "Action:" in response:
            action_str = response.split("Action:")[-1].strip().split("\n")[0]
            tool_name, args = parse_action(action_str)
            observation = tools[tool_name](**args)
            history += f"\nObservation: {observation}\n"

    return {"answer": "Could not determine answer", "steps": max_steps}
```

이 함수는 실행 가능한 완제품이 아니라 흐름을 설명하는 골격이다. 프롬프트는 `생각:`, `움직임:`, `답:`이라는 한국어 표지를 요청하지만 파서는 `Action:`과 `Answer:`만 찾는다. 요청대로 한국어 형식을 출력하면 행동 실행과 종료 판정이 작동하지 않을 수 있다. `parse_action`도 정의되어 있지 않으며 도구 목록과 인수 스키마를 모델에 제공하는 부분이 없다.

실제 구현에는 일관된 응답 스키마, 허용한 도구 이름과 인수 검증, 오류·시간 초과 처리, 중단 조건과 실행 예산이 필요하다. 문자열에 `Answer:`가 포함되었다는 이유만으로 종료하면 인용문이나 도구 설명 안의 표지를 잘못 인식할 수 있다. 응답 전체를 평가하거나 임의 코드를 실행하는 파서 대신 구조화된 호출을 검증해야 한다.

모든 기록을 history에 계속 붙이므로 긴 작업에서는 문맥 길이와 비용도 증가한다. 도구 결과는 외부 데이터이며 사용자나 시스템의 지시와 같은 권한을 갖지 않는다. 짧은 행동 이유와 검증 가능한 도구·결과 기록은 유용하지만, 모델이 출력한 설명을 내부 사고 과정의 충실한 보고라고 가정할 수는 없다.

## 3. 다른 작업 방식과의 비교

| 방식 | 외부 상호작용 | 주의할 점 |
|---|---|---|
| 외부 도구 없이 답하기 | 없다 | 새 정보나 환경 상태를 확인하기 어렵다 |
| 도구 호출 중심의 루프 | 있다 | 결과 선택·해석과 종료 조건을 검증해야 한다 |
| ReAct 형태의 루프 | 설명과 행동을 결합한다 | 잘못된 설명·관찰 해석이 다음 행동으로 이어질 수 있다 |

외부 근거는 오류를 줄이는 데 도움을 줄 수 있지만 검색 결과 자체의 오류, 오래된 정보, 프롬프트 인젝션까지 없애지는 않는다. 단일 에이전트와 여러 에이전트 중 어느 구성이 더 적합한지도 작업 분해, 통신 비용과 평가 결과에 따라 결정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
ReAct의 행동·관찰 루프가 외부 도구 없이 답하는 방식과 어떻게 다른가?

</div>

??? success "연습문제 1 풀이"

    도구를 호출해 환경이나 외부 자료에서 새 정보를 받고 그 결과를 다음 행동에 반영한다. 정보 수집과 계획 수정에 도움이 될 수 있지만 관찰이 정확한지, 해석과 행동이 적절한지는 별도로 검증해야 한다. 형식만으로 오류가 방지되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
제시된 함수에서 한국어 표지와 영어 파서가 충돌하는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    모델에는 움직임·답이라는 표지를 요청하지만 함수는 Action·Answer 문자열만 확인한다. 한국어 응답이면 조건문을 통과하지 못할 수 있다. 실행기와 모델이 같은 스키마를 사용하고, 허용한 도구·인수·종료 상태를 구조적으로 검증해야 한다. parse_action 등의 외부 정의도 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
여러 에이전트로 나누는 것이 언제나 단일 에이전트보다 유리한가?

</div>

??? success "연습문제 3 풀이"

    아니다. 독립적인 하위 작업에는 병렬 처리와 역할별 검토가 도움이 될 수 있지만 통신·조정·중복 작업 비용이 늘어난다. 같은 모델의 오류가 서로 상관될 수도 있다. 작업의 의존성과 문맥 공유 범위를 정하고 정확도·비용·완료 시간으로 비교해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
도구 결과에 실행을 유도하는 문장이 포함되어 있을 때 실행기는 어떻게 다루어야 하는가?

</div>

??? success "연습문제 4 풀이"

    조회한 문장은 작업의 근거 데이터로 취급하고 상위 지시나 도구 권한으로 취급하지 않는다. 허용 도구와 인수, 작업 범위 및 필요한 권한을 실행기가 검증한다. 재시도·시간·비용 한도와 행동 로그를 두고 오류·부작용을 확인한다. 모델의 자연어 설명만을 권한 검사로 사용하지 않는다.



## 정리하며

ReAct는 도구 관찰을 다음 행동에 반영하는 작업 루프다. 응답 형식과 파서, 도구 스키마와 종료 조건을 일치시켜야 하며, 외부 정보와 모델 설명의 사실성은 별도로 확인한다.

**참고 문헌**

1. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.
