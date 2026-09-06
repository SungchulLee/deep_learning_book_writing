# ReAct: 따지고 움직이기
## 학습 목표

- ReAct라는 틀을 이해한다
- 생각-행동-살핌 되풀이를 짠다
- ReAct를 금융 연구 일에 쓴다

## 핵심 생각

**ReAct**(Yao 외, 2023)는 따짐의 자취와 행동을 번갈아 놓는다:

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

## 구현

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

## ReAct와 다른 방식

| 방법 | 따짐 | 움직임 | 뿌리내림 |
|--------|----------|--------|-----------|
| 생각의 사슬만 | 있다 | 없다 | 없다 |
| 움직임만 | 없다 | 있다 | 있다 |
| ReAct | 있다 | 있다 | 있다 |

ReAct는 따짐을 찾아온 사실에 뿌리내려 헛것 지어내기를 줄인다.

## 참고 문헌

1. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.

## 연습문제

**연습문제 1.**
ReAct 얼거리를 밝혀라. 따짐과 움직임을 번갈아 하면 큰 말 모델 몸소 움직이개의 성능이 왜 나아지는가?

??? success "연습문제 1 풀이"
    ReAct(따지고 움직이기)는 큰 말 모델이 **생각**(무엇을 할지 따지기), **행동**(연장 부르기나 API 요청 실행), **살핌**(결과 읽기)을 번갈아 하도록 시킨다. 이 번갈아 하기가 도움되는 까닭은 이렇다. (1) 따짐의 자취가 결정 과정을 읽어 내고 벌레잡기 쉽게 한다. (2) 모델이 살핀 것에 따라 계획을 그때그때 고칠 수 있다. (3) 답을 실제 연장의 내놓음에 뿌리내려 헛것 지어내기를 줄인다. (4) 짜임 있는 꼴이 모델이 걸음을 건너뛰거나 뒷받침 없는 주장을 하지 못하게 막는다.

---

**연습문제 2.**
큰 말 모델 바탕 몸소 움직이개 체계의 핵심 조각은 무엇인가? 저마다의 몫을 설명하여라.

??? success "연습문제 2 풀이"
    핵심 조각: (1) **큰 말 모델 등뼈**: 시킴을 읽어 내고 계획을 만드는 따짐 엔진. (2) **연장 사이**: 몸소 움직이개가 부를 수 있는 API나 함수의 모음(찾기, 계산기, 코드 실행). (3) **기억**: 맥락을 담는 짧은 기억(대화 발자취, 메모장)과 긴 기억(벡터 데이터베이스, 앎 곳간). (4) **계획 단원**: 일을 작은 일로 쪼개는 전략(생각의 사슬, 생각의 나무, 일 쪼개기). (5) **행동 실행기**: 큰 말 모델의 내놓음을 짜임 있는 연장 부르기로 뜯어 읽고 실행을 다룬다.

---

**연습문제 3.**
복잡한 일에서 하나가 움직이는 얼개와 여럿이 움직이는 얼개를 견주어라. 맞바꿈은 무엇인가?

??? success "연습문제 3 풀이"
    **하나가 움직이기**: 큰 말 모델 하나가 따짐, 계획, 연장 쓰기를 모두 맡는다. 세우고 벌레잡기가 더 쉽지만 맥락 창에 매이고 여러 분야의 전문성이 필요할 때 힘겨우며 무너지면 끝인 곳이 하나 있다. **여럿이 움직이기**: 특화된 몸소 움직이개 여럿(보기로 코드 짜는 이, 연구하는 이, 검토하는 이)이 저마다 좁힌 능력으로 어울린다. 여러 재주가 필요한 복잡한 일에 낫고 나란히 실행할 수 있으며 맡은 몫마다 특화된 시킴말을 쓸 수 있다. 맞바꿈: 여럿이 움직이는 체계는 지휘하기 어렵고 값이 비싸며(큰 말 모델을 여러 번 부른다) 서로 맞추기에 어그러지거나 몸소 움직이개끼리 몸짓이 어긋날 수 있다.

---

**연습문제 4.**
연장을 쓸 수 있는 큰 말 모델 몸소 움직이개를 펼칠 때 어떤 안전 걱정거리가 생기는가? 어떻게 덜 수 있는가?

??? success "연습문제 4 풀이"
    걱정거리: (1) **뜻하지 않은 행동**: 잘못 읽어 내어 해로운 명령(파일 지우기, 전자우편 보내기)을 실행할 수 있다. (2) **시킴말 끼워넣기**: 찾아온 글월에 든 맞서는 들임이 몸소 움직이개의 몸짓을 가로챌 수 있다. (3) **끝나지 않는 되풀이**: 같은 행동을 되풀이하며 갇힐 수 있다. (4) **자료 새어 나감**: 연장을 부르며 민감한 앎을 뜻하지 않게 드러낼 수 있다. 덜어 내기: 모래상자 실행 환경, 민감한 행동에 사람의 확인 받기, 허용 목록, 내놓음 거르기, 횟수 제한, 최대 걸음 수, 모든 행동의 지켜보기와 기록 남기기.
