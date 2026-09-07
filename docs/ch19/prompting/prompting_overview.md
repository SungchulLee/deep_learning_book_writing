# 시킴말 재주 개요
## 학습 목표

- 곱게 다듬기에서 시킴말로의 틀 바뀜을 이해한다
- 잘 듣는 시킴말의 짜임을 가려낸다
- 시킴말 전략을 복잡도와 쓰임새로 나눈다

## 시킴말이라는 틀

예로부터의 자연어 다루기는 뒤따르는 일마다 그 일에 맞춘 곱게 다듬기가 필요했다. 큰 말 모델은 근본적으로 다른 틀을 들여왔다. 곧 매개변수를 조금도 고치지 않고 자연어 시킴으로 모델의 몸짓에 조건을 거는 **시킴말**이다.

엄밀히 말하면 일에 맞춘 매개변수 $\theta_{\text{task}}$을 배우는 대신 다음을 채우는 들임 $\text{prompt}(x)$을 짓는다.

$$\hat{y} = \text{LLM}(\text{prompt}(x); \theta_{\text{pretrained}})$$

이러면 이름표 붙인 익힘 자료와 일마다의 모델 복사본이 필요 없어진다.

## 시킴말의 짜임

잘 짜인 시킴말은 대개 네 조각을 담는다:

1. **체계 시킴**: 맡을 몫의 정의와 몸짓의 제약
2. **맥락**: 그 일과 맞닿는 바탕 앎
3. **보기**(없어도 됨): 들임-내놓음 시범
4. **물음**: 다룰 구체적인 들임

```
[System] You are a financial analyst specializing in equity research.
         Always cite specific metrics and provide confidence levels.

[Context] Company: NVIDIA (NVDA)
          Q3 FY2025 Revenue: \$35.1B (+94% YoY)
          Data Center Revenue: \$30.8B (+112% YoY)

[Query] Provide a brief assessment of NVIDIA's growth trajectory.
```

## 시킴말 전략의 갈래

| 전략 | 핵심 생각 | 알맞은 곳 |
|----------|---------|---------|
| 영 발 | 보기 없이 곧바른 시킴 | 단순한 갈래 매기기, 뽑기 |
| 몇 발 | 들임-내놓음 보기를 넣는다 | 꼴에 민감한 일, 드문 무늬 |
| 생각의 사슬 | 한 걸음씩 따지기를 이끌어 낸다 | 수학, 논리, 여러 걸음 살피기 |
| 생각의 나무 | 여러 따짐 길을 살펴본다 | 복잡한 계획, 가장 좋게 하기 |
| 스스로 한결같기 | 여러 길에 대한 다수결 | 셈하기, 사실 물음 |
| ReAct | 따짐과 행동을 번갈아 한다 | 연장 쓰기, 앎 모으기 |

## 참고 문헌

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
2. Liu, P., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP."

## 연습문제

**연습문제 1.**
영 발, 몇 발, 생각의 사슬 시킴말을 밝혀라. 저마다 보기를 들어라.

??? success "연습문제 1 풀이"
    **영 발**: 모델이 일의 시킴만 받는다. 보기: "Translate to French: Hello" => "Bonjour". **몇 발**: 시킴말에 보기가 들어 있다. 보기: "English: cat => French: chat. English: dog => French: " => "chien". **생각의 사슬**: 시킴말이 한 걸음씩 따지는 것을 보여 준다. 보기: "물음: 가게에 사과가 5개 있고 2개를 팔면 몇 개가 남는가? 한 걸음씩 생각해 보자. 가게는 사과 5개로 시작한다. 2개를 팔면 5-2=3개가 남는다. 답: 3." 생각의 사슬은 셈하기, 논리, 여러 걸음 따짐 일에 특히 잘 듣는다.

---

**연습문제 2.**
생각의 사슬 시킴말은 왜 따져 보는 일의 성능을 높이는가? 한계는 무엇인가?

??? success "연습문제 2 풀이"
    생각의 사슬이 도움이 되는 까닭은 다음과 같다: (1) 복잡한 문제를 다룰 만한 잔걸음으로 쪼갠다. (2) 중간 걸음이 모델의 속셈 너머의 "일하는 기억" 노릇을 한다. (3) 따져 봄이 또렷이 드러난 익히기 자료(교과서, 길잡이)를 흉내낸다. **한계**: (1) 토막 씀씀이와 늦음이 는다. (2) 그럴듯하지만 틀린 따져 봄 사슬을 낼 수 있다("충실하지 않은 생각의 사슬"). (3) 여러 걸음의 따져 봄이 필요 없는 일에는 도움이 되지 않는다. (4) 성능이 본보기 따져 봄 사슬의 품질에 크게 기댄다. (5) 작은 모델은 아예 덕을 보지 못할 수 있다.

---

**연습문제 3.**
앞선 시킴말 전략으로서 스스로 한결같음, 생각의 나무, 쉬운 것부터 시킴말을 견주어라.

??? success "연습문제 3 풀이"
    **스스로 한결같기**: 생각의 사슬 따짐 길을 여럿 뽑아 마지막 답을 다수결로 정해, 길 하나만 쓰는 것보다 튼튼하게 한다. **생각의 나무**: 걸음마다 너비 우선이나 깊이 우선으로 따짐 가지를 여럿 살펴보며 큰 말 모델이 어느 가지가 유망한지 값매김해 되돌아갈 수 있게 한다. **쉬운 것부터**: 복잡한 문제를 점점 어려워지는 작은 문제로 쪼개어 차례로 풀고 앞선 답 위에 쌓는다. 스스로 한결같기가 가장 단순하고(짜임을 바꾸지 않는다), 생각의 나무가 가장 유연하지만 값이 비싸며, 쉬운 것부터는 조합해 두루 통하는 데 뛰어나다.

---

**연습문제 4.**
생각의 사슬 따지기를 쓰는 수학 말 문제 시킴말을 꾸며라. "기차가 시속 60마일로 2.5시간, 그다음 시속 80마일로 1.5시간 달린다. 전체 거리는 얼마인가?"로 시험하여라.

??? success "연습문제 4 풀이"
    시킴말:
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
    바라는 내놓음: 첫 구간의 거리는 $60 \times 2.5 = 150$마일. 둘째 구간의 거리는 $80 \times 1.5 = 120$마일. 온 거리는 $150 + 120 = 270$마일.
