# 생각의 사슬 시킴말
## 학습 목표

- 생각의 사슬 따지기와 그것이 언제 도움되는지 이해한다
- 영 발과 몇 발 생각의 사슬 시킴말을 짠다
- 생각의 사슬이 따짐 일의 성능을 왜 낫게 하는지 살핀다

## 핵심 생각

**생각의 사슬 시킴말**(Wei 외, 2022)은 마지막 답에 앞서 가운데 따짐 걸음을 이끌어 낸다:

$$x \xrightarrow{\text{standard}} y \quad \text{vs.} \quad x \xrightarrow{\text{CoT}} r_1, r_2, \ldots, r_n, y$$

## 영 발 생각의 사슬

Kojima 외(2022)는 **"한 걸음씩 생각해 보자"**를 덧붙이면 따짐이 크게 나아짐을 찾아냈다:

```
Q: A company's stock price was \$150. It dropped 20% on Monday,
   then rose 25% on Tuesday. What is the final price?

A: Let's think step by step.
   Step 1: Starting price = \$150
   Step 2: After 20% drop: \$150 × 0.80 = \$120
   Step 3: After 25% rise: \$120 × 1.25 = \$150
   Final price: \$150
```

## 몇 발 생각의 사슬

따짐 사슬을 드러낸 시범을 준다:

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

## 생각의 사슬이 도움될 때

생각의 사슬은 다음에서 가장 크게 나아지게 한다:

1. **여러 걸음 셈하기**: 차례차례 하는 셈
2. **논리 따지기**: 삼단논법, 연역 사슬
3. **말로 된 문제**: 자연어를 수학으로 옮기기

단순한 갈래 매기기나 무늬 짝짓기에는 생각의 사슬이 **거의 도움되지 않는다**.

## 핵심 결과

| 잣대 | 보통 | 생각의 사슬 | 나아짐 |
|-----------|---------|-----|-------------|
| GSM8K(수학) | 56.5% | 74.4% | +17.9%p |
| SVAMP(수학) | 68.9% | 79.0% | +10.1%p |
| StrategyQA | 65.4% | 73.2% | +7.8%p |

*PaLM 540B 결과(Wei 외, 2022)*

## 수학적 분석

걸음마다 정확도가 $p$이고 연산이 $n$번 필요한 여러 걸음 문제에서:

- **Direct prediction**: Accuracy $\approx p^n$
- **CoT**: Accuracy $\approx p$ per step with self-correction

For $p = 0.95$, $n = 5$: direct $\approx 0.77$, CoT $\approx 0.95$ per step.

## 참고 문헌

1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs." *NeurIPS*.
2. Kojima, T., et al. (2022). "Large Language Models are Zero-Shot Reasoners." *NeurIPS*.

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
    Expected output: Distance for first segment: $60 \times 2.5 = 150$ miles. Distance for second segment: $80 \times 1.5 = 120$ miles. Total distance: $150 + 120 = 270$ miles.
