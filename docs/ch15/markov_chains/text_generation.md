# 글 만들기

text_generation.py (모듈 09) 마르코프 사슬로 글 만들기

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
text_generation.py (단원 09)

마르코프 사슬로 글 만들기
====================================

Location: 06_markov_chain/03_applications/
난이도: ⭐⭐ 기초
예상 시간: 2시간

학습 목표:
- 글 자료로 마르코프 모형 세우기
- n-그램 모형으로 새 글 만들기
- k차 마르코프 사슬 이해하기
- 글 분석 쓰임새 구현하기

수학적 바탕:
글을 위한 k차 마르코프 사슬:
- 상태 = 낱말 k개의 늘어놓음(k-그램)
- 옮김 = 다음 낱말의 확률
- P(w_{n+1} | w_n, w_{n-1}, ..., w_1) = P(w_{n+1} | w_n, ..., w_{n-k+1})
"""

import numpy as np
from collections import defaultdict, Counter
import random

# ========================================================================
# 메인
# ========================================================================


class MarkovTextGenerator:
    """마르코프 사슬로 글 만들기."""
    
    def __init__(self, order=1):
        """
        글 만들개 첫값 잡기.
        
        매개변수:
            order (int): 마르코프 사슬의 차수(살필 앞선 낱말의 수)
        """
        self.order = order
        self.model = defaultdict(Counter)
        self.start_states = []
    
    def train(self, text):
        """글로 모형 익히기."""
        words = text.split()
        
        # 시작할 수 있는 상태 저장
        for i in range(len(words) - self.order):
            state = tuple(words[i:i+self.order])
            if i == 0 or words[i-1] in '.!?':
                self.start_states.append(state)
            
            # 옮김 세기
            next_word = words[i+self.order]
            self.model[state][next_word] += 1
    
    def generate(self, length=50, seed=None):
        """주어진 길이의 글 만들기."""
        if seed:
            random.seed(seed)
        
        # 무작위 시작 상태로 시작
        current_state = random.choice(self.start_states if self.start_states else list(self.model.keys()))
        result = list(current_state)
        
        for _ in range(length - self.order):
            if current_state not in self.model:
                break
            
            # 다음 낱말의 확률 얻기
            next_words = self.model[current_state]
            total = sum(next_words.values())
            
            # 다음 낱말 표집
            choices = list(next_words.keys())
            weights = [next_words[w]/total for w in choices]
            next_word = random.choices(choices, weights=weights)[0]
            
            result.append(next_word)
            current_state = tuple(result[-self.order:])
        
        return ' '.join(result)


# 사용 예
if __name__ == "__main__":
    sample_text = """
    날쌘 갈색 여우가 게으른 개를 뛰어넘는다. 개는 나무 아래에서 자고 있었다.
    브루클린에 나무가 자란다. 브루클린은 뉴욕의 자치구이다. 뉴욕은 결코 잠들지 않는다.
    날쌘 고양이가 공원을 가로질러 달린다. 공원은 봄에 아름답다.
    봄은 뜰에 새 생명을 불어넣는다. 뜰에는 꽃이 많다.
    """
    
    print("MARKOV CHAIN TEXT GENERATION")
    print("=" * 70)
    
    for order in [1, 2]:
        print(f"\\nOrder {order} Markov Chain:")
        gen = MarkovTextGenerator(order=order)
        gen.train(sample_text)
        
        for i in range(3):
            print(f"  Generated {i+1}: {gen.generate(length=20)}")```

## 2. 논의

이 구현은 깔끔하고 읽기 좋은 PyTorch 코드로 마르코프 사슬의 핵심 개념을 보인다. 모듈로 나뉘어 있어 부분마다 따로 살펴보고 다른 일감이나 자료 묶음에 맞춰 고치기 쉽다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 확률 과정 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 마르코프 사슬에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
글 만들기 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_markovtextgenerator():
        model = MarkovTextGenerator(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 글 만들기

이 구현은 깔끔하고 읽기 좋은 PyTorch 코드로 마르코프 사슬의 핵심 개념을 보인다.

고갱이 갈래는 `MarkovTextGenerator`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
