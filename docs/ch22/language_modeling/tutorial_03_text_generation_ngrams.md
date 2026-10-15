# n-그램으로 텍스트 생성

바이그램 생성기는 직전 단어에 따른 분포에서 다음 단어를 선택한다. 같은 학습 빈도를 사용해도 선택 규칙에 따라 결과가 달라진다. 이 예제는 그리디 디코딩, 온도 샘플링, top-k 샘플링을 비교한다. 생성 문장이 자연스러워 보이는지와 모델의 확률·일반화 성능은 별도로 평가해야 한다.

## 1. 코드와 기존 출력

```python
"""
길잡이 03: n-그램으로 글 만들어 내기
==========================================

이 길잡이는 n-그램 말 모델로 글을 만들어 내는 법을 보여 준다.
여러 만들어 내기 전략과 그 성질을 살펴본다.

표집 전략:
1. 욕심쟁이: w_next = argmax P(w | 맥락)
2. 마구잡이: 온 분포에서 뽑기
3. 온도: P'(w) = P(w)^(1/T) / Z
4. 상위 k: 가장 그럴듯한 낱말 k개에서 뽑기
"""

import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Optional
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class TextGeneratorBigram:
    """바이그램 말 모델을 쓴 글 생성기."""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        for sentence in corpus:
            words = sentence.lower().split()
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)
    
    def get_next_word_distribution(self, context: str) -> List[Tuple[str, float]]:
        context = context.lower()
        next_word_counts = self.bigram_counts[context]
        if not next_word_counts:
            prob = 1.0 / len(self.vocab)
            return [(word, prob) for word in self.vocab]
        total_count = sum(next_word_counts.values())
        distribution = [(word, count / total_count) 
                       for word, count in next_word_counts.items()]
        distribution.sort(key=lambda x: x[1], reverse=True)
        return distribution
    
    def generate_greedy(self, max_length: int = 20) -> str:
        current_word = self.START_TOKEN
        generated = []
        for _ in range(max_length):
            distribution = self.get_next_word_distribution(current_word)
            if not distribution:
                break
            next_word = distribution[0][0]
            if next_word == self.END_TOKEN:
                break
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            current_word = next_word
        return ' '.join(generated)
    
    def generate_temperature(self, temperature: float = 1.0,
                            max_length: int = 20) -> str:
        current_word = self.START_TOKEN
        generated = []
        for _ in range(max_length):
            distribution = self.get_next_word_distribution(current_word)
            if not distribution:
                break
            words, probs = zip(*distribution)
            modified_probs = [p ** (1.0 / temperature) for p in probs]
            total = sum(modified_probs)
            modified_probs = [p / total for p in modified_probs]
            next_word = random.choices(words, weights=modified_probs, k=1)[0]
            if next_word == self.END_TOKEN:
                break
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            current_word = next_word
        return ' '.join(generated)
    
    def generate_top_k(self, k: int = 5, max_length: int = 20) -> str:
        current_word = self.START_TOKEN
        generated = []
        for _ in range(max_length):
            distribution = self.get_next_word_distribution(current_word)
            if not distribution:
                break
            top_k_distribution = distribution[:min(k, len(distribution))]
            words, probs = zip(*top_k_distribution)
            total = sum(probs)
            probs = [p / total for p in probs]
            next_word = random.choices(words, weights=probs, k=1)[0]
            if next_word == self.END_TOKEN:
                break
            if next_word != self.START_TOKEN:
                generated.append(next_word)
            current_word = next_word
        return ' '.join(generated)


if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
    ]
    gen = TextGeneratorBigram()
    gen.train(corpus)
    
    print("Greedy:", gen.generate_greedy())
    for temp in [0.5, 1.0, 1.5]:
        print(f"Temp {temp}:", gen.generate_temperature(temp))
```

**기존 출력:**

```
Greedy: the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on
Temp 0.5: the dog are friends
Temp 1.0: the cat sat on the cat sat on the cat sat on the cat sat on the cat and the
Temp 1.5: the cat sat on the cat and the cat sat on the cat sat on the log
```

## 2. 빈도 분포와 생성 종료

관측한 문맥에서는 다음 단어의 상대 빈도를 사용한다. 미관측 문맥에서는 저장된 어휘의 균등 분포로 대체한다. 이 어휘에는 시작·종료 토큰도 들어 있다. 시작 토큰을 선택하면 출력에는 추가하지 않지만 문맥은 시작으로 바뀌고, 종료 토큰을 선택하면 생성을 끝낸다. 따라서 fallback은 일반 단어만 고르는 규칙과 다르다.

max_length는 생성 반복 횟수의 상한이며 반환되는 단어 수와 항상 같지는 않다. 상한까지 종료 토큰을 뽑지 못하면 미완성 문장도 반환한다. train은 빈도를 누적하므로 같은 객체에 반복 호출하면 기존 관측이 남는다. 학습 전의 빈 어휘에 대한 균등 분포도 유효하게 정의되어 있지 않다.

## 3. 그리디 디코딩과 온도

그리디 디코딩은 각 단계의 최대 확률 후보를 선택한다. 고정된 모델과 동점 처리에서는 결과가 일정하지만, 단계별 최선이 문장 전체 확률의 최선을 보장하지는 않는다. 빈도 동점은 후보 순서에 영향을 받으며 미관측 문맥의 후보는 set 순서에 의존한다. 확률이 높은 전이를 따라 반복 고리에 들어갈 수도 있다.

온도 T>0의 분포는 P_T(w)=P(w)^(1/T)/Σ_v P(v)^(1/T)다. 낮은 온도는 높은 확률에 무게를 더 주고 높은 온도는 분포를 평평하게 한다. T→0에서는 최대 확률 후보들에 질량이 모이며 동점이 있으면 단일 그리디 선택과 다르다. T→∞에서는 현재 양의 확률을 가진 후보들 사이의 균등 분포에 접근한다. 관측 문맥에서 확률 0인 단어까지 자동으로 생성하지는 않는다.

현재 구현은 T의 양수 여부를 검사하지 않고 확률의 거듭제곱을 직접 계산한다. T=0은 정의되지 않고 매우 작은 T에서는 값이 모두 underflow되어 정규화가 실패할 수 있다. 이론적 극한과 해당 수치 구현의 안전성을 구분한다.

## 4. top-k와 문맥의 한계

top-k는 확률순 상위 k개를 남기고 그 안에서 다시 정규화해 샘플링한다. 작은 k는 선택 폭을 줄이고 큰 k는 낮은 확률의 후보를 포함할 수 있다. 어느 k도 문장의 정확성·자연스러움을 보장하지 않는다. 현재 코드는 k의 상한만 후보 수에 맞추므로 양의 정수인지도 확인해야 한다.

바이그램은 직전 단어 하나만 사용하므로 긴 문장의 주제·문법 의존성을 충분히 표현하지 못한다. 다른 샘플링 규칙이 없는 장기 문맥을 만들어 주지는 않는다. 신경 언어 모델은 더 긴 문맥을 다룰 수 있지만 그 역시 학습·평가와 생성 품질 검토가 필요하다. 본문에는 별도 검증·시험 성능 측정이 없으며 기존 출력도 재실행하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
P=(0.5,0.3,0.1,0.1)에 대해 T=0.5와 T=2의 온도 분포를 계산하고 정규화와 반올림 오차를 설명하라.

</div>

??? success "연습문제 1 풀이"

    T=0.5에서는 제곱한 값 (0.25,0.09,0.01,0.01)의 합이 0.36이다. 정규화하면 (25/36,1/4,1/36,1/36)이고 합은 정확히 1이다. T=2에서는 (√0.5,√0.3,√0.1,√0.1)을 그 합으로 나눈다. 약 (0.3747,0.2902,0.1676,0.1676)이며 표시 자릿수의 반올림 때문에 표시값의 합은 정확히 1이 아닐 수 있다. 온도 감소는 높은 확률에 더 집중시키고 증가는 상대적으로 평평하게 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
고정 k가 뾰족한 분포와 평평한 분포에서 갖는 한계 및 top-p 샘플링의 후보 선택을 설명하라.

</div>

??? success "연습문제 2 풀이"

    한 후보에 질량이 집중되면 큰 k가 낮은 확률 후보를 많이 포함할 수 있다. 반대로 평평한 분포에서 작은 k는 비슷하게 가능한 후보를 제외한다. top-p는 확률 내림차순으로 누적합이 p 이상이 되는 최소 접두 후보 집합을 남겨 정규화한다. 후보 수는 분포에 따라 달라지지만 생성 품질을 보장하지 않는다. 본문의 생성 클래스에는 top-p가 구현되어 있지 않다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 빔 서치의 점수와 탐색 범위를 설명하고 완료 문장을 반환한다는 보장이 있는지 검토하라.

</div>

??? success "연습문제 3 풀이"

    점수는 다음 단어 확률의 자연로그 합이며 단계마다 상위 beam_width개의 진행 가설만 남긴다. 제한된 빔 탐색은 전체 최적 문장을 보장하지 않고 길이 정규화가 없는 로그확률은 짧은 가설에 유리할 수 있다. 코드는 종료 가설뿐 아니라 반복 종료 시 남은 미완료 빔도 complete에 넣는다. 후보가 없어 break한 경우에도 이전 빔이 그대로 추가되므로 종료 전의 높은 점수로 완료 문장을 이길 수 있다. 따라서 완료된 문장 중 최선을 반환한다고 설명할 수 없다. 완료·미완료 가설의 종료 정책, 양의 빔 폭과 동점 처리를 명시하고 이 함수를 클래스 메서드로 연결해야 한다.

    ```python
    def generate_beam_search(self, beam_width: int = 3, max_length: int = 20) -> str:
        # 빔마다: (로그 확률, 낱말 목록, 마지막 낱말)
        beams = [(0.0, [], self.START_TOKEN)]
        complete = []
        
        for _ in range(max_length):
            candidates = []
            for log_prob, words, last_word in beams:
                distribution = self.get_next_word_distribution(last_word)
                for next_word, prob in distribution:
                    if prob <= 0:
                        continue
                    new_log_prob = log_prob + math.log(prob)
                    if next_word == self.END_TOKEN:
                        complete.append((new_log_prob, words))
                    elif next_word != self.START_TOKEN:
                        candidates.append((new_log_prob, words + [next_word], next_word))
            
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
        
        # 아직 안 끝난 빔을 후보에 더하기
        for log_prob, words, _ in beams:
            complete.append((log_prob, words))
        
        if not complete:
            return ""
        
        best = max(complete, key=lambda x: x[0])
        return ' '.join(best[1])
    ```


## 정리하며

생성 규칙은 동일한 바이그램 분포에서도 다른 문장을 만든다. 온도·후보 제한의 효과와 수치적 조건을 구분하고, 시작·종료·미관측 문맥 및 미완료 가설의 처리까지 확인해야 결과를 올바르게 해석할 수 있다.
