# n-그램으로 글 만들어 내기

Text generation using n-gram language models demonstrates the practical application of learned probability distributions. By sampling from the conditional distribution $P(w \mid \text{context})$ at each step, we can generate novel text that reflects the statistical patterns learned from the training corpus. Different sampling strategies -- greedy, random, temperature, and top-k -- offer various trade-offs between diversity and coherence.

## 코드

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
    """바이그램 말 모델을 쓴 글 만들개."""
    
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

## 논의

욕심쟁이 풀기는 늘 가장 그럴듯한 다음 낱말을 골라 늘 같은 내놓음을 낸다. 단순하고 빠르지만 모델이 확률 높은 고리에 갇혀 되풀이되는 글을 내기 쉽다. 보기로 여러 맥락 뒤에서 "the"가 늘 가장 그럴듯하다면 욕심쟁이 풀기는 "the"로 뒤덮인 차례를 낸다. 또 늘 같다는 성질 때문에, 한 걸음에서 덜 그럴듯한 낱말을 골라야 뒤에 더 좋은 낱말이 나오는 전체적으로 가장 좋은 차례를 놓칠 수 있다.

Temperature sampling modifies the probability distribution before sampling by raising each probability to the power $1/T$ and renormalizing. When $T < 1$, the distribution becomes sharper (more peaked around the mode), producing more conservative and repetitive text. When $T > 1$, the distribution flattens, giving lower-probability words more chance of selection and producing more creative but potentially incoherent text. At the extreme, $T \to 0$ converges to greedy decoding, while $T \to \infty$ converges to uniform random sampling over the vocabulary.

상위 k 표집은 후보를 가장 그럴듯한 낱말 $k$개로 제한한 뒤 그 잘라 낸 분포에서 뽑는다. 이러면 조리를 무너뜨릴 만큼 그럴듯하지 않은 낱말이 나오는 것을 막으면서 확률 높은 자리 안에서 여러 갈래임은 지킨다. n-그램 만들개의 핵심 한계는 먼 거리의 조리를 지키지 못한다는 것이다. 곧 바이그램 모델은 앞 낱말 하나만 "기억"하므로 만든 글이 금세 주제에서 벗어난다. 이 때문에 더 긴 맥락을 담는 숨은 상태를 지니는 신경 말 모델로 나아가게 된다.

## 연습문제

**연습문제 1.**
Given a distribution $P = \{0.5, 0.3, 0.1, 0.1\}$ over words $\{A, B, C, D\}$, compute the temperature-adjusted distribution for $T = 0.5$ and $T = 2.0$. Verify that they sum to 1.

??? success "연습문제 1 풀이"
    For $T = 0.5$: $P'(w) \propto P(w)^{1/0.5} = P(w)^2$
    
    - $A: 0.5^2 = 0.25$, $B: 0.3^2 = 0.09$, $C: 0.1^2 = 0.01$, $D: 0.1^2 = 0.01$
    - 합 = 0.36. 고르게 맞춘 값: $A: 0.694$, $B: 0.250$, $C: 0.028$, $D: 0.028$. 합 = 1.0.
    
    For $T = 2.0$: $P'(w) \propto P(w)^{1/2} = \sqrt{P(w)}$
    
    - $A: \sqrt{0.5} = 0.707$, $B: \sqrt{0.3} = 0.548$, $C: \sqrt{0.1} = 0.316$, $D: \sqrt{0.1} = 0.316$
    - 합 = 1.887. 고르게 맞춘 값: $A: 0.375$, $B: 0.290$, $C: 0.168$, $D: 0.168$. 합 = 1.0.
    
    온도가 낮으면 무게가 가장 그럴듯한 낱말에 몰리고, 높으면 무게가 더 고르게 퍼진다.

---

**연습문제 2.**
확률 분포가 아주 뾰족할 때와 아주 평평할 때, 붙박이 $k$을 쓰는 상위 k 표집이 왜 탈이 날 수 있는지 밝혀라. 알갱이(상위 p) 표집은 이 한계를 어떻게 다루는가?

??? success "연습문제 2 풀이"
    붙박이 $k$을 쓸 때, 분포가 아주 뾰족하면(보기로 한 낱말의 확률이 0.95) $k = 50$개를 넣는 것은 뽑힐 수도 있는 확률 낮고 상관없는 후보를 잔뜩 더해 좋음을 떨어뜨린다. 거꾸로 분포가 평평하면(여러 낱말의 확률이 비슷하면) $k = 5$은 그럴듯한 후보를 빼 버려 여러 갈래임을 줄인다.
    
    알갱이(상위 p) 표집은 후보 모음의 크기를 그때그때 맞춘다. 곧 쌓인 확률이 문턱값 $p$(보기로 0.95)을 넘는 가장 작은 낱말 모음을 고른다. 한 낱말이 판을 잡으면 알갱이가 작고(아마 1~2개), 분포가 평평하면 알갱이가 커져 여러 낱말을 담는다. 이렇게 만들어 내기 걸음마다 그 자리의 분포 꼴에 맞춰 여러 갈래임과 좋음의 맞바꿈을 조절한다.

---

**연습문제 3.**
걸음마다 `beam_width`개의 가설을 지니고 점수가 가장 높은 온전한 차례를 돌려주는 `generate_beam_search` 메서드를 `TextGeneratorBigram` 클래스에 짜라.

??? success "연습문제 3 풀이"
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
