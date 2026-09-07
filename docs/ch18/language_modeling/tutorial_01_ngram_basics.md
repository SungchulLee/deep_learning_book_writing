# n-그램 말 모델의 기초

n-그램 말 모델은 말 나타내기에서 가장 단순하고 읽어 내기 쉬운 방식으로, 앞선 $n-1$개 낱말을 보고 낱말의 확률을 어림한다. 이는 요즘 신경 말 모델이 서 있는 바탕이며, 그 센 점과 한계를 아는 것이 깊은 배움의 나아감을 제대로 헤아리는 데 꼭 필요한 맥락을 준다.

## 코드

```python
"""
길잡이 01: n-그램 말 모델 — 기초
==============================================

이 길잡이는 n-그램 말 모델의 근본 개념을 소개한다.
말 나타내기에서 가장 단순하고 읽어 내기 쉬운 방식이다.

학습 목표:
--------------------
1. n-그램이 무엇이고 왜 쓸모 있는지 이해한다
2. 유니그램, 바이그램, 트라이그램 모델 세우기
3. 최대 가능도 어림으로 확률 셈하기
4. n-그램 모델의 한계를 이해한다

수학적 바탕:
------------------------
n-그램은 글에서 잇닿은 낱말 n개의 차례이다.

유니그램 모델(n=1):
- P(w) = count(w) / total_words
- 낱말끼리 안 얽힌다고 가정한다

바이그램 모델(n=2):
- P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
- 맥락 낱말 하나를 쓴다

트라이그램 모델(n=3):
- P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})
- 맥락 낱말 둘을 쓴다

차례의 확률:
- P(w_1, w_2, ..., w_n) = P(w_1) * P(w_2|w_1) * ... * P(w_n|w_{n-k},...,w_{n-1})
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import math

# ========================================================================
# 메인
# ========================================================================


class UnigramModel:
    """낱말끼리 안 얽힌다고 가정하는 유니그램 말 모델."""
    
    def __init__(self):
        self.word_counts = Counter()
        self.total_words = 0
        self.vocab = set()
    
    def train(self, corpus: List[str]) -> None:
        for sentence in corpus:
            words = sentence.lower().split()
            self.word_counts.update(words)
            self.total_words += len(words)
            self.vocab.update(words)
        print(f"Trained unigram model on {self.total_words} words")
        print(f"Vocabulary size: {len(self.vocab)} unique words")
    
    def probability(self, word: str) -> float:
        word = word.lower()
        if word not in self.vocab:
            return 0.0
        return self.word_counts[word] / self.total_words
    
    def log_probability(self, word: str) -> float:
        prob = self.probability(word)
        if prob == 0.0:
            return float('-inf')
        return math.log2(prob)
    
    def sentence_log_probability(self, sentence: str) -> float:
        words = sentence.lower().split()
        log_prob = 0.0
        for word in words:
            log_prob += self.log_probability(word)
            if log_prob == float('-inf'):
                return float('-inf')
        return log_prob


class BigramModel:
    """낱말마다 앞 낱말에 조건을 거는 바이그램 말 모델."""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        total_bigrams = 0
        for sentence in corpus:
            words = sentence.lower().split()
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                if w1 != self.START_TOKEN:
                    self.vocab.add(w1)
                if w2 != self.END_TOKEN:
                    self.vocab.add(w2)
                total_bigrams += 1
        print(f"Trained bigram model on {total_bigrams} bigrams")
        print(f"Vocabulary size: {len(self.vocab)} unique words")
    
    def probability(self, word: str, context: str) -> float:
        word = word.lower()
        context = context.lower()
        if context not in self.unigram_counts:
            return 0.0
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        return bigram_count / context_count if context_count > 0 else 0.0
    
    def sentence_log_probability(self, sentence: str) -> float:
        words = sentence.lower().split()
        words = [self.START_TOKEN] + words + [self.END_TOKEN]
        log_prob = 0.0
        for i in range(len(words) - 1):
            context, word = words[i], words[i + 1]
            prob = self.probability(word, context)
            if prob == 0.0:
                return float('-inf')
            log_prob += math.log2(prob)
        return log_prob


class TrigramModel:
    """낱말마다 앞선 낱말 둘에 조건을 거는 트라이그램 말 모델."""
    
    def __init__(self):
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        total_trigrams = 0
        for sentence in corpus:
            words = sentence.lower().split()
            words = [self.START_TOKEN, self.START_TOKEN] + words + [self.END_TOKEN]
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i + 1], words[i + 2]
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
                for w in [w1, w2, w3]:
                    if w not in [self.START_TOKEN, self.END_TOKEN]:
                        self.vocab.add(w)
                total_trigrams += 1
        print(f"Trained trigram model on {total_trigrams} trigrams")
        print(f"Vocabulary size: {len(self.vocab)} unique words")
    
    def probability(self, word: str, context1: str, context2: str) -> float:
        word = word.lower()
        context1 = context1.lower()
        context2 = context2.lower()
        trigram_count = self.trigram_counts[context1][context2][word]
        bigram_count = self.bigram_counts[context1][context2]
        return trigram_count / bigram_count if bigram_count > 0 else 0.0


def demonstrate_ngram_models():
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog played",
        "cats and dogs are friends",
        "the quick brown fox jumps"
    ]
    
    unigram = UnigramModel()
    unigram.train(corpus)
    
    bigram = BigramModel()
    bigram.train(corpus)
    
    trigram = TrigramModel()
    trigram.train(corpus)
    
    test_sentences = ["the cat sat", "the dog played", "the elephant danced"]
    for sent in test_sentences:
        print(f"  '{sent}': bigram log P = {bigram.sentence_log_probability(sent):.2f}")


if __name__ == "__main__":
    demonstrate_ngram_models()
```

## 논의

n낱말 모델은 사슬 법칙과 마르코프 가정을 써서 월의 확률을 조건부 확률의 곱으로 쪼갠다. 한낱말 모델은 낱말마다 서로 아랑곳없다고 보고 $P(w) = \text{count}(w) / N$을 셈한다. 여기서 $N$은 온 낱말 수다. 이는 낱말 잦기는 담지만 앞뒤 흐름을 모두 버린다. 두낱말 모델은 바로 앞 낱말 하나를 조건으로 삼아 $P(w_i \mid w_{i-1}) = \text{count}(w_{i-1}, w_i) / \text{count}(w_{i-1})$을 쓰며, "the the"보다 "the cat"이 더 그럴듯하다는 단순한 매임을 담는다. 세낱말 모델은 이를 앞 낱말 둘로 넓혀 더 긴 무늬를 담되 세는 수가 더 성글어진다.

n-그램 모델의 근본 맞바꿈은 나타내는 힘과 자료의 성김 사이에 있다. $n$이 커질수록 더 먼 얽힘을 담아낼 수 있지만 가능한 n-그램의 수가 지수로 늘어나, 올바른 n-그램이 익힘 자료에 한 번도 안 나올 가능성이 점점 커진다. 이 "확률 0" 문제는 본 적 없는 바이그램이나 트라이그램 하나 때문에 월 전체의 확률이 0이 되어 그 들임에 모델을 쓸 수 없게 만든다. 이 한계 때문에 (이어지는 길잡이에서 다룰) 부드럽게 하기 재주가 나왔고, 끝내는 이어진 나타냄으로 두루 통하는 신경 말 모델의 발전으로 이어졌다.

여기 짜보기에서는 수치가 든든하도록 내내 로그 확률을 쓴다. 작은 확률을 여럿 곱하면 뜨는 수가 아래로 넘치지만 로그 확률을 더하면 수치가 든든하다. $\log P(w_1, \ldots, w_n) = \sum_i \log P(w_i \mid \text{context}_i)$ 덕에 내내 로그 밭에서 셈하고, 풀이할 때만 확률로 되돌리면 된다.

## 연습문제

**연습문제 1.**
Given the corpus `["the cat sat", "the dog sat", "a cat ran"]`, compute by hand the bigram probability $P(\text{sat} \mid \text{cat})$ and the sentence log probability (base 2) for the sentence "the cat sat" under a bigram model.

??? success "연습문제 1 풀이"
    먼저 경계 토막을 더한 뒤 바이그램을 센다:
    
    - 월 1: `<s> the cat sat </s>` — 바이그램: (`<s>`,the), (the,cat), (cat,sat), (sat,`</s>`)
    - 월 2: `<s> the dog sat </s>` — 바이그램: (`<s>`,the), (the,dog), (dog,sat), (sat,`</s>`)
    - 월 3: `<s> a cat ran </s>` — 바이그램: (`<s>`,a), (a,cat), (cat,ran), (ran,`</s>`)
    
    $P(\text{sat} \mid \text{cat}) = \text{count}(\text{cat}, \text{sat}) / \text{count}(\text{cat}) = 1 / 2 = 0.5$
    
    For "the cat sat": $P(\text{the} \mid \langle s \rangle) = 2/3$, $P(\text{cat} \mid \text{the}) = 1/2$, $P(\text{sat} \mid \text{cat}) = 1/2$, $P(\langle /s \rangle \mid \text{sat}) = 2/2 = 1$.
    
    $\log_2 P = \log_2(2/3) + \log_2(1/2) + \log_2(1/2) + \log_2(1) \approx -0.585 - 1.0 - 1.0 + 0 = -2.585$

---

**연습문제 2.**
낱말이 저마다 낱말 곳간에 있을 수 있는데도 트라이그램 모델이 "the elephant danced"에 확률 0을 주는 까닭을 밝혀라. 자료의 성김 문제는 $n$에 따라 어떻게 커지는가?

??? success "연습문제 2 풀이"
    The trigram model computes $P(w_i \mid w_{i-2}, w_{i-1})$. Even if "elephant" and "danced" appear in the vocabulary individually, the specific trigram (`<s>`, `<s>`, the), (the, elephant, ...) etc., may never have been observed. Since the MLE estimate is count-based, any unobserved trigram gets probability zero.
    
    Data sparsity scales exponentially with $n$. For a vocabulary of size $V$, there are $V^n$ possible n-grams. With $V = 10{,}000$: unigrams = $10^4$, bigrams = $10^8$, trigrams = $10^{12}$. Most of these will never appear in any realistic training corpus, making higher-order n-grams increasingly sparse. This is why practical n-gram models rarely go beyond $n = 5$, and even trigram models require smoothing.

---

**연습문제 3.**
맥락 낱말이 주어질 때 가장 그럴듯한 다음 낱말과 그 확률을 돌려주는 `most_likely_next(self, context: str) -> str` 메서드를 `BigramModel` 클래스에 짜라.

??? success "연습문제 3 풀이"
    ```python
    def most_likely_next(self, context: str) -> Tuple[str, float]:
        """맥락이 주어질 때 가장 그럴듯한 다음 낱말 돌려주기."""
        context = context.lower()
        if context not in self.bigram_counts:
            return ("<unk>", 0.0)
        
        next_word_counts = self.bigram_counts[context]
        total = self.unigram_counts[context]
        
        best_word = max(next_word_counts, key=next_word_counts.get)
        best_prob = next_word_counts[best_word] / total
        
        return (best_word, best_prob)
    
    # 쓰는 법:
    # model = BigramModel()
    # model.train(corpus)
    # word, prob = model.most_likely_next("the")
    # print(f"Most likely after 'the': '{word}' (P={prob:.4f})")
    ```
