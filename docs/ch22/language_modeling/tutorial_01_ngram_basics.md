# n-그램 언어 모델 기초 예제

이 예제는 소문자 공백 단어의 빈도로 유니그램·바이그램·트라이그램을 추정한다. 신경망 학습이 아니라 최대우도 빈도 계산이다. 실제 관측된 문맥과 종료 전이, 미관측 사건의0확률을 비교해 고정 문맥의 이점·한계를 이해한다.

## 1. 코드와 기존 출력

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

**기존 출력:**

```
Trained unigram model on 28 words
Vocabulary size: 17 unique words
Trained bigram model on 33 bigrams
Vocabulary size: 17 unique words
Trained trigram model on 33 trigrams
Vocabulary size: 17 unique words
  'the cat sat': bigram log P = -inf
  'the dog played': bigram log P = -3.13
  'the elephant danced': bigram log P = -inf
```

## 2. 빈도와 문맥 길이

유니그램 p(w)=C(w)/N은 각 단어의 빈도를 사용하고 순서를 구분하지 않는다. 바이그램 p(w|h)=C(h,w)/Σ_v C(h,v), 트라이그램 p(w|h1,h2)=C(h1,h2,w)/Σ_v C(h1,h2,v)는 최근1개·2개 단어에 조건을 건다. 고차일수록 표현할 문맥은 길어지지만 자료가 희소해 평가 성능이 반드시 좋아지지는 않는다.

각 train은 기존 빈도를 초기화하지 않고 누적한다. 입력의 소문자화·공백 분리도 정의의 일부다. 구두점·미등록 단어와 한국어 형태 분석을 자동 처리하지 않으므로 다른 토큰화에서 얻은 확률과 직접 비교하지 않는다.

## 3. 시작·종료와0확률

바이그램은 시작1개·종료1개, 트라이그램은 시작2개·종료1개를 추가한다. 분모의 문맥 빈도는 다음 사건을 가진 횟수이며 일반적인 전체 단어 빈도와 역할이 다르다. 끝 단어 다음의 종료까지 관측되어야 문장 전체 확률을 계산할 수 있다.

기본 시연 말뭉치에는 sat 다음이 on이고 바로 종료하지 않는다. 따라서 the cat sat이 문법적으로 자연스러워도 이 바이그램의 종료 전이가0이면 전체 로그확률은−∞다. 단어 자체가 어휘에 있더라도 조합·종료가 없으면0이 될 수 있다. ‘본 적 없는 단어’와 ‘본 적 없는 n-그램’은 별개다.

## 4. 로그 합과 확률 모델의 한계

문장 로그확률은 예측 사건들의 log2 확률 합이다. 작은 양수를 곱할 때 underflow를 줄이지만, 실제0확률의−∞를 유한값으로 바꾸는 것은 아니다. empty sentence는 바이그램에서 시작→종료라는 사건을 검사하고 유니그램에서는 빈합0을 반환하므로 같은 길이·경계 모형이 아니다.

미관측 문맥의 확률은0으로 두며 전체 후보에 대한 정상화된 fallback은 없다. 평활화·UNK를 설계할 때 시작·종료·예측 어휘와 분모를 일치시켜야 한다. 차수 증가가 필요한 데이터를 일률적으로 지수 배수만큼 요구한다거나 실전 n이 반드시5이하라는 제한은 없다. 가능한 조합 수|V|^n과 실제 관측·저장량·평가의 희소성을 구분한다.

## 5. 다음 단어 선택과 평가 범위

most_likely_next는 관측된 문맥의 최대 빈도 후보와 상대 빈도를 반환하는 추가 메서드 예다. 반환값은 문자열 하나가 아니라 `(단어,확률)`이다. 동점은 사전 순서의 영향을 받고, 미관측 문맥의 UNK·0 반환은 정상화된 예측 분포와 다르다. 관측 사전이 비어 있는 경우도 별도 계약이 필요하다.

이 파일은 고정 자료의 빈도·로그확률을 시연하며 별도 검증·시험 PPL을 측정하지 않는다. 비교에는 동일 어휘·UNK·토큰화·종료 규칙과 독립 자료를 사용한다. 코드·기존 출력은 유지하며 이 편집에서 다시 실행하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
말뭉치 the cat sat / the dog sat / a cat ran에서 P(sat|cat)과 종료 포함 the cat sat의 log2 확률을 계산하라.

</div>

??? success "연습문제 1 풀이"

    P(sat|cat)=1/2다. 시작→the는2/3, the→cat은1/2, cat→sat은1/2, sat→종료는1이므로 문장 확률1/6, log2는약−2.585다. 이 연습 자료와 본문 기본 시연의 문장들은 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
어휘에 있는 단어들로 만든 문장도0확률을 얻을 수 있는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    관측된 n-그램이나 문장 종료 전이가 없으면 MLE가0을 줄 수 있다. 가능한 조합은|V|^n이지만 실제 저장은 관측 조합만으로 가능하다. 차수·전처리·자료에 따라 희소성·성능을 검증하고 정상화된 평활화·UNK 정책을 설계한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
가장 가능성 높은 다음 단어와 확률을 반환하는 아래 most_likely_next의 계약을 설명하라.

</div>

??? success "연습문제 3 풀이"

    반환 타입은 Tuple[str,float]이며 미관측 문맥에는 UNK·0을 돌려준다. 관측된 비어 있지 않은 후보에서 최대 빈도를 고르고 문맥 횟수로 나눈다. 동점 처리·빈 후보·정규화된 fallback과 모델에 실제 메서드로 연결하는 과정은 별도로 정의해야 한다.

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


## 정리하며

n-그램 빈도 모델은 문맥과 종료 사건이 관측되었는지에 민감하다. 로그 계산·희소 조합·어휘 밖 단어를 구분하고 차수·평활화·평가의 토큰화·경계 계약을 일치시킨다.
