# n-그램 부드럽게 하기 재주

부드럽게 하기 재주는 n-그램 말 모델의 확률 0 문제를 다룬다. 최대 가능도 어림에서는 본 적 없는 n-그램의 확률이 0이 된다. 부드럽게 하지 않으면 본 적 없는 바이그램 하나 때문에 월 전체의 확률이 0으로 주저앉아 새 자료에 두루 통하지 못한다. 이 길잡이에서는 바탕이 되는 세 방식, 곧 라플라스 부드럽게 하기, k 더하기 부드럽게 하기, 선형 사이 끼움을 다룬다.

## 1. 코드

```python
"""
길잡이 02: n-그램 부드럽게 하기 재주
=========================================

이 길잡이는 n-그램 모델의 확률 0 문제를 다루는 부드럽게 하기 재주를 다룬다.
본 적 없는 n-그램을 다루려면 부드럽게 하기가 꼭 필요하다.

라플라스 부드럽게 하기(하나 더하기):
P_laplace(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)

k 더하기 부드럽게 하기:
P_add-k(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k*V)

선형 사이 끼움:
P_interp(w_i | w_{i-1}) = lambda_2 * P_ML(w_i | w_{i-1}) + lambda_1 * P_ML(w_i)
"""

import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

# ========================================================================
# 메인
# ========================================================================


class LaplaceBigramModel:
    """라플라스(하나 더하기) 부드럽게 하기를 쓴 바이그램 말 모델."""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        word_counts = Counter()
        for sentence in corpus:
            words = sentence.lower().split()
            word_counts.update(words)
        
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= min_freq}
        self.vocab.add(self.UNK_TOKEN)
        self.vocab_size = len(self.vocab)
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
    
    def probability(self, word: str, context: str) -> float:
        word = word.lower()
        context = context.lower()
        if word not in self.vocab:
            word = self.UNK_TOKEN
        if context not in self.vocab and context != self.START_TOKEN:
            context = self.UNK_TOKEN
        
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        numerator = bigram_count + 1
        denominator = context_count + self.vocab_size
        return numerator / denominator
    
    def perplexity(self, test_corpus: List[str]) -> float:
        total_log_prob = 0.0
        total_words = 0
        for sentence in test_corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            for i in range(len(words) - 1):
                prob = self.probability(words[i + 1], words[i])
                total_log_prob += math.log2(prob)
                total_words += 1
        cross_entropy = -total_log_prob / total_words
        return 2 ** cross_entropy


class InterpolatedBigramModel:
    """선형 사이 끼움 부드럽게 하기를 쓴 바이그램 모델."""
    
    def __init__(self, lambda2: float = 0.7, lambda1: float = 0.3):
        assert abs(lambda2 + lambda1 - 1.0) < 1e-6
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_words = 0
        self.vocab = set()
        self.START_TOKEN = "<s>"
        self.END_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        word_counts = Counter()
        for sentence in corpus:
            words = sentence.lower().split()
            word_counts.update(words)
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= min_freq}
        self.vocab.add(self.UNK_TOKEN)
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [word if word in self.vocab else self.UNK_TOKEN 
                    for word in words]
            words = [self.START_TOKEN] + words + [self.END_TOKEN]
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                if w1 != self.START_TOKEN:
                    self.unigram_counts[w1] += 1
                    self.total_words += 1
    
    def probability(self, word: str, context: str) -> float:
        p_unigram = (self.unigram_counts.get(word, 0) + 1) / (self.total_words + len(self.vocab))
        context_count = sum(self.bigram_counts[context].values())
        p_bigram = self.bigram_counts[context][word] / context_count if context_count > 0 else 0.0
        return self.lambda2 * p_bigram + self.lambda1 * p_unigram


if __name__ == "__main__":
    train_corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
    ]
    test_corpus = ["the cat plays", "a dog jumps"]
    
    laplace = LaplaceBigramModel()
    laplace.train(train_corpus)
    print(f"Laplace perplexity: {laplace.perplexity(test_corpus):.2f}")
```

**출력:**

```
Laplace perplexity: 9.43
```

## 2. 논의

라플라스 부드럽게 하기(하나 더하기)는 가장 단순한 방식이다. 곧 바이그램 셈마다 1을 더하고 분모를 낱말 곳간 크기 $V$만큼 맞춘다. 이러면 확률이 0인 바이그램이 없어지지만 분포가 크게 일그러질 수 있다. 낱말 곳간이 50,000개일 때 10번만 나온 맥락 낱말에 대해 가능한 바이그램마다 1을 더하면 확률 무게 대부분이 본 적 없는 사건으로 옮겨 가고 실제로 본 바이그램에는 아주 조금만 남는다. k 더하기 부드럽게 하기는 소수 셈 $k < 1$을 써서 이를 넓힌 것으로, 덜 세게 나눠 주어 실제 분포를 더 잘 지킨다.

선형 사이 메우기는 차수가 다른 모델을 아울러 밑바탕부터 다른 길을 간다. 세는 수를 손보는 대신 두낱말 확률과 한낱말 확률의 짐 실은 평균을 셈한다. $P_{\text{interp}}(w_i \mid w_{i-1}) = \lambda_2 P_{\text{bigram}} + \lambda_1 P_{\text{unigram}}$이다. 어떤 앞뒤 흐름에서 두낱말 모델의 확률이 0이면 한낱말 몫이 기댈 어림을 준다. 사이 메우기 짐 $\lambda_1$과 $\lambda_2$은 남겨 둔 자료에서 EM 알고리즘으로 가장 좋게 잡을 수 있어, 붙박인 매끄럽게 하기 상수보다 자료에 더 기댄다.

어떤 부드럽게 하기 재주를 고르느냐에는 단순함, 셈 값, 모델의 좋음 사이의 맞바꿈이 있다. 라플라스는 짜기 쉽지만 흔히 지나치게 부드러워진다. k 더하기는 웃매개변수 $k$을 다듬어야 한다. 선형 사이 끼움은 더 원칙 있지만 섞는 무게를 가장 좋게 해야 한다. 실전에서는 고친 크네저-네이 부드럽게 하기 같은 더 정교한 방법이 깎은 셈과, 낱말이 나타나는 맥락의 여러 갈래임을 나타내는 낮은 차수 분포로 물러나기를 써서 가장 좋은 헷갈림도를 얻는다.

## 연습문제

**연습문제 1.**
낱말 곳간 크기가 $V = 100$이고 맥락 낱말이 50번 나오며 그 뒤에 오는 서로 다른 낱말이 10개일 때, (가) 맥락 뒤에 10번 나온 낱말과 (나) 본 적 없는 낱말의 라플라스 부드럽게 한 확률을 셈하여라.

??? success "연습문제 1 풀이"
    Using $P(w \mid c) = (\text{count}(c, w) + 1) / (\text{count}(c) + V)$:
    
    (가) 본 낱말(세는 수 = 10): $P = (10 + 1) / (50 + 100) = 11/150 \approx 0.0733$
    
    부드럽게 하지 않으면 $P = 10/50 = 0.200$이다. 라플라스 부드럽게 하기가 이를 절반 넘게 줄였다.
    
    (나) 못 본 낱말(세는 수 = 0): $P = (0 + 1) / (50 + 100) = 1/150 \approx 0.00667$
    
    본 적 없는 낱말이 $100 - 10 = 90$개 있고 이들이 확률 무게의 $90/150 = 0.60$을 함께 받는 반면, 본 낱말 10개는 $60/150 = 0.40$을 나눠 갖는다. 라플라스 부드럽게 하기가 본 적 없는 사건에 무게를 지나치게 줄 수 있음을 보여 준다.

---

**연습문제 2.**
$\lambda_2 + \lambda_1 = 1$인 선형 사이 메우기에서, 낱말 사전의 모든 낱말에 대해 $P_{\text{unigram}}(w) > 0$이면 $P_{\text{bigram}}(w \mid c) = 0$이더라도 메운 확률이 늘 양수임을 밝혀라.

??? success "연습문제 2 풀이"
    $\lambda_1 > 0$이고 $\lambda_2 > 0$일 때 $P_{\text{interp}}(w \mid c) = \lambda_2 P_{\text{bigram}}(w \mid c) + \lambda_1 P_{\text{unigram}}(w)$이 주어졌다고 하자.
    
    When $P_{\text{bigram}}(w \mid c) = 0$:
    
    $$
    P_{\text{interp}}(w \mid c) = \lambda_2 \cdot 0 + \lambda_1 \cdot P_{\text{unigram}}(w) = \lambda_1 \cdot P_{\text{unigram}}(w)
    $$
    
    $\lambda_1 > 0$이고 (가정에 따라) $P_{\text{unigram}}(w) > 0$이므로 $P_{\text{interp}}(w \mid c) > 0$이다. $\square$
    
    이는 가장 낮은 차수 모델이 낱말 곳간의 모든 항목에 양의 확률을 주기만 하면, 선형 사이 끼움이 낮은 차수 모델로 물러나 높은 차수 모델의 확률 0을 본디 다룬다는 것을 보여 준다.

---

**연습문제 3.**
다짐 묶음에서 헷갈림도를 가장 작게 하여 Add-k 매끄럽게 하기의 가장 좋은 $k$을 찾는 격자 찾기 함수를 짜라. $k \in \{0.01, 0.05, 0.1, 0.25, 0.5, 1.0\}$을 시험하여라.

??? success "연습문제 3 풀이"
    ```python
    def find_optimal_k(train_corpus, val_corpus, k_values=None):
        if k_values is None:
            k_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        
        best_k = None
        best_ppl = float('inf')
        
        for k in k_values:
            model = AddKBigramModel(k=k)
            model.train(train_corpus)
            ppl = model.perplexity(val_corpus)
            print(f"  k={k:.2f}: perplexity = {ppl:.2f}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_k = k
        
        print(f"\n  Best k = {best_k} with perplexity = {best_ppl:.2f}")
        return best_k, best_ppl
    ```

## 정리하며

**다룬 것** — n-그램 부드럽게 하기 재주

라플라스 부드럽게 하기(하나 더하기)는 가장 단순한 방식이다.

고갱이 갈래는 `LaplaceBigramModel`, `InterpolatedBigramModel`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
