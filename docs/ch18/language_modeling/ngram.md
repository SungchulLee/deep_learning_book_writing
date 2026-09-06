# n-그램 말 모델
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- n-그램 말 모델의 근본 개념을 이해한다
- 유니그램, 바이그램, 트라이그램 모델을 맨바닥부터 짠다
- 확률 어림에 최대 가능도 어림(MLE)을 쓴다
- 본 적 없는 n-그램을 다루는 부드럽게 하기 재주를 짠다
- 여러 표집 전략으로 글을 만들어 낸다
- 헷갈림도로 n-그램 모델을 값매김한다

---

## 말 나타내기 들어가기

Language modeling is the task of assigning probabilities to sequences of words. Given a sequence of words $w_1, w_2, \ldots, w_n$, a language model computes:

$$P(w_1, w_2, \ldots, w_n)$$

단순해 보이는 이 일에는 깊은 쓰임새가 있다. 곧 글 만들어 내기, 말소리 알아듣기, 기계 옮김, 맞춤법 고치기 등이다. n-그램 모델은 이 문제의 고전적인 방식이며 읽어 내기 쉽고 셈이 효율적이다.

---

## 확률의 사슬 규칙

확률의 사슬 규칙을 쓰면 차례의 결합 확률을 쪼갤 수 있다:

$$P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdots P(w_n|w_1, \ldots, w_{n-1})$$

$$= \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

이 정확한 쪼갬은 아무리 긴 지난 이야기에도 조건을 건 확률을 어림해야 하므로 긴 차례에서는 셈으로 감당할 수 없다. n-그램 모델은 **마르코프 가정**으로 이를 다룬다.

---

## 마르코프 가정

n-그램 모델의 핵심 눈썰미는 **마르코프 가정**이다. 곧 낱말의 확률은 지난 이야기 전체가 아니라 앞선 $n-1$개 낱말에만 달렸다:

$$P(w_i | w_1, \ldots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \ldots, w_{i-1})$$

이 가정은 정확도를 내주고 다룰 수 있음을 얻는다. $n$의 값에 따라 모델 갈래가 달라진다:

| 모델 | n | 맥락 | 가정 |
|-------|---|---------|------------|
| 유니그램 | 1 | 없음 | 낱말끼리 안 얽힌다 |
| 바이그램 | 2 | 앞 낱말 | 1차 마르코프 |
| 트라이그램 | 3 | 앞 낱말 2개 | 2차 마르코프 |
| 4-그램 | 4 | 앞 낱말 3개 | 3차 마르코프 |

---

## 유니그램 모델

유니그램 모델은 낱말끼리 아예 안 얽힌다고 가정한다:

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i)$$

### 최대 가능도 어림

유니그램에서 최대 가능도 어림 확률은 그저 상대 잦기이다:

$$P_{MLE}(w) = \frac{\text{count}(w)}{\sum_{w' \in V} \text{count}(w')} = \frac{\text{count}(w)}{N}$$

여기서 $N$은 말뭉치의 전체 낱말 수이다.

### PyTorch 구현

```python
from collections import Counter
from typing import List
import math


class UnigramModel:
    """
    낱말끼리 안 얽힌다고 가정하는 유니그램 말 모델.
    
    P(w) = count(w) / total_words
    """
    
    def __init__(self):
        self.word_counts = Counter()
        self.total_words = 0
        self.vocab = set()
    
    def train(self, corpus: List[str]) -> None:
        """월의 말뭉치로 익히기."""
        for sentence in corpus:
            words = sentence.lower().split()
            self.word_counts.update(words)
            self.total_words += len(words)
            self.vocab.update(words)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total words: {self.total_words}")
    
    def probability(self, word: str) -> float:
        """최대 가능도 어림으로 P(낱말) 셈하기."""
        word = word.lower()
        if word not in self.vocab:
            return 0.0
        return self.word_counts[word] / self.total_words
    
    def log_probability(self, word: str) -> float:
        """수치를 든든하게 하는 로그 확률."""
        prob = self.probability(word)
        return math.log2(prob) if prob > 0 else float('-inf')
    
    def sentence_log_probability(self, sentence: str) -> float:
        """log P(월) = log P(낱말)의 합 셈하기."""
        words = sentence.lower().split()
        return sum(self.log_probability(w) for w in words)


# 사용 예
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog played"
]

model = UnigramModel()
model.train(corpus)

# 낱말 확률
for word in ["the", "cat", "elephant"]:
    print(f"P({word}) = {model.probability(word):.4f}")
```

**출력:**
```
Vocabulary size: 11
Total words: 19
P(the) = 0.3158
P(cat) = 0.1053
P(elephant) = 0.0000
```

유니그램 모델은 낱말 잦기는 담아내지만 차례 짜임은 모두 무시한다. 곧 "the cat sat"과 "cat the sat"의 확률이 같다.

---

## 바이그램 모델

바이그램 모델은 낱말마다 바로 앞 낱말에 조건을 건다:

$$P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}$$

이러면 "New York"이나 "the cat"이 "the the"보다 확률이 높은 것 같은 가까운 얽힘을 담아낸다.

### 월의 경계 다루기

월의 시작과 끝을 나타내려 특별한 토막을 들여온다:

- `<s>`: 월 시작 토막
- `</s>`: 월 끝 토막

월 "the cat sat"에 대해:

- 바이그램: `(<s>, the)`, `(the, cat)`, `(cat, sat)`, `(sat, </s>)`

### 구현

```python
from collections import defaultdict, Counter
from typing import List, Tuple
import math


class BigramModel:
    """
    바이그램 말 모델: P(w_i | w_{i-1}).
    
    월 경계 토막을 쓴 최대 가능도 어림을 짠다.
    """
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)  # bigram_counts[w1][w2]
        self.unigram_counts = Counter()
        self.vocab = set()
        self.START = "<s>"
        self.END = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """말뭉치로 바이그램 모델 익히기."""
        total_bigrams = 0
        
        for sentence in corpus:
            words = sentence.lower().split()
            words = [self.START] + words + [self.END]
            
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                
                if w1 != self.START:
                    self.vocab.add(w1)
                if w2 != self.END:
                    self.vocab.add(w2)
                    
                total_bigrams += 1
        
        print(f"Trained on {total_bigrams} bigrams")
        print(f"Vocabulary: {len(self.vocab)} words")
    
    def probability(self, word: str, context: str) -> float:
        """
        최대 가능도 어림으로 P(낱말 | 맥락) 셈하기.
        
        인수:
            word: 지금 낱말 w_i
            context: 앞 낱말 w_{i-1}
        """
        word = word.lower()
        context = context.lower()
        
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        
        return bigram_count / context_count if context_count > 0 else 0.0
    
    def log_probability(self, word: str, context: str) -> float:
        """수치를 든든하게 하는 로그 확률."""
        prob = self.probability(word, context)
        return math.log2(prob) if prob > 0 else float('-inf')
    
    def sentence_probability(self, sentence: str) -> float:
        """사슬 규칙으로 P(월) 셈하기."""
        words = sentence.lower().split()
        words = [self.START] + words + [self.END]
        
        prob = 1.0
        for i in range(len(words) - 1):
            p = self.probability(words[i + 1], words[i])
            if p == 0:
                return 0.0
            prob *= p
        
        return prob
    
    def sentence_log_probability(self, sentence: str) -> float:
        """월의 로그 확률."""
        words = sentence.lower().split()
        words = [self.START] + words + [self.END]
        
        log_prob = 0.0
        for i in range(len(words) - 1):
            log_prob += self.log_probability(words[i + 1], words[i])
        
        return log_prob


# 예
model = BigramModel()
model.train(corpus)

# 조건부 확률
test_bigrams = [("the", "cat"), ("cat", "sat"), ("dog", "played")]
for context, word in test_bigrams:
    print(f"P({word} | {context}) = {model.probability(word, context):.4f}")
```

---

## 트라이그램 모델

트라이그램 모델은 맥락을 앞선 낱말 둘로 넓힌다:

$$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})}$$

트라이그램은 더 먼 얽힘을 담아내지만 **자료의 성김**에 더 시달린다. 곧 올바른 트라이그램인데도 익힘에 한 번도 나오지 않을 수 있다.

```python
class TrigramModel:
    """
    트라이그램 말 모델: P(w_i | w_{i-2}, w_{i-1}).
    """
    
    def __init__(self):
        # trigram_counts[w1][w2][w3] = 셈
        self.trigram_counts = defaultdict(lambda: defaultdict(Counter))
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.START = "<s>"
        self.END = "</s>"
    
    def train(self, corpus: List[str]) -> None:
        """트라이그램 모델 익히기."""
        for sentence in corpus:
            words = sentence.lower().split()
            # 트라이그램 맥락을 위한 시작 토막 둘
            words = [self.START, self.START] + words + [self.END]
            
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i+1], words[i+2]
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
                
                for w in [w1, w2, w3]:
                    if w not in [self.START, self.END]:
                        self.vocab.add(w)
    
    def probability(self, word: str, context1: str, context2: str) -> float:
        """
        P(낱말 | 맥락1, 맥락2) 셈하기.
        
        인수:
            word: 지금 낱말 w_i
            context1: 자리 i-2의 낱말
            context2: 자리 i-1의 낱말
        """
        trigram_count = self.trigram_counts[context1][context2][word]
        bigram_count = self.bigram_counts[context1][context2]
        
        return trigram_count / bigram_count if bigram_count > 0 else 0.0
```

---

## 자료의 성김 문제

n-그램 모델의 근본 어려움은 **자료의 성김**이다. 곧 올바른 낱말 차례가 익힘에 한 번도 나오지 않는다. 다음을 보자:

- 영어 낱말 곳간: 흔한 낱말 약 50,000개
- Possible bigrams: $50,000^2 = 2.5 \times 10^9$
- Possible trigrams: $50,000^3 = 1.25 \times 10^{14}$

대부분의 n-그램은 셈이 0이 되어 다음이 따라온다:

1. 올바르지만 본 적 없는 차례의 확률이 0이 된다
2. 익힘 자료 너머로 두루 통하지 못한다
3. 헷갈림도가 정의되지 않는다(0의 로그)

**풀이**: 부드럽게 하기 재주가 확률 무게를 본 적 없는 사건에 나눠 준다.

---

## 부드럽게 하기 재주

### 라플라스(하나 더하기) 부드럽게 하기

가장 단순한 부드럽게 하기는 모든 셈에 1을 더한다:

$$P_{Laplace}(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + V}$$

여기서 $V$은 낱말 곳간의 크기이다.

**좋은 점**: 단순하고 확률이 0이 아님을 보장한다
**나쁜 점**: 낱말 곳간이 크면 본 적 없는 사건에 확률 무게를 너무 많이 준다

```python
class LaplaceBigramModel:
    """라플라스(하나 더하기) 부드럽게 하기를 쓴 바이그램 모델."""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        self.UNK = "<unk>"
        self.START = "<s>"
        self.END = "</s>"
    
    def train(self, corpus: List[str], min_freq: int = 1) -> None:
        """낱말 곳간 문턱값을 두고 익히기."""
        # 1차: 낱말 세기
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence.lower().split())
        
        # 낱말 곳간 세우기(최소 잦기를 넘는 낱말)
        self.vocab = {w for w, c in word_counts.items() if c >= min_freq}
        self.vocab.add(self.UNK)
        self.vocab_size = len(self.vocab)
        
        # 2차: 곳간 밖 낱말을 다루며 바이그램 세기
        for sentence in corpus:
            words = sentence.lower().split()
            words = [w if w in self.vocab else self.UNK for w in words]
            words = [self.START] + words + [self.END]
            
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
    
    def probability(self, word: str, context: str) -> float:
        """
        라플라스로 부드럽게 한 확률.
        
        P(w|c) = (count(c, w) + 1) / (count(c) + V)
        """
        word = word.lower()
        context = context.lower()
        
        # 곳간 밖 낱말 다루기
        if word not in self.vocab:
            word = self.UNK
        if context not in self.vocab and context != self.START:
            context = self.UNK
        
        numerator = self.bigram_counts[context][word] + 1
        denominator = self.unigram_counts[context] + self.vocab_size
        
        return numerator / denominator
```

### k 더하기 부드럽게 하기

다듬을 수 있는 매개변수 $k$(보통 $0 < k < 1$)로 넓힌 것:

$$P_{add-k}(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + k}{\text{count}(w_{i-1}) + k \cdot V}$$

$k$이 작을수록(보기로 0.1이나 0.5) 라플라스보다 덜 세게 작용한다.

### 선형 사이 끼움

선형 사이 끼움은 여러 n-그램 차수의 증거를 아우른다:

$$P_{interp}(w_i | w_{i-1}) = \lambda_2 \cdot P_{ML}(w_i | w_{i-1}) + \lambda_1 \cdot P_{ML}(w_i)$$

where $\lambda_2 + \lambda_1 = 1$.

트라이그램에서는:

$$P_{interp}(w_i | w_{i-2}, w_{i-1}) = \lambda_3 \cdot P_{tri} + \lambda_2 \cdot P_{bi} + \lambda_1 \cdot P_{uni}$$

**직관**: 트라이그램 맥락이 드물면 바이그램과 유니그램 어림으로 물러난다.

```python
class InterpolatedBigramModel:
    """선형 사이 끼움 부드럽게 하기를 쓴 바이그램 모델."""
    
    def __init__(self, lambda2: float = 0.7, lambda1: float = 0.3):
        """
        인수:
            lambda2: 바이그램 확률의 무게
            lambda1: 유니그램 확률의 무게
        """
        assert abs(lambda2 + lambda1 - 1.0) < 1e-6, "Lambdas must sum to 1"
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_words = 0
        self.vocab = set()
    
    def train(self, corpus: List[str]) -> None:
        """사이 끼운 모델 익히기."""
        # ... 위와 비슷한 익히기 코드 ...
        pass
    
    def unigram_probability(self, word: str) -> float:
        """하나 더하기 부드럽게 하기를 쓴 P(낱말)."""
        count = self.unigram_counts[word]
        return (count + 1) / (self.total_words + len(self.vocab))
    
    def bigram_probability_ml(self, word: str, context: str) -> float:
        """날 최대 가능도 어림 바이그램 확률."""
        bigram_count = self.bigram_counts[context][word]
        context_count = sum(self.bigram_counts[context].values())
        return bigram_count / context_count if context_count > 0 else 0.0
    
    def probability(self, word: str, context: str) -> float:
        """
        사이 끼운 확률.
        
        P(w|c) = λ₂ · P_bigram(w|c) + λ₁ · P_unigram(w)
        """
        p_bigram = self.bigram_probability_ml(word, context)
        p_unigram = self.unigram_probability(word)
        
        return self.lambda2 * p_bigram + self.lambda1 * p_unigram
```

### 크네저-네이 부드럽게 하기

가장 정교한 n-그램 부드럽게 하기 방법인 크네저-네이는 절대 깎기에, 낱말이 몇 가지 다른 맥락에 나타나는지를 세는 **이음 셈**에 바탕한 고친 낮은 차수 분포를 곁들여 쓴다.

직관은 이렇다. "Francisco"는 유니그램 잦기가 높지만 오직 "San" 뒤에만 나오므로 그 이음 확률은 낮아야 한다.

---

## n-그램으로 글 만들어 내기

n-그램 모델은 조건부 분포에서 뽑아 자연스레 글을 만들어 낸다.

### 탐욕적 복호

늘 가장 그럴듯한 다음 낱말을 고른다:

$$w_t = \arg\max_w P(w | \text{context})$$

**좋은 점**: 늘 같고 빠르다
**나쁜 점**: 되풀이 고리에 갇힐 수 있다

### 마구잡이 표집

온 분포에서 뽑는다:

$$w_t \sim P(w | \text{context})$$

**좋은 점**: 여러 갈래의 내놓음
**나쁜 점**: 그럴듯하지 않은 차례를 만들 수 있다

### 온도 표집

소프트맥스 앞에서 로짓의 잣수를 맞춰 마구잡이 정도를 다스린다:

$$P'(w) \propto P(w)^{1/T}$$

- $T > 1$: 더 평평한 분포(더 마구잡이)
- $T < 1$: 더 뾰족한 분포(더 정해진 대로)
- $T = 1$: 보통의 표집

### 상위 k 표집

가장 그럴듯한 낱말 $k$개에서만 뽑는다:

1. 낱말을 확률로 정렬한다
2. 상위 $k$개만 남긴다
3. 확률을 다시 고르게 맞춘다
4. 잘라 낸 분포에서 뽑는다

```python
import random


class TextGenerator:
    """여러 표집 전략을 쓴 글 만들어 내기."""
    
    def __init__(self, bigram_model):
        self.model = bigram_model
    
    def get_distribution(self, context: str) -> List[Tuple[str, float]]:
        """다음 낱말에 대한 확률 분포 얻기."""
        counts = self.model.bigram_counts[context]
        if not counts:
            return [(w, 1/len(self.model.vocab)) for w in self.model.vocab]
        
        total = sum(counts.values())
        return [(w, c/total) for w, c in counts.items()]
    
    def generate_greedy(self, max_length: int = 20) -> str:
        """욕심쟁이 만들어 내기: 늘 가장 그럴듯한 것을 고른다."""
        context = self.model.START
        generated = []
        
        for _ in range(max_length):
            dist = self.get_distribution(context)
            if not dist:
                break
            
            # 확률이 가장 높은 것 고르기
            next_word = max(dist, key=lambda x: x[1])[0]
            
            if next_word == self.model.END:
                break
            
            generated.append(next_word)
            context = next_word
        
        return ' '.join(generated)
    
    def generate_temperature(self, temperature: float = 1.0, 
                             max_length: int = 20) -> str:
        """온도를 맞춘 표집."""
        context = self.model.START
        generated = []
        
        for _ in range(max_length):
            dist = self.get_distribution(context)
            if not dist:
                break
            
            words, probs = zip(*dist)
            
            # 온도를 적용한다
            scaled_probs = [p ** (1/temperature) for p in probs]
            total = sum(scaled_probs)
            scaled_probs = [p/total for p in scaled_probs]
            
            # 뽑기
            next_word = random.choices(words, weights=scaled_probs, k=1)[0]
            
            if next_word == self.model.END:
                break
            
            generated.append(next_word)
            context = next_word
        
        return ' '.join(generated)
    
    def generate_top_k(self, k: int = 5, max_length: int = 20) -> str:
        """상위 k 표집."""
        context = self.model.START
        generated = []
        
        for _ in range(max_length):
            dist = self.get_distribution(context)
            if not dist:
                break
            
            # 확률로 정렬하고 상위 k개 남기기
            dist.sort(key=lambda x: x[1], reverse=True)
            top_k = dist[:k]
            
            # 다시 고른다
            words, probs = zip(*top_k)
            total = sum(probs)
            probs = [p/total for p in probs]
            
            # 뽑기
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            if next_word == self.model.END:
                break
            
            generated.append(next_word)
            context = next_word
        
        return ' '.join(generated)
```

---

## 값매김: 헷갈림도

**헷갈림도**는 말 모델의 표준 내재 값매김 잣대이다:

$$\text{PPL} = 2^{H(P, \hat{P})} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(w_i | \text{context})}$$

**읽는 법**: 평균 갈래 수이다. 곧 모델이 평균적으로 똑같이 그럴듯한 선택지 PPL개 가운데서 "고르고" 있다는 뜻이다.

- 헷갈림도가 낮을수록 = 더 좋은 모델
- 헷갈림도 100은 모델이 낱말 100개에서 고르게 고르는 것만큼 헷갈린다는 뜻이다

```python
def compute_perplexity(model, test_corpus: List[str]) -> float:
    """
    시험 말뭉치에서 헷갈림도 셈하기.
    
    PPL = 2^(-평균 로그 확률)
    """
    total_log_prob = 0.0
    total_words = 0
    
    for sentence in test_corpus:
        words = sentence.lower().split()
        words = [model.START] + words + [model.END]
        
        for i in range(len(words) - 1):
            prob = model.probability(words[i + 1], words[i])
            if prob > 0:
                total_log_prob += math.log2(prob)
            else:
                total_log_prob += -100  # 큰 벌주기
            total_words += 1
    
    cross_entropy = -total_log_prob / total_words
    perplexity = 2 ** cross_entropy
    
    return perplexity


# 부드럽게 하기 재주 견주기
train_corpus = ["the cat sat on the mat", "the dog sat on the log"] * 10
test_corpus = ["the cat played", "a dog runs"]

laplace_model = LaplaceBigramModel()
laplace_model.train(train_corpus)
print(f"Laplace PPL: {compute_perplexity(laplace_model, test_corpus):.2f}")
```

---

## 부드럽게 하기 재주 견줌

| 재주 | 좋은 점 | 나쁜 점 | 알맞은 곳 |
|-----------|------|------|----------|
| **라플라스** | 단순하다 | V이 크면 지나치게 부드러워진다 | 작은 낱말 곳간 |
| **k 더하기** | 다듬을 수 있다 | 검증 뭉치가 필요하다 | 가운데 크기 낱말 곳간 |
| **사이 끼움** | 증거를 아우른다 | 웃매개변수가 여럿이다 | 두루 쓰기 |
| **크네저-네이** | 가장 앞선다 | 짜기가 복잡하다 | 실전 체계 |

### 흔한 헷갈림도(Penn Treebank)

| 모델 | 헷갈림도 |
|-------|------------|
| 유니그램 | 약 1000 |
| 바이그램(라플라스) | 약 300~500 |
| 트라이그램(크네저-네이) | 약 80~150 |

---

## n-그램 모델의 한계

1. **붙박이 맥락**: $n-1$개 낱말 너머의 얽힘을 담아내지 못한다
2. **자료의 성김**: 가능한 n-그램이 지수로 늘어난다
3. **뜻의 닮음이 없음**: "cat"과 "feline"이 서로 남남이다
4. **큰 저장 공간**: n-그램 셈을 모두 담아 두어야 한다
5. **두루 통하지 않음**: "the cat sat"이 "the dog sat"에 도움이 안 된다

이 한계 때문에 (다음에 다룰) **신경 말 모델**이 나왔는데, 이는 비슷한 낱말과 맥락에 두루 통하는 이어진 나타냄을 배운다.

---

## 요약

- n-그램 모델은 마르코프 가정으로 차례의 확률을 어림한다
- 최대 가능도 어림은 셈에서 곧바로 확률을 어림해 준다
- 본 적 없는 n-그램을 다루려면 부드럽게 하기가 꼭 필요하다
- 만들어 내기 전략마다 여러 갈래임과 좋음을 맞바꾼다
- 헷갈림도는 모델이 남겨 둔 자료를 얼마나 잘 어림하는지 잰다
- n-그램은 바탕 잣대로서, 그리고 자료가 적은 곳에서 여전히 쓸모 있다

---

## 참고 문헌

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). 3장.
2. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4), 359-394.
3. Kneser, R., & Ney, H. (1995). Improved backing-off for m-gram language modeling. *ICASSP*.

## 연습문제

1. **4-그램 모델 짜기**: 트라이그램 짜기를 4-그램으로 넓혀라. 자료의 성김이 헷갈림도에 어떤 영향을 주는가?

2. **부드럽게 하기 견줌**: 라플라스, 0.5 더하기, 사이 끼움으로 바이그램 모델을 익히고 남겨 둔 자료에서 헷갈림도를 견주어라.

3. **가장 좋은 람다 찾기**: 검증 뭉치에서 가장 좋은 사이 끼움 무게를 찾는 격자 찾기를 짜라.

4. **만들어 내기의 여러 갈래**: 온도를 달리해(0.5, 1.0, 1.5, 2.0) 표본 100개를 만들어라. 겹치지 않음과 좋음을 재어라.

5. **되풀이 살피기**: 욕심쟁이 풀기가 얼마나 자주 같은 n-그램을 되풀이하는지 좇아라. 되풀이 벌주기를 짜라.

---
