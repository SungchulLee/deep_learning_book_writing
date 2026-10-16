# n-그램 언어 모델과 평활화

언어 모델은 다음 토큰의 조건부 분포를 통해 텍스트의 확률을 표현한다. n-그램은 최근 n−1토큰의 빈도로 조건부 확률을 추정하는 고전적인 기준 모델이다. 여기서는 소문자화·공백 분리한 단어를 사용하며 Python의 Counter로 구현한다. PyTorch 뉴럴 넷트웍이나 의미 embedding을 학습하는 예제는 아니다.

## 1. 사슬 규칙과 제한된 문맥

길이 M의 토큰 열에는 정확한 사슬 규칙이 성립한다.

$$p(w_{1:M})=\prod_{i=1}^{M}p(w_i\mid w_{<i}).$$

사슬 규칙 자체가 계산 불가능한 것은 아니다. 모든 긴 문맥을 별도 빈도로 추정하면 자료가 부족해지는 것이 문제다. n-그램은 마지막 n−1개로 문맥을 제한한다. 유니그램은 문맥 없음, 바이그램은 직전1개, 트라이그램은2개다. 차수를 높이면 표현할 국소 문맥은 길어지지만 희소성·저장량·추정 분산도 커져 평가 성능이 반드시 좋아지지는 않는다.

## 2. 유니그램의 상대 빈도

어휘 V의 단어 빈도 C(w), 총 토큰 N에서 MLE는 p(w)=C(w)/N이다. 같은 단어들로 이루어진 같은 길이의 열은 순서를 바꿔도 같은 확률을 갖는다. 종료·길이 모형이 없는 유니그램의 곱은 길이를 고정한 토큰 열의 분포와 모든 길이의 문장 분포를 구분해야 한다.

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

**기존 출력:**

```
Vocabulary size: 11
Total words: 19
P(the) = 0.3158
P(cat) = 0.1053
P(elephant) = 0.0000
```

train은 기존 빈도를 초기화하지 않고 누적한다. 미등록 단어는0확률, 로그는−∞다. 빈 문장의 로그합은0이지만 모든 문장의 종료 확률을 모델링한다는 뜻은 아니다. 위 고정 말뭉치는6단어 문장3개로 총18토큰·어휘9개이고 the는6번·cat은2번이다. 따라서 P(the)=1/3, P(cat)=1/9가 현재 코드·자료의 값이다. 기존 출력의19토큰·어휘11개·다른 확률은 이 입력과 일치하지 않으므로 검증된 실행값으로 인용하지 않는다.

## 3. 바이그램과 종료 토큰

문맥 h의 다음 단어 빈도를 세면 p_ML(w|h)=C(h,w)/Σ_v C(h,v)다. 문장 끝의 단어도 종료 토큰을 예측하도록 하고 시작 토큰은 조건으로만 사용한다. `<s> the cat sat </s>`처럼 경계를 포함하면 일반 단어와 문장 종료를 같은 예측 사건 집합에서 다룰 수 있다.

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

**기존 출력:**

```
Trained on 21 bigrams
Vocabulary: 9 words
P(cat | the) = 0.3333
P(sat | cat) = 0.5000
P(played | dog) = 0.5000
```

이 BigramModel은 outgoing 빈도를 unigram_counts에 저장하므로 보통 단어의 단순 빈도 사전과 구분한다. vocab에는 시작·종료를 제외하지만 bigram_counts에는 종료 예측이 있다. MLE의 분모는 outgoing 전체 합을 사용하므로 관측 문맥에서 종료를 포함한 분포를 계산할 수 있다. 미관측 문맥에는0을 반환해 정상화된 대체 분포를 제공하지 않는다. 긴 문장 확률의 곱은 underflow가 날 수 있어 로그합을 사용한다.

## 4. 트라이그램과 희소성

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

앞의 두 시작 토큰과 마지막 종료 토큰을 포함해 세 단어 빈도/앞 두 단어 문맥 빈도로 계산한다. 이 클래스는 학습 때 소문자화하지만 probability에서는 입력을 소문자로 바꾸지 않아 호출자가 전처리를 맞춰야 한다. 미관측 문맥은0이다. 문법적으로 가능한 조합이라도 학습에 없으면 MLE는0을 줄 수 있고, 평가 정답에0이 하나라도 있으면 정확한 음의 로그우도·퍼플렉서티는 무한대가 된다.

어휘 크기를 고정하면 가능한 n-그램 조합은 |V|^n이지만 모든 조합을 저장해야 하는 것은 아니다. 관측된 조합만 저장할 수 있고 차수·자료에 따라 메모리가 달라진다. UNK 폴리시와 평활화는 미관측 단어와 미관측 문맥을 어떻게 처리하는지 구분해야 한다.

## 5. Add-k 평활화와 예측 어휘

고정된 예측 어휘 V에서 k>0일 때 다음 식을 쓸 수 있다.

$$p_k(w\mid h)=\frac{C(h,w)+k}{\sum_{v\in V}C(h,v)+k|V|}.$$

후보 집합과 분모가 같아야 합이1이다. 시작은 조건, 종료는 예측 사건으로 포함할지 명시한다. 분자에 양수를 더했다는 사실만으로 임의의 어휘·분모가 정상화되는 것은 아니다.

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
        """낱말 곳간 임계값을 두고 익히기."""
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

위 LaplaceBigramModel은 드문 단어를 UNK로 바꾸지만 **종료 토큰을 vocab에 넣지 않은 채 빈도에는 따로 센다**. probability('</s>',h)는 종료를 UNK로 바꾸어 종료 빈도를 사용하지 않는다. 반면 문맥 분모에는 종료 전이 빈도도 들어 있어, 종료가 관측된 문맥에서 저장한 어휘 위의 확률 합이1보다 작아질 수 있다. 따라서 현재 코드는 유효한 종료 포함 Laplace 언어 모델의 완성 구현이 아니다. min_freq·어휘·UNK·종료 사건과 train 재호출 시 빈도 초기화도 함께 맞춰야 한다.

## 6. 보간과 Kneser–Ney의 아이디어

여러 차수의 **정상화된** 분포를 비음수 λ_j, Σ_j λ_j=1로 합하면 보간 분포가 된다. 고정 가중 보간은 문맥 빈도가 드물다고 자동으로 웨이트를 바꾸는 backoff와는 다르다. 미관측 문맥의 높은 차수 분포가 전부0이면 단순 합의 전체 질량도 줄어들 수 있어 그 경우의 재정규화·가중 폴리시를 정해야 한다.

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

InterpolatedBigramModel의 train은 pass다. 어휘·빈도·total_words가 채워지지 않아 생성 직후에는 유니그램 분모가0이 될 수 있다. λ의 합만 확인하고 비음수 조건은 확인하지 않는다. 데이터 학습·미관측 문맥 처리·어휘 계약 없이 이 클래스로 보간 실험이 완성되었다고 설명할 수 없다.

Kneser–Ney는 높은 차수의 관측 빈도를 discount하고 낮은 차수에서는 단어의 총 빈도 대신 **서로 다른 선행 문맥 수** 같은 continuation 통계를 활용한다. 특정 이름 뒤에만 자주 나오는 단어와 여러 문맥 뒤에 나타나는 단어를 구분하려는 목적이다. 단일·수정 discount와 backoff 규칙의 구체적 구현은 별개이며 이 페이지에는 Kneser–Ney 코드가 없다. [언어 모델 교재](https://web.stanford.edu/~jurafsky/slp3/3.pdf)의 평활화 설명과 대조해 읽는다.

## 7. 생성 분포와 모델 확률의 차이

탐욕적 생성은 매 단계 최대 확률 토큰을 선택하지만 전체 문장의 최고 확률을 보장하지 않는다. 확률 샘플링은 다양한 열을 만들고, 온도 T>0에서 p_T(w)∝p(w)^(1/T)로 분포를 바꿀 수 있다. top-k는 k개 후보를 남겨 재정규화하며 k≥1 등 유효 범위가 필요하다.

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

TextGenerator는 model.probability를 호출하지 않고 bigram_counts를 직접 정상화한다. 따라서 Laplace·보간 모델을 넣더라도 그 평활화 확률로 생성하는 것이 아니다. 관측 문맥에는 관측 후보만, 없는 문맥에는 vocab의 균일 분포를 사용한다. 종료가 vocab에 없으면 그 fallback에서는 종료를 직접 뽑지 못할 수 있다. max_length 제한이 실제 종료 확률과 같지는 않다.

온도0·음수·극단값과 top-k0·빈 어휘·동점·재현성의 계약도 정의해야 한다. 동일 확률의 선택 순서는 자료 구조 순서의 영향을 받을 수 있다. 생성 전략을 바꾼 평가와 원래 모델의 정답 로그우도는 별개다.

## 8. 퍼플렉서티와 기존 평가 코드

동일 토큰화·어휘·경계·평가 집합에서 M개 예측 사건의 퍼플렉서티는 다음과 같다.

$$\operatorname{PPL}=\exp\left[-\frac1M\sum_i\log p(w_i\mid h_i)\right].$$

정답에 부여한 확률들의 기하평균의 역수다. 동등한 크기의 균일 선택지라는 해석은 직관이며 실제로 매번 그 개수의 후보가 있다는 뜻은 아니다. 낮은 PPL은 같은 평가 계약에서 더 높은 평균 정답 로그확률을 뜻하지만 생성의 사실성·유용성·의미적 정확도를 모두 보장하지 않는다. 토큰화가 다르면 숫자를 직접 비교할 수 없다.

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

**기존 출력:**

```
Laplace PPL: 8.49
```

compute_perplexity는 종료를 포함한 각 문장 길이+1개 사건을 센다. 0확률에−∞ 대신 임의의−100 log2 값을 넣어 유한한 점수를 만든다. 이는 정확한 PPL이 아니고 smoothing을 구현한 것도 아니다. 빈 평가 집합에서는 분모0이다. 더욱이 위 Laplace 클래스는 종료 처리·정규화가 맞지 않아 기존8.49 출력을 유효한 언어 모델의 비교 점수로 인용하면 안 된다.

## 9. 비교와 한계

차수·평활화는 동일 말뭉치·학습 어휘·UNK·토큰화·문장 경계에서 검증하고 최종 시험을 분리한다. 데이터 전처리·분할·사전 크기를 모르는 Penn Treebank의 ‘흔한 PPL’ 범위를 고정 순위로 제시하지 않는다. 코드가 구현하지 않은 Kneser–Ney 성능을 이 시연에 연결하지도 않는다.

n-그램은 제한된 문맥과 개별 빈도를 사용해 희소 조합 사이의 파라미터 공유가 약하다. 그러나 낮은 차수 보간·공유 단어 빈도 덕분에 한 문장의 통계가 다른 문장에 전혀 영향을 주지 않는다는 설명도 지나치다. 뉴럴 언어 모델의 분산 표현은 다른 공유 방식을 제공하며 실제 이점은 데이터·학습·평가 조건에서 확인한다.

## 연습문제

1. **평가 사건 정의**: 문장이 단어4개일 때 바이그램 PPL에서 종료를 포함하면 몇 사건인가? → 시작 토큰을 조건으로 첫 단어부터 종료까지5개다. 시작 토큰 자체의 예측은 세지 않는다.

2. **평활화의 질량**: 종료 빈도를 분모에 넣고 예측 어휘에는 종료를 빼면 어떤 문제가 생기는가? → 어휘 후보에 분배되지 않은 빈도가 남아 질량이1보다 작을 수 있다. 후보·UNK·종료 변환과 분모를 일치시켜야 한다.

3. **보간의 조건**: λ 합1만으로 정상화가 보장되는가? → 비음수 웨이트와 각 구성 분포의 정상화가 필요하다. 미관측 문맥에서0함수를 반환한 부분 모델은 그 조건을 만족하지 않는다.

4. **생성 계약**: TextGenerator에 Laplace 모델을 넣으면 Laplace 확률로 생성하는가? → 아니다. 저장 빈도를 직접 사용하므로 평활화 함수를 우회한다. 생성 후보와 학습 모델의 확률 정의를 연결해야 한다.

5. **0확률의 평가**: 정답 확률0을 log2 값−100으로 치환하면 정확 PPL인가? → 아니다. 정확 로그우도는−∞·PPL은∞다. 임의 치환은 정의를 바꾸며 빈 평가 집합도 별도로 처리해야 한다.

## 정리하며

n-그램 모델은 짧은 문맥의 빈도로 확률을 추정한다. 어휘·UNK·종료 토큰과 분모가 맞아야 평활화가 정상화되고, 생성·평가도 같은 모델 확률을 사용해야 한다. 현재 코드의 미완성 보간·종료 처리·PPL 절단과 기존 출력의 불일치를 구분해 읽는다.

**참고 문헌**

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). 3장.
2. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4), 359-394.
3. Kneser, R., & Ney, H. (1995). Improved backing-off for m-gram language modeling. *ICASSP*.
