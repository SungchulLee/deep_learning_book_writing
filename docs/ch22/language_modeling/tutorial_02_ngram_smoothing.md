# n-그램 확률 평활화

최대우도 n-그램 모델은 관측하지 못한 조합에 확률 0을 준다. 문장에 그런 전이가 하나만 있어도 문장 전체 확률이 0이 된다. 평활화는 관측한 사건의 확률 일부를 다른 사건에 배분하는 방법이다. 여기서는 라플라스 평활화, Add-k 평활화, 유니그램·바이그램의 선형 보간을 비교한다. 각 방법은 같은 예측 어휘와 문장 경계 규칙 아래에서 정규화되어야 한다.

## 1. 코드와 기존 출력

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

**기존 출력:**

```
Laplace perplexity: 9.43
```

## 2. 라플라스와 Add-k 평활화

예측 어휘 크기가 V이고 문맥 c의 다음 단어 빈도 합이 C(c)라면 Add-k는 P(w|c)=[C(c,w)+k]/[C(c)+kV]다. k>0이면 어휘 안의 모든 사건이 양의 확률을 얻는다. k=1이 라플라스 평활화다. 문맥이 드물고 어휘가 크면 kV가 관측 빈도보다 커져 미관측 사건에 많은 확률이 배분된다. 작은 k는 이 효과를 줄이지만 검증 성능이 반드시 개선되는 것은 아니다.

현재 LaplaceBigramModel은 종료 토큰을 빈도에는 넣으면서 예측 어휘에는 넣지 않는다. probability에 종료 토큰을 주면 UNK로 바꾸므로 실제로 관측한 종료 전이와 다른 사건을 조회한다. 분모에는 종료 빈도가 남아 있어 저장된 어휘에 대한 확률 합도 1보다 작아질 수 있다. 따라서 이 코드의 퍼플렉서티를 올바르게 정규화된 문장 모델의 평가값으로 해석하면 안 된다. 어휘·UNK·종료 사건을 일치시키는 설계가 먼저 필요하다.

train은 기존 빈도를 초기화하지 않으면서 어휘를 다시 만든다. 반복 학습하면 이전 어휘 기준의 빈도와 새 어휘가 섞일 수 있다. 빈 평가 자료의 퍼플렉서티도 정의하거나 예외 처리해야 한다. 기존 코드와 출력은 그대로 두었다.

## 3. 선형 보간의 조건

유니그램과 바이그램을 P(w|c)=λ₂P₂(w|c)+λ₁P₁(w)로 혼합할 수 있다. 두 성분이 같은 예측 어휘에서 정규화되고 λ₁,λ₂≥0, λ₁+λ₂=1이면 혼합도 확률 분포다. 특히 λ₁>0이고 유니그램이 모든 사건에 양의 확률을 주면 관측하지 못한 바이그램도 양의 확률을 얻는다. 웨이트의 합이 1이라는 조건만으로는 양수성이나 정규화가 보장되지 않는다.

현재 InterpolatedBigramModel은 학습을 구현하지만 두 성분의 예측 사건이 일치하지 않는다. 바이그램은 종료를 세고 유니그램·어휘는 종료를 제외한다. 미관측 문맥의 바이그램 성분은 전부 0이므로 어휘 위 확률 합이 λ₁만 남는다. probability는 학습 때의 소문자화·UNK 변환도 적용하지 않는다. 어휘 밖 임의 문자열마다 유니그램의 가산 확률을 주는 동작은 고정된 어휘의 정규화된 분포가 아니다.

이 조건들을 바로잡은 모델에서는 검증 자료의 로그우도로 혼합 웨이트를 선택할 수 있다. 격자 탐색이나 혼합 모델의 EM 추정이 가능하지만 본문 코드는 웨이트 최적화를 구현하지 않는다. 바이그램이 관측되지 않았을 때 낮은 차수 모델을 사용하는 보간과, 특정 조건에서만 낮은 차수로 전환하는 백오프도 구분한다.

## 4. 평가와 방법 선택

평활화 강도와 혼합 웨이트는 학습 자료와 분리한 검증 자료로 고르고 최종 비교는 별도 시험 자료에서 수행한다. 토큰화·어휘·UNK·종료 처리와 퍼플렉서티의 예측 사건 수를 같게 해야 수치 비교가 가능하다. Kneser–Ney 계열은 낮은 차수 분포에 다양한 선행 문맥의 수를 반영하는 중요한 대안이지만, 모든 자료에서 같은 방법이나 설정이 최선이라고 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
예측 어휘 크기 V=100, 문맥 빈도 50, 관측된 다음 단어 종류 10개일 때, 해당 문맥 뒤에 10번 나온 단어와 한 번도 나오지 않은 단어의 라플라스 확률을 계산하라.

</div>

??? success "풀이"

    관측 단어는 (10+1)/(50+100)=11/150≈0.0733, 미관측 단어는 1/150≈0.00667이다. 관측 단어 10개의 총 빈도는 50이므로 이들이 받는 확률 합은 (50+10)/150=0.40이다. 나머지 90개는 90/150=0.60을 받는다. 관측 단어 하나의 MLE 10/50=0.20과 비교하면 평활화의 영향이 크다. 여기서 V는 종료 사건을 포함할지까지 일관되게 정한 예측 어휘 크기다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
선형 보간에서 미관측 바이그램의 확률이 양수가 되기 위한 웨이트 조건과, 혼합이 정규화되기 위한 조건을 설명하라.

</div>

??? success "풀이"

    λ₁>0이고 P₁(w)>0이면 P₂(w|c)=0인 사건도 P(w|c)=λ₁P₁(w)>0이다. λ₁=0이면 이 결론은 성립하지 않는다. 정규화에는 두 성분이 같은 사건 집합에서 각각 합 1이고 웨이트가 음이 아니며 합 1이어야 한다. 미관측 문맥에서 P₂를 모두 0으로 두면 혼합의 합이 λ₁이므로 별도의 정상화된 fallback이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 격자 탐색이 Add-k의 k를 선택하려면 어떤 모델 인터페이스와 자료 분리가 필요한지 설명하라.

</div>

??? success "풀이"

    후보 k마다 같은 학습 자료로 새 모델을 만들고 검증 퍼플렉서티가 가장 작은 값을 선택한다. 아래 AddKBigramModel은 이 페이지에 정의되어 있지 않아 그대로 실행 가능한 완성 구현이 아니다. 어휘·UNK·종료 규칙을 고정한 train과 perplexity가 먼저 필요하다. 모든 후보가 무한대나 유효하지 않은 값을 내는 경우도 처리해야 한다. 검증 자료로 k를 선택한 뒤 최종 시험 자료로 성능을 보고한다.

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

평활화와 보간은 미관측 조합에 확률을 배분하지만 어휘·종료·UNK와 분모가 일치해야 한다. 현재 예제의 종료 사건 및 미관측 문맥 처리는 정규화 조건을 만족하지 않으므로 출력 수치만으로 방법의 성능을 비교하지 않는다.
