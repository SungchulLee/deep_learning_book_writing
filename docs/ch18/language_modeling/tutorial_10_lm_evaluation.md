# 실습 10

길잡이 10: 말 모델 값매김. 말 모델을 두루 살피는 값매김 잣대와 방법.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 말 모델 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
길잡이 10: 말 모델 값매김
=======================================

말 모델을 두루 살피는 값매김 잣대와 방법.

값매김의 갈래:
1. 내재 잣대(헷갈림도, 글자당 비트)
2. 바깥 잣대(뒤따르는 일의 성능)
3. 사람의 값매김
4. 여러 갈래임과 좋음의 잣대

핵심 잣대:
-----------

1. 헷갈림도(PPL):
   PPL = exp(-1/N ∑ log P(w_i | context))
   - 낮을수록 좋다
   - 모델이 시험 자료에 얼마나 "놀라는지" 잰다

2. 글자당 비트(BPC):
   BPC = -1/(N*log(2)) ∑ log P(w_i | context)
   - 낮을수록 좋다
   - 글자 수로 고르게 맞춤

3. 엇갈린 엔트로피:
   H = -1/N ∑ log P(w_i | context)
   - 곧바른 손실 잣대
   - 헷갈림도 = exp(H)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import List, Dict
import math

# ========================================================================
# 메인
# ========================================================================


class LanguageModelEvaluator:
    """말 모델을 두루 살피는 값매김 꾸러미."""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.to(device)
    
    def compute_perplexity(self, test_corpus: List[str]) -> float:
        """
        시험 말뭉치에서 헷갈림도 셈하기.
        
        헷갈림도 = exp(평균 음의 로그 가능도)
        
        인수:
            test_corpus: 시험 월의 목록
            
        반환값:
            헷갈림도 점수(낮을수록 좋다)
        """
        self.model.eval()
        total_loss = 0
        total_words = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        with torch.no_grad():
            for sentence in test_corpus:
                words = sentence.lower().split()
                words = [self.vocab.START_TOKEN] + words + [self.vocab.END_TOKEN]
                indices = [self.vocab.word_to_idx(w) for w in words]
                
                if len(indices) < 2:
                    continue
                
                # 들임과 목표 만들기
                input_seq = torch.tensor([indices[:-1]], dtype=torch.long).to(self.device)
                target_seq = torch.tensor([indices[1:]], dtype=torch.long).to(self.device)
                
                # 순전파
                if hasattr(self.model, 'lstm') or hasattr(self.model, 'rnn'):
                    logits, _ = self.model(input_seq)
                else:
                    logits = self.model(input_seq)
                
                # 손실을 계산한다
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
                
                total_loss += loss.item()
                total_words += len(indices) - 1
        
        # 헷갈림도 = exp(평균 손실)
        avg_loss = total_loss / total_words
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_bits_per_character(self, test_corpus: List[str]) -> float:
        """
        글자당 비트 잣대 셈하기.
        
        인수:
            test_corpus: 시험 월의 목록
            
        반환값:
            글자당 비트 점수(낮을수록 좋다)
        """
        total_log_prob = 0
        total_chars = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for sentence in test_corpus:
                words = sentence.lower().split()
                words = [self.vocab.START_TOKEN] + words + [self.vocab.END_TOKEN]
                indices = [self.vocab.word_to_idx(w) for w in words]
                
                if len(indices) < 2:
                    continue
                
                input_seq = torch.tensor([indices[:-1]], dtype=torch.long).to(self.device)
                target_seq = torch.tensor(indices[1:], dtype=torch.long).to(self.device)
                
                if hasattr(self.model, 'lstm') or hasattr(self.model, 'rnn'):
                    logits, _ = self.model(input_seq)
                else:
                    logits = self.model(input_seq)
                
                # 로그 확률 얻기
                log_probs = torch.log_softmax(logits[0], dim=-1)
                
                # 목표 낱말의 로그 확률 더하기
                for i, target_word_idx in enumerate(target_seq):
                    total_log_prob += log_probs[i, target_word_idx].item()
                
                # 본디 월의 글자 세기
                total_chars += len(sentence)
        
        # BPC = -log_prob / (chars * log(2))
        bpc = -total_log_prob / (total_chars * math.log(2))
        
        return bpc
    
    def evaluate_generation_diversity(self, num_samples: int = 100,
                                     max_length: int = 20) -> Dict:
        """
        만든 표본의 여러 갈래임 값매김하기.
        
        잣대:
        - 겹치지 않는 n-그램
        - 자기 BLEU(낮을수록 여러 갈래)
        - 엔트로피
        
        인수:
            num_samples: 만들 표본의 개수
            max_length: 표본마다의 최대 길이
            
        반환값:
            여러 갈래임 잣대의 사전
        """
        from tutorial_09_conditional_generation import GenerationStrategies
        
        # 표본 만들기
        samples = []
        for _ in range(num_samples):
            start_token = torch.tensor([[self.vocab.word_to_idx(self.vocab.START_TOKEN)]])
            generated = GenerationStrategies.nucleus_sampling(
                self.model, start_token, max_length=max_length,
                vocab=self.vocab
            )
            
            # 글로 바꾸기
            tokens = generated[0].tolist()
            words = [self.vocab.idx_to_word(idx) for idx in tokens
                    if idx != self.vocab.word_to_idx(self.vocab.PAD_TOKEN)]
            samples.append(' '.join(words))
        
        # 지표를 계산한다
        metrics = {}
        
        # 1. 겹치지 않는 표본
        unique_samples = len(set(samples))
        metrics['unique_samples'] = unique_samples
        metrics['repetition_rate'] = 1 - (unique_samples / num_samples)
        
        # 2. 겹치지 않는 n-그램
        for n in [2, 3, 4]:
            all_ngrams = []
            for sample in samples:
                words = sample.split()
                ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
            
            if all_ngrams:
                unique_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)
                metrics[f'unique_{n}grams'] = unique_ngrams / total_ngrams
        
        # 3. 평균 길이
        metrics['avg_length'] = np.mean([len(s.split()) for s in samples])
        
        # 4. 낱말 곳간 덮음
        all_words = set()
        for sample in samples:
            all_words.update(sample.split())
        metrics['vocab_coverage'] = len(all_words)
        
        return metrics


class BenchmarkSuite:
    """말 모델의 표준 잣대."""
    
    @staticmethod
    def penn_treebank_benchmark(model, vocab):
        """Penn Treebank 잣대로 값매김하기."""
        # 실제로는 PTB 자료를 읽어 들인다
        # 자리 채움 짜기
        print("Penn Treebank Benchmark")
        print("-" * 50)
        print("Vocabulary: ~10k words")
        print("Training: ~1M words")
        print("Validation: ~70k words")
        print("Test: ~80k words")
        print()
        print("State-of-the-art perplexities:")
        print("  LSTM (3-layer): ~60-80")
        print("  Transformer (6-layer): ~50-70")
        print("  AWD-LSTM: ~57")
        print("  Transformer-XL: ~54")
    
    @staticmethod
    def wikitext_benchmark(model, vocab):
        """WikiText-2 잣대로 값매김하기."""
        print("WikiText-2 Benchmark")
        print("-" * 50)
        print("Vocabulary: ~33k words")
        print("Training: ~2M words")
        print("More challenging than PTB (longer context)")
        print()
        print("State-of-the-art perplexities:")
        print("  LSTM (2-layer): ~100-120")
        print("  Transformer (6-layer): ~80-100")
        print("  GPT-2 (small): ~30-40")


def demonstrate_evaluation():
    """값매김 잣대 보이기."""
    
    print("=" * 70)
    print("Language Model Evaluation Metrics")
    print("=" * 70)
    
    print("""
1. 헷갈림도
-------------
- 가장 흔한 내재 잣대
- 모델이 시험 자료를 얼마나 잘 어림하는지 잰다
- 낮을수록 좋다
- 읽는 법: "평균적으로 모델이 낱말 PPL개 가운데서 고르고 있다"

흔한 값:
  n-그램 모델: 200~400 (PTB)
  LSTM: 80~120 (PTB)
  변환기: 60~80 (PTB)
  크게 미리 익힘: 20~40 (PTB)

한계:
- 만들어 낸 글의 좋음과 곧바로 이어지지는 않는다
- 낱말 곳간이 다르면 견줄 수 없다
- 뜻의 조리는 재지 못한다


2. 글자당 비트
---------------------
- 고르게 맞춘 잣대
- 토막내기가 달라도 견줄 수 있다
- 낮을수록 좋다
- 글자 수준 모델에 쓴다


3. 여러 갈래임 잣대
--------------------
- 다른 n-그램: 전체 n-그램에 대한 겹치지 않는 것의 비
- 자기 BLEU: 만든 표본끼리의 BLEU 점수(낮을수록 여러 갈래)
- 엔트로피: 낱말 분포의 섀넌 엔트로피
- 되풀이 비율: 되풀이되는 차례의 잦기


4. 사람의 값매김
-------------------
값매김할 갈래:
- 매끄러움: 말법의 올바름
- 조리: 논리의 흐름
- 한결같음: 어긋남 없음
- 맞닿음: 주제에 맞음
- 사실다움: 참됨

매김 잣대:
- 리커트 잣대(1~5)
- 짝 견줌
- 가장 좋음-가장 나쁨 잣대


5. 뒤따르는 일
-------------------
구체적인 쓰임새에서 값매김하기:
- 글 이어 쓰기
- 물음 답하기
- 간추리기
- 옮김
- 대화


값매김의 좋은 버릇:
---------------------------
1. 잣대를 여럿 쓴다(내재 + 바깥)
2. 믿음 구간을 함께 알린다
3. 여러 분야에서 시험하기
4. 사람의 값매김을 넣는다
5. 치우침 살피기
6. 효율 재기(빠르기, 기억 공간)
7. 어그러지는 방식 값매김하기
8. 센 바탕 잣대와 견주기
    """)


def compare_models():
    """여러 모델을 견주는 틀."""
    
    print("\n" + "=" * 70)
    print("Model Comparison Framework")
    print("=" * 70)
    
    results = {
        'N-gram': {
            'perplexity': 350,
            'speed': 'Very Fast',
            'memory': 'Low',
            'diversity': 'Low'
        },
        'LSTM': {
            'perplexity': 100,
            'speed': 'Medium',
            'memory': 'Medium',
            'diversity': 'Medium'
        },
        'Transformer': {
            'perplexity': 70,
            'speed': 'Slow (train), Fast (inference)',
            'memory': 'High',
            'diversity': 'High'
        },
        'GPT-2': {
            'perplexity': 35,
            'speed': 'Medium',
            'memory': 'Very High',
            'diversity': 'Very High'
        }
    }
    
    print("\nModel Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<15} {'Perplexity':<12} {'Speed':<20} {'Memory':<10}")
    print("-" * 70)
    
    for model, metrics in results.items():
        print(f"{model:<15} {metrics['perplexity']:<12} "
              f"{metrics['speed']:<20} {metrics['memory']:<10}")


if __name__ == "__main__":
    demonstrate_evaluation()
    compare_models()
    
    print("""

익힘 문제:
1. 말 모델의 BLEU 점수 짜기
2. 여러 갈래임을 재려 자기 BLEU 셈하기
3. A/B 시험 얼거리 짜기
4. 잣대를 그려 보는 대시보드 만들기
5. 엇갈린 엔트로피 쪼갬 살피기 짜기
6. 사람의 값매김 사이 만들기
7. 잣대와 사람의 판단 사이 얽힘 재기
8. 모델 눈금 맞추기 값매김하기(믿음도와 정확도)
9. 맞서는 보기로 시험하기
10. 만든 글의 공정함과 치우침 재기
    """)```

## 2. 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 말 모델에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
실습 10 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_languagemodelevaluator():
        model = LanguageModelEvaluator(...)
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

**다룬 것** — 실습 10

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

고갱이 갈래는 `LanguageModelEvaluator`, `BenchmarkSuite`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
