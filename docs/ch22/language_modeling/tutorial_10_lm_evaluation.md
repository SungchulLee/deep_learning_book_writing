# 언어 모델 평가

언어 모델 평가는 정답 텍스트에 주는 확률과 생성 결과의 품질을 구분한다. 퍼플렉서티는 정답 토큰 예측의 평균 로그로스를 나타내고, 생성 다양성은 출력 표본의 분포적 특성을 보여 준다. 어느 하나만으로 유창성·관련성·사실성을 모두 판단할 수는 없다. 이 페이지의 평가 함수와 고정 설명용 수치도 구분해서 읽는다.

## 1. 코드와 기존 출력

    ```python
    """
    길잡이 10: 말 모델 밸류
    =======================================

    말 모델을 두루 살피는 밸류 잣대와 방법.

    밸류의 클래스:
    1. 내재 잣대(헷갈림도, 글자당 비트)
    2. 바깥 잣대(뒤따르는 일의 성능)
    3. 사람의 밸류
    4. 여러 클래스임과 좋음의 잣대

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
       - 곧바른 로스 잣대
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
        """말 모델을 두루 살피는 밸류 꾸러미."""

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

                    # 로스를 계산한다
                    loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))

                    total_loss += loss.item()
                    total_words += len(indices) - 1

            # 헷갈림도 = exp(평균 로스)
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
            만든 표본의 여러 클래스임 밸류하기.

            잣대:
            - 겹치지 않는 n-그램
            - 자기 BLEU(낮을수록 여러 클래스)
            - 엔트로피

            인수:
                num_samples: 만들 표본의 개수
                max_length: 표본마다의 최대 길이

            반환값:
                여러 클래스임 잣대의 사전
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
            """Penn Treebank 잣대로 밸류하기."""
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
            """WikiText-2 잣대로 밸류하기."""
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
        """밸류 잣대 보이기."""

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


    3. 여러 클래스임 잣대
    --------------------
    - 다른 n-그램: 전체 n-그램에 대한 겹치지 않는 것의 비
    - 자기 BLEU: 만든 표본끼리의 BLEU 점수(낮을수록 여러 클래스)
    - 엔트로피: 낱말 분포의 섀넌 엔트로피
    - 되풀이 비율: 되풀이되는 차례의 잦기


    4. 사람의 밸류
    -------------------
    밸류할 클래스:
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
    구체적인 쓰임새에서 밸류하기:
    - 글 이어 쓰기
    - 물음 답하기
    - 간추리기
    - 옮김
    - 대화


    밸류의 좋은 버릇:
    ---------------------------
    1. 잣대를 여럿 쓴다(내재 + 바깥)
    2. 믿음 구간을 함께 알린다
    3. 여러 분야에서 시험하기
    4. 사람의 밸류를 넣는다
    5. 치우침 살피기
    6. 효율 재기(빠르기, 기억 공간)
    7. 어그러지는 방식 밸류하기
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
    2. 여러 클래스임을 재려 자기 BLEU 셈하기
    3. A/B 시험 얼거리 짜기
    4. 잣대를 그려 보는 대시보드 만들기
    5. 엇갈린 엔트로피 쪼갬 살피기 짜기
    6. 사람의 밸류 사이 만들기
    7. 잣대와 사람의 판단 사이 얽힘 재기
    8. 모델 눈금 맞추기 밸류하기(믿음도와 정확도)
    9. 맞서는 보기로 시험하기
    10. 만든 글의 공정함과 치우침 재기
        """)
    ```

**기존 출력:**

    ```
    ======================================================================
    Language Model Evaluation Metrics
    ======================================================================

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


    3. 여러 클래스임 잣대
    --------------------
    - 다른 n-그램: 전체 n-그램에 대한 겹치지 않는 것의 비
    - 자기 BLEU: 만든 표본끼리의 BLEU 점수(낮을수록 여러 클래스)
    - 엔트로피: 낱말 분포의 섀넌 엔트로피
    - 되풀이 비율: 되풀이되는 차례의 잦기


    4. 사람의 밸류
    -------------------
    밸류할 클래스:
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
    구체적인 쓰임새에서 밸류하기:
    - 글 이어 쓰기
    - 물음 답하기
    - 간추리기
    - 옮김
    - 대화


    밸류의 좋은 버릇:
    ---------------------------
    1. 잣대를 여럿 쓴다(내재 + 바깥)
    2. 믿음 구간을 함께 알린다
    3. 여러 분야에서 시험하기
    4. 사람의 밸류를 넣는다
    5. 치우침 살피기
    6. 효율 재기(빠르기, 기억 공간)
    7. 어그러지는 방식 밸류하기
    8. 센 바탕 잣대와 견주기
        

    ======================================================================
    Model Comparison Framework
    ======================================================================

    Model Performance Comparison:
    ----------------------------------------------------------------------
    Model           Perplexity   Speed                Memory    
    ----------------------------------------------------------------------
    N-gram          350          Very Fast            Low       
    LSTM            100          Medium               Medium    
    Transformer     70           Slow (train), Fast (inference) High      
    GPT-2           35           Medium               Very High 


    익힘 문제:
    1. 말 모델의 BLEU 점수 짜기
    2. 여러 클래스임을 재려 자기 BLEU 셈하기
    3. A/B 시험 얼거리 짜기
    4. 잣대를 그려 보는 대시보드 만들기
    5. 엇갈린 엔트로피 쪼갬 살피기 짜기
    6. 사람의 밸류 사이 만들기
    7. 잣대와 사람의 판단 사이 얽힘 재기
    8. 모델 눈금 맞추기 밸류하기(믿음도와 정확도)
    9. 맞서는 보기로 시험하기
    10. 만든 글의 공정함과 치우침 재기
        
    ```

## 2. 퍼플렉서티의 정의와 집계

예측 대상 토큰이 N개일 때 PPL=exp[−Σ_t ln P(w_t|문맥)/N]이다. 같은 토큰화·어휘·자료·문맥과 경계 규칙에서는 낮은 값이 정답에 높은 평균 로그확률을 주었다는 뜻이다. 실제로 매번 PPL개 후보 중 균등하게 고른다는 뜻은 아니다. 토큰화나 UNK 처리·문맥 길이가 다르면 수치를 직접 순위화하지 않는다.

compute_perplexity는 문장마다 시작 토큰을 입력하고 종료 토큰까지 목표로 삼아 합산 로스를 구한다. reduction=sum으로 토큰을 합산하는 점은 맞다. 다만 ignore_index=0이 실제 PAD ID와 같아야 하고, 분모 total_words는 제외 ID 여부와 관계없이 모든 목표를 센다. 제외된 토큰이 있으면 유효 토큰 수와 맞춰야 한다. 빈 평가 말뭉치는 0 나눗셈이 되고 큰 로스는 exp에서 overflow될 수 있다.

문장마다 상태를 초기화하고 소문자 공백 토큰화를 사용한다. 문서 문맥을 연속해서 평가하는 프로토콜이나 subword GPT 모델의 평가와 같지 않다. 모델 출력도 직접 B×T×V 텐서 또는 RNN의 (logits,hidden) 튜플이어야 한다. 미래 정보가 누출되는 모델의 낮은 PPL은 올바른 자기회귀 성능이 아니다.

## 3. 문자당 비트의 비교 조건

문자당 비트는 같은 원문에 할당한 음의 log2 확률을 동일한 문자 수 정의로 나눈 값이다. 토큰화가 달라도 비교하려면 두 모델이 같은 원문 사건을 표현하고 로스 없는 인코딩·경계·문자/바이트 단위를 일치시켜야 한다. 숫자를 문자 수로 나누기만 하면 자동으로 공정해지는 것은 아니다.

현재 compute_bits_per_character는 소문자 공백 단어와 UNK로 바꾼 목표·종료의 로그확률을 더하면서 원래 sentence의 len을 분모로 쓴다. 대소문자·공백 정보와 미등록 단어의 철자가 소실되므로 원문 문자열에 대한 엄밀한 부호 길이로 볼 수 없다. 모든 문장이 빈 문자열이면 문자 수가 0이다. PPL 함수와 달리 목표 ID 0도 로그확률 합에 포함한다는 차이도 있다.

## 4. 생성 다양성 지표

unique_samples는 서로 다른 전체 표본 수이고 repetition_rate는 1−서로 다른 표본 수/전체 표본 수다. 이는 한 문장 내부의 반복 토큰 비율과 다르다. unique_n_grams 계열은 모든 표본의 n-그램을 모아 종류 수를 총 출현 수로 나눈 corpus-level distinct-n이다. 짧은 출력과 표본 수에 민감하며 무작위 문자열도 높게 나올 수 있다.

현재 생성 평가는 PAD만 제거하고 시작·종료 토큰은 문자열에 남긴다. 평균 길이·distinct-n·어휘 종류 수에도 이 특수 토큰들이 영향을 준다. vocab_coverage는 비율이 아니라 관측한 서로 다른 문자열 토큰 수다. num_samples=0은 유효하지 않으며 생성 시작 텐서는 CPU에 만들어져 GPU 모델과 장치가 맞지 않는다. 호출하는 top-p 구현의 경계 처리 한계도 이어받는다.

## 5. 수치·벤치마크와 사람 평가

BenchmarkSuite는 전달받은 모델·어휘를 평가하지 않고 고정 안내를 출력한다. compare_models의 퍼플렉서티·속도·메모리·다양성도 하드코딩한 예시이며 이번 실행이나 표준 벤치마크에서 측정한 순위가 아니다. 기존 출력의 ‘최신 성능’ 표현과 고정 범위를 현재 성능 자료로 사용하지 않는다. main에서는 평가기를 생성하거나 시험 자료의 점수를 계산하지 않는다.

실제 비교는 자료 버전·분할·토큰화·문맥·종료 처리·학습 자원과 평가 코드를 기록하고 독립 시험 자료에서 수행한다. 생성에서는 동일 조건의 표본으로 유창성·관련성·사실성·반복을 살핀다. 사람 평가에는 명확한 채점 기준·평가자 간 일치·가능하면 가려진 모델 이름과 표본 불확실성을 함께 고려한다. 다양성 증가와 품질 향상은 별개의 결론이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
같은 두 평가 배치의 평균 로스가 1과 3이고 유효 토큰 수가 각각 10과 90이면 전체 PPL의 집계 방법을 설명하라.

</div>

??? success "풀이"

    전체 평균 음의 로그우도는 (10×1+90×3)/(10+90)=2.8이고 PPL은 exp(2.8)≈16.44다. 배치 평균의 단순 평균 2를 지수화한 exp(2)≈7.39와 다르다. 현재 PPL 함수는 합산 로스를 쓰지만 분모에서도 ignore_index에 해당하는 토큰을 제외해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
compute_perplexity와 문자당 비트에 필요한 입력·분모 계약을 설명하라.

</div>

??? success "풀이"

    평가 말뭉치와 예측 사건 수가 비어 있지 않고 어휘·특수 ID·모델 인터페이스·장치가 맞아야 한다. PPL 분모는 로스에 실제 포함된 목표 토큰 수다. 문자당 비트는 동일 원문과 일관된 문자 단위의 부호 확률을 요구한다. 소문자화·공백 분리·UNK로 정보가 소실된 단어 로그확률을 원문 길이로 나눈 값에는 그 한계가 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
문장 전체 중복률, distinct-n, vocab_coverage가 각각 측정하는 것과 품질 지표로서의 한계를 설명하라.

</div>

??? success "풀이"

    전체 중복률은 완전히 같은 표본의 재등장을, distinct-n은 corpus의 n-그램 종류 수/출현 수를 센다. 현재 vocab_coverage는 어휘 대비 비율이 아니라 관측 단어 종류 수다. 특수 토큰·표본 수·길이에 영향을 받으며 의미 없는 무작위 출력도 다양하게 보일 수 있다. 관련성·유창성·사실성과 함께 평가해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 test_languagemodelevaluator가 실제 평가기와 맞는 검증인지 검토하고 점검할 사례를 제시하라.

</div>

??? success "풀이"

    LanguageModelEvaluator는 nn.Module이 아니므로 직접 호출·parameters·backward를 요구하는 아래 함수는 맞지 않는 자리표시자다. 검증 설계에는 알려진 고정 확률의 PPL, 유효 목표 집계, 빈 자료·문자 수 0, PAD·EOS 처리, distinct-n의 중복 집계, 장치 일치가 포함되어야 한다. 평가 함수는 no_grad를 쓰므로 경사 생성을 성공 기준으로 삼지 않는다. 이 편집에서는 모델이나 이 테스트를 실행하지 않는다.

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


## 정리하며

PPL·문자당 비트·다양성은 서로 다른 대상을 측정한다. 토큰화·원문 사건·유효 분모와 생성 표본의 처리를 맞추고, 고정 안내 수치를 실제 벤치마크와 구분해야 평가 결과를 해석할 수 있다.
