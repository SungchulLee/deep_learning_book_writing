# 추론

학습된 Seq2Seq 모델에 텍스트를 넣고 탐욕 선택이나 빔 탐색으로 출력 토큰을 얻는 도구다. 이 페이지의 실행부는 무작위 초기 모델을 사용하므로 출력 문자열은 학습된 번역 결과가 아니다.

## 1. 코드

```python
"""
Seq2Seq 모델을 위한 추론 스크립트
탐욕적 복호, 빔 탐색, 번역 도구를 담고 있다
"""

import torch
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class Seq2SeqInference:
    """
    Seq2Seq 모델을 위한 추론 도구
    
    인수:
        model: 학습된 Seq2Seq 모델
        src_vocab: 원본 어휘 (토큰을 색인으로)
        trg_vocab: 표적 어휘 (토큰을 색인으로)
        device: 추론을 돌릴 장치
        sos_idx: 순차열 시작 토큰의 색인
        eos_idx: 순차열 끝 토큰의 색인
        pad_idx: 덧댐 토큰의 색인
    """
    
    def __init__(self, model, src_vocab, trg_vocab, device, sos_idx=1, eos_idx=2, pad_idx=0):
        self.model = model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        # 거꾸로 된 어휘 만들기 (색인에서 토큰으로)
        self.idx_to_src = {idx: token for token, idx in src_vocab.items()}
        self.idx_to_trg = {idx: token for token, idx in trg_vocab.items()}
    
    def tokenize_source(self, text):
        """
        원본 텍스트를 토큰으로 나눈다
        
        인수:
            text: 원본 텍스트 문자열이나 토큰의 목록
            
        반환값:
            tokens: 토큰 색인의 목록
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = text
        
        # 색인으로 바꾸기
        indices = [self.src_vocab.get(token, self.src_vocab.get('<unk>', 3)) 
                   for token in tokens]
        
        return indices
    
    def detokenize_target(self, indices):
        """
        표적의 색인을 다시 텍스트로 바꾼다
        
        인수:
            indices: 토큰 색인의 목록이나 텐서
            
        반환값:
            text: 토큰을 되돌린 텍스트 문자열
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()
        
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.sos_idx]:
                token = self.idx_to_trg.get(idx, '<unk>')
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def greedy_decode(self, src_text, max_len=50):
        """
        탐욕적 복호로 번역을 만든다
        
        인수:
            src_text: 원본 텍스트 문자열이나 토큰의 목록
            max_len: 만들 순차열의 최대 길이
            
        반환값:
            translation: 번역된 텍스트
            attention_weights: 어텐션 웨이트 (있을 때)
        """
        self.model.eval()
        
        # 원본을 토큰으로 나누기
        src_indices = self.tokenize_source(src_text)
        src_tensor = torch.tensor([src_indices]).to(self.device)
        src_lengths = torch.tensor([len(src_indices)]).to(self.device)
        
        # 생성
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                if hasattr(self.model, 'encoder'):
                    # 어텐션이 있는 모델
                    output, attention = self.model.generate(
                        src_tensor, max_len, self.sos_idx, self.eos_idx, src_lengths
                    )
                    attention = attention[0].cpu()
                else:
                    output = self.model.generate(
                        src_tensor, max_len, self.sos_idx, self.eos_idx, src_lengths
                    )
                    attention = None
            else:
                raise ValueError("Model doesn't have generate method")
        
        # 토큰을 다시 글로
        translation = self.detokenize_target(output[0])
        
        return translation, attention
    
    def beam_search_decode(self, src_text, beam_width=5, max_len=50, length_penalty=0.6):
        """
        빔 탐색으로 번역을 만든다
        
        인수:
            src_text: 원본 텍스트 문자열이나 토큰의 목록
            beam_width: 빔의 수
            max_len: 만들 순차열의 최대 길이
            length_penalty: 길이 정규화 벌점
            
        반환값:
            translation: 번역된 텍스트
            score: 가장 좋은 번역의 점수
        """
        self.model.eval()
        
        # 원본을 토큰으로 나누기
        src_indices = self.tokenize_source(src_text)
        src_tensor = torch.tensor([src_indices]).to(self.device)
        src_lengths = torch.tensor([len(src_indices)]).to(self.device)
        
        # 생성
        with torch.no_grad():
            if hasattr(self.model, 'beam_search'):
                output, score = self.model.beam_search(
                    src_tensor, beam_width, max_len, 
                    self.sos_idx, self.eos_idx, src_lengths, length_penalty
                )
            else:
                raise ValueError("Model doesn't have beam_search method")
        
        # 토큰을 다시 글로
        translation = self.detokenize_target(output[0])
        
        return translation, score
    
    def translate_batch(self, src_texts, method='greedy', **kwargs):
        """
        텍스트 배치를 번역한다
        
        인수:
            src_texts: 원본 텍스트의 목록
            method: 복호 방법 ('greedy' 또는 'beam')
            **kwargs: 복호에 쓸 추가 인자
            
        반환값:
            translations: 번역된 텍스트의 목록
        """
        translations = []
        
        for src_text in src_texts:
            if method == 'greedy':
                translation, _ = self.greedy_decode(src_text, **kwargs)
            elif method == 'beam':
                translation, _ = self.beam_search_decode(src_text, **kwargs)
            else:
                raise ValueError(f"Unknown decoding method: {method}")
            
            translations.append(translation)
        
        return translations
    
    def interactive_translate(self):
        """대화식 번역 모드"""
        print("Interactive Translation Mode")
        print("Enter 'quit' to exit")
        print("-" * 50)
        
        while True:
            src_text = input("\nSource: ").strip()
            
            if src_text.lower() == 'quit':
                break
            
            if not src_text:
                continue
            
            # 탐욕적 복호
            translation_greedy, attention = self.greedy_decode(src_text)
            print(f"Greedy: {translation_greedy}")
            
            # 빔 탐색 (쓸 수 있으면)
            try:
                translation_beam, score = self.beam_search_decode(src_text)
                print(f"Beam:   {translation_beam} (score: {score:.2f})")
            except:
                pass
    
    def visualize_attention(self, src_text, translation, attention_weights):
        """
        어텐션 웨이트 그려 보기
        
        인수:
            src_text: 원본 텍스트
            translation: 번역된 텍스트
            attention_weights: 어텐션 웨이트 텐서
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 토큰으로 나누기
            src_tokens = src_text.lower().split()
            trg_tokens = translation.split()
            
            # 어텐션을 실제 길이에 맞게 잘라 내기
            attention = attention_weights[:len(trg_tokens), :len(src_tokens)].numpy()
            
            # 그림
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attention, xticklabels=src_tokens, yticklabels=trg_tokens,
                       cmap='viridis', ax=ax, cbar=True)
            ax.set_xlabel('Source')
            ax.set_ylabel('Target')
            ax.set_title('Attention Weights')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and seaborn required for visualization")


class BLEU:
    """
    BLEU 점수 계산기
    """
    
    @staticmethod
    def compute_bleu(reference, hypothesis, max_n=4):
        """
        BLEU 점수를 계산한다
        
        인수:
            reference: 기준 번역 (문자열이나 토큰의 목록)
            hypothesis: 가설 번역 (문자열이나 토큰의 목록)
            max_n: n-그램의 최대 차수
            
        반환값:
            bleu_score: BLEU 점수
        """
        if isinstance(reference, str):
            reference = reference.split()
        if isinstance(hypothesis, str):
            hypothesis = hypothesis.split()
        
        # n-그램마다 정밀도 계산
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = BLEU._get_ngrams(reference, n)
            hyp_ngrams = BLEU._get_ngrams(hypothesis, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0)
                continue
            
            # 맞은 개수 세기
            matches = sum(min(ref_ngrams.get(ng, 0), hyp_ngrams.get(ng, 0)) 
                         for ng in hyp_ngrams)
            
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)
        
        # 짧음 벌점
        bp = BLEU._brevity_penalty(len(reference), len(hypothesis))
        
        # 정밀도의 기하 평균
        if min(precisions) > 0:
            log_precision_sum = sum(torch.log(torch.tensor(p)) for p in precisions)
            geo_mean = torch.exp(log_precision_sum / max_n)
            bleu_score = bp * geo_mean.item()
        else:
            bleu_score = 0
        
        return bleu_score
    
    @staticmethod
    def _get_ngrams(tokens, n):
        """토큰에서 n-그램을 뽑는다"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    @staticmethod
    def _brevity_penalty(ref_len, hyp_len):
        """짧음 벌점을 계산한다"""
        if hyp_len > ref_len:
            return 1.0
        else:
            return torch.exp(torch.tensor(1 - ref_len / hyp_len)).item()


if __name__ == "__main__":
    # 사용 예
    from encoder import BasicEncoder
    from decoder import AttentionDecoder
    from seq2seq_model import Seq2SeqAttention
    
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 임시 어휘 만들기
    src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    trg_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # 임시 낱말 몇 개 더하기
    words = ['hello', 'world', 'how', 'are', 'you', 'good', 'morning']
    for i, word in enumerate(words):
        src_vocab[word] = i + 4
        trg_vocab[word] = i + 4
    
    # 모델 생성
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    
    encoder = BasicEncoder(
        input_size=INPUT_DIM,
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    decoder = AttentionDecoder(
        output_size=OUTPUT_DIM,
        embedding_dim=256,
        hidden_size=1024,
        encoder_hidden_size=1024,
        num_layers=2,
        rnn_type='LSTM'
    )
    
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    
    # 추론 객체 만들기
    inference = Seq2SeqInference(model, src_vocab, trg_vocab, device)
    
    # 번역 시험
    src_text = "hello world"
    translation, attention = inference.greedy_decode(src_text)
    
    print(f"Source: {src_text}")
    print(f"Translation: {translation}")
    
    # BLEU 점수 시험
    reference = "good morning world"
    hypothesis = "good morning"
    bleu_score = BLEU.compute_bleu(reference, hypothesis)
    print(f"\nBLEU score: {bleu_score:.4f}")
```

**출력:**

```
Source: hello world
Translation: 

BLEU score: 0.0000
```

## 2. 논의

추론 래퍼의 원문 토큰화는 소문자·공백 분리다. 전처리 페이지의 구두점 분리와 다르므로 학습에 사용한 규칙과 맞춰야 한다. 사전 객체가 아닌 token→index 딕셔너리를 받으며 특수 토큰 ID도 모델과 일치해야 한다. 빈 문자열은 길이 0 입력을 만들어 패킹에서 실패할 수 있다.

탐욕 디코딩은 model.generate에 위임한다. encoder 속성만으로 `(output,attention)` 반환을 가정하므로 기본 Seq2Seq처럼 텐서만 반환하는 모델에는 맞지 않는다. 빔 탐색은 해당 메서드가 있는 모델만 사용할 수 있다. translate_batch는 실제 배치 연산이 아니라 문장별 순차 호출이다.

타깃 복원은 첫 EOS에서 중단하고 PAD·SOS를 제외한다. 시각화는 이 복원된 토큰 수로 어텐션 행을 자르므로 생성 단계와의 대응, 학습·추론 토큰화가 같은지 확인해야 한다. 웨이트는 관련성의 계산 계수이며 인과적인 중요도 자체는 아니다.

BLEU 구현에는 중요한 한계가 있다. n-그램 정밀도의 분모에 전체 발생 수가 아니라 서로 다른 n-그램 종류 수를 사용하므로, 반복되는 일치가 있으면 정밀도가 1을 넘을 수도 있다. 표준 clipped precision의 분모는 후보 n-그램 발생 수의 합이어야 한다. 빈 가설은 brevity penalty에서 0 나눗셈이 생긴다. 짧은 문장은 높은 차수 n-그램이 없어 기본 4-그램 점수가 0이 될 수 있다. 이 함수의 수치를 표준 코퍼스 BLEU로 보고해서는 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
학습 전처리와 추론 토큰화가 다르면 어떤 문제가 생기는가?

</div>

??? success "풀이"
    같은 단어나 문장 부호가 다른 토큰으로 나뉘어 OOV와 입력 ID가 달라질 수 있다. 학습 때의 토큰화·정규화·사전과 특수 토큰 규약을 동일하게 사용해야 한다. 무작위 모델의 출력 문자열로 번역 품질을 평가할 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 BLEU 정밀도의 분모가 잘못된 이유를 간단한 반복 예로 설명하라.

</div>

??? success "풀이"
    후보와 정답이 모두 a a라면 유니그램 일치 발생 수는 2다. 현재 코드는 후보의 유니그램 종류 수 1로 나누어 2를 얻는다. 올바른 분모는 발생 수 합 2여서 정밀도는 1이다. n-그램 종류와 횟수를 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
greedy, beam과 translate_batch의 역할을 구분하라.

</div>

??? success "풀이"
    greedy는 단계별 최대 로짓 토큰을 선택하는 모델 메서드에 위임한다. beam은 제한된 여러 후보를 유지하는 근사 탐색이다. translate_batch는 문장을 하나씩 처리하므로 병렬 배치 추론과 다르다. 각 방식의 점수·종료 조건과 실제 시간을 맞춰 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_seq2seqinference():
        model = Seq2SeqInference(...)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

추론에서는 학습 때의 전처리와 사전, 모델 반환 규약을 유지한다. 검색 방식과 시각화의 의미를 구분하고, 여기의 간단한 BLEU 함수를 표준 평가기로 사용하지 않는다.
