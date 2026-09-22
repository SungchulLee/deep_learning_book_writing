"""inference — inference 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch09/seq2seq/inference.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
            attention_weights: 어텐션 가중치 (있을 때)
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
        텍스트 묶음을 번역한다
        
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
        어텐션 가중치 그려 보기
        
        인수:
            src_text: 원본 텍스트
            translation: 번역된 텍스트
            attention_weights: 어텐션 가중치 텐서
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
