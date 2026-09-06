# Seq2Seq 모델

완전한 Seq2Seq 모델. 부호기와 복호기를 엮어 온전한 구조를 만든다.

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순차열 모델의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
"""
완전한 순차열 대 순차열 모델
부호기와 복호기를 엮어 온전한 구조를 만든다
"""

import torch
import torch.nn as nn
import random
from encoder import BasicEncoder
from decoder import BasicDecoder, AttentionDecoder

# ========================================================================
# 메인
# ========================================================================


class Seq2Seq(nn.Module):
    """
    어텐션이 없는 기본 순차열 대 순차열 모델
    
    인수:
        encoder: 부호기 모듈
        decoder: 복호기 모듈
        device: 돌릴 장치
    """
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        seq2seq 모델을 지나는 순전파
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            trg: 표적 순차열 (배치 크기, trg_len)
            teacher_forcing_ratio: 교사 강요를 쓸 확률
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            outputs: 시각마다의 예측 (배치 크기, trg_len, output_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 복호기의 출력을 담을 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 원본 순차열 부호화
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 복호기의 첫 입력은 <sos> 토큰이다
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 한 번에 토큰 하나씩 복호
        for t in range(1, trg_len):
            # 복호기를 지나는 순전파
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            # 예측 담기
            outputs[:, t] = output
            
            # 교사 강요를 쓸지 정하기
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 예측한 토큰 얻기
            top1 = output.argmax(1)
            
            # 다음 입력은 정답이거나 예측한 토큰이다
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        탐욕적 복호로 순차열을 만든다
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            max_len: 만들 순차열의 최대 길이
            sos_idx: 순차열 시작 토큰의 색인
            eos_idx: 순차열 끝 토큰의 색인
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            generated: 만들어진 순차열 (배치 크기, seq_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 원본 부호화
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # <sos> 토큰으로 시작
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
            
            generated = [decoder_input]
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len):
                # 순전파
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                
                # 예측한 토큰 얻기
                predicted = output.argmax(1).unsqueeze(1)
                
                # 끝난 순차열 갱신
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # 예측 담기
                generated.append(predicted)
                
                # 모든 순차열이 끝나면 멈추기
                if finished.all():
                    break
                
                # 다음 입력
                decoder_input = predicted
            
            # 모든 예측 이어 붙이기
            generated = torch.cat(generated, dim=1)
            
        return generated


class Seq2SeqAttention(nn.Module):
    """
    어텐션 장치가 있는 순차열 대 순차열 모델
    
    인수:
        encoder: 부호기 모듈
        decoder: 어텐션이 있는 복호기 모듈
        device: 돌릴 장치
        pad_idx: 덧댐 토큰의 색인
    """
    
    def __init__(self, encoder, decoder, device, pad_idx=0):
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def create_mask(self, src):
        """덧댐 토큰의 가림막을 만든다"""
        mask = (src != self.pad_idx)
        return mask
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """
        어텐션이 있는 seq2seq 모델을 지나는 순전파
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            trg: 표적 순차열 (배치 크기, trg_len)
            teacher_forcing_ratio: 교사 강요를 쓸 확률
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            outputs: 시각마다의 예측 (배치 크기, trg_len, output_size)
            attentions: 시각마다의 어텐션 가중치 (배치 크기, trg_len, src_len)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 출력과 어텐션 가중치를 담을 텐서
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # 원본 순차열의 가림막 만들기
        mask = self.create_mask(src)
        
        # 원본 순차열 부호화
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # 복호기의 첫 입력은 <sos> 토큰이다
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 한 번에 토큰 하나씩 복호
        for t in range(1, trg_len):
            # 어텐션이 있는 복호기를 지나는 순전파
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, cell, mask
            )
            
            # 예측과 어텐션 가중치 담기
            outputs[:, t] = output
            attentions[:, t] = attention_weights
            
            # 교사 강요를 쓸지 정하기
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 예측한 토큰 얻기
            top1 = output.argmax(1)
            
            # 다음 입력은 정답이거나 예측한 토큰이다
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2, src_lengths=None):
        """
        어텐션과 함께 탐욕적 복호로 순차열을 만든다
        
        인수:
            src: 원본 순차열 (배치 크기, src_len)
            max_len: 만들 순차열의 최대 길이
            sos_idx: 순차열 시작 토큰의 색인
            eos_idx: 순차열 끝 토큰의 색인
            src_lengths: 원본 순차열의 실제 길이
            
        반환값:
            generated: 만들어진 순차열 (배치 크기, seq_len)
            all_attentions: 걸음마다의 어텐션 가중치 (배치 크기, seq_len, src_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 가림막 만들기
            mask = self.create_mask(src)
            
            # 원본 부호화
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # <sos> 토큰으로 시작
            decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
            
            generated = [decoder_input]
            all_attentions = []
            finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
            
            for _ in range(max_len):
                # 순전파
                output, hidden, cell, attention_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs, cell, mask
                )
                
                # 어텐션 가중치 담기
                all_attentions.append(attention_weights.unsqueeze(1))
                
                # 예측한 토큰 얻기
                predicted = output.argmax(1).unsqueeze(1)
                
                # 끝난 순차열 갱신
                finished |= (predicted.squeeze(1) == eos_idx)
                
                # 예측 담기
                generated.append(predicted)
                
                # 모든 순차열이 끝나면 멈추기
                if finished.all():
                    break
                
                # 다음 입력
                decoder_input = predicted
            
            # 모든 예측과 어텐션 이어 붙이기
            generated = torch.cat(generated, dim=1)
            all_attentions = torch.cat(all_attentions, dim=1)
            
        return generated, all_attentions
    
    def beam_search(self, src, beam_width=5, max_len=50, sos_idx=1, eos_idx=2, 
                   src_lengths=None, length_penalty=0.6):
        """
        빔 탐색으로 순차열을 만든다
        
        인수:
            src: 원본 순차열 (1, src_len) — 순차열 하나만
            beam_width: 빔의 수
            max_len: 만들 순차열의 최대 길이
            sos_idx: 순차열 시작 토큰의 색인
            eos_idx: 순차열 끝 토큰의 색인
            src_lengths: 원본 순차열의 실제 길이
            length_penalty: 길이 정규화 벌점 (alpha)
            
        반환값:
            best_sequence: 가장 좋은 순차열
            best_score: 가장 좋은 순차열의 점수
        """
        self.eval()
        
        with torch.no_grad():
            # 원본 부호화
            mask = self.create_mask(src)
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # 빔 초기화
            # 빔마다: (순차열, 점수, 숨은 상태, 세포 상태)
            beams = [(torch.full((1, 1), sos_idx, dtype=torch.long).to(self.device), 
                     0.0, hidden, cell)]
            completed = []
            
            for step in range(max_len):
                candidates = []
                
                for sequence, score, h, c in beams:
                    # 순차열이 이미 끝났으면 건너뛰기
                    if sequence[0, -1].item() == eos_idx:
                        completed.append((sequence, score, h, c))
                        continue
                    
                    # 마지막 토큰 얻기
                    last_token = sequence[:, -1].unsqueeze(1)
                    
                    # 순전파
                    output, new_h, new_c, _ = self.decoder(
                        last_token, h, encoder_outputs, c, mask
                    )
                    
                    # 상위 k개 예측 얻기
                    log_probs = torch.log_softmax(output, dim=1)
                    top_probs, top_indices = log_probs.topk(beam_width)
                    
                    # 새 후보 만들기
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        new_sequence = torch.cat([sequence, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + prob.item()
                        candidates.append((new_sequence, new_score, new_h, new_c))
                
                # 상위 beam_width개 후보 고르기
                candidates.sort(key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)
                beams = candidates[:beam_width]
                
                # 모든 빔이 끝나면 멈추기
                if len(beams) == 0:
                    break
            
            # 남은 빔을 끝난 목록에 더하기
            completed.extend(beams)
            
            # 가장 좋은 순차열 고르기
            if completed:
                best = max(completed, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty))
                return best[0], best[1]
            else:
                return beams[0][0], beams[0][1]


if __name__ == "__main__":
    # 사용 예
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 초매개변수
    input_vocab_size = 10000
    output_vocab_size = 10000
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.1
    
    # 부호기와 복호기 만들기
    encoder = BasicEncoder(
        input_size=input_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=True,
        rnn_type='LSTM'
    ).to(device)
    
    decoder = AttentionDecoder(
        output_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size * 2,  # 부호기가 양방향이라 2를 곱한다
        encoder_hidden_size=hidden_size * 2,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type='LSTM'
    ).to(device)
    
    # seq2seq 모델 만들기
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    
    # 예제 데이터
    batch_size = 32
    src_len = 20
    trg_len = 25
    
    src = torch.randint(3, input_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(3, output_vocab_size, (batch_size, trg_len)).to(device)
    
    # 순전파
    outputs, attentions = model(src, trg, teacher_forcing_ratio=0.5)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attentions.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")```

## 논의

이 구현은 클래스 두 개(`Seq2Seq`, `Seq2SeqAttention`)를 정의하며, 이들이 어우러져 완전한 순차열 모델 구조를 이룬다. 클래스마다 별개의 부품을 감싸므로 코드가 모듈식이고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 순차열 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 훑으며 핵심 설계 결정을 찾아라. 구체적인 구현 선택 세 가지를 열거하고 각각이 순차열 모델에 알맞은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 블록의 수를 설정할 수 있도록 `Seq2Seq`을 확장하라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 훑는다. (보통의 파이썬 리스트가 아니라) `nn.ModuleList`을 써야 PyTorch가 최적화를 위해 모든 매개변수를 등록한다. `for n in [2, 4, 8]: model = Seq2Seq(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
