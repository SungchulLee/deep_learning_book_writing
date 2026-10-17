# Seq2Seq 모델

기본 인코더·디코더와 어텐션 버전을 연결하고, 티처 포싱 순전파·탐욕 생성·빔 탐색을 구분하는 예제다. 출력의 형태 확인은 학습이나 번역 평가와 다르다.

## 1. 코드

??? note "코드 (403줄)"

    ```python
    """
    완전한 순차열 대 순차열 모델
    인코더와 디코더를 엮어 온전한 구조를 만든다
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
            encoder: 인코더 모듈
            decoder: 디코더 모듈
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

            # 디코더의 출력을 담을 텐서
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

            # 원본 순차열 부호화
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

            # 디코더의 첫 입력은 <sos> 토큰이다
            decoder_input = trg[:, 0].unsqueeze(1)

            # 한 번에 토큰 하나씩 복호
            for t in range(1, trg_len):
                # 디코더를 지나는 순전파
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
            encoder: 인코더 모듈
            decoder: 어텐션이 있는 디코더 모듈
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
                attentions: 시각마다의 어텐션 웨이트 (배치 크기, trg_len, src_len)
            """
            batch_size = src.shape[0]
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.output_size

            # 출력과 어텐션 웨이트를 담을 텐서
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
            attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)

            # 원본 순차열의 가림막 만들기
            mask = self.create_mask(src)

            # 원본 순차열 부호화
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

            # 디코더의 첫 입력은 <sos> 토큰이다
            decoder_input = trg[:, 0].unsqueeze(1)

            # 한 번에 토큰 하나씩 복호
            for t in range(1, trg_len):
                # 어텐션이 있는 디코더를 지나는 순전파
                output, hidden, cell, attention_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs, cell, mask
                )

                # 예측과 어텐션 웨이트 담기
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
                all_attentions: 스텝마다의 어텐션 웨이트 (배치 크기, seq_len, src_len)
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

                    # 어텐션 웨이트 담기
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

        # 초파라메터
        input_vocab_size = 10000
        output_vocab_size = 10000
        embedding_dim = 256
        hidden_size = 512
        num_layers = 2
        dropout = 0.1

        # 인코더와 디코더 만들기
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
            hidden_size=hidden_size * 2,  # 인코더가 양방향이라 2를 곱한다
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
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    ```


**출력:**

```
Model: Seq2SeqAttention
Input shape: torch.Size([32, 20])
Target shape: torch.Size([32, 25])
Output shape: torch.Size([32, 25, 10000])
Attention shape: torch.Size([32, 25, 20])
Total parameters: 57,565,969
```

## 2. 논의

두 클래스는 대안 모델이다. forward는 trg의 첫 토큰을 시작 입력으로 삼아 위치 1부터 예측을 저장한다. outputs[:,0]은 0으로 남으므로 로스에서 첫 위치를 제외해야 한다. teacher_forcing_ratio의 난수 선택은 단계별로 한 번 이루어져 배치 전체에 공통 적용된다. Python random을 사용하므로 torch 시드만으로 선택 순서가 고정되지 않는다.

어텐션 마스크는 src가 pad_idx와 다른지로 만든다. 길이 패킹 후 인코더 출력 폭이 원래 src 폭보다 짧아지면 마스크·어텐션 저장 텐서와 크기가 맞지 않을 수 있다. 양방향 상태가 이미 이어 붙여져 반환되므로 디코더의 레이어수·은닉 차원도 맞춰야 한다. ConvEncoder처럼 초기 상태를 반환하지 않는 인코더를 바로 대체할 수는 없다.

generate는 탐욕 선택을 쓰고 모든 표본이 EOS를 한 번 이상 생성하면 멈춘다. 먼저 끝난 표본도 계속 생성하므로 그 표본의 첫 EOS 뒤는 평가에서 제외해야 한다. 반환에는 SOS가 포함되며 max_len은 추가 생성 횟수다. 호출 후 평가 모드를 유지한다.

beam_search는 배치 1을 가정하며 빔 너비가 어휘 크기를 넘지 않아야 한다. 로그 점수 합을 SOS까지 포함한 길이의 거듭제곱으로 나눠 순위를 정한다. 최종 반환 score는 정규화 전 원점수다. 제한 길이에 도달한 미완료 후보도 비교에 포함하므로 반드시 EOS로 끝난 번역을 반환한다고 볼 수 없다. 빔 탐색은 제한된 후보를 유지하는 근사 탐색이지 전체 최대 확률 시퀀스의 보장이 아니다.

실행부의 trg는 무작위 토큰이라 첫 위치도 실제 SOS라고 보장되지 않는다. 따라서 제시된 형태 출력은 배선 시연이며 실제 학습 데이터의 정렬과 특수 토큰 규약은 별도로 구성해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
forward에서 타깃 첫 위치와 이후 로스 위치를 어떻게 맞춰야 하는가?

</div>

??? success "풀이"
    trg[:,0]은 시작 입력이며 출력 첫 위치는 예측하지 않아 0으로 남는다. outputs[:,1:]를 trg[:,1:]와 비교하고 PAD 타깃은 제외한다. SOS·EOS ID는 실제 사전과 일치해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
배치 생성에서 첫 EOS 뒤의 토큰이 남는 이유와 처리 방법을 설명하라.

</div>

??? success "풀이"
    종료 여부는 누적하지만 완료된 표본의 다음 입력·출력을 고정하지 않는다. 다른 표본을 기다리는 동안 추가 토큰이 생길 수 있으므로 표본별 첫 EOS에서 잘라 평가하거나 완료된 표본을 별도로 마스킹한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
빔 탐색의 원점수와 길이 정규화 점수를 구분하라.

</div>

??? success "풀이"
    원점수는 선택한 토큰들의 로그 확률 합이다. 순위는 이를 길이의 거듭제곱으로 나눈 값으로 결정하지만 함수는 원점수를 반환한다. 길이 정의에는 SOS가 포함되고 미완료 후보도 있을 수 있으므로 외부에서 점수를 재현할 때 같은 규칙을 사용해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

학습 순전파, 탐욕 생성과 빔 탐색은 입력과 점수 규약이 다르다. 타깃 첫 위치, 마스크 길이, 상태 차원과 종료 이후 토큰을 확인해야 실제 학습·평가에 연결할 수 있다.
