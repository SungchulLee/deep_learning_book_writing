# 코드: 프랑스어→영어 번역

이 구현은 바다나우(덧셈) 어텐션을 갖춘 부호기-복호기 구조로 프랑스어 문장을 영어로 옮기는 완전한 순차열 대 순차열 번역 시스템을 세운다. 데이터 확보, 텍스트 전처리, 어휘 만들기, 모델 정의, 교사 강요를 쓰는 학습, 탐욕적 복호, 어텐션 시각화까지 아우른다.

**출처.** [PyTorch seq2seq 번역 실습](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)을 고쳐 쓴 것으로, 자세한 주석을 붙여 하나의 독립된 스크립트로 다시 짰다.

---

## 1. 구조

이 모델은 함께 움직이는 세 부품으로 이루어진다.

```
Source: "je suis étudiant"
         ↓
   ┌──────────────────────┐
   │   Encoder (GRU)      │
   │                      │
   │  x₁ → h₁            │
   │  x₂ → h₂            │  → all encoder hidden states H = [h₁, h₂, ..., h_T]
   │  ...                 │  → final hidden state h_T
   │  x_T → h_T           │
   └──────────────────────┘
              ↓
   ┌──────────────────────┐
   │   Bahdanau Attention  │
   │                      │
   │  score(sₜ, hⱼ) =     │
   │    Vᵀ tanh(Wa sₜ +   │  → context vector cₜ = Σⱼ αₜⱼ hⱼ
   │         Ua hⱼ)       │  → attention weights αₜ
   └──────────────────────┘
              ↓
   ┌──────────────────────┐
   │   Decoder (GRU)      │
   │                      │
   │  [embed(yₜ₋₁); cₜ]  │
   │       → GRU          │  → ŷₜ (log-softmax over target vocab)
   │       → Linear       │
   └──────────────────────┘
              ↓
Target: "i am a student"
```

### 수식으로 나타내기

**부호기.** 원본 토큰 $(x_1, \ldots, x_T)$에 대해 다음과 같다.

$$\mathbf{e}_t = \text{Embed}(x_t) \in \mathbb{R}^H$$

$$\mathbf{h}_t^{enc} = \text{GRU}(\mathbf{e}_t, \mathbf{h}_{t-1}^{enc}) \in \mathbb{R}^H$$

**어텐션.** 복호기의 $t$번째 걸음에서 바다나우 덧셈 어텐션은 복호기 상태 $\mathbf{s}_t$과 부호기 상태 $\mathbf{h}_j^{enc}$ 사이의 정렬 점수를 계산한다.

$$\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_t + \mathbf{U}_a \mathbf{h}_j)$$

$$\alpha_{t,j} = \frac{\exp(\text{score}(\mathbf{s}_t, \mathbf{h}_j))}{\sum_{k=1}^{T} \exp(\text{score}(\mathbf{s}_t, \mathbf{h}_k))}$$

$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{t,j} \, \mathbf{h}_j^{enc}$$

**복호기.** 걸음마다 복호기는 임베딩한 이전 토큰과 어텐션 문맥을 이어 붙여 GRU에 넣는다.

$$\mathbf{s}_t = \text{GRU}([\text{Embed}(y_{t-1}); \mathbf{c}_t], \mathbf{s}_{t-1})$$

$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t)$$

확률 $p = 0.5$의 교사 강요는 학습 중에 정답인 이전 토큰 $y_{t-1}$을 넣을지 모델 자신의 예측 $\hat{y}_{t-1}$을 넣을지 무작위로 고른다.

---

## 2. 데이터 파이프라인

### 내려받기와 전처리

데이터셋은 탭으로 나눈 텍스트 파일로 배포되는 Tatoeba 프로젝트의 영어-프랑스어 문장 쌍이다. 전처리는 유니코드 문자를 ASCII로 정규화하고, 모든 텍스트를 소문자로 바꾸고, 문장 부호 앞에 공백을 넣는다.

```python
SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 10

def unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```

토큰이 10개 미만이고 영어 문장이 흔한 머리말(`"i am"`, `"he is"`, `"they are"` 따위)로 시작하는 쌍만 걸러 내어, 작은 모델을 학습시키기에 알맞은 단순 평서문으로 이루어진 데이터셋을 얻는다.

### 어휘 만들기

`Lang` 클래스는 낱말↔색인의 양방향 대응과 낱말 빈도를 지닌다.

```python
class Lang:
    def __init__(self, name: str):
        self.name = name
        self.word2index: dict[str, int] = {}
        self.word2count: dict[str, int] = {}
        self.index2word: dict[int, str] = {0: "SOS", 1: "EOS"}
        self.n_words: int = 2  # SOS와 EOS

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```

두 언어의 어휘를 만든 뒤 문장 쌍을 덧댄 정수 텐서로 바꾸고 배치 크기 32의 `DataLoader`으로 감싼다.

---

## 3. 모델 구현

### 부호기

부호기는 원본 토큰을 저마다 임베딩하고 한 층짜리 GRU로 순차열을 처리하며 자리마다 숨은 상태를 모은다.

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        """
        매개변수
        ----------
        x : LongTensor, 모양 (B, T)

        반환값
        -------
        all_hidden : Tensor, 모양 (B, T, H)
        last_hidden : Tensor, 모양 (1, B, H)
        """
        hidden = None
        all_hidden_list = []
        for t in range(x.size(1)):
            embedded = self.dropout(self.embedding(x[:, t].unsqueeze(1)))
            output, hidden = self.gru(embedded, hidden)
            all_hidden_list.append(hidden.squeeze(0).unsqueeze(1))
        all_hidden = torch.cat(all_hidden_list, dim=1)
        return all_hidden, hidden
```

(순차열 전체를 한 번에 처리하지 않고) 한 걸음씩 도는 반복문을 쓴 것은 이해를 돕기 위해서이다. 시각마다 숨은 상태가 어떻게 달라지는지 뚜렷이 드러난다. 실전 코드에서는 순차열 전체를 `self.gru`에 한 번에 넘기는 편이 효율적이다.

### 바다나우 어텐션

어텐션 모듈은 복호기의 질의와 부호기의 모든 열쇠 사이의 덧셈 정렬 점수를 계산한다.

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        query : (B, 1, H) — 복호기의 숨은 상태
        keys  : (B, T, H) — 부호기의 모든 숨은 상태

        문맥 (B, 1, H)과 가중치 (B, 1, T)를 돌려준다
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)  # (B, 1, T)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)        # (B, 1, H)
        return context, weights
```

`self.Wa(query) + self.Ua(keys)`에서 방송이 통하는 까닭은 `query`의 모양이 $(B, 1, H)$이고 `self.Ua(keys)`의 모양이 $(B, T, H)$이기 때문이다. PyTorch가 질의를 부호기의 $T$개 자리에 걸쳐 퍼뜨린다.

### 어텐션 복호기

복호기는 임베딩과 어텐션과 GRU를 엮어 자기회귀 생성 반복문을 이룬다.

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_hidden, all_encoder_hidden, target_tensor=None):
        batch_size = all_encoder_hidden.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=DEVICE
        ).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden

        decoder_outputs, attentions = [], []
        for t in range(MAX_LENGTH):
            dec_out, decoder_hidden, attn_w = self._step(
                decoder_input, decoder_hidden, all_encoder_hidden
            )
            decoder_outputs.append(dec_out)
            attentions.append(attn_w)

            if target_tensor is not None and random.random() < TEACHER_FORCING_RATIO:
                decoder_input = target_tensor[:, t].unsqueeze(1)
            else:
                _, topi = dec_out.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions
```

핵심 설계 선택은 다음과 같다.

- **GRU의 입력 차원이 $2H$이다**: 임베딩한 토큰($H$)과 어텐션 문맥($H$)을 이어 붙인 것이다.
- **교사 강요 비율 0.5**: 노출 편향(언제나 정답을 봄)과 학습의 불안정함(언제나 제 예측을 봄) 사이에서 균형을 잡는다. 교육 과정에 기댄 대안은 예정된 표본 추출 절을 보라.
- **argmax 예측에 `detach()` 걸기**: 자유 실행 모드에서 이산적인 표본 추출 연산으로 기울기가 흐르는 것을 막는다.

---

## 4. 학습

학습은 학습률 $10^{-3}$의 Adam 최적화와, 복호기의 로그 소프트맥스 출력에 대한 음의 로그 가능도 손실을 쓴다.

```python
def train_epoch(dataloader, encoder, decoder, enc_opt, dec_opt, criterion):
    total_loss = 0.0
    for input_t, target_t in dataloader:
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        all_hidden, last_hidden = encoder(input_t)
        dec_out, _, _ = decoder(last_hidden, all_hidden, target_t)

        loss = criterion(
            dec_out.view(-1, dec_out.size(-1)),
            target_t.view(-1),
        )
        loss.backward()
        enc_opt.step()
        dec_opt.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

출력을 $(B \cdot T, V)$으로, 표적을 $(B \cdot T)$으로 모양을 바꾸어 복호기의 모든 자리에 대해 손실을 한꺼번에 계산한다. 덧댄 토큰(색인 0)도 손실에 이바지하는데, 실전 시스템이라면 `nn.NLLLoss`에 `ignore_index=0`을 넘겨 가릴 것이다.

걸러 낸 데이터셋(약 1만 쌍)으로 80세대를 학습하며, 요즘 GPU에서는 5~10분, CPU에서는 20~30분쯤 걸린다.

---

## 5. 평가와 추론

### 탐욕적 복호

추론할 때 복호기는 교사 강요 없이 돌며 걸음마다 확률이 가장 높은 토큰을 고른다.

```python
def evaluate_sentence(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        inp_t = tensor_from_sentence(input_lang, sentence).unsqueeze(0)
        padded = torch.zeros(1, MAX_LENGTH, dtype=torch.long, device=DEVICE)
        padded[0, :inp_t.size(1)] = inp_t

        all_hidden, last_hidden = encoder(padded)
        dec_out, _, attentions = decoder(last_hidden, all_hidden)

        decoded_words = []
        for step in range(MAX_LENGTH):
            _, topi = dec_out[:, step, :].topk(1)
            idx = topi[0, 0].item()
            if idx == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word.get(idx, "<UNK>"))
    return decoded_words, attentions
```

이는 걸음마다 argmax를 취하는 탐욕적 탐색이다. 더 나은 번역을 원하면 빔 탐색(빔 탐색 절 참고)이 여러 가설을 유지하여 계산은 더 들지만 대체로 더 좋은 결과를 낸다.

### 어텐션 시각화

어텐션 가중치 행렬 $\boldsymbol{\alpha} \in \mathbb{R}^{T' \times T}$은 복호기가 생성 걸음마다 어떤 원본 토큰에 주목하는지 드러낸다. 이 행렬을 열지도로 그리면 모델의 정렬 거동을 해석할 수 있는 증거가 된다.

```python
def show_attention(input_sentence, output_words, attentions):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    n_out = len(output_words)
    n_in = len(input_sentence.split(" ")) + 1  # <EOS> 때문에 +1
    cax = ax.matshow(attentions[0, :n_out, :n_in].cpu().numpy(), cmap="bone")
    fig.colorbar(cax)
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
    ax.set_yticklabels([""] + output_words)
    plt.tight_layout()
    plt.show()
```

잘 학습된 모델에서는 어순이 지켜지는 단순한 번역이면 어텐션 행렬이 단조로운 대각선에 가깝고, 어순이 다른 언어 쌍에서는 특유의 교차 무늬가 나타난다(프랑스어와 영어 사이의 형용사-명사 순서 뒤바뀜 따위).

---

## 6. 코드 실행하기

```bash
python seq2seq_attention.py
```

이 스크립트는 Tatoeba 데이터셋을 저절로 내려받고 80세대를 학습한 뒤 부호기와 복호기의 가중치를 `model/`에 저장하고, 무작위 쌍과 따로 떼어 둔 시험 문장 네 개로 평가하며 어텐션을 그려 보인다.

### 기대 출력

```
Data ready.
  Pairs: 11445
  fra vocab: 4601
  eng vocab: 2991
Example pair: ['je suis pret .', 'i m ready .']

0m 45s (- 11m 15s) (epoch 5 6%) loss=3.8234
1m 28s (- 10m 12s) (epoch 10 12%) loss=2.9156
...
11m 02s (- 0m 0s) (epoch 80 100%) loss=0.5483

> je suis desole si c est une question idiote
= i m sorry if it s a stupid question
< i m sorry if it s a stupid question . <EOS>
```

---

## 7. 초매개변수

| 매개변수 | 값 | 참고 |
|-----------|-------|-------|
| `HIDDEN_SIZE` | 128 | 부호기와 복호기 모두의 GRU 숨은 차원 |
| `BATCH_SIZE` | 32 | 학습 미니배치 크기 |
| `TEACHER_FORCING_RATIO` | 0.5 | 복호기 걸음마다 정답을 쓸 확률 |
| `MAX_LENGTH` | 10 | 문장의 최대 길이(토큰). 더 긴 쌍은 걸러 낸다 |
| `lr` | 0.001 | Adam 학습률 |
| `n_epochs` | 80 | 학습 세대 수 |
| `dropout_p` | 0.1 | 임베딩에 적용하는 드롭아웃 |

---

## 8. 다른 절과의 이음새

이 구현은 7장 곳곳의 개념을 아우른다.

- **낱말 임베딩 (7.1절)**: 부호기와 복호기 모두 토큰 색인을 조밀한 벡터로 보내는 학습된 `nn.Embedding` 층을 쓴다. 이 임베딩은 사전 학습 벡터를 쓰지 않고 나머지 모델과 함께 학습한다.
- **RNN과 숨은 상태 (7.2절)**: 부호기와 복호기의 GRU 세포가 순차 정보를 쌓는 숨은 상태를 지니며, RNN 절의 순환식을 그대로 구현한다.
- **LSTM과 GRU의 문 (7.3절)**: GRU의 갱신 문과 재설정 문이 순차열에 걸쳐 기억을 골라 지키게 해 주어, 이런 번역 순차열에서 기본 RNN을 괴롭혔을 기울기 소실을 누그러뜨린다.
- **바다나우 어텐션 (7.4절)**: 덧셈 어텐션 장치가 고정된 문맥 벡터의 정보 병목을 없애, 복호기가 생성 걸음마다 쓸모 있는 부호기 상태에 물어볼 수 있게 한다.
- **부호기-복호기 틀 (7.5절)**: 전체 구조가 어텐션으로 보강한 문맥 벡터와 함께 부호기-복호기 틀을 실현한다.
- **교사 강요 (7.5절)**: 학습은 수렴 속도와 노출 편향의 균형을 잡으려고 확률적인 교사 강요를 쓴다.

---

## 9. 퀀트 금융으로의 확장

같은 부호기-복호기-어텐션 구조를 여러 금융 순차열 변환 과제에 쓸 수 있다.

- **텍스트에서 신호 만들기**: 애널리스트 보고서 문장을 부호화하고, 부호화된 금융 서술에 조건을 두어 정형화된 매매 신호(방향, 크기, 확신)를 복호한다.
- **주문 집행**: 상위 주문의 명세(수량, 급함, 제약)를 부호화하고, 집행 품질을 최적으로 하는 하위 주문 행동(지정가, 수량, 시점)의 순차열을 복호한다.
- **자산군 간 번역**: 거시 지표나 수익률 곡선의 움직임을 부호화하고 주식 팩터 수익률 예측을 복호한다. 사실상 자산군의 언어 사이를 "번역"하는 셈이다.
- **보고서 요약**: 긴 실적 발표 회의록을 부호화하고, 포트폴리오 운용자에게 중요한 정보를 짚어 주는 간결한 요약을 복호한다.

어느 경우든 어텐션 장치는 해석 가능성이라는 결정적인 이점을 준다. 어텐션 가중치가 어떤 입력 요소가 출력 결정에 가장 큰 영향을 주었는지 짚어 주어, 금융 응용에서 흔히 요구되는 설명 가능성을 뒷받침한다.

---

## 연습문제

**연습문제 1.**
순차열 대 순차열 모델의 부호기-복호기 구조를 설명하라.

??? success "연습문제 1 풀이"
    부호기는 입력 순차열을 처리하여 문맥 벡터(마지막 숨은 상태)로 눌러 담는다. 복호기는 문맥과 앞서 만든 토큰에 조건을 두고 출력 순차열을 한 토큰씩 만든다. 교사 강요는 학습 중에 정답 토큰을 쓰고, 자기회귀 복호는 모델의 예측을 쓴다.

---

**연습문제 2.**
기본 seq2seq 모델의 정보 병목 문제란 무엇인가? 어텐션은 그것을 어떻게 푸는가?

??? success "연습문제 2 풀이"
    입력 순차열 전체를 크기가 고정된 문맥 벡터 하나로 눌러 담아야 하므로 긴 순차열에서는 정보를 잃는다. 어텐션은 복호기가 부호기의 모든 숨은 상태를 '돌아보며' 복호 걸음마다 쓸모 있는 정보를 고를 수 있게 한다. 그러면 병목이 사라진다.

---

**연습문제 3.**
seq2seq 모델에 대해 빔 너비가 $k$인 빔 탐색 복호를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def beam_search(model, src, beam_width=5, max_len=50):
        beams = [(0.0, [BOS])]  # (log_prob, tokens)
        for _ in range(max_len):
            candidates = []
            for score, tokens in beams:
                logits = model.decode(src, tokens)
                top_k = logits[-1].topk(beam_width)
                for prob, idx in zip(*top_k):
                    candidates.append((score + prob.item(), tokens + [idx.item()]))
            beams = sorted(candidates, reverse=True)[:beam_width]
        return beams[0][1]
    ```

---

**연습문제 4.**
텍스트 생성에서 탐욕적 복호와 빔 탐색과 표본 추출 전략을 견주어라.

??? success "연습문제 4 풀이"
    탐욕적 복호는 걸음마다 argmax를 고르므로 빠르지만 전체적으로 가장 좋은 순차열을 놓칠 수 있다. 빔 탐색은 상위 $k$개의 가설을 유지하여 품질이 낫지만 비싸고 되풀이가 생길 수 있다. 표본 추출은 확률 분포에서 뽑으므로 다양하지만 다스리기 어렵다. 핵(top-p) 표본 추출은 다양성과 품질의 균형을 잡는다.

## 정리하며

이 마당은 구조、데이터 파이프라인、모델 구현、학습을 차례로 짚었다.
