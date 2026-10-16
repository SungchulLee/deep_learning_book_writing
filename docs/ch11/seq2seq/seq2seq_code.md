# 코드: 프랑스어→영어 번역

이 페이지는 GRU 인코더와 가산 어텐션 디코더를 이용한 프랑스어→영어 번역의 구조를 설명한다. 코드가 여러 조각으로 제시되어 있으며, 데이터 로딩·배치 생성과 디코더의 _step 등 일부 정의는 빠져 있다. 본문만으로 독립 실행 가능한 전체 시스템이 완성된 것은 아니다.

**출처.** [PyTorch seq2seq 번역 실습](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)을 고쳐 쓴 것으로, 자세한 주석을 붙여 하나의 독립된 스크립트로 다시 짰다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

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

**인코더.** 원본 토큰 $(x_1, \ldots, x_T)$에 대해 다음과 같다.

$$\mathbf{e}_t = \text{Embed}(x_t) \in \mathbb{R}^H$$

$$\mathbf{h}_t^{enc} = \text{GRU}(\mathbf{e}_t, \mathbf{h}_{t-1}^{enc}) \in \mathbb{R}^H$$

**어텐션.** 디코더의 $t$번째 스텝에서 바다나우 덧셈 어텐션은 갱신 전 디코더 상태 $\mathbf{s}_{t-1}$와 인코더 상태 $\mathbf{h}_j^{enc}$ 사이의 정렬 점수를 계산한다.

$$\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_j) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j)$$

$$\alpha_{t,j} = \frac{\exp(\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_j))}{\sum_{k=1}^{T} \exp(\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_k))}$$

$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{t,j} \, \mathbf{h}_j^{enc}$$

**디코더.** 스텝마다 디코더는 임베딩한 이전 토큰과 어텐션 문맥을 이어 붙여 GRU에 넣는다.

$$\mathbf{s}_t = \text{GRU}([\text{Embed}(y_{t-1}); \mathbf{c}_t], \mathbf{s}_{t-1})$$

$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t)$$

확률 $p = 0.5$의 티처 포싱는 학습 중에 정답인 이전 토큰 $y_{t-1}$을 넣을지 모델 자신의 예측 $\hat{y}_{t-1}$을 넣을지 무작위로 고른다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

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

`Lang` 클래스는 단어↔색인의 양방향 대응과 단어 빈도를 지닌다.

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

Lang에는 SOS=0, EOS=1이 예약되어 있고 별도의 PAD·UNK ID는 정의되지 않았다. 아래 추론 코드의 0 패딩은 SOS와 같은 ID를 사용한다. 사전 구축과 배치 생성의 전체 과정은 이 페이지에 제시되지 않는다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 3. 모델 구현

### 인코더

인코더는 원본 토큰을 저마다 임베딩하고 한 레이어짜리 GRU로 시퀀스를 처리하며 자리마다 은닉 상태를 모은다.

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
        파라메터
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

(시퀀스 전체를 한 번에 처리하지 않고) 한 스텝씩 도는 반복문을 쓴 것은 이해를 돕기 위해서이다. 시각마다 은닉 상태가 어떻게 달라지는지 뚜렷이 드러난다. 실전 코드에서는 시퀀스 전체를 `self.gru`에 한 번에 넘기는 편이 효율적이다.

### 바다나우 어텐션

어텐션 모듈은 디코더의 질의와 인코더의 모든 키 사이의 덧셈 정렬 점수를 계산한다.

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        """
        query : (B, 1, H) — 디코더의 숨은 상태
        keys  : (B, T, H) — 인코더의 모든 숨은 상태

        문맥 (B, 1, H)과 웨이트 (B, 1, T)를 돌려준다
        """
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)  # (B, 1, T)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)        # (B, 1, H)
        return context, weights
```

`self.Wa(query) + self.Ua(keys)`에서 브로드캐스팅이 통하는 까닭은 `query`의 모양이 $(B, 1, H)$이고 `self.Ua(keys)`의 모양이 $(B, T, H)$이기 때문이다. PyTorch가 질의를 인코더의 $T$개 자리에 걸쳐 퍼뜨린다.

### 어텐션 디코더

디코더는 임베딩과 어텐션과 GRU를 엮어 자기회귀 생성 반복문을 이룬다.

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
- **티처 포싱 비율 0.5**: 노출 바이어스(언제나 정답을 봄)과 학습의 불안정함(언제나 제 예측을 봄) 사이에서 균형을 잡는다. 교육 과정에 기댄 대안은 스케줄드 샘플링 절을 보라.
- **argmax와 detach**: 선택된 정수 토큰에는 일반적인 미분이 흐르지 않는다. detach가 순환 은닉 상태의 시간 역전파까지 끊는다는 뜻은 아니다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 4. 학습

학습은 러닝레이트 $10^{-3}$의 Adam 최적화와, 디코더의 로그 소프트맥스 출력에 대한 음의 로그 우도 로스를 쓴다.

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

출력과 정답을 토큰별로 펼쳐 음의 로그 우도를 계산한다. 하지만 이 페이지에서는 0이 SOS이자 패딩으로 쓰이므로, 실전에서는 별도의 PAD ID를 두고 인코더 길이·어텐션 마스크·로스 제외를 일관되게 처리해야 한다. ignore_index만 추가해도 인코더나 어텐션의 패딩 영향까지 없어지는 것은 아니다.

학습 시간과 최종 품질은 데이터 버전, 필터링, 하드웨어와 구현에 따라 달라진다. 이 페이지에는 전체 학습 실행과 측정 조건이 없어 고정된 분 단위 실행 시간을 제시할 수 없다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 5. 평가와 추론

### 탐욕 디코딩

추론할 때 디코더는 티처 포싱 없이 돌며 스텝마다 확률이 가장 높은 토큰을 고른다.

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

추론은 매 단계 argmax를 고르는 탐욕 선택이다. 빔 탐색은 여러 후보를 유지하지만 실제 품질 향상을 보장하지 않으며 점수와 종료 조건을 따로 설계해야 한다. 이 평가 함수는 eval 모드를 직접 설정하지 않고, 사전에 없는 단어와 너무 긴 문장에 대한 처리도 별도로 필요하다.

### 어텐션 시각화

어텐션 웨이트 행렬 $\boldsymbol{\alpha} \in \mathbb{R}^{T' \times T}$은 디코더가 생성 스텝마다 어떤 원본 토큰에 주목하는지 드러낸다. 이 행렬을 열지도로 그리면 모델의 정렬 거동을 해석할 수 있는 증거가 된다.

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

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 6. 코드 실행하기

```bash
python seq2seq_attention.py
```

아래 명령과 기대 출력은 전체 시스템을 구성했을 때의 사용 예시다. 해당 실행 파일과 데이터 다운로드·저장·분할 코드가 이 페이지에 모두 제공된 것은 아니므로, 이를 현재 코드 조각의 검증된 실행 결과나 독립 테스트 성능으로 해석해서는 안 된다.

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

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 7. 하이퍼파라미터

| 파라메터 | 값 | 참고 |
|    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

-----|    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

----|    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

----|
| `HIDDEN_SIZE` | 128 | 인코더와 디코더 모두의 GRU 숨은 차원 |
| `BATCH_SIZE` | 32 | 학습 미니배치 크기 |
| `TEACHER_FORCING_RATIO` | 0.5 | 디코더 스텝마다 정답을 쓸 확률 |
| `MAX_LENGTH` | 10 | 문장의 최대 길이(토큰). 더 긴 쌍은 걸러 낸다 |
| `lr` | 0.001 | Adam 러닝레이트 |
| `n_epochs` | 80 | 학습 세대 수 |
| `dropout_p` | 0.1 | 임베딩에 적용하는 드롭아웃 |

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 8. 다른 절과의 이음새

이 구현은 12장 곳곳의 개념을 아우른다.

- **단어 임베딩 (12.1절)**: 인코더와 디코더 모두 토큰 색인을 조밀한 벡터로 보내는 학습된 `nn.Embedding` 레이어를 쓴다. 이 임베딩은 사전 학습 벡터를 쓰지 않고 나머지 모델과 함께 학습한다.
- **RNN과 은닉 상태 (12.2절)**: 인코더와 디코더의 GRU 세포가 순차 정보를 쌓는 은닉 상태를 지니며, RNN 절의 순환식을 그대로 구현한다.
- **LSTM과 GRU의 문 (12.3절)**: GRU의 갱신 문과 재설정 문이 시퀀스에 걸쳐 기억을 골라 지키게 해 주어, 이런 번역 시퀀스에서 기본 RNN을 괴롭혔을 그래디언트 소실을 누그러뜨린다.
- **바다나우 어텐션 (12.5절)**: 덧셈 어텐션 메커니즘이 고정된 문맥 벡터의 정보 병목을 없애, 디코더가 생성 스텝마다 쓸모 있는 인코더 상태에 물어볼 수 있게 한다.
- **인코더-디코더 틀 (12.4절)**: 전체 구조가 어텐션으로 보강한 문맥 벡터와 함께 인코더-디코더 틀을 실현한다.
- **티처 포싱 (12.4절)**: 학습은 수렴 속도와 노출 바이어스의 균형을 잡으려고 확률적인 티처 포싱를 쓴다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

## 9. 퀀트 금융으로의 확장

인코더·디코더의 관점은 텍스트 요약이나 시계열 변환에도 적용할 수 있다. 다만 연속값의 출력 분포, 데이터 분할과 실제 예측 시점의 정보 조건을 새로 정해야 한다. 번역 모델을 그대로 연결하면 유용한 금융 예측이나 최적 액션이 나온다고 볼 수 없다.

어텐션 웨이트는 가중합에 사용한 계수이며 출력 결정의 인과적 기여도와 같지 않다. 설명에 활용하려면 값 벡터와 이후 계산, 입력 변경에 따른 반응도 함께 검토해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Seq2Seq 모델의 인코더-디코더 구조를 설명하라.

</div>

??? success "연습문제 1 풀이"
    인코더는 입력 시퀀스를 처리하여 문맥 벡터(마지막 은닉 상태)로 눌러 담는다. 디코더는 문맥과 앞서 만든 토큰에 조건을 두고 출력 시퀀스를 한 토큰씩 만든다. 티처 포싱는 학습 중에 정답 토큰을 쓰고, 자기회귀 복호는 모델의 예측을 쓴다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
기본 Seq2Seq의 고정 문맥 제약과 어텐션이 완화하는 부분을 설명하라.

</div>

??? success "연습문제 2 풀이"
    기본 모델은 입력을 고정 크기 최종 상태로 요약한다. 어텐션은 디코더가 여러 위치의 인코더 표현에 다시 접근하게 한다. 정보 접근 경로를 늘리지만 표현의 로스와 학습·계산 제약까지 모두 제거하지는 않는다.

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 빔 탐색 조각이 이 페이지의 모델에 바로 연결되는지 검토하고 점수·종료 조건의 누락을 설명하라.

</div>

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

    `BOS`와 `model.decode`는 현재 코드에 정의되지 않았다. 또한 로짓을 그대로 더하는 것은 로그 확률 합이 아니므로 log_softmax가 필요하다. EOS 처리, 완료 후보 관리와 길이 점수, 순환 상태 전달도 추가해야 한다. 이 조각은 완성된 빔 디코더가 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
탐욕 선택, 빔 탐색과 확률적 샘플링의 차이를 설명하라.

</div>

??? success "연습문제 4 풀이"
    탐욕 선택은 각 단계 최대 확률을 고르고, 빔 탐색은 제한된 여러 후보를 점수에 따라 유지한다. 둘 다 전체 최적 품질을 보장하지 않는다. 샘플링은 분포에서 뽑아 다양성을 얻으며 온도·top-k·top-p로 후보 분포를 조절할 수 있다. 품질과 다양성은 공통 평가 기준에서 비교한다.

## 정리하며

번역 모델의 핵심은 토큰 표현, 순환 상태, 어텐션과 다음 토큰 로스의 연결이다. 이 페이지의 조각에는 누락된 정의가 있으며 PAD·SOS 규약과 평가 모드도 보완이 필요하다. 기대 출력과 어텐션 그림을 검증된 번역 품질이나 인과적 설명으로 해석하지 않는다.
