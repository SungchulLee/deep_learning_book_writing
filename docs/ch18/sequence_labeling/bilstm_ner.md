# 이름 알아보기를 위한 두 방향 LSTM
## 학습 목표

- 이름 알아보기에서 두 방향 맥락이 왜 중요한지 이해한다
- 두 방향 LSTM 바탕 차례 이름표 붙이기 모델을 짠다
- 꼴 특징을 위해 글자 수준 묻힘을 넣는다
- 가장 좋은 성능을 위해 두 방향 LSTM과 CRF를 아우른다

## 이름 알아보기에 왜 두 방향 LSTM인가?

이름 있는 것을 정확히 알아보려면 **두 방향 맥락**이 필요하다:

- **왼쪽 맥락**: "CEO **Satya Nadella**" — 직함이 사람임을 가리킨다
- **오른쪽 맥락**: "**Microsoft** announced" — 움직씨가 조직임을 가리킨다
- **두 방향 모두**: "The **New York** Times" — 마디 전체가 필요하다

## 구조

```
Word: "Apple"  "Inc"  "announced"  "profits"
        ↓        ↓        ↓           ↓
    [Embedding] [Emb]   [Emb]       [Emb]
        ↓        ↓        ↓           ↓
    →LSTM→   →LSTM→   →LSTM→     →LSTM→   (Forward)
    ←LSTM←   ←LSTM←   ←LSTM←     ←LSTM←   (Backward)
        ↓        ↓        ↓           ↓
    [Concat]  [Concat] [Concat]   [Concat]
        ↓        ↓        ↓           ↓
    [Linear]  [Linear] [Linear]   [Linear]
        ↓        ↓        ↓           ↓
      B-ORG    I-ORG      O           O
```

## 수학적 정식화

### 앞 방향 LSTM과 뒤 방향 LSTM

**앞 방향 LSTM**은 왼쪽에서 오른쪽으로 다룬다:

$$\overrightarrow{h}_t = \text{LSTM}(x_t, \overrightarrow{h}_{t-1})$$

**뒤 방향 LSTM**은 오른쪽에서 왼쪽으로 다룬다:

$$\overleftarrow{h}_t = \text{LSTM}(x_t, \overleftarrow{h}_{t+1})$$

### 이어 붙인 나타냄

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2d}$$

### 내보냄 점수

$$e_t = W_o \cdot h_t + b_o \in \mathbb{R}^{|L|}$$

## PyTorch 구현

### 기본 두 방향 LSTM 이름 알아보기

```python
import torch
import torch.nn as nn
from typing import Optional

class BiLSTMNER(nn.Module):
    """
    이름 알아보기를 위한 두 방향 LSTM.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # 두 방향이면 이것이 두 배
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        
        self.num_tags = num_tags
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        # 토큰 임베딩
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)
        
        # 두 방향 LSTM 부호화
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(embeds)
        
        lstm_out = self.dropout(lstm_out)
        
        # 분류
        logits = self.classifier(lstm_out)
        
        # 이름표가 있으면 손실을 셈한다
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, self.num_tags),
                labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def predict(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.argmax(outputs['logits'], dim=-1)
```

### 글자 묻힘을 갖춘 두 방향 LSTM

글자 수준 특징은 꼴(앞가지, 뒷가지, 대문자 쓰기)을 담아낸다:

```python
class CharLSTM(nn.Module):
    """글자 수준 LSTM 부호기."""
    
    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int = 25,
        char_hidden_dim: int = 50
    ):
        super().__init__()
        
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=0
        )
        
        self.char_lstm = nn.LSTM(
            char_embedding_dim,
            char_hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        인수:
            char_ids: (batch, max_words, max_chars)
        반환값:
            char_repr: (batch, max_words, char_hidden_dim)
        """
        batch_size, max_words, max_chars = char_ids.shape
        
        # 다루기 위해 꼴 바꾸기
        char_ids = char_ids.view(-1, max_chars)
        
        char_embeds = self.char_embedding(char_ids)
        _, (h_n, _) = self.char_lstm(char_embeds)
        
        # 앞 방향과 뒤 방향의 마지막 상태 이어 붙이기
        char_repr = torch.cat([h_n[0], h_n[1]], dim=-1)
        char_repr = char_repr.view(batch_size, max_words, -1)
        
        return char_repr


class BiLSTMCharNER(nn.Module):
    """글자 수준 묻힘을 갖춘 두 방향 LSTM 이름 알아보기."""
    
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int,
        word_embedding_dim: int = 100,
        char_embedding_dim: int = 25,
        char_hidden_dim: int = 50,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_encoder = CharLSTM(
            char_vocab_size, char_embedding_dim, char_hidden_dim
        )
        
        # 낱말 묻힘과 글자 묻힘 이어 붙이기
        lstm_input_dim = word_embedding_dim + char_hidden_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.num_tags = num_tags
    
    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        # 낱말 묻힘
        word_embeds = self.word_embedding(word_ids)
        
        # 글자 수준 나타냄
        char_repr = self.char_encoder(char_ids)
        
        # 이어 붙인다
        combined = torch.cat([word_embeds, char_repr], dim=-1)
        combined = self.dropout(combined)
        
        # 두 방향 LSTM
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)
        
        # 분류
        logits = self.classifier(lstm_out)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_tags), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}
```

### 두 방향 LSTM-CRF

```python
class BiLSTMCRF(nn.Module):
    """이름 알아보기를 위해 CRF 층을 얹은 두 방향 LSTM."""
    
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF 층(넘어가기를 배운다)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.num_tags = num_tags
    
    def _get_emissions(self, input_ids, attention_mask=None):
        embeds = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out)
    
    def forward(self, input_ids, attention_mask, labels):
        emissions = self._get_emissions(input_ids, attention_mask)
        
        # CRF 앞먹임: 로그 나눔 함수와 참값 점수 셈하기
        gold_score = self._score_sentence(emissions, labels, attention_mask)
        partition = self._forward_algorithm(emissions, attention_mask)
        
        loss = (partition - gold_score).mean()
        return {'loss': loss, 'emissions': emissions}
    
    def _forward_algorithm(self, emissions, mask):
        """로그 나눔 함수 셈하기."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # 초기화한다
        alpha = self.start_transitions + emissions[:, 0]
        
        for t in range(1, seq_len):
            alpha_t = []
            for tag in range(num_tags):
                emit = emissions[:, t, tag].unsqueeze(1)
                trans = self.transitions[tag].unsqueeze(0)
                score = alpha + emit + trans
                alpha_t.append(torch.logsumexp(score, dim=1))
            
            new_alpha = torch.stack(alpha_t, dim=1)
            alpha = torch.where(
                mask[:, t].unsqueeze(1).bool(),
                new_alpha,
                alpha
            )
        
        return torch.logsumexp(alpha + self.end_transitions, dim=1)
    
    def _score_sentence(self, emissions, tags, mask):
        """참값 차례의 점수 셈하기."""
        batch_size, seq_len, _ = emissions.shape
        
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for t in range(1, seq_len):
            emit = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            trans = self.transitions[tags[:, t], tags[:, t-1]]
            score += (emit + trans) * mask[:, t].float()
        
        # 끝 넘어가기
        seq_lens = mask.sum(dim=1).long()
        last_tags = tags.gather(1, (seq_lens - 1).unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def decode(self, input_ids, attention_mask):
        """비터비 풀기."""
        emissions = self._get_emissions(input_ids, attention_mask)
        return self._viterbi_decode(emissions, attention_mask)
    
    def _viterbi_decode(self, emissions, mask):
        """가장 좋은 이름표 차례 찾기."""
        batch_size, seq_len, num_tags = emissions.shape
        
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, t].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score, indices = next_score.max(dim=1)
            
            history.append(indices)
            score = torch.where(mask[:, t].unsqueeze(1).bool(), next_score, score)
        
        score += self.end_transitions
        
        # 되짚기
        best_tags = []
        for idx in range(batch_size):
            seq_len_i = mask[idx].sum().int().item()
            _, best_last = score[idx].max(dim=0)
            
            tags = [best_last.item()]
            for hist in reversed(history[:seq_len_i - 1]):
                tags.append(hist[idx, tags[-1]].item())
            tags.reverse()
            best_tags.append(tags)
        
        return best_tags
```

## 견줌: 두 방향 LSTM과 변환기

| 갈래 | 두 방향 LSTM | 변환기 |
|--------|--------|-------------|
| 필요한 익힘 자료 | 적다 | 많다 |
| 익히기 빠르기 | 더 빠르다 | 더 느리다 |
| 미룸 빠르기 | 더 느리다 | 더 빠르다(묶음) |
| 성능 | 좋다 | 가장 좋다 |
| 글자 특징 | 더하기 쉽다 | 아래낱말이 다룬다 |
| 미리 익힘의 이점 | GloVe/Word2Vec | BERT/RoBERTa |

## 모범 사례

1. **미리 익힌 낱말 묻힘을 쓴다**(GloVe, fastText)
2. 꼴을 위해 **글자 수준 특징을 더한다**
3. 짜임 있는 어림을 위해 **CRF와 아우른다**
4. 층 사이에 **떨구기를 쓴다**(0.3~0.5)
5. **기울기 자르기**(최대 노름 1.0~5.0)
6. 검증 F1을 보고 **일찍 멈추기**

## 요약

두 방향 LSTM 모델은 여전히 이름 알아보기의 센 바탕이다:

- 두 방향 맥락을 잘 담아낸다
- 익힘 자료가 적어도 잘 된다
- 글자 특징으로 넓히기 쉽다
- CRF 층이 경계 찾기를 낫게 한다

## 연습문제

**연습문제 1.**
BIO 이름표 방식을 설명하여라. 월 "Barack Obama visited New York City"에 이름표를 어떻게 붙이겠는가?

??? success "연습문제 1 풀이"
    BIO 이름표에서 **B-X**는 갈래 X인 것의 시작을, **I-X**는 것의 안쪽(이어짐)을, **O**는 어느 것에도 들지 않는 토막을 나타낸다.

    | 토막 | 이름표 |
    |-------|-----|
    | Barack | B-PER |
    | Obama | I-PER |
    | visited | O |
    | New | B-LOC |
    | York | I-LOC |
    | City | I-LOC |

    B 이름표는 같은 갈래의 것이 잇달아 나올 때(보기로 "Obama Trump"를 서로 다른 PER 둘로) 가리는 데 꼭 필요하다.

---

**연습문제 2.**
차례 이름표 붙이기에서 자리마다 따로 소프트맥스로 갈래를 매기는 대신 두 방향 LSTM 위에 CRF 층을 얹는 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    자리마다 따로 소프트맥스를 쓰면 이름표를 서로 아랑곳없이 다루어 넘어감의 매임을 버린다. 보기로 O 뒤에 I-PER을 미루어 볼 수 있는데, BIO 이름 붙이기에서는 옳지 않다. CRF 켜는 이름표 이음 전체의 함께 확률을 모델로 삼아, 어떤 이름표 넘어감이 옳은지 담는 넘어감 행렬 $A_{ij}$을 배운다. 이음의 CRF 점수는 $s(x, y) = \sum_t (E_{y_t, t} + A_{y_t, y_{t+1}})$이며 $E$은 두 방향 LSTM에서 나온 내보냄 점수다. 이러면 비터비 풀기로 두루 앞뒤가 맞는 미루어 봄을 얻는다.

---

**연습문제 3.**
것 수준에서의 이름 알아보기 값매김에 쓰는 정밀도, 재현율, F1 점수를 설명하여라. 것 수준 값매김이 토막 수준보다 왜 더 빡빡한가?

??? success "연습문제 3 풀이"
    **개체 수준** 따지기는 개체의 테두리와 갈래가 모두 딱 맞아야 한다. 정밀도 = (옳게 미루어 본 개체) / (미루어 본 개체 모두). 재현율 = (옳게 미루어 본 개체) / (참 개체 모두). $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$이다. 이는 토막 수준보다 깐깐하다. 참 개체가 "New York City"인데 "New York"으로 미루어 보면 토막 수준에서는 반쯤 점수를 받지만(토막 3개 가운데 2개가 맞다) 개체 수준에서는 0점이다(테두리가 어긋난다). 개체 수준의 자가 참 세상의 쓸모를 더 잘 비춘다.

---

**연습문제 4.**
두 방향 LSTM은 양쪽 방향의 맥락을 어떻게 담아내는가? 자리 $t$에서의 숨은 상태 셈을 적어라.

??? success "연습문제 4 풀이"
    두 방향 LSTM은 $x_1, \ldots, x_T$을 다루는 앞으로 가는 LSTM과 $x_T, \ldots, x_1$을 다루는 뒤로 가는 LSTM으로 이루어진다.

    $$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

    자리 $t$의 마지막 나타냄은 이어 붙인 $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$이며, 왼쪽 앞뒤 흐름($\overrightarrow{h}_t$을 거쳐)과 오른쪽 앞뒤 흐름($\overleftarrow{h}_t$을 거쳐)을 모두 담는다. 개체명 알아내기에서는 둘레 낱말이 양쪽에서 걸리는 일이 잦으므로 이것이 종요롭다.
