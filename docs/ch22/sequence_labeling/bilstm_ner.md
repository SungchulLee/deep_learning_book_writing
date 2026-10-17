# BiLSTM을 이용한 개체명 인식

양방향 LSTM은 각 토큰에서 왼쪽·오른쪽 문맥을 결합한다. 직함·동사·이름의 다른 부분이 개체의 유형과 경계를 판단하는 단서가 될 수 있지만, 특정 동사 하나가 조직임을 확정하지는 않는다. The New York Times처럼 전체 언급을 판단할 때 내부의 New York만 별도 장소로 태깅할지는 데이터 지침에 달렸다.

## 1. 문맥 피처와 출력 로짓

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

방향별 은닉 크기가 d이면 두 방향을 연결한 표현은2d차원이다. LSTM은 은닉·셀 상태를 각각 갱신하므로 h_t=LSTM(x_t,h_{t−1}) 표기는 셀 상태를 생략한 약식이다. 연결 피처를 선형 변환해 K개 태그 로짓을 만든다. 양방향은 뒤 토큰도 사용해 전체 문장 NER에 적합할 수 있지만 미래 입력이 없는 온라인 과제에는 그대로 적용할 수 없다.

## 2. 기본 BiLSTMNER

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
        
        # 이름표가 있으면 로스를 셈한다
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

입력은 B×L 토큰 ID와 선택적 attention_mask·labels다. default hidden_dim256이면 방향별128, 연결256차원에서 B×L×K 로짓을 만든다. hidden_dim이 홀수이면 두 방향 합은2 floor(hidden_dim/2)라 선형 입력과 맞지 않는다. 단어 임베딩은 무작위 초기화이며 사전 학습 임베딩을 로드하는 코드가 없다.

mask가 있으면 유효 길이로 pack한다. 양의 길이·오른쪽 연속 패딩을 가정하며 unpack은 배치의 최대 유효 길이로 줄어들 수 있다. labels가 원래 고정 패딩 길이를 유지하면 펼친 logits와 원소 수가 달라 CE에서 실패할 수 있다. total_length 또는 목표·마스크 정렬 규칙이 필요하다.

로스는−100 레이블만 무시한다. attention_mask만0으로 두고 패딩 labels를O로 채우면 CE에서 자동으로 제외되지 않는다. 모든 레이블이−100일 때 평균의 유효 분모도 고려해야 한다. predict는 no_grad를 쓰지만 eval로 전환하지 않아 호출자가 학습 모드이면 dropout이 남는다. 반환 argmax에는 패딩 위치를 제거하는 별도 절차도 없다.

## 3. 문자 피처를 결합하는 모델

```python
class CharLSTM(nn.Module):
    """글자 수준 LSTM 인코더."""
    
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

CharLSTM은 B×L×M 문자 ID를 단어별 시퀀스로 펼쳐 방향별 최종 은닉을 연결한다. 기본 char_hidden_dim50이면 단어마다50차원이고 단어 임베딩100과 합쳐150차원의 입력이 된다. char_hidden_dim도 홀수이면 실제 출력2 floor(H/2)와 선언한 입력 차원이 달라질 수 있다.

문자 길이 packing·mask가 없으므로 짧은 단어의 padding 문자도 LSTM 상태 갱신에 참여한다. embedding의 padding_idx0이 padding을0벡터로 만들어도 recurrent bias·은닉 갱신이 멈추는 것은 아니다. 전부 padding인 단어도 최종 피처가 항상0이 아니다.

BiLSTMCharNER는 attention_mask를 인수로 받지만 실제 단어 LSTM의 packing·피처·로스 마스크에 사용하지 않는다. 단어 embedding에도 padding_idx가 없다. 양방향 문맥이 패딩 길이에 영향을 받을 수 있어 입력 mask가 있다는 사실만으로 패딩 처리가 완성되었다고 설명하면 안 된다. 문자 피처는 접두·접미·철자 패턴을 학습할 수 있지만 규칙을 자동으로 정확히 인식하거나 모든 희귀 개체를 해결하지는 않는다.

## 4. BiLSTMCRF의 학습·디코딩 구분

```python
class BiLSTMCRF(nn.Module):
    """이름 알아보기를 위해 CRF 레이어를 얹은 두 방향 LSTM."""
    
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
        
        # CRF 레이어(넘어가기를 배운다)
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
        
        # CRF 피드포워드: 로그 나눔 함수와 참값 점수 셈하기
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

이 페이지의 CRF는 정답 점수와 forward 알고리즘에서 모두 `transitions[next,prev]`를 사용한다. 그러나 `_viterbi_decode`는 broadcast 방식으로 `transitions[prev,next]`를 사용해 **학습한 분포와 다른 방향의 점수로 디코딩**한다. [별도 CRF 절](crf.md)의 구현과도 오류 위치가 같지 않으므로 각각의 경로를 확인해야 한다.

이 BiLSTMCRF의 인코더는 attention_mask를 받아도 packing하지 않는다. CRF의 오른쪽 패딩 mask가 태그 점수를 제외해도 BiLSTM의 역방향 상태가 패딩을 읽는 문제를 자동 해결하지는 않는다. CRF는 첫 위치가 유효한 비어 있지 않은 연속 prefix를 가정하고 패딩 tags도 유효 ID가 필요하다. CE용−100은 gather 전에 마스킹되지 않는다.

decode는 eval·no_grad를 자체 적용하지 않으며 BIO 강제 제약도 없다. CRF를 붙였다고 항상 올바른 BIO나 더 높은 F1을 보장하지 않는다. 모드·제약·전이 방향·유효 길이를 맞춘 뒤 같은 span 지표로 비교해야 한다.

## 5. 다른 모델과 비교하기

BiLSTM과 트랜스포머의 데이터 요구·속도·정확도를 하나의 고정 순위로 정할 수 없다. 사전 학습 규모·모델 크기·시퀀스 길이·장치·배치·토큰화·도메인이 영향을 준다. 무작위 BiLSTM과 대규모 사전 학습 인코더를 비교하면 구조뿐 아니라 사전 학습의 효과도 섞인다.

문자 피처·dropout·경사 clipping·CRF·임베딩 초기화는 선택 가능한 설계다. 검증 자료에서 목적에 맞게 선택하고 최고 checkpoint와 독립 평가를 사용한다. 이 페이지는 구조 조각을 제공할 뿐 여러 모델을 학습한 통제 실험 결과를 제공하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
hidden_dim=255와 char_hidden_dim=51을 그대로 넣을 때 생길 수 있는 차원 문제를 설명하라.

</div>

??? success "풀이"

    방향별 바닥 나눗셈 때문에 실제 연결 차원은254와50이다. 그러나 선형 또는 다음 LSTM은255·51로 선언한 값을 기대해 맞지 않을 수 있다. 양방향 연결 차원을 일관되게 정의해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기본 모델에서 packing 뒤 로짓과 labels의 길이가 다를 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    pad_packed_sequence가 배치의 최대 유효 길이로 반환하지만 labels는 원래 패딩 길이를 유지할 수 있다. CE는 펼친 원소 수가 같아야 하므로 total_length 또는 목표 정렬이 필요하다. −100 로스 마스크와 입력 packing도 서로 다른 기능이다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
문자 padding_idx=0과 attention_mask 인수가 패딩을 완전히 해결하는가?

</div>

??? success "풀이"

    패딩 0 벡터도 LSTM의 상태 갱신을 멈추지 않고 문자 길이를 pack하지 않는다. 문자 결합 모델은 attention_mask를 사용하지 않아 단어 패딩도 읽는다. 각 단계의 유효 길이·목표 마스크를 실제로 연결해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
이 페이지의 CRF 학습과 Viterbi의 전이 방향을 비교하라.

</div>

??? success "풀이"

    정답·forward는 [next,prev]로 일치하지만 Viterbi는 [prev,next]로 계산한다. 비대칭 전이면 학습한 분포의 최적 경로를 구하지 않는다. 방향을 통일하고 eval·no_grad와 BIO 제약도 필요에 맞게 적용해야 한다.



## 정리하며

BiLSTM은 양방향 문맥을 만들고 문자 피처·CRF와 결합할 수 있다. 그러나 차원·packing·목표 마스크·모드와 전이 방향이 맞아야 의미 있는 모델이 된다. 현재 코드의 패딩·디코딩 한계와 실제 학습·평가 범위를 구분한다.
