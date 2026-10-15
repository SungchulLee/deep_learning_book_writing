# 시퀀스 레이블링을 위한 조건부 랜덤 필드

선형 사슬 조건부 랜덤 필드(CRF)는 입력 X가 주어졌을 때 레이블 열 Y의 조건부 분포를 모델링한다. 신경 인코더의 위치별 점수와 인접 레이블 전이 점수를 합해 전체 열을 정규화한다. 점수는 실수값의 잠재 함수이며 그 자체가 확률·우도인 것은 아니다. CRF를 붙이는 것만으로 BIO 규칙이 자동 강제되지는 않는다.

## 1. 시퀀스 점수와 전이 방향

길이 n의 입력에서 K개 태그의 emission E∈R^(n×K), 전이 A를 두자. 이 절의 수식에서는 **A[a,b]를 이전 a→다음 b**의 점수로 정의한다. 시작 s·종료 e를 포함하면 다음과 같다.

$$S(X,Y)=s_{y_1}+\sum_{i=1}^{n}E_{i,y_i}
+\sum_{i=2}^{n}A_{y_{i-1},y_i}+e_{y_n}.$$

$$p(Y\mid X)=\exp[S(X,Y)-\log Z(X)],\qquad
Z(X)=\sum_{Y'\in\mathcal Y(X)}\exp S(X,Y').$$

전이만 이웃 레이블을 연결하더라도 E는 전체 입력 X를 보는 BiLSTM·트랜스포머에서 만들 수 있다. 유효 태그 열 집합에 제한을 두려면 시작·종료·전이 제약을 일관되게 적용한다. 학습용 정답은 같은 집합 안에 있어야 한다.

## 2. 순방향 알고리즘과 로그우도

모든 K^n개 열을 열거하는 대신, i위치의 b로 끝나는 부분 열들의 로그합 α_i(b)를 계산한다.

$$\alpha_1(b)=s_b+E_{1,b},$$

$$\alpha_i(b)=E_{i,b}+\operatorname{logsumexp}_{a}[\alpha_{i-1}(a)+A_{a,b}],$$

$$\log Z(X)=\operatorname{logsumexp}_{b}[\alpha_n(b)+e_b].$$

logsumexp는 큰 값의 지수 overflow 등을 피하도록 안정적으로 계산한다. 전체 경로가 금지된 경우처럼 모든 입력이−∞이면 별도의 유효 경로 처리가 필요하다. 복잡도는 O(nK²)이고 순방향 값만 갱신하면 O(K) 상태를 유지할 수 있다. 실제 배치·자동미분 중간 텐서·Viterbi 역추적 저장량은 따로 계산한다.

정답 Y*의 음의 로그우도는 L=log Z−S(X,Y*)다. 제약과 점수가 같은 정상화된 분포이면0이상이다. emission 경사는 주변확률−정답 지시값이며 전이 경사는 해당 전이의 기대 횟수−관측 횟수다. 전체 시퀀스 손실의 평균과 토큰 수로 나누는 평균은 서로 다른 가중 방식이다.

## 3. Viterbi와 코드의 방향 불일치

Viterbi는 logsumexp 대신 max를 쓰고 최적 이전 태그를 저장한다. 마지막 종료 점수를 더해 끝 태그를 선택하고 역추적한다. 시퀀스의 최적 경로와 위치별 주변확률의 argmax는 일반적으로 같지 않다.

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class CRF(nn.Module):
    """
    차례 이름표 붙이기를 위한 조건부 무작위 마당 층.
    
    여기 짠 것이 받치는 것:
    - 덧대기를 곁들인 배치 셈하기
    - 수치를 든든하게 하는 로그 자리 앞먹임 알고리즘
    - 미룸을 위한 비터비 풀기
    - 가리기로 거는 넘어가기 제약
    """
    
    def __init__(
        self,
        num_tags: int,
        batch_first: bool = True,
        pad_tag_id: Optional[int] = None
    ):
        """
        CRF 층 첫자리매김.
        
        인수:
            num_tags: 이름표의 개수(START와 END를 쓰면 함께)
            batch_first: True이면 들임이 (batch, seq, features)이다
            pad_tag_id: 덧대기용 이름표 번호(손실에서 뺀다)
        """
        super().__init__()
        
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.pad_tag_id = pad_tag_id
        
        # 넘어가기 행렬: transitions[i, j] = j -> i의 점수
        # 효율적인 셈을 위한 (next_tag, current_tag) 번호 매김
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 시작과 끝 넘어가기 점수
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """매개변수를 고른 분포로 첫자리매김."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        음의 로그 가능도 손실 셈하기.
        
        인수:
            emissions: 내보냄 점수 (batch, seq_len, num_tags)
            tags: 참값 이름표 (batch, seq_len)
            mask: 불 마스크 (batch, seq_len). 쓸 수 있는 자리는 True
            reduction: 'none', 'mean', 또는 'sum'
            
        반환값:
            손실 값(reduction에 따라 홑값이거나 표본마다)
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # 참값 차례의 점수 셈하기
        gold_score = self._compute_score(emissions, tags, mask)
        
        # 나눔 함수 셈하기(모든 차례에 대한 로그-합-지수)
        partition = self._compute_partition(emissions, mask)
        
        # 음의 로그 가능도
        nll = partition - gold_score
        
        if reduction == 'none':
            return nll
        elif reduction == 'mean':
            return nll.mean()
        elif reduction == 'sum':
            return nll.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        참값 이름표 차례의 점수 셈하기.
        
        인수:
            emissions: (batch, seq_len, num_tags)
            tags: (batch, seq_len)
            mask: (batch, seq_len)
            
        반환값:
            배치 안 차례마다의 점수 (batch,)
        """
        batch_size, seq_len, _ = emissions.shape
        
        # 시작 넘어가기 점수
        score = self.start_transitions[tags[:, 0]]
        
        # 첫 자리의 내보냄 점수
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for i in range(1, seq_len):
            # 넘어가기 점수: tags[i-1] -> tags[i]
            # 쓸 수 있는 자리에만 더하기
            trans_score = self.transitions[tags[:, i], tags[:, i-1]]
            
            # 자리 i의 내보냄 점수
            emit_score = emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1)
            
            # 가리기: 쓸 수 있는 자리만 고치기
            score += (trans_score + emit_score) * mask[:, i].float()
        
        # 끝 넘어가기 점수
        # 차례마다 마지막으로 쓸 수 있는 자리 찾기
        seq_lengths = mask.sum(dim=1).long()
        last_tags = tags.gather(1, (seq_lengths - 1).unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        앞먹임 알고리즘으로 로그 나눔 함수 셈하기.
        
        인수:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            
        반환값:
            차례마다의 로그 나눔 함수 (batch,)
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # 시작 넘어가기와 첫 내보냄으로 첫자리매김
        # alpha: (batch, num_tags)
        alpha = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            # 가능한 모든 넘어가기에 퍼뜨리기
            # alpha_expand: (batch, num_tags, 1)
            # transitions: (num_tags, num_tags) -> (batch, num_tags, num_tags)로 퍼뜨림
            # emissions_expand: (batch, 1, num_tags)
            
            alpha_expand = alpha.unsqueeze(2)  # (batch, num_tags, 1)
            emit_scores = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)
            
            # 아무 이름표에서 아무 이름표로 넘어가는 점수
            # (batch, num_tags, num_tags)
            scores = alpha_expand + self.transitions + emit_scores
            
            # 앞 이름표에 대한 로그-합-지수
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch, num_tags)
            
            # 가리기: 덧댄 자리는 옛 alpha 유지
            mask_i = mask[:, i].unsqueeze(1)  # (batch, 1)
            alpha = torch.where(mask_i, new_alpha, alpha)
        
        # 끝 넘어가기를 더하고 마지막 로그-합-지수를 셈하기
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        비터비 알고리즘으로 가장 그럴듯한 이름표 차례 찾기.
        
        인수:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            
        반환값:
            배치 항목마다의 이름표 차례 목록
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool,
                            device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """
        비터비 풀기 짜기.
        
        인수:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            
        반환값:
            가장 좋은 이름표 차례의 목록
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # 시작 넘어가기로 첫자리매김
        # score: (batch, num_tags)
        score = self.start_transitions + emissions[:, 0]
        
        # 뒤 가리개 갈무리
        history = []
        
        for i in range(1, seq_len):
            # 모든 넘어가기에 퍼뜨리기
            # score: (batch, num_tags, 1)
            # transitions: (num_tags, num_tags)
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            
            # (prev_tag, current_tag) 짝마다의 점수
            # (batch, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission
            
            # 지금 이름표마다 가장 좋은 앞 이름표
            # (batch, num_tags)
            next_score, indices = next_score.max(dim=1)
            
            # 뒤 가리개 갈무리
            history.append(indices)
            
            # 가리기: 덧댄 자리는 옛 점수 유지
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i, next_score, score)
        
        # 끝 넘어가기 더하기
        score = score + self.end_transitions
        
        # 가장 좋은 마지막 이름표 얻기
        seq_lengths = mask.sum(dim=1).long()
        
        # 되짚기
        best_tags_list = []
        
        for idx in range(batch_size):
            # 마지막 자리의 가장 좋은 이름표
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            # 발자취를 되짚기
            for hist in reversed(history[:seq_lengths[idx] - 1]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            
            # 앞 차례를 얻으려 뒤집기
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list
```

위 코드에는 중요한 전이 방향 불일치가 있다. `_compute_score`의 정답 점수는 `transitions[next,prev]`를 쓰지만 partition·Viterbi는 `transitions[prev,next]`로 계산한다. 비대칭 전이에서는 정답 점수와 정규화가 **다른 모델**이므로 반환값이 올바른 CRF 음의 로그우도가 아니며 음수가 될 수도 있다. 전이 방향을 한 관례로 통일하기 전에는 이 구현이 정상적인 CRF 학습·추론을 수행한다고 설명할 수 없다.

기본 입력은 B×L×K emission, B×L 태그·bool mask다. batch_first=False일 때 시간·배치 축을 바꾼다. 첫 위치가 유효하고 적어도1토큰이며 mask는 오른쪽 패딩 형태의 연속 prefix라고 가정한다. 첫 위치는 mask를 확인하지 않고 사용하고, 마지막 태그도 mask 합−1로 고르므로 빈 열·중간이 비는 mask에는 맞지 않는다.

패딩 태그도0~K−1의 유효 ID여야 한다. gather·전이 인덱싱 후 mask를 곱하므로−100 같은 CE ignore_index를 그대로 넣으면 안 된다. `pad_tag_id`는 저장만 하고 mask 생성·태그 제외에 사용하지 않는다. 입력 길이·태그 범위 검증도 별도로 필요하다.

## 4. BiLSTM과 트랜스포머 연결

```python
class BiLSTMCRF(nn.Module):
    """
    차례 이름표 붙이기를 위한 두 방향 LSTM-CRF 모델.
    
    구조:
        묻힘 → 양방향 LSTM → 선형 → CRF
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        pad_token_id: int = 0,
        pad_tag_id: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # 두 방향이면 숨은 크기가 두 배
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # LSTM의 내놓음을 이름표 공간으로 내리쬐기
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF 층
        self.crf = CRF(num_tags, batch_first=True, pad_tag_id=pad_tag_id)
    
    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        두 방향 LSTM에서 내보냄 점수 얻기.
        
        인수:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        반환값:
            내보냄 점수 (batch, seq_len, num_tags)
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # 효율적인 LSTM 셈을 위해 덧댄 차례 묶기
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(embeddings)
        
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        return emissions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        익히기용 손실 셈하기.
        
        인수:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        반환값:
            음의 로그 가능도 손실
        """
        emissions = self._get_emissions(input_ids, attention_mask)
        
        mask = attention_mask.bool() if attention_mask is not None else None
        loss = self.crf(emissions, labels, mask=mask, reduction='mean')
        
        return loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        비터비 풀기로 이름표 차례 어림하기.
        
        인수:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        반환값:
            어림한 이름표 차례의 목록
        """
        emissions = self._get_emissions(input_ids, attention_mask)
        
        mask = attention_mask.bool() if attention_mask is not None else None
        return self.crf.decode(emissions, mask=mask)
```

BiLSTMCRF는 embedding·양방향 LSTM·선형 emission·CRF를 연결한다. hidden_dim//2를 방향별 크기로 쓰고 선형 입력은 hidden_dim이므로 hidden_dim은 짝수여야 한다. packed sequence는 양의 유효 길이와 오른쪽 패딩을 가정한다. unpack 때 total_length를 지정하지 않아 출력은 배치의 최대 유효 길이로 줄어들 수 있다. 입력 전체 패딩 길이를 유지한다고 가정하지 말고 emission·목표·mask의 대응을 명시한다.

predict는 eval·no_grad를 자체 적용하지 않는다. 추론 호출자가 모드와 경사 추적을 정해야 dropout과 메모리 사용을 제어할 수 있다. CRF의 mask와 LSTM의 packing, embedding의 padding_idx는 서로 다른 기능이다.

```python
from transformers import AutoModel

class TransformerCRF(nn.Module):
    """
    이름 알아보기를 위해 CRF 층을 얹은 변환기(BERT/RoBERTa).
    """
    
    def __init__(
        self,
        model_name: str,
        num_tags: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        hidden_size = self.transformer.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """변환기에서 내보냄 점수 얻기."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        return emissions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """CRF 손실 셈하기."""
        emissions = self._get_emissions(input_ids, attention_mask)
        loss = self.crf(emissions, labels, mask=attention_mask.bool())
        return loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """비터비 풀기로 어림하기."""
        emissions = self._get_emissions(input_ids, attention_mask)
        return self.crf.decode(emissions, mask=attention_mask.bool())
```

TransformerCRF는 사전 학습 인코더의 각 토큰 특징에서 태그 점수를 만든다. 특수 토큰·서브워드 목표를 실제 CRF 태그 열에 어떻게 정렬할지 정해야 한다. Transformer의 attention_mask를 그대로 CRF mask로 쓰면 `[CLS]`·`[SEP]`도 보통 유효 위치가 된다. 뒤 서브워드를−100으로 무시하는 CE 관례를 이 CRF에 그대로 쓸 수 없다. 유효 단어를 모아 연속 열로 만들거나 그에 맞는 별도 설계가 필요하다.

freeze_encoder는 인코더 매개변수의 경사를 끌 뿐 model.train()에서 dropout을 자동 고정하지는 않는다. 사전 학습 토큰화·입력과 학습하려는 태그 집합의 계약을 함께 확인한다.

## 5. IOB2 제약의 적용

```python
def apply_iob2_constraints(
    crf: CRF,
    tag_to_idx: dict,
    penalty: float = -10000.0
):
    """
    CRF 넘어가기에 IOB2 넘어가기 제약 걸기.
    
    올바르지 않은 넘어가기는 큰 음수 점수를 받는다.
    
    인수:
        crf: 고칠 CRF 층
        tag_to_idx: 이름표 이름에서 번호로의 대응
        penalty: 올바르지 않은 넘어가기의 점수
    """
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    num_tags = len(tag_to_idx)
    
    with torch.no_grad():
        for i in range(num_tags):
            for j in range(num_tags):
                from_tag = idx_to_tag[j]
                to_tag = idx_to_tag[i]
                
                # I-X는 B-X나 I-X 뒤에만 올 수 있다
                if to_tag.startswith('I-'):
                    entity_type = to_tag[2:]
                    valid_prev = {f'B-{entity_type}', f'I-{entity_type}'}
                    
                    if from_tag not in valid_prev:
                        crf.transitions.data[i, j] = penalty
        
        # 시작 넘어가기 제약: I-로 시작할 수 없다
        for i in range(num_tags):
            tag = idx_to_tag[i]
            if tag.startswith('I-'):
                crf.start_transitions.data[i] = penalty
```

이 함수는 `transitions[next,prev]` 관례로 금지 전이·시작 점수를−10000으로 **한 번 대입**한다. partition·Viterbi의 반대 관례와도 맞지 않는다. 또한 유한 벌점은 경로를 수학적으로 제거하지 않으며 optimizer가 이후 값을 바꿀 수 있다. 매 순전파에서 적용하는 고정 마스크 등 지속적인 제약과 구분한다.

진짜−∞를 적용하는 설계에서도 금지 점수×패딩0은NaN이 될 수 있으므로 mask 처리와 수치 연산 순서를 검토해야 한다. 학습 정규화·정답 점수·디코딩에서 같은 경로 집합을 쓰는 것이 핵심이다. IOB2의 종료에는 O·B·I가 가능하고 시작에는 I를 허용하지 않는다.

## 6. 학습 루프의 인수·평가 계약

```python
def train_ner_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    CRF를 갖춘 이름 알아보기 모델의 익히기 되풀이.
    """
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        # 학습
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            loss = model(input_ids, labels, attention_mask)
            loss.backward()
            
            # 안정성을 위한 기울기 자르기
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 검증
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model.predict(input_ids, attention_mask)
                
                # 어림과 이름표 모으기
                for pred, label, mask in zip(
                    predictions,
                    labels.cpu().numpy(),
                    attention_mask.cpu().numpy()
                ):
                    seq_len = mask.sum()
                    all_preds.append(pred[:seq_len])
                    all_labels.append(label[:seq_len].tolist())
        
        # F1 점수 셈하기
        f1 = compute_entity_f1(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  New best model saved!")
    
    return best_f1
```

루프의 `model(input_ids,labels,attention_mask)`는 BiLSTMCRF의 인수 순서다. TransformerCRF는 `(input_ids,attention_mask,labels)`라 같은 호출을 쓰면 두 텐서가 뒤바뀐다. 키워드 인수나 공통 인터페이스가 필요하며 모델 자체의 장치도 호출 전에 맞춰야 한다.

scheduler는 배치마다 인수 없이 step하므로 배치 단위 정책을 가정한다. epoch 단위나 검증 지표를 요구하는 스케줄러와 그대로 호환되는 범용 루프는 아니다. 출력 손실도 단순 배치 평균이어서 마지막 작은 배치가 다른 가중을 받을 수 있다.

`compute_entity_f1`은 정의되어 있지 않다. 태그 ID를 문자열·span으로 바꾸는 매핑, 특수·패딩 위치와 무효 태그 정책을 포함한 정확 span 평가가 별도로 필요하다. best_f1이0에서 시작하므로 개선이 없으면 파일을 저장하지 않고, 저장한 최고 모델을 끝에 복원하지도 않는다. 검증을 재사용한 최고 점수와 독립 시험 성능을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드에서 CRF NLL이 올바른 정규화 확률의 손실이 아닐 수 있는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"

    정답 점수는 전이 [next,prev], partition과 Viterbi는 [prev,next]를 쓴다. 비대칭 전이에서 같은 점수 함수를 사용하지 않아 log Z−정답 점수가 정상화된 −log 확률이 아니다. 모든 경로의 전이 관례를 통일해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
−100 패딩 태그와 중간에 False가 있는 mask를 그대로 사용할 수 있는가?

</div>

??? success "연습문제 2 풀이"

    gather·전이 인덱싱을 마스크보다 먼저 하므로 −100 태그는 맞지 않는다. 유효 ID를 채우고 오른쪽 연속 prefix 마스크를 사용해야 하며 첫 위치도 유효해야 한다. 빈 시퀀스는 별도 계약이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
apply_iob2_constraints의 유한 점수 대입은 영구적인 강제 제약인가?

</div>

??? success "연습문제 3 풀이"

    아니다. −10000은 유한값이고 학습 가능한 가중치라 갱신될 수 있다. 코드의 전이 방향도 partition과 반대다. 올바른 관례에서 학습·정답·디코딩에 지속적으로 같은 금지 경로를 적용해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
공통 학습 루프에 TransformerCRF를 넣을 때 인수와 서브워드 목표를 점검하라.

</div>

??? success "연습문제 4 풀이"

    TransformerCRF의 두 번째 인수는 attention_mask인데 루프는 labels를 보낸다. 키워드나 공통 인터페이스로 맞춘다. 특수 토큰·무시 서브워드의 −100 목표를 CRF 태그로 바로 쓰지 말고 유효 연속 열과 마스크를 구성해야 한다.



## 정리하며

CRF는 같은 시퀀스 점수로 정답·정규화·Viterbi를 계산해야 한다. 현재 구현의 전이 방향 불일치와 마스크·패딩·모델 인수 계약은 학습 결과의 의미에 직접 영향을 준다. 유효 BIO 제약과 신경 인코더의 문맥 처리를 구분하고 정확 span 평가를 연결한다.

**참고 문헌**

1. Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML*.

2. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.

3. Ma, X., & Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. *ACL*.

4. Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF Models for Sequence Tagging. *arXiv*.
