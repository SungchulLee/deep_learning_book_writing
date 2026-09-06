# 차례 이름표 붙이기를 위한 조건부 무작위 마당
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 선형 사슬 CRF의 수학적 세우기를 이끌어 낸다
- CRF와 신경 차례 모델의 관계를 이해한다
- 효율적인 앞먹임 알고리즘으로 PyTorch에 CRF 층을 짠다
- 음의 로그 가능도 손실로 CRF 모델을 익힌다
- 가장 좋은 차례 어림을 위해 비터비 풀기를 한다
- CRF 층을 두 방향 LSTM 및 변환기 얼개와 아우른다

## 들어가며

Conditional Random Fields (CRFs) are discriminative probabilistic models for sequence labeling that model the conditional probability $P(\mathbf{Y}|\mathbf{X})$ directly. Unlike independent token classifiers, CRFs capture dependencies between adjacent labels, making them particularly effective for NER where tag transitions follow specific patterns (e.g., I-PER can only follow B-PER or I-PER).

## 수학적 바탕

### 문제의 얼개

다음이 주어졌다고 하자.

- Input sequence: $\mathbf{X} = (x_1, x_2, \ldots, x_n)$
- Output labels: $\mathbf{Y} = (y_1, y_2, \ldots, y_n)$
- Label set: $\mathcal{L} = \{l_1, l_2, \ldots, l_k\}$

### 선형 사슬 CRF 모델

조건부 확률을 다음과 같이 정한다:

$$
P(\mathbf{Y}|\mathbf{X}) = \frac{1}{Z(\mathbf{X})} \exp\left(\sum_{i=1}^{n} s(y_i, \mathbf{X}, i) + \sum_{i=1}^{n} t(y_{i-1}, y_i)\right)
$$

여기서:

- $s(y_i, \mathbf{X}, i)$: **Emission score** - how likely is label $y_i$ at position $i$ given input
- $t(y_{i-1}, y_i)$: **Transition score** - how likely is transitioning from $y_{i-1}$ to $y_i$
- $Z(\mathbf{X})$: **Partition function** - normalization constant

### 점수 함수

차례의 전체 점수는 다음과 같다:

$$
\text{Score}(\mathbf{X}, \mathbf{Y}) = \sum_{i=1}^{n} E_{y_i, i} + \sum_{i=1}^{n} T_{y_{i-1}, y_i}
$$

여기서:

- $\mathbf{E} \in \mathbb{R}^{n \times k}$: Emission matrix from neural encoder
- $\mathbf{T} \in \mathbb{R}^{k \times k}$: Transition matrix (learnable parameters)

### 나눔 함수

나눔 함수는 가능한 모든 이름표 차례에 걸쳐 더한다:

$$
Z(\mathbf{X}) = \sum_{\mathbf{Y}' \in \mathcal{L}^n} \exp\left(\text{Score}(\mathbf{X}, \mathbf{Y}')\right)
$$

Direct computation is $O(k^n)$ - intractable. We use the **forward algorithm**.

### 앞먹임 알고리즘

앞먹임 변수를 정한다:

$$
\alpha_i(y) = \sum_{\mathbf{Y}_{1:i-1}} \exp\left(\sum_{j=1}^{i} E_{y_j, j} + \sum_{j=2}^{i} T_{y_{j-1}, y_j}\right)
$$

이는 자리 $i$에서 이름표 $y$으로 끝나는 모든 부분 차례의 점수 합을 나타낸다.

**되돌이 관계식**:

$$
\alpha_1(y) = \exp(E_{y,1} + T_{\text{START}, y})
$$

$$
\alpha_i(y) = \sum_{y' \in \mathcal{L}} \alpha_{i-1}(y') \cdot \exp(T_{y', y} + E_{y, i})
$$

**나눔 함수**:

$$
Z(\mathbf{X}) = \sum_{y \in \mathcal{L}} \alpha_n(y) \cdot \exp(T_{y, \text{END}})
$$

**Complexity**: $O(n \cdot k^2)$ - linear in sequence length, quadratic in label count.

### 로그 공간에서 셈하기

수치를 든든히 하려고 로그 자리에서 셈한다:

$$
\log \alpha_i(y) = \text{logsumexp}_{y'}\left(\log \alpha_{i-1}(y') + T_{y', y}\right) + E_{y, i}
$$

여기서:

$$
\text{logsumexp}(a_1, \ldots, a_m) = \log\left(\sum_{j=1}^{m} \exp(a_j)\right)
$$

## 손실 함수

### 음의 로그 가능도

익히기 목표는 음의 로그 가능도를 가장 작게 하는 것이다:

$$
\mathcal{L} = -\log P(\mathbf{Y}^*|\mathbf{X}) = -\text{Score}(\mathbf{X}, \mathbf{Y}^*) + \log Z(\mathbf{X})
$$

Where $\mathbf{Y}^*$ is the ground truth sequence.

### 기울기 셈하기

내보냄 점수에 대한 기울기:

$$
\frac{\partial \mathcal{L}}{\partial E_{y,i}} = P(y_i = y | \mathbf{X}) - \mathbb{1}[y_i^* = y]
$$

넘어가기 점수에 대한 기울기:

$$
\frac{\partial \mathcal{L}}{\partial T_{y', y}} = \sum_{i=2}^{n} P(y_{i-1} = y', y_i = y | \mathbf{X}) - \sum_{i=2}^{n} \mathbb{1}[y_{i-1}^* = y', y_i^* = y]
$$

## 비터비 풀기

미룸 때에는 가장 그럴듯한 차례를 찾는다:

$$
\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y}|\mathbf{X}) = \arg\max_{\mathbf{Y}} \text{Score}(\mathbf{X}, \mathbf{Y})
$$

### 비터비 알고리즘

다음과 같이 정한다:

$$
v_i(y) = \max_{\mathbf{Y}_{1:i-1}} \text{Score}(\mathbf{X}_{1:i}, \mathbf{Y}_{1:i-1}, y)
$$

**되돌이**:

$$
v_1(y) = E_{y,1} + T_{\text{START}, y}
$$

$$
v_i(y) = \max_{y' \in \mathcal{L}}\left(v_{i-1}(y') + T_{y', y}\right) + E_{y, i}
$$

**뒤 가리개**:

$$
b_i(y) = \arg\max_{y' \in \mathcal{L}}\left(v_{i-1}(y') + T_{y', y}\right)
$$

**되짚기**:

$$
y_n^* = \arg\max_{y}\left(v_n(y) + T_{y, \text{END}}\right)
$$

$$
y_i^* = b_{i+1}(y_{i+1}^*) \quad \text{for } i = n-1, \ldots, 1
$$

## PyTorch 구현

### CRF 층 단원

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class CRF(nn.Module):
    """
    차례 이름표 붙이기를 위한 조건부 무작위 마당 층.
    
    여기 짠 것이 받치는 것:
    - 덧대기를 곁들인 묶음 셈하기
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
            묶음 안 차례마다의 점수 (batch,)
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
            묶음 항목마다의 이름표 차례 목록
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

### 두 방향 LSTM-CRF 모델

```python
class BiLSTMCRF(nn.Module):
    """
    차례 이름표 붙이기를 위한 두 방향 LSTM-CRF 모델.
    
    구조:
        Embedding → BiLSTM → Linear → CRF
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

### 변환기-CRF 모델

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

## 넘어가기 제약

### IOB2 제약 쓰기

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

## 익히기 되풀이

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

## 계산에 대한 고려

### 시간 복잡도

| 연산 | 복잡도 |
|-----------|------------|
| Forward (partition function) | $O(n \cdot k^2)$ |
| Viterbi decoding | $O(n \cdot k^2)$ |
| Backward pass | $O(n \cdot k^2)$ |

여기서 $n$은 차례의 길이, $k$은 이름표의 개수이다.

### 기억 공간에서 헤아릴 점

- Transition matrix: $O(k^2)$ parameters
- Forward variables: $O(n \cdot k)$ per sequence
- Backpointers for Viterbi: $O(n \cdot k)$

### 묶음 셈하기

위에 짠 것은 효율을 위해 묶음 셈하기를 받쳐 준다:

- 묶음 차원에 걸친 벡터 연산
- 길이가 들쭉날쭉한 차례를 위한 가림 셈하기
- GPU로 빨라진 행렬 연산

## 요약

조건부 무작위 마당은 다음으로 신경 차례 이름표 붙이개를 낫게 한다:

1. **넘어가기 나타내기**: 배울 수 있는 넘어가기 점수로 이름표 사이의 얽힘을 담아낸다
2. **전체 고르게 맞추기**: 모든 차례에 걸쳐 제대로 된 확률 분포를 보장한다
3. **짜임 있는 어림**: 비터비 알고리즘으로 가장 좋은 차례를 찾는다
4. **제약 강제하기**: 가림으로 올바르지 않은 이름표 넘어가기를 막는다

힘센 신경 부호기(두 방향 LSTM, 변환기)와 CRF 풀개를 아우른 것은, 특히 이름표 사이 얽힘이 중요할 때 이름 알아보기의 센 방식으로 남아 있다.

## 참고 문헌

1. Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML*.

2. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.

3. Ma, X., & Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. *ACL*.

4. Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF Models for Sequence Tagging. *arXiv*.

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
    Independent softmax treats each position's label independently, ignoring transition constraints. For example, it might predict I-PER following O, which is invalid in BIO tagging. A CRF layer models the joint probability of the entire label sequence, learning a transition matrix $A_{ij}$ that captures which tag transitions are valid. The CRF score for a sequence is $s(x, y) = \sum_t (E_{y_t, t} + A_{y_t, y_{t+1}})$, where $E$ is the emission score from BiLSTM. This ensures globally consistent predictions via Viterbi decoding.

---

**연습문제 3.**
것 수준에서의 이름 알아보기 값매김에 쓰는 정밀도, 재현율, F1 점수를 설명하여라. 것 수준 값매김이 토막 수준보다 왜 더 빡빡한가?

??? success "연습문제 3 풀이"
    **Entity-level** evaluation requires both the entity boundary and type to be exactly correct. Precision = (correctly predicted entities) / (total predicted entities). Recall = (correctly predicted entities) / (total gold entities). $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$. This is stricter than token-level because: a prediction "New York" when the gold entity is "New York City" gets partial credit at the token level (2/3 tokens correct) but zero credit at the entity level (boundary mismatch). Entity-level metrics better reflect real-world utility.

---

**연습문제 4.**
두 방향 LSTM은 양쪽 방향의 맥락을 어떻게 담아내는가? 자리 $t$에서의 숨은 상태 셈을 적어라.

??? success "연습문제 4 풀이"
    A BiLSTM consists of a forward LSTM processing $x_1, \ldots, x_T$ and a backward LSTM processing $x_T, \ldots, x_1$:

    $$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

    The final representation at position $t$ is the concatenation $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$, which captures both left context (through $\overrightarrow{h}_t$) and right context (through $\overleftarrow{h}_t$). This is critical for NER since entity recognition often depends on surrounding words in both directions.
