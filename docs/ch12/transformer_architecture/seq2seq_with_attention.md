# 어텐션을 결합한 GRU Seq2Seq

GRU 인코더·디코더에 Bahdanau 어텐션을 연결한 Seq2Seq 모델이다. 트랜스포머와 비교할 순환 모델이며, 셀프 어텐션으로 RNN을 대체한 구조가 아니다.

## 1. 코드

```python
"""
주의를 쓰는 수열 대 수열 모형
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- attention_mechanisms ---
"""
순환 신경망을 위한 주의 얼개
"""

# ========================================================================
# 메인
# ========================================================================

class BahdanauAttention(nn.Module):
    """바다나우(더하기) 주의"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        # query: [batch, hidden] - 디코더의 숨은 상태
        # keys: [batch, seq_len, hidden] - 인코더의 출력
        scores = self.V(torch.tanh(
            self.W1(query).unsqueeze(1) + self.W2(keys)
        ))  # [batch, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        return context, attention_weights

class LuongAttention(nn.Module):
    """루옹(곱하기) 주의"""
    def __init__(self, hidden_size, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, keys):
        if self.method == 'dot':
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        elif self.method == 'general':
            scores = torch.bmm(self.W(query).unsqueeze(1), keys.transpose(1, 2))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, keys).squeeze(1)
        return context, attention_weights


# ---------------------------------------------------------------------------
# 가린 소프트맥스 도구
# ---------------------------------------------------------------------------
# 많은 수열 과제에서 입력의 길이가 제각각이다. 주의 점수를 셈할 때
# 실제 수열 길이를 넘는 자리는 -inf로 가려서 소프트맥스가
# 그 자리에 확률 0을 주도록 해야 한다.
# 그러면 채움 토큰에 주의하지 않게 되며, 트랜스포머의 인코더 자기 주의와
# 교차 주의 모두에 꼭 필요하다.

def masked_softmax(X, valid_lens):
    """유효 길이를 넘는 자리를 가리고 소프트맥스를 한다.

    인수:
        X: 꼴이 (batch_size, num_queries, num_keys)인 3차원 텐서
        valid_lens: 1차원 텐서 (batch_size,) 또는 2차원 텐서 (batch_size, num_queries)
            성분마다 그 질의에 유효한 열쇠가 몇 개인지 알려 준다.
    반환값:
        X와 같은 꼴의 소프트맥스 출력이며 가린 자리는 0이다.
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        # 배치 원소마다 모든 질의에 같은 유효 길이를 쓴다
        valid_lens = valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    # 가림을 만든다: valid_len 이상인 자리에 -1e6을 주어 exp(-1e6) ≈ 0이 되게 한다
    X_flat = X.reshape(-1, shape[-1])
    maxlen = X_flat.size(1)
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_lens[:, None]
    X_flat[~mask] = -1e6

    return F.softmax(X_flat.reshape(shape), dim=-1)

# ========================================================================
# 메인
# ========================================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden, attn_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_size, hidden_size)
        self.decoder = Decoder(tgt_vocab, embed_size, hidden_size)
    
    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        outputs = []
        for t in range(tgt.size(1)):
            output, hidden, _ = self.decoder(tgt[:, t:t+1], hidden, encoder_outputs)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    pass
```

## 2. 논의

Encoder는 정수 토큰 $(B,S)$를 임베딩한 뒤 GRU로 처리해 $(B,S,H)$ 출력과 $(1,B,H)$ 마지막 상태를 반환한다. Decoder는 한 시점의 입력 $(B,1)$, 은닉 상태와 전체 인코더 출력을 받아 $(B,V_t)$ 로짓을 반환한다.

디코더는 이전 은닉 상태로 원문에 어텐션하고 문맥 벡터를 현재 토큰 임베딩과 이어 GRU에 넣는다. Seq2SeqWithAttention은 전달받은 tgt의 각 토큰을 순서대로 사용하므로 목표를 한 토큰 옮기는 일은 호출자가 해야 한다. 같은 tgt를 그대로 정답으로 주면 다음 토큰 학습이 아니다.

원문 패딩 마스크와 길이 처리가 없고 임베딩에도 padding_idx가 없다. 패딩이 GRU 상태와 어텐션에 영향을 줄 수 있다. 이 코드는 학습 루프, 자유 생성이나 EOS 종료를 구현하지 않으며 빈 목표 시퀀스에서는 출력 목록을 stack할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
원문 어휘 크기 V, 임베딩 E, 은닉 H인 Encoder의 파라메터 수를 구하라.

</div>

??? success "풀이"
    임베딩은 VE, 단층 GRU는 $3HE+3H^2+6H$이므로 합은 $VE+3HE+3H^2+6H$다. PyTorch GRU는 입력·은닉 경로의 바이어스를 각각 갖는다. 생성자에 기본값은 없어 수치 답을 위해 V·E·H를 지정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 어텐션에 드롭아웃을 추가할 위치와 해석을 설명하라.

</div>

??? success "풀이"
    BahdanauAttention 내부에서 소프트맥스 뒤, 키 벡터 가중합 앞에 적용한다. Decoder가 이미 받은 context에 드롭아웃을 적용하는 것은 웨이트 드롭아웃과 다르다. 학습 시 일부 연결을 무작위로 제외하며 웨이트 행 합이 항상 1이 되거나 분포가 더 고르게 된다고 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
목표 길이 T, 원문 길이 S인 이 모델의 어텐션 점수 크기와 순차 계산을 설명하라.

</div>

??? success "풀이"
    각 디코더 시점은 S개 원문 위치의 점수를 구한다. 전체 점수 수는 배치당 TS이며 계산은 대략 $O(TSH^2)$다. 이 구현은 W2(keys)를 매 시점 다시 계산하므로 단순 가중합의 O(TSH)보다 큰 투영 비용이 있다. 디코더 은닉 상태는 GRU로 연결되어 T방향 계산이 순차적이다.

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

이 예제는 어텐션을 결합한 순환 Seq2Seq다. 입력·목표 시프트, 한 시점 디코더 규약과 패딩 처리를 확인해야 하며, 모델 정의만으로 번역 학습·생성이 완성되지는 않는다.
