# 번역 학습

앞의 GRU Seq2Seq 모델, 교차 엔트로피 로스와 Adam 옵티마이저를 준비하는 짧은 예제다. 함수 이름과 달리 데이터 로딩, 로스 계산과 웨이트 갱신을 수행하지 않는다.

## 1. 코드

```python
"""
주의를 쓰는 Seq2Seq 학습
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


# --- seq2seq_with_attention ---
"""
주의를 쓰는 수열 대 수열 모형
"""

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

# ========================================================================
# 메인
# ========================================================================

# 간단한 학습 보기
def train_simple_example():
    # 초파라메터
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    embed_size = 256
    hidden_size = 512
    batch_size = 32
    
    model = Seq2SeqWithAttention(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Training script ready!")
    print("\nTo train: Create your own dataset and training loop")

if __name__ == '__main__':
    train_simple_example()
```

**출력:**

```
Model created with 4702697 parameters
Training script ready!

To train: Create your own dataset and training loop
```

## 2. 논의

src_vocab_size와 tgt_vocab_size는 각각 1000, 임베딩은 256, 은닉은 512다. 모델의 4,702,697개 파라메터는 임베딩·GRU·어텐션·출력 레이어를 모두 합한 수이며, 보존된 출력도 준비 완료 메시지다. batch_size=32는 선언만 되어 사용되지 않는다.

실제 번역 학습에는 원문·목표문 쌍, 어휘·특수 토큰과 배치 패딩이 필요하다. 목표 입력은 BOS부터 시작하고 정답은 다음 토큰부터 EOS까지 맞춘다. 반환 로짓 $(B,T,V)$와 정수 정답 $(B,T)$를 로스 함수가 기대하는 축으로 배치해야 한다.

현재 CrossEntropyLoss의 기본값은 PAD 토큰을 별도로 제외하지 않는다. 패딩 마스크와 ignore_index를 맞추고, 역전파·옵티마이저 갱신·검증과 생성 평가를 추가해야 한다. 이 페이지에 실제 학습 로스, 번역 정확도나 BLEU 결과는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
제시된 모델의 파라메터 4,702,697개를 구성 요소별로 나누어라.

</div>

??? success "풀이"
    인코더 임베딩 256,000개와 GRU 1,182,720개, 디코더 임베딩 256,000개와 GRU 1,969,152개, 어텐션 525,825개, 출력 레이어 513,000개다. 합계는 4,702,697개이며 로스 함수와 Adam 생성은 모델의 학습 파라메터를 늘리지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
준비 코드만으로 학습이 이루어지지 않는 이유와 필요한 단계를 설명하라.

</div>

??? success "풀이"
    모델 호출·로스 계산·zero_grad·backward·step이 모두 없다. 데이터를 구성하고 목표 시프트와 패딩 처리를 정한 뒤 그 단계들을 반복해야 한다. 평가 데이터는 학습·설정 선택에 쓴 분할과 역할을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 Seq2Seq의 로스에서 소프트맥스와 패딩을 어떻게 처리해야 하는가?

</div>

??? success "풀이"
    CrossEntropyLoss는 로짓을 받으므로 모델 출력에 소프트맥스를 먼저 적용하지 않는다. 어휘 축을 맞추고 PAD 정답을 ignore_index로 제외한다. 로스 평균을 보고할 때는 유효 목표 토큰 수를 기준으로 집계한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_train translation():
        model = Train Translation(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

이 페이지는 번역 학습의 구성 요소를 생성하는 준비 단계다. 출력의 파라메터 수를 학습 결과로 오해하지 말고, 목표 정렬·패딩·실제 갱신과 평가 과정을 별도로 구성해야 한다.
