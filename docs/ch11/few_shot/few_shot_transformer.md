# 소수 예시 트랜스포머

트랜스포머 기반 소수 예시 학습. 자기 주의로 받침 집합과 물음 집합을 함께 다루는 오늘날의 접근법으로,

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

## 코드

```python
"""
트랜스포머 기반 소수 예시 학습

자기 주의로 받침 집합과 물음 집합을 함께 다루어,
모델이 보기 사이의 관계를 따져 볼 수 있게 하는 오늘날의 접근법.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========================================================================
# 메인
# ========================================================================


class PositionalEncoding(nn.Module):
    """
    묻힘에 자리 정보를 더한다.
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # 위치 인코딩 행렬을 만든다
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerFewShotClassifier(nn.Module):
    """
    트랜스포머 기반 소수 예시 가려내개.
    
    모델은 받침 보기와 물음 보기를 함께 다루어, 주의 장치가
    집합을 넘나들며 보기를 견주고 이어 보게 한다.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super(TransformerFewShotClassifier, self).__init__()
        
        # 입력 묻힘
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 분류 머리
        self.classifier = nn.Linear(d_model, 1)
    
    def forward(self, support, support_labels, query):
        """
        인수:
            support: (n_support, input_dim)
            support_labels: (n_support,)
            query: (n_query, input_dim)
        
        반환값:
            logits: (n_query, n_classes)
        """
        n_support = support.shape[0]
        n_query = query.shape[0]
        n_classes = support_labels.max().item() + 1
        
        # 받침과 물음을 합친다
        all_examples = torch.cat([support, query], dim=0)  # (n_support + n_query, input_dim)
        
        # d_model로 쏘아 넣는다
        embeddings = self.input_projection(all_examples)  # (n_support + n_query, d_model)
        
        # 자리 부호를 더한다
        embeddings = self.pos_encoder(embeddings.unsqueeze(0)).squeeze(0)
        
        # 트랜스포머를 씌운다
        transformed = self.transformer(embeddings.unsqueeze(0)).squeeze(0)
        
        # 다시 받침과 물음으로 쪼갠다
        support_transformed = transformed[:n_support]
        query_transformed = transformed[n_support:]
        
        # 물음마다 부류마다의 로짓을 셈한다
        logits = []
        for query_emb in query_transformed:
            class_logits = []
            for c in range(n_classes):
                # 이 부류의 받침 보기를 얻는다
                class_mask = (support_labels == c)
                class_support = support_transformed[class_mask]
                
                # 물음과 부류 원형 사이의 닮음을 셈한다
                query_expanded = query_emb.unsqueeze(0).expand(class_support.shape[0], -1)
                similarities = F.cosine_similarity(query_expanded, class_support, dim=1)
                class_logit = similarities.mean()
                class_logits.append(class_logit)
            
            logits.append(torch.stack(class_logits))
        
        return torch.stack(logits)


class SetTransformer(nn.Module):
    """
    소수 예시 학습을 위한 집합 기반 트랜스포머.
    
    유도 집합 주의를 써서 받침 집합과 물음 집합의
    순열에 흔들리지 않는 표현을 만든다.
    """
    def __init__(self, input_dim, d_model=128, num_heads=4, num_inds=32):
        super(SetTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 유도 집합 주의 블록
        self.inducing_points = nn.Parameter(torch.randn(num_inds, d_model))
        
        self.mab1 = MultiheadAttentionBlock(d_model, num_heads)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads)
        
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        인수:
            x: (batch_size, set_size, input_dim)
        
        반환값:
            output: (batch_size, d_model) - 집합 묻힘
        """
        # 입력을 쏘아 넣는다
        x = self.input_proj(x)  # (batch_size, set_size, d_model)
        
        # 유도 집합 주의
        batch_size = x.shape[0]
        inds = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        h = self.mab1(inds, x)  # (batch_size, num_inds, d_model)
        h = self.mab2(h, h)     # (batch_size, num_inds, d_model)
        
        # 유도점 위에서 모은다
        output = h.mean(dim=1)  # (batch_size, d_model)
        
        return self.output_proj(output)


class MultiheadAttentionBlock(nn.Module):
    """
    Set Transformer를 위한 여러 머리 주의 블록.
    """
    def __init__(self, d_model, num_heads):
        super(MultiheadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, query, key_value):
        """
        인수:
            query: (batch_size, query_len, d_model)
            key_value: (batch_size, kv_len, d_model)
        """
        # 어텐션
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.layer_norm(query + attn_out)
        
        # 순전파 신경망
        ffn_out = self.ffn(query)
        query = self.layer_norm2(query + ffn_out)
        
        return query


def train_transformer_fewshot(model, support, support_labels, query, query_labels, optimizer):
    """
    트랜스포머 소수 예시 모델의 학습 걸음.
    """
    model.train()
    optimizer.zero_grad()
    
    # 순전파
    logits = model(support, support_labels, query)
    
    # 손실을 계산한다
    loss = F.cross_entropy(logits, query_labels)
    
    # 역전파
    loss.backward()
    optimizer.step()
    
    # 정확도를 계산한다
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


# 사용 예
if __name__ == "__main__":
    # 모형 설정
    input_dim = 784  # 편 28x28 그림
    d_model = 128
    nhead = 4
    num_layers = 2
    
    # 모델 생성
    model = TransformerFewShotClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    # 5-갈래 5-예시 과제 보기
    n_way = 5
    k_shot = 5
    n_query = 15
    
    support = torch.randn(n_way * k_shot, input_dim)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, input_dim)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # 순전파
    logits = model(support, support_labels, query)
    print(f"Logits shape: {logits.shape}")  # (15, 5)
    
    # 학습
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_transformer_fewshot(
        model, support, support_labels, query, query_labels, optimizer
    )
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Set Transformer를 쓴 보기
    set_model = SetTransformer(input_dim=input_dim, d_model=128, num_heads=4)
    support_set = support.unsqueeze(0)  # 배치 차원을 더한다
    set_embedding = set_model(support_set)
    print(f"Set embedding shape: {set_embedding.shape}")  # (1, 128)```

## 논의

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 4개(`PositionalEncoding`, `TransformerFewShotClassifier`, `SetTransformer`, `MultiheadAttentionBlock`)를 정한다. 클래스마다 서로 다른 부품을 감싸 코드를 모듈 방식으로 만들고 넓히기 쉽게 한다. `forward` 메서드가 파이토치가 자동 미분에 쓰는 계산 그래프를 정한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `PositionalEncoding`에 든 학습 가능한 매개변수의 총 개수를 셈하라. 가중치와 편향을 모두 넣어 층별로 나누어 보여라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
`PositionalEncoding`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = PositionalEncoding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.
