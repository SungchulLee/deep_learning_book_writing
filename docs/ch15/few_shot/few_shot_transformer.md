# 퓨샷 트랜스포머

서포트와 쿼리의 피처를 하나의 시퀀스로 묶어 트랜스포머로 변환한 뒤 클래스별 유사도로 분류하는 예제다. 별도의 SetTransformer는 서포트 집합 전체를 하나의 벡터로 요약한다. 두 모델의 출력과 순서 의존성을 구분해 읽는다.

## 1. 코드

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
        
        # 순전파 뉴럴 넷트웍
        ffn_out = self.ffn(query)
        query = self.layer_norm2(query + ffn_out)
        
        return query


def train_transformer_fewshot(model, support, support_labels, query, query_labels, optimizer):
    """
    트랜스포머 소수 예시 모델의 학습 스텝.
    """
    model.train()
    optimizer.zero_grad()
    
    # 순전파
    logits = model(support, support_labels, query)
    
    # 로스를 계산한다
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
    
    # 5-클래스 5-예시 과제 보기
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
    print(f"Set embedding shape: {set_embedding.shape}")  # (1, 128)
```

**출력:**

```
Logits shape: torch.Size([15, 5])
Loss: 1.6069, Accuracy: 0.2667
Set embedding shape: torch.Size([1, 128])
```

## 2. 논의

입력 서포트 $(M,D)$와 쿼리 $(Q,D)$를 연결해 $(1,M+Q,d)$로 변환한다. 위치 표현과 트랜스포머를 적용한 뒤 각 쿼리와 클래스별 서포트의 코사인 유사도를 평균 내어 $(Q,N)$ 로짓을 만든다. classifier 선형 레이어는 선언되어 있지만 forward에서 쓰이지 않으므로 이 로스로 학습되지 않는다. 라벨은 0부터 N−1까지 연속적이고 모든 클래스에 서포트가 있어야 한다.

어텐션 마스크가 없으므로 한 쿼리의 표현은 다른 쿼리에도 의존한다. 이는 쿼리 집합을 함께 활용하는 전이적(transductive) 평가에 해당한다. 쿼리 라벨을 입력하지는 않지만 쿼리를 하나씩 처리하는 평가와 조건이 다르다. 위치 표현도 있으므로 서포트 순서를 바꾸면 결과가 달라질 수 있다. 클래스별로 정렬한 순서가 불필요한 단서가 되는지도 점검한다. 기본 위치 버퍼는 총 100개 표본까지만 지원한다.

SetTransformer는 유도점 32개가 입력 집합에 어텐션하고, 그 결과에 다시 셀프 어텐션을 적용한 뒤 평균을 낸다. 입력 (B,M,D)에 대해 (B,d)를 반환하며 클래스별 분류 로짓은 제공하지 않는다. 위치 표현이 없고 평가 모드에서 입력 순서를 바꾸어도 같은 집합 요약을 얻는 구조다. 원 논문의 모든 블록을 그대로 구현한 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
PositionalEncoding의 학습 파라메터와 버퍼를 구분하고 입력 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    사인·코사인 위치 표는 register_buffer로 등록되므로 학습 파라메터는 0개다. 모델 이동과 state_dict 저장에는 포함된다. 이 구현은 짝수 d_model을 가정하며 시퀀스 길이는 max_len 이하여야 한다. 트랜스포머에는 별도로 d_model이 nhead로 나누어떨어져야 하는 조건이 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 코드에 어텐션 드롭아웃이 있는지, 추가하려면 어디를 바꿔야 하는지 설명하라.

</div>

??? success "연습문제 2 풀이"
    TransformerFewShotClassifier의 TransformerEncoderLayer는 dropout=0.1로 생성되어 어텐션 드롭아웃을 포함한다. MultiheadAttentionBlock의 nn.MultiheadAttention은 기본값 0을 사용하므로 생성자에 dropout을 지정할 수 있다. 반환된 어텐션 웨이트에 나중에 드롭아웃을 적용해도 이미 계산된 attn_out에는 반영되지 않는다. 드롭아웃이 항상 더 균등한 어텐션을 만든다는 보장은 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
총 표본 수 L=M+Q일 때 트랜스포머와 유도점 기반 집합 요약의 어텐션 비용을 비교하라.

</div>

??? success "연습문제 3 풀이"
    일반 셀프 어텐션의 쌍별 계산은 O(L²d)다. 유도점 수를 I라 하면 이 SetTransformer의 두 어텐션은 O(IMd+I²d)다. 선형 투영과 FFN 비용은 별도다. I를 고정하면 입력 집합 길이에 대한 쌍별 계산은 선형으로 늘지만, 집합을 I개 표현으로 압축하는 설계가 과제에 적절한지는 평가해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

표본을 함께 처리하는 트랜스포머 분류기와 집합 요약기는 서로 다른 출력을 만든다. 위치·쿼리 문맥에 따른 순서 및 평가 조건, 미사용 분류 레이어와 최대 길이를 확인한다.
