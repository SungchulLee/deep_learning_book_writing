# 뜻 묻힘 방법
## 개요

뜻 묻힘 방법은 손으로 정한 속성을 글 뭉치에서 배운 이어진 벡터 표현으로 갈아 끼운다. 부류 이름은 닮은 개념끼리 가까이 놓이는 뜻 공간에 묻히고, 그래서 말에 담긴 앎을 거쳐 영 예시 옮김이 가능해진다.

## 속성에서 묻힘으로

### 속성의 한계

속성 기반 ZSL에는 다음이 필요하다.

- 알맞은 속성을 정할 전문가의 앎
- 부류마다의 값비싼 표시
- 이진이나 낱낱의 표현은 미묘함을 잃을 수 있다
- 속성이 가름에 쓸 정보를 다 담지 못할 수 있다

### 묻힘의 이점

뜻 묻힘은 다음을 준다.

- **손 표시 필요 없음**: 큰 글 뭉치로 미리 학습한다
- **풍부한 표현**: 이어진 벡터가 미묘한 닮음을 담는다
- **조합의 뜻**: 개념 사이의 관계가 지켜진다
- **규모 키우기**: 이름이 있는 부류라면 무엇이든 묻을 수 있다

## 낱말 묻힘 방법

### Word2Vec

Word2Vec은 둘레 낱말을 맞히며 묻힘을 배운다.

**Skip-gram**: 대상 낱말에서 둘레를 맞힌다

$$P(w_{context} | w_{target}) = \frac{\exp(\mathbf{v}_{context}^\top \mathbf{v}_{target})}{\sum_w \exp(\mathbf{v}_w^\top \mathbf{v}_{target})}$$

**CBOW**: 둘레에서 대상을 맞힌다

$$P(w_{target} | w_{context_1}, \ldots, w_{context_k})$$

**성질**:

- 뜻의 관계를 담는다: 왕 - 남자 + 여자 ≈ 여왕
- 흔한 차원: 100~300
- 수십억 낱말로 미리 학습한다

### GloVe(전역 벡터)

GloVe은 낱말이 함께 나오는 통계를 최적화한다.

$$J = \sum_{i,j} f(X_{ij}) (\mathbf{v}_i^\top \mathbf{v}_j + b_i + b_j - \log X_{ij})^2$$

여기서 $X_{ij}$은 함께 나온 횟수이고 $f$은 무게 주는 함수이다.

### FastText

FastText은 Word2Vec에 낱말 아래 정보를 더해 넓힌다.

- 낱말을 글자 n-그램의 자루로 나타낸다
- 어휘 밖 낱말을 다룬다
- 형태 정보를 담는다

$$\mathbf{v}_{word} = \sum_{g \in \text{ngrams}(word)} \mathbf{v}_g$$

### 견줌

| 방법 | 핵심 특징 | 어휘 밖 다루기 | 속도 |
|--------|-------------|--------------|-------|
| Word2Vec | 둘레 맞히기 | 없음 | 빠름 |
| GloVe | 함께 나옴 통계 | 없음 | 빠름 |
| FastText | 낱말 아래 묻힘 | 있음 | 보통 |
| BERT | 맥락에 따르고 양방향 | 있음 | 느림 |

## ZSL에 묻힘 쓰기

### 부류 이름 묻기

이름이 "zebra"인 부류라면 다음과 같다.

```python
import gensim.downloader as api

# 미리 학습된 묻힘을 불러온다
word_vectors = api.load("glove-wiki-gigaword-300")

# 부류 묻힘을 얻는다
zebra_embedding = word_vectors["zebra"]  # 꼴: (300,)
```

### 여러 낱말로 된 부류 이름

"polar bear" 같은 겹낱말 이름이라면 다음과 같다.

```python
def get_class_embedding(class_name, word_vectors):
    """
    여러 낱말로 될 수 있는 부류 이름의 묻힘을 얻는다.
    """
    words = class_name.lower().split()
    embeddings = []
    
    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
    
    if not embeddings:
        raise ValueError(f"No words in '{class_name}' found in vocabulary")
    
    # 낱말 묻힘을 평균 낸다
    return np.mean(embeddings, axis=0)
```

### 뜻의 짜임

묻힘은 갈래의 관계를 자연스럽게 담는다.

```python
def visualize_semantic_structure(class_names, word_vectors):
    """
    t-SNE로 부류 묻힘을 그려 본다.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # 묻힘을 얻는다
    embeddings = np.array([get_class_embedding(c, word_vectors) 
                          for c in class_names])
    
    # 차원을 줄인다
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 그림
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    for i, name in enumerate(class_names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title("Semantic Embedding Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
```

## 시각-뜻 어울림 배우기

### 정식화

시각 특징과 뜻 묻힘 사이의 어울림 함수를 배운다.

$$F: \mathcal{V} \times \mathcal{S} \rightarrow \mathbb{R}$$

예측 규칙은 다음과 같다.

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} F(\phi(\mathbf{x}), \mathbf{s}_c)$$

### 묻힘 공간 접근법

**시각을 뜻으로 쏘아 넣기**:

$$f_v(\mathbf{v}) \in \mathcal{S}$$

학습: $\min_\theta \sum_{(\mathbf{x}, y)} \|f_\theta(\phi(\mathbf{x})) - \mathbf{s}_y\|^2$

**뜻을 시각으로 쏘아 넣기**:

$$f_s(\mathbf{s}) \in \mathcal{V}$$

**함께 쓰는 묻힘 공간**:

$$f_v(\mathbf{v}), f_s(\mathbf{s}) \in \mathcal{E}$$

### 순위 손실

회귀 대신 순위 매기기를 써서 맞는 부류가 더 높은 점수를 받게 한다.

$$\mathcal{L} = \sum_{(\mathbf{x}, y)} \sum_{c \neq y} \max(0, \Delta + F(\phi(\mathbf{x}), \mathbf{s}_c) - F(\phi(\mathbf{x}), \mathbf{s}_y))$$

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSemanticEmbedding(nn.Module):
    """
    ZSL을 위한 시각-뜻 묻힘 모델.
    
    시각 특징을 뜻 묻힘 공간으로 쏘아 넣는다.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 256):
        super().__init__()
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
        )
        
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, embedding_dim),
        )
    
    def encode_visual(self, v):
        """시각 특징을 부호로 바꾼다."""
        return F.normalize(self.visual_encoder(v), dim=1)
    
    def encode_semantic(self, s):
        """뜻 묻힘을 부호로 바꾼다."""
        return F.normalize(self.semantic_encoder(s), dim=1)
    
    def compatibility(self, v, s):
        """
        어울림 점수를 셈한다(코사인 닮음).
        """
        v_emb = self.encode_visual(v)
        s_emb = self.encode_semantic(s)
        return torch.sum(v_emb * s_emb, dim=1)
    
    def forward(self, v, s_positive, s_negative):
        """
        순위 손실 셈을 위한 앞먹임.
        
        인수:
            v: 시각 특징 (batch_size, visual_dim)
            s_positive: 맞는 부류의 묻힘 (batch_size, semantic_dim)
            s_negative: 틀린 부류의 묻힘 (batch_size, semantic_dim)
        
        반환값:
            양과 음의 어울림 점수
        """
        v_emb = self.encode_visual(v)
        s_pos_emb = self.encode_semantic(s_positive)
        s_neg_emb = self.encode_semantic(s_negative)
        
        pos_score = torch.sum(v_emb * s_pos_emb, dim=1)
        neg_score = torch.sum(v_emb * s_neg_emb, dim=1)
        
        return pos_score, neg_score

class RankingLoss(nn.Module):
    """
    어울림 학습을 위한 여백 기반 순위 손실.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        """
        순위 손실을 셈한다.
        
        손실 = max(0, margin + neg_score - pos_score)
        """
        loss = torch.clamp(self.margin + neg_score - pos_score, min=0)
        return loss.mean()

def train_vse_model(model, dataloader, class_embeddings, seen_classes, 
                    epochs=50, lr=0.001, margin=0.2):
    """
    시각-뜻 묻힘 모델을 익힌다.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = RankingLoss(margin=margin)
    
    # 부류 묻힘 텐서를 미리 셈해 둔다
    class_emb_tensor = torch.stack([
        torch.tensor(class_embeddings[c], dtype=torch.float32)
        for c in seen_classes
    ])
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_v, batch_y in dataloader:
            # 양의 묻힘을 얻는다
            s_positive = torch.stack([
                torch.tensor(class_embeddings[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            # 음의 묻힘을 뽑는다(어려운 음의 보기 캐기)
            batch_size = len(batch_y)
            neg_indices = torch.randint(0, len(seen_classes), (batch_size,))
            # 음의 보기가 양의 보기와 다르게 한다
            for i, y in enumerate(batch_y):
                while seen_classes[neg_indices[i]] == y:
                    neg_indices[i] = torch.randint(0, len(seen_classes), (1,)).item()
            
            s_negative = class_emb_tensor[neg_indices]
            
            # 순전파
            pos_score, neg_score = model(batch_v, s_positive, s_negative)
            loss = criterion(pos_score, neg_score)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
```

## ConSE: 뜻 묻힘의 볼록 결합

### 개념

ConSE(뜻 묻힘의 볼록 결합)은 본 부류의 확률로 무게 준 평균 뜻 표현을 만든 다음 가장 가까운 못 본 부류를 찾는다.

### 수식으로 나타내기

**1단계: 본 부류의 확률**

본 부류로 가려내개를 익힌다.

$$P(y^s | \mathbf{x}) = \text{softmax}(W^\top \phi(\mathbf{x}))$$

**2단계: 쏘아 넣은 묻힘**

본 부류 묻힘의 무게 준 평균을 셈한다.

$$\hat{\mathbf{z}} = \sum_{c \in \mathcal{Y}^s} P(c | \mathbf{x}) \cdot \mathbf{s}_c$$

**3단계: 최근접 이웃**

가장 가까운 못 본 부류를 찾는다.

$$\hat{y} = \arg\min_{c \in \mathcal{Y}^u} \|\hat{\mathbf{z}} - \mathbf{s}_c\|$$

### 위 T개 ConSE

효율과 튼튼함을 위해 확률이 가장 높은 본 부류 T개만 쓴다.

$$\hat{\mathbf{z}} = \sum_{c \in \text{Top}_T(\mathcal{Y}^s | \mathbf{x})} \frac{P(c | \mathbf{x})}{\sum_{c' \in \text{Top}_T} P(c' | \mathbf{x})} \cdot \mathbf{s}_c$$

### PyTorch 구현

```python
class ConSE(nn.Module):
    """
    뜻 묻힘의 볼록 결합(ConSE).
    
    본 부류의 확률로 뜻 공간에 쏘아 넣는다.
    """
    
    def __init__(self, visual_dim: int, n_seen_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_seen_classes)
        )
    
    def forward(self, x):
        """본 부류의 로짓을 얻는다."""
        return self.classifier(x)
    
    def predict_unseen(self, x, seen_classes, unseen_classes, 
                       class_embeddings, top_t=None):
        """
        ConSE으로 못 본 부류를 맞힌다.
        
        인수:
            x: 시각 특징 (batch_size, visual_dim)
            seen_classes: 본 부류 이름 목록
            unseen_classes: 못 본 부류 이름 목록
            class_embeddings: 부류 이름을 묻힘으로 옮기는 사전
            top_t: 쓸 위 부류의 개수(None이면 모두)
        
        반환값:
            맞힌 못 본 부류의 첨자
        """
        self.eval()
        
        with torch.no_grad():
            # 본 부류의 확률을 얻는다
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)  # (batch, n_seen)
            
            # 본 부류의 묻힘을 얻는다
            seen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in seen_classes
            ])  # (n_seen, emb_dim)
            
            if top_t is not None:
                # 위 T개 부류만 쓴다
                top_probs, top_indices = torch.topk(probs, k=top_t, dim=1)
                # 다시 고른다
                top_probs = top_probs / top_probs.sum(dim=1, keepdim=True)
                
                # 무게 준 묻힘을 셈한다
                projected = torch.zeros(x.size(0), seen_embs.size(1))
                for i in range(x.size(0)):
                    for j in range(top_t):
                        projected[i] += top_probs[i, j] * seen_embs[top_indices[i, j]]
            else:
                # 모든 부류를 쓴다
                projected = probs @ seen_embs  # (배치, emb_dim)
            
            # 못 본 부류의 묻힘을 얻는다
            unseen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in unseen_classes
            ])  # (n_unseen, emb_dim)
            
            # 가장 가까운 못 본 부류를 찾는다(코사인 닮음)
            projected_norm = F.normalize(projected, dim=1)
            unseen_norm = F.normalize(unseen_embs, dim=1)
            
            similarities = projected_norm @ unseen_norm.T  # (batch, n_unseen)
            predictions = torch.argmax(similarities, dim=1)
            
            return predictions

def train_conse_classifier(model, dataloader, seen_classes, epochs=50, lr=0.001):
    """
    ConSE을 위한 본 부류 가려내개를 익힌다.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            labels = torch.tensor([class_to_idx[y] for y in batch_y])
            
            logits = model(batch_x)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/len(dataloader):.4f}, "
                  f"Acc: {100*correct/total:.2f}%")
```

## 견줌: 어울림 배우기와 ConSE

### 방법의 차이

| 갈래 | 어울림 배우기 | ConSE |
|--------|----------------------|-------|
| **학습** | 끝에서 끝까지 시각-뜻 | 본 부류 가려내개 |
| **쏘아 넣기** | 배운 묻힘 공간 | 무게 준 평균 |
| **두루 쓰임** | 어떤 어울림 함수든 | 선형 결합 |
| **풀이 가능성** | 낮음 | 높음(본 부류 무게를 거쳐) |

### 무엇을 언제 쓸 것인가

**어울림 배우기**:

- 큰 학습 데이터셋
- 끝에서 끝까지 익히는 쪽이 낫다
- 복잡한 시각-뜻 관계

**ConSE**:

- 학습 데이터가 모자랄 때
- 미리 학습된 가려내개를 쓸 수 있을 때
- 풀이할 수 있는 예측이 필요할 때

### 성능 견줌

```python
def compare_methods(X_train, y_train, X_test, y_test,
                   seen_classes, unseen_classes, class_embeddings):
    """
    어울림 배우기와 ConSE을 견준다.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # 데이터 로더 생성
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(range(len(y_train)))
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    visual_dim = X_train.shape[1]
    semantic_dim = len(class_embeddings[seen_classes[0]])
    
    # 방법 1: 어울림 배우기
    print("Training Compatibility Model...")
    compat_model = VisualSemanticEmbedding(visual_dim, semantic_dim)
    train_vse_model(compat_model, train_loader, class_embeddings, 
                    seen_classes, epochs=50)
    
    # 방법 2: ConSE
    print("\nTraining ConSE Model...")
    conse_model = ConSE(visual_dim, len(seen_classes))
    train_conse_classifier(conse_model, train_loader, seen_classes, epochs=50)
    
    # 평가한다
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # 어울림 예측
    compat_model.eval()
    unseen_embs = torch.tensor(
        [class_embeddings[c] for c in unseen_classes],
        dtype=torch.float32
    )
    
    with torch.no_grad():
        v_emb = compat_model.encode_visual(X_test_tensor)
        s_emb = compat_model.encode_semantic(unseen_embs)
        scores = v_emb @ s_emb.T
        compat_preds = torch.argmax(scores, dim=1).numpy()
    
    # ConSE 예측
    conse_preds = conse_model.predict_unseen(
        X_test_tensor, seen_classes, unseen_classes, class_embeddings
    ).numpy()
    
    # 정확도를 셈한다
    compat_acc = (compat_preds == y_test).mean()
    conse_acc = (conse_preds == y_test).mean()
    
    print(f"\nResults:")
    print(f"Compatibility Learning: {compat_acc*100:.2f}%")
    print(f"ConSE: {conse_acc*100:.2f}%")
    print(f"Random Baseline: {100/len(unseen_classes):.2f}%")
```

## 여러 낱말과 겹낱말 부류 이름 다루기

### 어려움

실제 부류 가운데 겹낱말 이름이 많다.

- "polar bear"
- "fire truck"
- "German shepherd"

### 전략

**1. 낱말 평균 내기**:

$$\mathbf{s}_c = \frac{1}{|words_c|} \sum_{w \in words_c} \mathbf{v}_w$$

**2. 무게 준 평균 내기**(TF-IDF 무게):

$$\mathbf{s}_c = \sum_{w \in words_c} \text{tfidf}(w) \cdot \mathbf{v}_w$$

**3. 어구 묻힘**:
FastText 같은 어구를 아는 모델을 쓰거나 어구 묻힘을 따로 익힌다.

**4. BERT·GPT 묻힘**:
부류 이름 전체에 대한 맥락 묻힘을 얻는다.

```python
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embedding(class_name, tokenizer, model):
    """
    부류 이름의 BERT 묻힘을 얻는다.
    """
    # 토큰으로 나누기
    inputs = tokenizer(class_name, return_tensors='pt', 
                       padding=True, truncation=True)
    
    # 묻힘을 얻는다
    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] 토큰 묻힘을 쓴다
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embedding.numpy()
```

## 요약

뜻 묻힘 방법은 속성 기반 ZSL에 대한 힘 있는 대안을 준다.

1. **미리 학습된 묻힘**(Word2Vec, GloVe, FastText)은 손 표시 없이도 풍부한 뜻의 관계를 담는다
2. **어울림 배우기**는 시각 공간과 뜻 공간을 맞추도록 끝에서 끝까지 익힌다
3. **ConSE**은 본 부류의 확률을 써서 풀이할 수 있는 영 예시 예측을 한다
4. **오늘날의 접근법**은 더 나은 뜻 표현을 위해 BERT·GPT의 맥락 묻힘을 끌어 쓴다

핵심 이점은 규모를 키우기 쉽다는 것이다. 이름이 있는 부류라면 무엇이든 곧바로 묻을 수 있어 표시하는 수고를 더하지 않고도 영 예시 알아보기를 할 수 있다.

## 연습문제

**연습문제 1.**
영 예시 학습을 정의하고 소수 예시 학습과 어떻게 다른지 설명하라.

??? success "연습문제 1 풀이"
    영 예시 학습은 학습 중에 한 번도 보지 못한 부류의 사례를, 본 부류와 못 본 부류를 잇는 딸린 정보(속성, 글 설명, 낱말 묻힘)를 써서 가려낸다. 소수 예시 학습은 이름표 붙은 보기를 몇 개 쓴다. 영 예시 학습은 대상 부류의 보기가 아예 없어도 되며 오로지 뜻 표현을 거친 앎의 옮김에 기댄다.

---

**연습문제 2.**
영 예시 학습의 중심 쏠림 문제와 그 다루는 법을 설명하라.

??? success "연습문제 2 풀이"
    차원이 높은 공간에서는 어떤 점('중심점')이 참 부류와 상관없이 다른 많은 점의 최근접 이웃이 된다. 그래서 최근접 이웃 기반 영 예시 분류에서 어떤 부류가 너무 자주 예측된다. 해법: 묻힘을 고르거나, 눈금 맞춘 점수 매기기를 쓰거나, 전용 거리 재기를 배운다.

---

**연습문제 3.**
영 예시 학습의 속성 기반 접근법과 묻힘 기반 접근법을 견주어라.

??? success "연습문제 3 풀이"
    속성 기반: 부류마다 이진이나 이어진 속성 벡터로 그려진다(이를테면 '줄무늬가 있다', '털이 있다'). 모델은 속성을 맞히는 법을 배운 다음 부류 원형과 맞춘다. 묻힘 기반: 시각 특징과 부류 설명(이를테면 부류 이름의 word2vec)을 함께 쓰는 공간으로 옮긴다. 묻힘 방법이 규모를 키우기 쉽고 손 표시가 덜 든다.

---

**연습문제 4.**
일반화된 영 예시 학습(GZSL)의 치우침 문제란 무엇인가?

??? success "연습문제 4 풀이"
    GZSL에서는 시험 때 본 부류와 못 본 부류가 함께 나온다. 모델은 본 부류로 익혔으므로 그쪽으로 치우친다. 해법: 눈금 맞춘 쌓기(본 부류 점수에서 치우침을 빼기), 본 부류와 못 본 부류를 가르는 분포 밖 알아채기, 또는 못 본 부류의 특징을 지어내는 생성 접근법.
