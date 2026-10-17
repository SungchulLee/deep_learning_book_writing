# 의미 임베딩 방법

의미 임베딩은 클래스 이름이나 설명을 텍스트에서 학습한 벡터로 나타낸다. 클래스별 속성 표시 비용을 줄일 수 있지만, 언어에서 가까운 개념이 이미지에서도 잘 구분되는 것은 아니다. 시각 피처와의 호환성 학습 및 ConSE의 확률 가중 평균을 비교한다.

## 1. 속성에서 임베딩으로

속성 기반 방법은 공통 속성을 정의하고 클래스별 값을 표시해야 한다. 텍스트 임베딩은 이미 학습한 언어 표현을 활용해 이 비용을 줄인다.

다만 이름이 중의적이거나 도메인 용어가 코퍼스에 없으면 적절한 표현을 얻기 어렵다. 클래스 이름을 벡터로 바꾸는 것과 시각 분류 성능이 확보되는 것은 별개다. 설명 문구와 언어, 토큰화·풀링 방식도 결과에 영향을 준다.

## 2. 단어 임베딩 방법

### Word2Vec

Word2Vec의 Skip-gram은 중심 단어에서 주변 단어를, CBOW는 주변 단어에서 중심 단어를 예측한다. 아래 소프트맥스 식은 입력용·출력용 임베딩을 구분하지 않은 간단한 표기다. 실제 학습에서는 음성 샘플링 등으로 전체 어휘 정규화의 비용을 줄일 수 있다.

**Skip-gram**: 대상 단어에서 둘레를 맞힌다

$$P(w_{context} | w_{target}) = \frac{\exp(\mathbf{v}_{context}^\top \mathbf{v}_{target})}{\sum_w \exp(\mathbf{v}_w^\top \mathbf{v}_{target})}$$

**CBOW**: 둘레에서 대상을 맞힌다

$$P(w_{target} | w_{context_1}, \ldots, w_{context_k})$$

일부 벡터 관계가 단어 유추를 표현할 수 있지만 모든 의미 관계에 성립하는 법칙은 아니다. 임베딩 차원과 학습 코퍼스는 사용하는 모델에 따라 달라진다. 서로 다른 모델의 벡터는 차원이 같아도 같은 좌표계가 아니다.

- 의미 관계를 담는다: 왕 - 남자 + 여자 ≈ 여왕
- 흔한 차원: 100~300
- 수십억 단어로 미리 학습한다

### GloVe(전역 벡터)

GloVe은 단어가 함께 나오는 통계를 최적화한다.

$$J = \sum_{i,j} f(X_{ij}) (\mathbf{v}_i^\top \mathbf{v}_j + b_i + b_j - \log X_{ij})^2$$

여기서 $X_{ij}$은 함께 나온 횟수이고 $f$은 무게 주는 함수이다.

### FastText

FastText는 단어를 구성하는 문자 n-그램 벡터를 활용한다. 단어 벡터와 서브워드 표현을 결합해 희귀 단어와 미등록 단어의 형태 정보를 활용할 수 있다. 원래 학습한 서브워드 정보를 포함한 모델이 있어야 하며, 미리 내보낸 단어 벡터 표만으로는 새 단어를 합성하지 못할 수 있다.

- 단어를 글자 n-그램의 자루로 나타낸다
- 어휘 밖 단어를 다룬다
- 형태 정보를 담는다

$$\mathbf{v}_{word} = \sum_{g \in \text{ngrams}(word)} \mathbf{v}_g$$

### 비교

| 방법 | 표현 방식 | 새 이름 처리 시 고려할 점 |
|---|---|---|
| Word2Vec | 단어별 정적 벡터 | 어휘에 없는 단어의 처리 필요 |
| GloVe | 동시 출현 통계 기반 정적 벡터 | 어휘와 토큰 표기 일치 필요 |
| FastText | 단어·문자 n-그램 | 서브워드 정보가 있는 모델 필요 |
| BERT 계열 | 문맥별 토큰 표현 | 토큰화와 풀링·설명 문구 선택 필요 |

---

## 3. ZSL에 임베딩 쓰기

### 클래스 이름 임베딩

아래 조각은 gensim을 통해 GloVe 벡터를 내려받는 예시다. 실제 파일을 준비해야 하며, 단어 벡터를 가져오는 것 자체가 이미지·텍스트 정렬 학습은 아니다.

이름이 "zebra"인 클래스라면 다음과 같다.

```python
import gensim.downloader as api

# 미리 학습된 묻힘을 불러온다
word_vectors = api.load("glove-wiki-gigaword-300")

# 부류 묻힘을 얻는다
zebra_embedding = word_vectors["zebra"]  # 꼴: (300,)
```

### 여러 단어로 된 클래스 이름

공백으로 나눈 단어들의 벡터를 평균 내는 간단한 방법이다. 누락 단어를 제외하므로 일부 단어만 남으면 원래 이름과 의미가 달라질 수 있다. np 임포트도 별도로 필요하다.

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

### 의미 구조 살펴보기

아래 t-SNE는 클래스 벡터를 2차원으로 시각화한다. perplexity=5이므로 클래스가 5개보다 많아야 한다. 가까운 점은 선택한 투영에서의 관계를 보여 주지만 전역 거리나 분류 가능성을 증명하지 않는다. 차원 축 자체에는 고정된 의미가 없다.

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

---

## 4. 시각·의미 호환성 학습하기

### 문제 정의

시각 피처와 의미 임베딩 사이의 호환성 함수를 배운다.

$$F: \mathcal{V} \times \mathcal{S} \rightarrow \mathbb{R}$$

예측 규칙은 다음과 같다.

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} F(\phi(\mathbf{x}), \mathbf{s}_c)$$

### 임베딩 공간 접근법

**시각을 뜻으로 투영**:

$$f_v(\mathbf{v}) \in \mathcal{S}$$

학습: $\min_\theta \sum_{(\mathbf{x}, y)} \|f_\theta(\phi(\mathbf{x})) - \mathbf{s}_y\|^2$

**뜻을 시각으로 투영**:

$$f_s(\mathbf{s}) \in \mathcal{V}$$

**함께 쓰는 임베딩 공간**:

$$f_v(\mathbf{v}), f_s(\mathbf{s}) \in \mathcal{E}$$

### 순위 로스

회귀 대신 순위 매기기를 써서 맞는 클래스가 더 높은 점수를 받게 한다.

$$\mathcal{L} = \sum_{(\mathbf{x}, y)} \sum_{c \neq y} \max(0, \Delta + F(\phi(\mathbf{x}), \mathbf{s}_c) - F(\phi(\mathbf{x}), \mathbf{s}_y))$$

### PyTorch 구현

predict_unseen은 unseen_classes 목록의 인덱스를 반환한다. 의미 벡터와 projected를 CPU에 생성하므로 CUDA 입력에 그대로 적용할 수 없다. train_conse_classifier는 원본 라벨을 class_to_idx로 변환하므로 라벨 자료형과 사전 키를 일치시켜야 한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSemanticEmbedding(nn.Module):
    """
    ZSL을 위한 시각-뜻 묻힘 모델.
    
    시각 피처를 뜻 묻힘 공간으로 쏘아 넣는다.
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
        """시각 피처를 부호로 바꾼다."""
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
        순위 로스 셈을 위한 피드포워드.
        
        인수:
            v: 시각 피처 (batch_size, visual_dim)
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
    어울림 학습을 위한 여백 기반 순위 로스.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        """
        순위 로스를 셈한다.
        
        로스 = max(0, margin + neg_score - pos_score)
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

---

## 5. ConSE: 의미 임베딩의 볼록 결합

### 개념

ConSE는 기존 학습 클래스 분류기의 확률로 클래스 의미 벡터를 가중 평균한다. 이 변환 자체에는 별도의 시각·의미 투영 학습이 필요 없다. 다만 사용할 학습 클래스 분류기와 의미 벡터는 미리 준비되어 있어야 한다.

### 수식으로 나타내기

**1단계: 학습 클래스의 확률**

분류기의 출력 열을 학습 클래스 목록과 일치시킨다. 예측 확률로 의미 벡터를 합산하며, 상위 T개만 사용할 때는 선택한 확률을 다시 정규화한다.

$$P(y^s | \mathbf{x}) = \text{softmax}(W^\top \phi(\mathbf{x}))$$

**2단계: 투영한 임베딩**

학습 클래스 임베딩의 가중 평균을 계산한다.

$$\hat{\mathbf{z}} = \sum_{c \in \mathcal{Y}^s} P(c | \mathbf{x}) \cdot \mathbf{s}_c$$

**3단계: 최근접 이웃**

아래 유클리드 최근접 이웃은 한 가지 규칙이다. 이 페이지의 ConSE 구현은 L2 정규화 후 코사인 유사도로 비교한다. 정규화 여부를 고정하여 수식과 구현의 결과를 구분한다.

$$\hat{y} = \arg\min_{c \in \mathcal{Y}^u} \|\hat{\mathbf{z}} - \mathbf{s}_c\|$$

### 상위 T개 클래스 사용하기

학습 클래스의 상위 T개 예측만 사용할 수 있다. 1≤T≤학습 클래스 수를 만족해야 하며 T는 검증 클래스에서 선택한다. 많은 클래스를 포함한다고 항상 더 좋아지지는 않는다.

$$\hat{\mathbf{z}} = \sum_{c \in \text{Top}_T(\mathcal{Y}^s | \mathbf{x})} \frac{P(c | \mathbf{x})}{\sum_{c' \in \text{Top}_T} P(c' | \mathbf{x})} \cdot \mathbf{s}_c$$

### PyTorch 구현

predict_unseen은 unseen_classes 목록의 인덱스를 반환한다. 의미 벡터와 projected를 CPU에 생성하므로 CUDA 입력에 그대로 적용할 수 없다. train_conse_classifier는 원본 라벨을 class_to_idx로 변환하므로 라벨 자료형과 사전 키를 일치시켜야 한다.

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
            x: 시각 피처 (batch_size, visual_dim)
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

---

## 6. 호환성 학습과 ConSE 비교

호환성 학습은 이미지·의미 쌍의 점수나 공통 공간을 직접 학습한다. ConSE는 학습 클래스의 분류 확률과 고정 의미 벡터로 새 클래스의 위치를 추정한다. 두 방법 모두 학습 클래스와 미학습 클래스의 관계를 활용하지만 학습 목표와 전이 경로가 다르다.

ConSE의 성능은 기존 분류기의 예측과 의미 공간의 클래스 배치에 의존한다. 호환성 학습은 표현을 조정할 수 있지만 학습 클래스에 과적합될 수 있다. 데이터 양이나 해석 가능성만으로 우열을 정하지 말고 독립적인 검증 과제로 비교한다.

### 성능 비교

compare_methods는 y_train 대신 표본 순번을 라벨로 넣으므로 원래 클래스 매핑이 사라진다. 이 상태로 학습 결과를 비교할 수 없다. y_test도 미학습 클래스의 로컬 인덱스에 맞춰야 한다. 올바른 라벨과 같은 분할·피처·학습 예산을 사용한 비교가 필요하다.

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

---

## 7. 여러 단어로 된 클래스 이름

### 주요 과제

여러 단어로 이루어진 이름은 단어 평균, 구 임베딩 또는 문맥 인코더로 나타낼 수 있다. 공백 분리와 단순 평균은 단어 순서와 숙어의 의미를 잃는다.

- "polar bear"
- "fire truck"
- "저먼 셰퍼드"

### 전략

**1. 단어 평균 내기**:

$$\mathbf{s}_c = \frac{1}{|words_c|} \sum_{w \in words_c} \mathbf{v}_w$$

**2. 가중 평균 내기**(TF-IDF 웨이트):

$$\mathbf{s}_c = \sum_{w \in words_c} \text{tfidf}(w) \cdot \mathbf{v}_w$$

**3. 구 임베딩**: 여러 단어를 하나의 구로 학습했거나 문장 수준 표현을 제공하는 모델을 사용한다. FastText의 문자 n-그램 기능 자체가 모든 복합어·구의 의미를 이해한다는 뜻은 아니다.

**4. 문맥 임베딩**: 클래스 이름을 설명 문장에 넣고 토큰 평균이나 특정 토큰을 풀링할 수 있다. 아래 함수는 BERT의 첫 토큰을 반환하지만 그것이 시각적 클래스 구분에 적합한지는 검증해야 한다. model.eval()과 장치 설정은 별도이며 CUDA 텐서는 CPU로 이동한 뒤 NumPy로 변환한다.

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

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
클래스 이름의 단어 벡터를 얻는 것만으로 제로샷 이미지 분류를 할 수 있는가?

</div>

??? success "풀이"
    이미지 피처와 같은 의미 좌표계에 정렬되어야 한다. 별도의 투영 학습이나 공동 사전 학습된 인코더가 필요하다. 텍스트 벡터는 클래스 설명을 제공하지만 입력 이미지와의 대응을 자동으로 만들지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ConSE의 상위 T개 웨이트가 볼록 결합을 이루는 이유를 설명하라.

</div>

??? success "풀이"
    선택한 클래스의 확률은 음이 아니고 그 합으로 나누면 합이 1이 된다. 따라서 투영 벡터는 선택한 클래스 의미 벡터의 볼록 결합이다. 이후 정규화한 방향을 코사인 비교에 사용할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
클래스 이름에서 일부 단어가 어휘에 없을 때 단순 평균의 한계를 설명하라.

</div>

??? success "풀이"
    누락 단어를 제외하면 클래스의 구별에 중요한 정보가 사라질 수 있다. 모든 단어가 누락되면 함수는 오류를 낸다. 서브워드 모델이나 설명 문장, 표준 이름 매핑 등을 검증할 수 있으며 서로 다른 이름을 같은 표현으로 만들지 않도록 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
성능 비교 코드에서 원래 라벨 대신 표본 순번을 쓰면 어떤 문제가 생기는가?

</div>

??? success "풀이"
    클래스별 감독 신호가 사라져 의미 벡터 사전의 키와 맞지 않거나 잘못된 클래스를 조회한다. TensorDataset에는 실제 원본 클래스 라벨을 넣고 모델별 로컬 인덱스 변환을 명시해야 한다. 테스트 예측 인덱스도 같은 클래스 목록으로 해석한다.

## 정리하며

텍스트 임베딩은 클래스의 의미 정보를 제공한다. 호환성 학습은 시각·의미 관계를 직접 학습하고 ConSE는 기존 분류 확률로 의미 벡터를 합성한다. 어휘·라벨 매핑과 좌표계 정렬을 확인하고, 시각화나 클래스 이름의 벡터화 자체를 분류 성능으로 해석하지 않는다.

**참고**: [Norouzi 등, Zero-Shot Learning by Convex Combination of Semantic Embeddings](https://arxiv.org/abs/1312.5650).
