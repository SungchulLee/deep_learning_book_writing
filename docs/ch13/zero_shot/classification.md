# 영 예시 분류

시각-뜻 묻힘 모델은 (CNN에서 나온) 시각 특징과 (낱말 묻힘 같은) 뜻 표현 사이의 깊은 신경망 옮김을 배운다. 이 절은 DeViSE, 겹선형 어울림 모델, 그리고 영 예시 학습을 위한 한 걸음 나아간 구조를 다룬다.

---

## 1. DeViSE: 깊은 시각-뜻 묻힘

### 구조 훑어보기

DeViSE(Frome 외, 2013)는 CNN 시각 특징을 뜻 묻힘 공간으로 쏘아 넣는 법을 배운다.

$$M: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^{d_s}$$

이때 다음이 성립한다.

$$M \cdot \mathbf{v}(\text{그림}) \approx \mathbf{s}(\text{부류 이름})$$

### 부품

**시각 부호기**: 미리 학습된 CNN(대개 VGG, ResNet)

- 끝에서 두 번째 층에서 특징을 뽑는다
- 학습 중에 얼리거나 미세 조정한다

**쏘아 넣는 망**: 배운 변환

- 선형 또는 비선형 옮김
- 시각 특징을 뜻 공간으로 쏘아 넣는다

**뜻 묻힘**: 미리 학습된 낱말 벡터

- Word2Vec, GloVe 또는 FastText
- 학습 중에는 (대개) 붙박이로 둔다

### 손실 함수

DeViSE은 여백 기반 순위 손실을 쓴다.

$$\mathcal{L} = \sum_{(\mathbf{x}, y)} \sum_{j \neq y} \max(0, \gamma - \mathbf{s}_y^\top \hat{\mathbf{v}} + \mathbf{s}_j^\top \hat{\mathbf{v}})$$

여기서 각 기호는 다음과 같다.

- $\hat{\mathbf{v}} = M \cdot \phi(\mathbf{x})$은 쏘아 넣은 시각 묻힘이다
- $\gamma$은 여백 매개변수이다
- $j \neq y$에 대한 합은 음의 보기를 뽑아 어림할 수 있다

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeViSE(nn.Module):
    """
    깊은 시각-뜻 묻힘 모델(DeViSE).
    
    CNN 특징을 낱말 묻힘 공간으로 쏘아 넣는다.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 300, dropout: float = 0.5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 시각 쏘아 넣기 망
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )
        
        # 뜻 쏘아 넣기(선택)
        self.semantic_projection = nn.Linear(semantic_dim, embedding_dim)
        
        # 소프트맥스의 온도(배울 수 있다)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def project_visual(self, visual_features):
        """시각 특징을 묻힘 공간으로 쏘아 넣는다."""
        projected = self.visual_projection(visual_features)
        return F.normalize(projected, p=2, dim=1)
    
    def project_semantic(self, semantic_embeddings):
        """뜻 묻힘을 묻힘 공간으로 쏘아 넣는다."""
        projected = self.semantic_projection(semantic_embeddings)
        return F.normalize(projected, p=2, dim=1)
    
    def forward(self, visual_features, semantic_pos, semantic_neg=None):
        """
        학습용 앞먹임.
        """
        v_proj = self.project_visual(visual_features)
        s_pos_proj = self.project_semantic(semantic_pos)
        
        pos_score = torch.sum(v_proj * s_pos_proj, dim=1)
        
        if semantic_neg is not None:
            s_neg_proj = self.project_semantic(semantic_neg)
            neg_score = torch.sum(v_proj * s_neg_proj, dim=1)
            return pos_score, neg_score
        
        return pos_score
    
    def predict(self, visual_features, class_embeddings, class_names):
        """시각 특징의 부류를 맞힌다."""
        self.eval()
        
        with torch.no_grad():
            v_proj = self.project_visual(visual_features)
            
            s_all = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in class_names
            ])
            s_proj = self.project_semantic(s_all)
            
            scores = v_proj @ s_proj.T
            pred_indices = torch.argmax(scores, dim=1)
            predictions = [class_names[i] for i in pred_indices.numpy()]
        
        return predictions, scores
```

---

## 2. 겹선형 어울림 모델

### 개념

겹선형 모델은 시각 차원과 뜻 차원 사이의 쌍마다의 어울림을 잡아낸다.

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

여기서 $W \in \mathbb{R}^{d_v \times d_s}$은 배운 가중치 행렬이다.

### 낮은 계수 어림

낮은 계수 인수분해로 매개변수를 줄인다.

$$W = U V^\top$$

이는 두 갈래를 모두 더 낮은 차원으로 쏘아 넣는 것과 같다.

$$F(\mathbf{v}, \mathbf{s}) = (U^\top \mathbf{v})^\top (V^\top \mathbf{s})$$

### 구현

```python
class BilinearCompatibility(nn.Module):
    """
    ZSL을 위한 겹선형 어울림 모델.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 hidden_dim: int = 512, use_low_rank: bool = True):
        super().__init__()
        
        self.use_low_rank = use_low_rank
        
        if use_low_rank:
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            self.W = nn.Parameter(torch.randn(visual_dim, semantic_dim) * 0.01)
            self.visual_bn = nn.BatchNorm1d(visual_dim)
    
    def forward(self, visual, semantic):
        """겹선형 어울림 점수를 셈한다."""
        if self.use_low_rank:
            v_proj = F.normalize(self.visual_proj(visual), dim=1)
            s_proj = F.normalize(self.semantic_proj(semantic), dim=-1)
            
            if len(s_proj.shape) == 2 and s_proj.shape[0] != visual.shape[0]:
                scores = v_proj @ s_proj.T
            else:
                scores = torch.sum(v_proj * s_proj, dim=1)
        else:
            visual = self.visual_bn(visual)
            if len(semantic.shape) == 2 and semantic.shape[0] != visual.shape[0]:
                scores = visual @ self.W @ semantic.T
            else:
                scores = torch.sum(visual @ self.W * semantic, dim=1)
        
        return scores
```

---

## 3. 손실 함수

### 여백 순위 손실

```python
class RankingLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        loss = torch.clamp(self.margin - pos_score + neg_score, min=0)
        return loss.mean()
```

### 세쌍 손실

```python
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return loss.mean()
```

### InfoNCE와 대조 손실

```python
def info_nce_loss(visual_emb, semantic_emb, temperature=0.07):
    """대조 학습을 위한 InfoNCE 손실."""
    v_norm = F.normalize(visual_emb, dim=1)
    s_norm = F.normalize(semantic_emb, dim=1)
    
    logits = v_norm @ s_norm.T / temperature
    labels = torch.arange(len(visual_emb), device=visual_emb.device)
    
    loss_v = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.T, labels)
    
    return (loss_v + loss_s) / 2
```

---

## 4. 한 걸음 나아간 구조

### 갈래를 넘나드는 주의

```python
class CrossModalAttention(nn.Module):
    """시각과 뜻을 맞추기 위한 갈래를 넘나드는 주의."""
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 hidden_dim: int = 256, n_heads: int = 8):
        super().__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, visual, semantic):
        v_proj = self.visual_proj(visual)
        s_proj = self.semantic_proj(semantic)
        
        if len(v_proj.shape) == 2:
            v_proj = v_proj.unsqueeze(1)
        s_proj = s_proj.unsqueeze(1)
        
        attended, _ = self.attention(
            query=s_proj, key=v_proj, value=v_proj
        )
        
        score = self.output_proj(attended.squeeze(1))
        return score.squeeze(-1)
```

### 두 가지 망

```python
class TwoBranchNetwork(nn.Module):
    """부호기를 따로 두는 두 가지 망."""
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 512):
        super().__init__()
        
        self.visual_branch = nn.Sequential(
            nn.Linear(visual_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim)
        )
        
        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, visual, semantic):
        v_emb = F.normalize(self.visual_branch(visual), dim=1)
        s_emb = F.normalize(self.semantic_branch(semantic), dim=-1)
        
        if len(s_emb.shape) == 2 and s_emb.shape[0] != visual.shape[0]:
            return v_emb @ s_emb.T
        else:
            return torch.sum(v_emb * s_emb, dim=1)
```

---

## 5. 학습의 좋은 버릇

### 어려운 음의 보기 캐기

모델이 가장 헷갈려 하는 음의 보기를 고른다.

```python
def hard_negative_mining(model, visual, positive_idx, all_embeddings, seen_classes):
    """표본마다 가장 어려운 음의 보기를 고른다."""
    with torch.no_grad():
        scores = model(visual, all_embeddings)
        
        # 양의 부류를 가린다
        for i, pos_i in enumerate(positive_idx):
            scores[i, pos_i] = -float('inf')
        
        # 가장 어려운 음의 보기를 고른다
        hard_neg_indices = torch.argmax(scores, dim=1)
    
    return all_embeddings[hard_neg_indices]
```

### 학습률 스케줄링

```python
import math

def cosine_warmup_scheduler(optimizer, num_epochs, warmup_epochs=10):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

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

## 정리하며

시각-뜻 묻힘 모델은 오늘날 ZSL의 등뼈를 이룬다.

1. **DeViSE**은 순위 손실을 써서 CNN 특징을 낱말 묻힘 공간으로 쏘아 넣는다
2. **겹선형 모델**은 갈래 사이의 풍부한 어울림을 잡아낸다
3. **갈래를 넘나드는 주의**는 촘촘한 맞춤을 배운다
4. **손실 함수**로는 순위, 세쌍, 교차 엔트로피, InfoNCE 변형이 있다

살펴야 할 핵심은 다음과 같다.

- (시각과 뜻 양쪽) 미리 학습된 특징이 매우 중요하다
- 어려운 음의 보기 캐기가 가름 학습을 좋게 한다
- 벌주기가 본 부류에 지나치게 맞추어지는 것을 막는다
- 알맞은 고르기가 학습을 안정되게 한다
