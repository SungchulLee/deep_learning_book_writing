# 제로샷 분류

시각·의미 임베딩 분류기는 이미지 피처와 클래스 설명을 비교할 수 있도록 투영을 학습한다. DeViSE의 순위 학습 아이디어, 쌍선형 점수와 두 인코더 구조를 살펴본다. 제시한 코드는 개념을 확장한 예제이며 각 원 논문의 구조를 그대로 재현하지는 않는다.

## 1. DeViSE: 깊은 시각-의미 임베딩

### 구조 훑어보기

DeViSE(Frome 외, 2013)는 CNN 시각 피처를 의미 임베딩 공간으로 투영하는 법을 배운다.

$$M: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^{d_s}$$

이때 다음이 성립한다.

$$M \cdot \mathbf{v}(\text{이미지}) \approx \mathbf{s}(\text{클래스 이름})$$

### 부품

**시각 인코더**: 이미지에서 피처를 추출한다. 원 DeViSE는 당시의 CNN을 사용했으며, 아래 구현은 이미 추출한 피처를 입력받는다. 백본의 동결·미세 조정은 별도로 정해야 한다.

- 끝에서 두 번째 레이어에서 피처를 뽑는다
- 학습 중에 얼리거나 미세 조정한다

**투영하는 망**: 배운 변환

- 선형 또는 비선형 옮김
- 시각 피처를 의미 공간으로 투영하는다

**의미 임베딩**: 미리 학습된 단어 벡터

- Word2Vec, GloVe 또는 FastText
- 학습 중에는 (대개) 고정로 둔다

### 로스 함수

DeViSE는 마진 기반 순위 로스를 쓴다.

$$\mathcal{L} = \sum_{(\mathbf{x}, y)} \sum_{j \neq y} \max(0, \gamma - \mathbf{s}_y^\top \hat{\mathbf{v}} + \mathbf{s}_j^\top \hat{\mathbf{v}})$$

여기서 각 기호는 다음과 같다.

- $\hat{\mathbf{v}} = M \cdot \phi(\mathbf{x})$은 투영한 시각 임베딩이다
- $\gamma$은 마진 파라메터이다
- $j \neq y$에 대한 합은 음의 보기를 뽑아 어림할 수 있다

### PyTorch 구현

아래 DeViSE는 시각·의미 양쪽을 투영하고 L2 정규화하는 변형이다. temperature는 선언만 되고 점수 계산에 사용되지 않는다. predict는 의미 벡터를 CPU에 만들고 인덱스를 바로 numpy로 변환하므로 CUDA 모델에 그대로 적용할 수 없다. BatchNorm1d 때문에 학습 배치 크기도 고려한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeViSE(nn.Module):
    """
    깊은 시각-뜻 묻힘 모델(DeViSE).
    
    CNN 피처를 낱말 묻힘 공간으로 쏘아 넣는다.
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
        """시각 피처를 묻힘 공간으로 쏘아 넣는다."""
        projected = self.visual_projection(visual_features)
        return F.normalize(projected, p=2, dim=1)
    
    def project_semantic(self, semantic_embeddings):
        """뜻 묻힘을 묻힘 공간으로 쏘아 넣는다."""
        projected = self.semantic_projection(semantic_embeddings)
        return F.normalize(projected, p=2, dim=1)
    
    def forward(self, visual_features, semantic_pos, semantic_neg=None):
        """
        학습용 피드포워드.
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
        """시각 피처의 부류를 맞힌다."""
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

## 2. 쌍선형 호환성 모델

### 개념

쌍선형 모델은 시각 차원과 의미 차원 사이의 쌍마다의 호환성을 잡아낸다.

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

여기서 $W \in \mathbb{R}^{d_v \times d_s}$은 배운 웨이트 행렬이다.

### 저랭크 분해

저랭크 분해로 파라메터를 줄인다.

$$W = U V^\top$$

이는 두 표현을 모두 더 낮은 차원으로 투영하는 것과 같다.

$$F(\mathbf{v}, \mathbf{s}) = (U^\top \mathbf{v})^\top (V^\top \mathbf{s})$$

### 구현

use_low_rank=True 경로는 비선형 MLP와 정규화를 포함하므로 수식의 단순 W=UVᵀ 분해와 정확히 같지는 않다. 또한 클래스 수와 배치 크기가 같으면 코드가 대응 쌍 입력으로 해석해 (B,C) 대신 (B,)를 반환한다. 입력 의미를 크기 비교만으로 구분하는 한계를 주의한다.

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

## 3. 로스 함수

### 마진 순위 로스

```python
class RankingLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        loss = torch.clamp(self.margin - pos_score + neg_score, min=0)
        return loss.mean()
```

### 트리플렛 로스

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

### InfoNCE와 대조 로스

행 i의 이미지와 열 i의 의미 벡터가 양성인 동일 크기 배치를 가정한다. 같은 클래스 설명이 여러 번 나오면 다른 열의 같은 설명도 음성으로 취급되어 목표가 충돌할 수 있다. 온도는 양수여야 한다.

```python
def info_nce_loss(visual_emb, semantic_emb, temperature=0.07):
    """대조 학습을 위한 InfoNCE 로스."""
    v_norm = F.normalize(visual_emb, dim=1)
    s_norm = F.normalize(semantic_emb, dim=1)
    
    logits = v_norm @ s_norm.T / temperature
    labels = torch.arange(len(visual_emb), device=visual_emb.device)
    
    loss_v = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.T, labels)
    
    return (loss_v + loss_s) / 2
```

---

## 4. 확장 구조

### 교차 모달 어텐션

2차원 시각 입력을 길이 1의 키·값으로 바꾸면 소프트맥스의 웨이트는 항상 1이다. 평가 모드에서 출력은 의미 쿼리와 무관하게 시각 값의 변환이 되므로 이 경우 의미별 정교한 정렬을 학습한다고 설명할 수 없다. 여러 시각 토큰을 제공하면 어텐션으로 선택할 수 있다.

```python
class CrossModalAttention(nn.Module):
    """시각과 뜻을 맞추기 위한 클래스를 넘나드는 주의."""
    
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

### 두 인코더 구조

시각과 의미를 공통 차원으로 투영한 뒤 정규화 내적으로 비교한다. 이 구현도 행 수가 같으면 대응 쌍으로 처리하므로 전체 클래스 점수 행렬을 원하는 호출에서는 배치·클래스 수가 우연히 같을 때를 구별해야 한다.

```python
class TwoBranchNetwork(nn.Module):
    """인코더를 따로 두는 두 가지 망."""
    
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

## 5. 학습 시 고려할 점

### 하드 네거티브 선택

양성 클래스의 점수를 제외하고 가장 높은 음성 점수를 선택한다. scores는 (B,C)여야 하며 positive_idx는 클래스 임베딩 행 번호와 일치해야 한다. seen_classes 인수는 사용되지 않는다. no_grad는 모델을 평가 모드로 바꾸지 않으므로 드롭아웃과 BatchNorm 상태도 고려한다. 어려운 표본만 고르면 라벨 오류의 영향이 커질 수 있다.

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

### 러닝레이트 스케줄링

0<warmup_epochs<num_epochs를 가정하는 코사인 스케줄 예시다. 반복 단위를 에포크로 정하고 scheduler.step 호출 위치를 일치시킨다. 시작 계수는 0이며, 명시한 학습 기간을 넘겨 호출하면 코사인 값이 다시 커질 수 있다.

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

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
순위 로스가 0이 되는 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    양성 점수가 음성 점수보다 마진 이상 크면 해당 항이 0이다. 모든 클래스의 점수를 확률로 정규화할 필요는 없다. 음성 샘플링을 사용하면 전체 음성에 대한 로스와 추정 방식이 달라질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
W=UVᵀ 분해의 차원과 파라메터 절감을 설명하라.

</div>

??? success "연습문제 2 풀이"
    U가 d_v×r, V가 d_s×r이면 공통 r차원 공간의 내적으로 쌍선형 점수를 계산한다. 파라메터 수는 d_v×d_s에서 r(d_v+d_s)로 바뀐다. 비선형 MLP를 넣으면 일반적으로 같은 고정 W의 분해로 해석할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배치 크기와 클래스 수가 같을 때 BilinearCompatibility의 출력 의미를 설명하라.

</div>

??? success "연습문제 3 풀이"
    코드는 두 행 수가 다를 때만 전체 점수 행렬을 만든다. 같으면 행끼리 곱해 (B,) 점수를 반환한다. 전체 후보 클래스 비교와 대응 쌍 비교는 별도 API나 명시적인 인수로 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
하나의 시각 토큰에 대한 교차 어텐션이 클래스 의미를 구분할 수 있는지 설명하라.

</div>

??? success "연습문제 4 풀이"
    키가 하나면 어텐션 웨이트가 1이므로 값의 가중합은 쿼리와 무관하다. 이 코드에는 의미 쿼리의 잔차 경로도 없으므로 평가 모드의 점수는 의미 입력을 구분하지 못한다. 여러 시각 토큰 또는 별도의 의미 결합 경로가 필요하다.

## 정리하며

시각·의미 정렬은 투영 구조와 순위·대조 로스로 학습한다. 대응 쌍 점수와 전체 클래스 점수 행렬을 구분하고, 의미 정보가 실제 출력에 영향을 주는지 확인한다. 사전 학습·입력 순서·장치와 음성 표본의 정의도 평가에 포함한다.
