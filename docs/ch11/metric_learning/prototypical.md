# 원형 망
## 들어가며

Snell 외(2017)가 들여온 원형 망은 소수 예시 학습에서 가장 우아하고 쓸모 있는 접근법에 든다. 핵심 통찰은 놀랍도록 단순하다. 부류는 그 받침 묻힘의 평균(원형)으로 나타낼 수 있고, 가려내기는 가장 가까운 원형을 찾는 것으로 이루어진다.

## 핵심 개념

### 원형이라는 생각

묻힘 함수 $f_\theta$과 부류 $c$의 받침 집합 $\mathcal{S}_c = \{(x_1^c, y_1^c), \ldots, (x_{K}^c, y_K^c)\}$이 주어지면 부류 $c$의 **원형**은 다음과 같다.

$$\mathbf{c}_c = \frac{1}{|\mathcal{S}_c|} \sum_{(x_i, y_i) \in \mathcal{S}_c} f_\theta(x_i)$$

이는 그저 배운 표현 공간에서 부류 묻힘의 무게중심이다.

### 거리로 가려내기

물음 $x$이 주어지면 모든 원형까지의 거리를 셈하고 소프트맥스를 씌워 가려낸다.

$$p(y = c | x) = \frac{\exp(-d(f_\theta(x), \mathbf{c}_c))}{\sum_{c'} \exp(-d(f_\theta(x), \mathbf{c}_{c'}))}$$

여기서 $d$은 거리 함수이다(대개 유클리드 거리의 제곱이다).

### 원형이 통하는 까닭

원형 망의 쓸모는 여러 렌즈로 이해할 수 있다.

**브레그만 벌어짐의 눈**: Snell 외(2017)는 어떤 브레그만 벌어짐에서든 무리의 평균과 견주는 것이 가장 좋은 가려내기임을 보였다. 유클리드 거리의 제곱은 브레그만 벌어짐이다.

**섞기 풀이**: 원형을 셈하는 것은 평균 내기로 데이터를 늘리는 한 가지 꼴로 볼 수 있어 일반화를 좋게 한다.

**튼튼함**: 여러 보기를 평균 내면 잡음 끼거나 별난 받침 보기의 영향이 줄어든다.

## 수학적 바탕

### 브레그만 벌어짐

엄밀히 볼록하고 미분할 수 있는 함수 $\phi$에 딸린 브레그만 벌어짐은 다음과 같다.

$$d_\phi(z, z') = \phi(z) - \phi(z') - (z - z')^T \nabla\phi(z')$$

**정리**: 브레그만 벌어짐 $d_\phi$을 쓰는 지수족 섞음 모델에서 기대 벌어짐을 가장 작게 하는 무리 대표는 무리의 평균이다.

유클리드 거리의 제곱은 $\phi(z) = \|z\|_2^2$에 딸린 브레그만 벌어짐이다.

$$d_{\phi}(z, z') = \|z\|^2 - \|z'\|^2 - 2z'^T(z - z') = \|z - z'\|^2$$

### 가우스 섞음 모델과의 이음

원형 망은 공분산을 나누어 쓰는 가우스 섞음을 배우는 것으로 볼 수 있다.

$$p(x | c) = \mathcal{N}(f_\theta(x); \mu_c, \sigma^2 I)$$

여기서 $\mu_c$은 원형이고 $\sigma^2$은 부류들이 나누어 쓴다. 이 모델 아래에서 부류 위의 뒤확률은 거리 제곱의 음수에 씌운 소프트맥스와 정확히 같다.

## PyTorch 구현

### 온전한 원형 망

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PrototypicalNetwork(nn.Module):
    """
    소수 예시 학습을 위한 원형 망.
    
    참고: Snell 외, "Prototypical Networks for Few-shot Learning"
    NeurIPS 2017
    """
    
    def __init__(
        self, 
        encoder: nn.Module,
        distance: str = 'euclidean',
        temperature: float = 1.0
    ):
        """
        인수:
            encoder: 묻힘 망 f_θ
            distance: 'euclidean' 또는 'cosine'
            temperature: 로짓의 눈금 인자
        """
        super().__init__()
        self.encoder = encoder
        self.distance = distance
        self.temperature = temperature
    
    def compute_prototypes(
        self, 
        support_embeddings: torch.Tensor, 
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        받침 묻힘에서 부류 원형을 셈한다.
        
        인수:
            support_embeddings: (n_support, embed_dim)
            support_labels: 값이 {0, ..., n_way-1}에 드는 (n_support,)
        
        반환값:
            prototypes: (n_way, embed_dim)
        """
        n_way = support_labels.max().item() + 1
        embed_dim = support_embeddings.size(1)
        
        prototypes = torch.zeros(n_way, embed_dim, device=support_embeddings.device)
        
        for c in range(n_way):
            class_mask = (support_labels == c)
            prototypes[c] = support_embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(
        self, 
        query_embeddings: torch.Tensor, 
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        물음에서 원형까지의 거리를 셈한다.
        
        인수:
            query_embeddings: (n_query, embed_dim)
            prototypes: (n_way, embed_dim)
        
        반환값:
            distances: (n_query, n_way)
        """
        if self.distance == 'euclidean':
            # 유클리드 거리의 제곱: ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q^T*p
            query_norm = (query_embeddings ** 2).sum(dim=1, keepdim=True)
            proto_norm = (prototypes ** 2).sum(dim=1, keepdim=True).t()
            cross_term = torch.mm(query_embeddings, prototypes.t())
            
            distances = query_norm + proto_norm - 2 * cross_term
            distances = torch.clamp(distances, min=0.0)
            
        elif self.distance == 'cosine':
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        return distances
    
    def forward(
        self, 
        support: torch.Tensor, 
        support_labels: torch.Tensor, 
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        물음 보기의 가려내기 로짓을 셈한다.
        
        인수:
            support: 받침 집합 그림 (n_support, *input_shape)
            support_labels: 받침 이름표 (n_support,)
            query: 물음 그림 (n_query, *input_shape)
        
        반환값:
            logits: (n_query, n_way) 가려내기 로짓
        """
        n_support = support.size(0)
        
        # 효율적인 배치 부호화를 위해 이어 붙인다
        all_images = torch.cat([support, query], dim=0)
        all_embeddings = self.encoder(all_images)
        
        # 다시 쪼갠다
        support_embeddings = all_embeddings[:n_support]
        query_embeddings = all_embeddings[n_support:]
        
        # 원형을 셈한다
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # 거리를 셈한다
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # 로짓으로 바꾼다(거리의 음수)
        logits = -distances / self.temperature
        
        return logits
    
    def loss(
        self, 
        support: torch.Tensor, 
        support_labels: torch.Tensor, 
        query: torch.Tensor, 
        query_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        에피소드의 손실과 정확도를 셈한다.
        
        반환값:
            loss: 교차 엔트로피 손실
            accuracy: 가려내기 정확도
        """
        logits = self.forward(support, support_labels, query)
        loss = F.cross_entropy(logits, query_labels)
        
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss, accuracy
```

### 표준 부호기 구조

```python
class Conv4Encoder(nn.Module):
    """
    소수 예시 학습에서 흔히 쓰는 4층 합성곱 부호기.
    """
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
```

## 학습 절차

### 에피소드 학습 되돌이

```python
def train_prototypical_network(
    model: PrototypicalNetwork,
    train_dataset,
    optimizer: torch.optim.Optimizer,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    n_episodes: int = 100,
    device: str = 'cuda'
):
    """
    원형 망을 한 시대 동안 익힌다.
    """
    model.train()
    total_loss = 0
    total_acc = 0
    
    for episode in range(n_episodes):
        # 에피소드를 뽑는다
        support, support_labels, query, query_labels = sample_episode(
            train_dataset, n_way, k_shot, n_query
        )
        
        support = support.to(device)
        support_labels = support_labels.to(device)
        query = query.to(device)
        query_labels = query_labels.to(device)
        
        # 앞먹임과 되돌림
        optimizer.zero_grad()
        loss, acc = model.loss(support, support_labels, query, query_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes
```

### 학습 설정

원형 망의 주요 초매개변수는 다음과 같다.

| 매개변수 | 흔한 값 | 비고 |
|-----------|--------------|-------|
| N-갈래(학습) | 20~60 | 학습 때 N을 키우면 도움이 될 때가 많다 |
| K-예시(학습) | 5~15 | 시험 때보다 예시를 많이 |
| 부류마다 물음 | 15 | 표준 선택 |
| 학습률 | 1e-3 | Adam 최적화기와 함께 |
| 묻힘 차원 | 64~1600 | 복잡도에 달렸다 |
| 시대마다 에피소드 | 100~1000 | 복잡한 과제일수록 더 많이 |

**학습 요령**: 시험 때보다 큰 N-갈래로 익혀라. 30-갈래로 익히고 5-갈래로 시험하면 성능이 좋아질 때가 많다.

## 변형과 확장

### 무한 섞음 원형

부류마다 원형을 여럿 두어 부류의 불확실성을 다룬다.

```python
class InfiniteMixturePrototypes(nn.Module):
    """
    섞음 모델을 써서 부류마다
    원형을 여럿 두어 나타낸다.
    """
    
    def __init__(self, encoder: nn.Module, n_prototypes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.n_prototypes = n_prototypes
    
    def compute_cluster_prototypes(
        self, 
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ):
        """
        k-평균으로 부류마다 원형을 여럿 찾는다.
        """
        from sklearn.cluster import KMeans
        
        n_way = support_labels.max().item() + 1
        embed_dim = support_embeddings.size(1)
        
        all_prototypes = []
        prototype_weights = []
        prototype_classes = []
        
        for c in range(n_way):
            class_mask = support_labels == c
            class_embeddings = support_embeddings[class_mask].cpu().numpy()
            
            n_samples = class_embeddings.shape[0]
            n_clusters = min(self.n_prototypes, n_samples)
            
            if n_clusters == 1:
                prototypes = class_embeddings.mean(axis=0, keepdims=True)
                weights = [1.0]
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(class_embeddings)
                prototypes = kmeans.cluster_centers_
                
                # 무리 크기로 무게를 준다
                labels = kmeans.labels_
                weights = [(labels == k).sum() / len(labels) 
                          for k in range(n_clusters)]
            
            for i, (proto, weight) in enumerate(zip(prototypes, weights)):
                all_prototypes.append(proto)
                prototype_weights.append(weight)
                prototype_classes.append(c)
        
        return (
            torch.tensor(all_prototypes, device=support_embeddings.device),
            torch.tensor(prototype_weights, device=support_embeddings.device),
            torch.tensor(prototype_classes, device=support_embeddings.device)
        )
```

### 과제에 맞추어 가는 원형

물음 정보를 바탕으로 원형을 맞추어 간다.

```python
class TaskAdaptivePrototypes(nn.Module):
    """
    받침-물음 쌍 위의 주의로 원형을 다듬는다.
    """
    
    def __init__(self, encoder: nn.Module, embed_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        
        # 어텐션 장치
        self.query_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, support, support_labels, query):
        n_support = support.size(0)
        n_way = support_labels.max().item() + 1
        
        # 부호화
        all_images = torch.cat([support, query], dim=0)
        all_embeddings = self.encoder(all_images)
        
        support_embeddings = all_embeddings[:n_support]
        query_embeddings = all_embeddings[n_support:]
        
        # 처음 원형
        prototypes = []
        for c in range(n_way):
            mask = support_labels == c
            prototypes.append(support_embeddings[mask].mean(dim=0))
        prototypes = torch.stack(prototypes)  # (n_way, embed_dim)
        
        # 물음에 주의를 주어 다듬는다
        # 물음: 원형, 열쇠와 값: 받침 전체
        refined_prototypes, _ = self.query_attention(
            prototypes.unsqueeze(0),
            support_embeddings.unsqueeze(0),
            support_embeddings.unsqueeze(0)
        )
        refined_prototypes = refined_prototypes.squeeze(0)
        
        # 본디 것과 다듬은 것을 합친다
        prototypes = 0.5 * prototypes + 0.5 * refined_prototypes
        
        # 거리를 셈한다
        distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        
        return -distances
```

### 반 지도 원형 망

이름표 없는 데이터로 원형을 다듬는다.

```python
class SemiSupervisedProtoNet(nn.Module):
    """
    부드러운 배정으로 이름표 없는 보기를 아우른다.
    """
    
    def __init__(self, encoder: nn.Module, n_refine_steps: int = 3):
        super().__init__()
        self.encoder = encoder
        self.n_refine_steps = n_refine_steps
    
    def forward(
        self, 
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
        unlabeled: Optional[torch.Tensor] = None
    ):
        # 모두 부호로 바꾼다
        support_emb = self.encoder(support)
        query_emb = self.encoder(query)
        
        n_way = support_labels.max().item() + 1
        
        # 이름표 붙은 받침에서 얻은 처음 원형
        prototypes = self._compute_prototypes(support_emb, support_labels, n_way)
        
        if unlabeled is not None:
            unlabeled_emb = self.encoder(unlabeled)
            
            # 원형을 되풀이하여 다듬는다
            for _ in range(self.n_refine_steps):
                # 이름표 없는 것을 원형에 부드럽게 배정한다
                distances = torch.cdist(unlabeled_emb, prototypes, p=2) ** 2
                soft_labels = F.softmax(-distances, dim=1)  # (n_unlabeled, n_way)
                
                # 원형을 다시 셈한다
                new_prototypes = []
                for c in range(n_way):
                    # 이름표 없는 것까지 넣은 무게 준 평균
                    labeled_mask = support_labels == c
                    labeled_contrib = support_emb[labeled_mask].sum(dim=0)
                    labeled_count = labeled_mask.sum()
                    
                    unlabeled_weights = soft_labels[:, c]
                    unlabeled_contrib = (unlabeled_emb * unlabeled_weights.unsqueeze(1)).sum(dim=0)
                    unlabeled_count = unlabeled_weights.sum()
                    
                    prototype = (labeled_contrib + unlabeled_contrib) / (labeled_count + unlabeled_count)
                    new_prototypes.append(prototype)
                
                prototypes = torch.stack(new_prototypes)
        
        # 마지막 가려내기
        distances = torch.cdist(query_emb, prototypes, p=2) ** 2
        return -distances
    
    def _compute_prototypes(self, embeddings, labels, n_way):
        prototypes = []
        for c in range(n_way):
            mask = labels == c
            prototypes.append(embeddings[mask].mean(dim=0))
        return torch.stack(prototypes)
```

## 이론적 분석

### 표본 복잡도

부류마다 받침 보기 $K$개를 쓰고 묻힘 차원이 $d$인 원형 망에서는 다음과 같다.

**원형 어림의 오차**:

$$\mathbb{E}\left[\|\hat{\mathbf{c}}_c - \mathbf{c}_c^*\|_2^2\right] = O\left(\frac{\sigma^2}{K}\right)$$

여기서 $\sigma^2$은 부류 묻힘의 흩어짐이다.

**가려내기 오차**: 너그러운 가정 아래에서 남는 가려내기 위험은 $O(1/\sqrt{K})$으로 줄어든다.

### 다른 방법과의 견줌

| 방법 | 원형 셈하기 | 가려내기 | 복잡도 |
|--------|----------------------|----------------|------------|
| 원형 망 | 평균 | 가장 가까운 무게중심 | $O(NK)$ |
| 맞춤 망 | 항등 | 주의로 무게 준 값 | $O(NK \cdot Q)$ |
| 관계 망 | 평균 | 배운 관계 | $O(N \cdot Q)$ |

## 실전 권고

### 거리 함수 고르기

**유클리드**(기본값):

- L2로 고른 묻힘과 잘 맞는다
- 차원이 높은 공간에서 더 낫다
- 셈이 효율적이다

**코사인**:

- 크기에 흔들리지 않는다
- 고르기 없이도 잘 굴러간다
- 묻힘의 크기가 들쭉날쭉할 때 도움이 될 수 있다

### 묻힘 고르기

```python
def forward_with_normalization(self, support, support_labels, query):
    # 부호화
    support_emb = self.encoder(support)
    query_emb = self.encoder(query)
    
    # L2로 고른다(성능이 좋아질 때가 많다)
    support_emb = F.normalize(support_emb, p=2, dim=1)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    
    # 앞먹임의 나머지...
```

### 데이터 늘리기

에피소드 학습 중에 세게 늘린다.

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(84, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## 실험 결과

### 잣대 성능(5-갈래 정확도)

| 방법 | Omniglot 1-예시 | Omniglot 5-예시 | mini-ImageNet 1-예시 | mini-ImageNet 5-예시 |
|--------|-----------------|-----------------|---------------------|---------------------|
| 맞춤 망 | 98.1% | 98.9% | 43.6% | 55.3% |
| **원형 망** | **98.8%** | **99.7%** | **49.4%** | **68.2%** |
| MAML | 98.7% | 99.9% | 48.7% | 63.1% |

## 요약

원형 망은 다음을 준다.

1. **단순함**: 부류의 평균을 셈하고 최근접 이웃으로 가려내면 그만이다
2. **효율**: 시험 때 되풀이 최적화가 없다
3. **든든한 성능**: 더 복잡한 방법과 겨룰 만하다
4. **이론적 바탕**: 브레그만 벌어짐 이론에 뿌리를 둔다

실무자를 위한 핵심 통찰은 다음과 같다.

- 시험 때보다 큰 N-갈래로 익혀라
- 고른 묻힘에 유클리드 거리의 제곱을 쓰라
- 복잡한 부류에는 원형을 여럿 두는 것을 생각해 보라
- 일반화에는 데이터 늘리기가 매우 중요하다

## 참고 문헌

1. Snell, J., et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
2. Fort, S. "Gaussian Prototypical Networks for Few-Shot Learning on Omniglot." CVPR Workshop 2017.
3. Allen, K., et al. "Infinite Mixture Prototypes for Few-Shot Learning." ICML 2019.
4. Ren, M., et al. "Meta-Learning for Semi-Supervised Few-Shot Classification." ICLR 2018.

## 연습문제

**연습문제 1.**
원형 망의 가려내기 규칙을 끌어내라.

??? success "연습문제 1 풀이"
    부류 원형을 셈한다. 곧 $c_k = \frac{1}{|S_k|}\sum_{(x,y)\in S_k} f_\phi(x)$이다. 물음 $x$을 원형까지의 거리로 가려낸다. 곧 $p(y=k|x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_j \exp(-d(f_\phi(x), c_j))}$이며 $d$은 대개 유클리드 거리의 제곱이다.

---

**연습문제 2.**
유클리드 거리의 제곱을 쓰는 원형 망이 묻힘 공간의 선형 가려내기와 같음을 증명하라.

??? success "연습문제 2 풀이"
    로그 확률은 $\log p(y=k|x) = -\|f(x) - c_k\|^2 + \text{const} = 2c_k^\top f(x) - \|c_k\|^2 + \text{const}$이다. 이는 가중치가 $w_k = 2c_k$이고 편향이 $b_k = -\|c_k\|^2$인 $f(x)$의 선형 함수이다. 그러므로 원형 망은 선형 가려내기만으로 충분한 표현을 배운다. $\square$

---

**연습문제 3.**
파이토치로 원형 망을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way):
        prototypes = [support_emb[support_labels == k].mean(0) for k in range(n_way)]
        prototypes = torch.stack(prototypes)  # (N, D)
        dists = torch.cdist(query_emb, prototypes)  # (Q, N)
        return F.cross_entropy(-dists, query_labels)
    ```

---

**연습문제 4.**
원형 망을 맞춤 망, 관계 망과 견주어라.

??? success "연습문제 4 풀이"
    원형 망: 부류 무게중심을 셈하고 유클리드 거리와 소프트맥스를 쓴다. 단순하고 쓸모 있다. 맞춤 망: 받침 집합 위에서 주의로 무게 준 kNN을 쓰며 받침 집합 전체를 쓴다. 관계 망: 신경망으로 거리 함수를 배운다. 원형 망은 단순해서 가장 널리 쓰이고, 관계 망이 가장 두루 쓰이며, 맞춤 망은 받침 집합의 크기가 바뀔 때 가장 잘 다룬다.
