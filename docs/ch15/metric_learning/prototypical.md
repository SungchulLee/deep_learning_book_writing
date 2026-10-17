# 프로토타입 네트워크

프로토타입 네트워크는 클래스별 서포트 임베딩의 평균을 계산하고 쿼리를 가까운 평균으로 분류한다. Snell 등(2017)이 제안한 방법으로, 에피소드마다 출력 클래스의 대표 벡터를 새로 계산한다. 이 페이지에서는 거리 기반 확률, 평균의 수학적 성질과 여러 확장 구조를 살펴본다.

## 1. 핵심 개념

### 클래스별 대표 벡터

임베딩 함수 $f_\theta$와 클래스 c의 서포트 집합 $S_c$로 프로토타입을 계산한다.

$$p_c=\frac1{|S_c|}\sum_{(x,y)\in S_c}f_\theta(x)$$

쿼리 x의 클래스 확률은 음의 거리를 소프트맥스로 변환한다. 온도 T는 양수다.

$$p(y=c\mid x,S)=\frac{\exp[-d(f_\theta(x),p_c)/T]}{\sum_j\exp[-d(f_\theta(x),p_j)/T]}$$

거리로 제곱 유클리드 거리를 쓰면 각 클래스가 하나의 대표 벡터로 요약된다. 평균은 여러 표본의 잡음을 줄일 수 있지만 이상치에 민감하며 여러 군집으로 이루어진 클래스를 충분히 나타내지 못할 수 있다. 프로토타입 계산은 데이터 어그멘테이션과는 다른 연산이다.

## 2. 수학적 기초

### 브레그만 발산과 평균

미분 가능하고 엄격히 볼록한 함수 φ의 브레그만 발산은 다음과 같다.

$$D_\phi(z,p)=\phi(z)-\phi(p)-(z-p)^\top\nabla\phi(p)$$

첫 번째 인수에 표본 z를 두고 두 번째 인수 p를 대표점으로 최적화하면 산술평균이 발산 합의 최솟값을 달성한다. 이는 발산의 방향에 의존하는 성질이며 임의의 데이터에서 평균 기반 분류기가 최적이라는 뜻은 아니다. φ(z)=‖z‖²이면 제곱 유클리드 거리가 된다.

### 가우시안 모형으로 해석하기

임베딩 z의 클래스 조건부 분포가 평균 p_c, 공분산 σ²I인 가우시안이고 클래스 사전확률이 같으면 다음과 같다.

$$p(c\mid z)=\operatorname{softmax}_c\left(-\frac{\|z-p_c\|^2}{2\sigma^2}\right)$$

이는 임베딩 공간의 확률 모형이다. 일반적인 뉴럴 넷트웍 f의 출력에 가우시안 밀도를 적용하는 것만으로 원래 입력 x의 정규화된 밀도 p(x|c)가 정의되는 것은 아니다.

## 3. PyTorch 구현

### 프로토타입 네트워크 구현

라벨은 0부터 연속적으로 매기고 각 클래스에 서포트가 있어야 한다. temperature는 양수여야 한다. 서포트와 쿼리를 함께 인코딩하므로 학습 모드의 BatchNorm 통계는 쿼리에도 의존한다. 반환값은 확률이 아닌 로짓이다.

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
        에피소드의 로스와 정확도를 셈한다.
        
        반환값:
            loss: 교차 엔트로피 로스
            accuracy: 가려내기 정확도
        """
        logits = self.forward(support, support_labels, query)
        loss = F.cross_entropy(logits, query_labels)
        
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss, accuracy
```

### 컨볼루션 인코더 구조

Conv4Encoder는 네 번 풀링한 뒤 적응형 평균 풀링을 적용하므로 최종 차원은 hidden_dim이다. 단순 평탄화 인코더와 달리 마지막 공간 크기에 비례해 차원이 늘지 않는다. 다만 네 번 풀링할 수 있을 만큼 입력이 커야 한다.

```python
class Conv4Encoder(nn.Module):
    """
    소수 예시 학습에서 흔히 쓰는 4레이어 컨볼루션 인코더.
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

---

## 4. 학습 절차

### 에피소드 학습 루프

sample_episode는 별도로 정의해야 하며 클래스·표본 추출과 로컬 라벨을 일치시켜야 한다. 입력은 device로 이동하지만 함수가 모델을 이동시키지는 않는다. n_episodes는 양수여야 한다.

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
        
        # 피드포워드와 되돌림
        optimizer.zero_grad()
        loss, acc = model.loss(support, support_labels, query, query_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes
```

### 학습 설정

N-way, K-shot, 클래스별 쿼리 수와 에피소드 수를 명시한다. 학습 N을 테스트보다 크게 두는 방법도 있지만 데이터셋과 인코더에 따라 효과가 달라지므로 검증 과제에서 선택한다. 온도와 임베딩 스케일은 예측 확률과 경사 크기에 영향을 준다. 모델 비교에는 같은 클래스 분할과 평가 에피소드를 사용한다.

---

## 5. 변형과 확장

### 클래스별 다중 프로토타입

아래 코드는 클래스별 KMeans로 고정 상한 개수의 군집을 계산한다. 원 논문의 비모수적 무한 혼합 모형을 구현한 것은 아니다. forward와 최종 클래스 확률 계산도 없다. 경사가 필요한 텐서를 그대로 numpy()로 변환하면 오류가 나며, detach 후 변환해도 군집화 경로의 자동 미분은 끊어진다.

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

### 어텐션으로 보정하는 프로토타입

이 구현의 어텐션은 프로토타입을 쿼리로, 서포트 임베딩을 키·값으로 사용한다. 변수 이름과 달리 분류할 쿼리 임베딩을 키·값으로 사용하지 않는다. 다만 공동 인코딩의 BatchNorm 통계를 통해 쿼리 배치가 간접 영향을 줄 수 있다.

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

### 준지도 프로토타입 네트워크

라벨 없는 표본을 현재 프로토타입에 소프트 할당하고 가중 평균으로 대표 벡터를 반복 보정한다. 미지 클래스가 섞이거나 초기 할당이 부정확하면 프로토타입이 잘못 이동할 수 있다. 쿼리 라벨은 사용하지 않으며 라벨 없는 집합과 평가 집합의 관계를 명시한다.

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

---

## 6. 이론적 분석

### 평균 추정의 오차

고정된 임베딩 공간에서 같은 클래스의 서포트가 독립·동일 분포이고 공분산 Σ가 유한하면 표본 평균의 오차는 다음과 같다.

$$\mathbb E\|\hat p_c-\mu_c\|^2=\frac{\operatorname{tr}\Sigma_c}{K}$$

차원별 분산이 σ²로 같으면 dσ²/K다. 평균 추정 오차가 줄어도 클래스 간 겹침이나 인코더의 바이어스는 남는다. 이 식만으로 분류 오차가 특정 속도로 줄어든다고 결론낼 수 없다.

### 분류 계산 비용

인코더 비용을 제외하고 N개 클래스, 클래스당 K개 서포트, Q개 전체 쿼리, d차원 임베딩을 가정한다. 프로토타입 계산은 O(NKd), 쿼리 거리 계산은 O(QNd)다. 모든 서포트와 비교하는 매칭 방식의 유사도 계산은 O(QNKd)다. 학습된 관계 함수를 쓰면 그 함수의 계산량도 포함한다.

## 7. 실전 권고

### 거리와 임베딩 스케일

제곱 유클리드 거리는 방향과 크기를 함께 반영하고 코사인 거리는 정규화를 통해 방향을 비교한다. 두 벡터가 단위 벡터이면 제곱 유클리드 거리는 코사인 거리의 두 배다. 하지만 정규화한 표본의 평균은 보통 단위 벡터가 아니므로 평균 뒤 재정규화 여부도 구분해야 한다.

### 임베딩 정규화

아래 함수는 정규화까지만 작성된 조각이다. 프로토타입, 거리와 반환값 계산은 추가해야 한다.

```python
def forward_with_normalization(self, support, support_labels, query):
    # 부호화
    support_emb = self.encoder(support)
    query_emb = self.encoder(query)
    
    # L2로 고른다(성능이 좋아질 때가 많다)
    support_emb = F.normalize(support_emb, p=2, dim=1)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    
    # 피드포워드의 나머지...
```

### 데이터 어그멘테이션

아래 변환은 RGB 이미지용 예시다. transforms 임포트가 필요하며 흑백 입력에 3채널 정규화 통계를 그대로 쓸 수 없다. 의미를 바꾸는 변형은 피하고 어그멘테이션 강도를 검증한다.

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

---

## 8. 실험 결과

논문의 정확도를 비교할 때는 클래스 분할, 인코더, 이미지 크기, 학습·테스트 N-way와 K-shot, 평가 에피소드 수 및 신뢰구간을 함께 확인한다. 서로 다른 논문에서 조건 없이 모은 수치를 직접적인 우열로 해석하지 않는다. 이 페이지의 코드로 측정한 벤치마크 결과는 제공하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
프로토타입 계산과 거리 기반 클래스 확률을 설명하라.

</div>

??? success "풀이"
    각 클래스의 서포트 임베딩을 평균 내어 대표 벡터를 만들고, 쿼리에서 각 대표 벡터까지 음의 거리를 로짓으로 사용한다. 온도로 나눈 뒤 소프트맥스를 적용한다. 학습에는 로짓을 교차 엔트로피에 직접 전달한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
제곱 유클리드 거리를 사용할 때 결정 경계가 임베딩 공간에서 선형임을 보여라.

</div>

??? success "풀이"
    $-\|z-p_c\|^2=-\|z\|^2+2p_c^\top z-\|p_c\|^2$이다. 첫 항은 모든 클래스에 공통이므로 소프트맥스와 argmax에서 제거할 수 있다. 유효 로짓은 $2p_c^\top z-\|p_c\|^2$다. 로그 확률 자체는 정규화 항 때문에 z의 선형 함수가 아니며, 원래 입력 공간의 경계도 일반적으로 선형이 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 로스 함수가 본문의 거리 정의와 같은지 확인하고 필요한 입력 조건을 설명하라.

</div>

??? success "풀이"
    torch.cdist의 기본값은 유클리드 거리이며 제곱하지 않았으므로 본문의 제곱 거리 로짓과 로스가 다르다. 순위는 같아도 확률과 경사는 달라진다. support_labels에는 0부터 n_way−1까지 빈 클래스가 없어야 한다.

    ```python
    def prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way):
        prototypes = [support_emb[support_labels == k].mean(0) for k in range(n_way)]
        prototypes = torch.stack(prototypes)  # (N, D)
        dists = torch.cdist(query_emb, prototypes)  # (Q, N)
        return F.cross_entropy(-dists, query_labels)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
프로토타입·매칭·관계 네트워크의 비교 단위를 설명하라.

</div>

??? success "풀이"
    프로토타입 네트워크는 클래스 평균과 거리를 사용한다. 매칭 네트워크는 개별 서포트의 유사도 웨이트로 라벨을 합산한다. 관계 네트워크는 피처 쌍의 관계 점수를 뉴럴 넷트웍으로 학습한다. 각 방법의 정확도와 비용은 인코더·에피소드 구성·비교 함수에 따라 달라진다.

## 정리하며

프로토타입 네트워크는 서포트 평균으로 새 클래스의 대표 벡터를 만들고 쿼리를 거리로 분류한다. 평균의 추정 성질과 분류 정확도 보장은 구분한다. 거리·온도·정규화와 에피소드 평가 조건을 맞춰 비교하고, 확장 코드의 미분 경로와 구현 범위를 확인한다.

**참고 문헌**

1. Snell, J., et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017.
2. Fort, S. "Gaussian Prototypical Networks for Few-Shot Learning on Omniglot." CVPR Workshop 2017.
3. Allen, K., et al. "Infinite Mixture Prototypes for Few-Shot Learning." ICML 2019.
4. Ren, M., et al. "Meta-Learning for Semi-Supervised Few-Shot Classification." ICLR 2018.
