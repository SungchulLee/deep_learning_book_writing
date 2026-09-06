# 한 예시 학습을 위한 샴 망
## 들어가며

샴 망은 한 예시 학습을 노리고 설계된 첫 신경망 구조에 든다. Bromley 외(1993)가 서명 검증을 위해 들여왔고 Koch 외(2015)가 한 예시 그림 알아보기로 널리 알렸다. 이 망은 배운 표현을 견주어 두 입력이 같은 부류에 드는지를 가려내는 법을 배운다.

## 구조 개관

### 쌍둥이 망 설계

샴 망은 같은 가중치를 나누어 쓰는 똑같은 부분 망 둘(쌍둥이)로 이루어진다. 부분 망마다 입력 하나를 다루고, 그 출력을 견주어 닮음을 가린다.

```
Input 1 ──→ [Encoder f_θ] ──→ Embedding 1 ──┐
                                            ├──→ [Distance/Similarity] ──→ Same/Different
Input 2 ──→ [Encoder f_θ] ──→ Embedding 2 ──┘
        (shared weights)
```

핵심 통찰은 **가중치 나누어 쓰기**이다. 두 입력이 똑같은 망을 지나므로 어느 가지가 다루든 닮은 입력은 닮은 묻힘을 낸다.

### 수식으로 나타내기

두 입력 $x_1$과 $x_2$이 주어지면 샴 망은 다음을 셈한다.

**묻힘**:

$$z_1 = f_\theta(x_1), \quad z_2 = f_\theta(x_2)$$

**닮음과 거리**:

$$s(x_1, x_2) = g(|z_1 - z_2|)$$

여기서 $f_\theta$은 부호기 망이고, $|z_1 - z_2|$은 성분마다의 절대 차이를 셈하며, $g$은 대개 그 차이를 닮음 점수로 옮기는 배운 함수이다.

### 가중치 나누어 쓰기가 중요한 까닭

가중치를 나누어 쓰면 배운 묻힘 함수가 다음을 갖추게 된다.

1. **대칭적**: 입력의 차례와 상관없이 같은 변환이 쓰인다
2. **한결같음**: 닮은 입력은 늘 닮은 자리로 옮겨진다
3. **효율적**: 망을 하나만 담아 두고 익히면 된다

## 부호기 구조

### 그림을 위한 합성곱 부호기

Omniglot을 위한 Koch 외의 본디 구조는 다음과 같다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseConvEncoder(nn.Module):
    """
    샴 망을 위한 합성곱 부호기.
    
    구조는 Koch 외(2015)를 따르되
    입력 크기에 맞추어 손질했다.
    """
    
    def __init__(
        self, 
        input_channels: int = 1, 
        input_size: int = 105,
        hidden_dim: int = 64,
        embedding_dim: int = 4096
    ):
        """
        인수:
            input_channels: 입력 채널 수(흑백이면 1)
            input_size: 입력 그림 크기(정사각 그림으로 놓는다)
            hidden_dim: 합성곱 거르개의 바탕 개수
            embedding_dim: 마지막 묻힘 차원
        """
        super().__init__()
        
        # 거르개 크기가 커지는 합성곱 층
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=10)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=7)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4)
        self.conv4 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4)
        
        # 합성곱 뒤의 편 크기를 셈한다
        self._conv_output_size = self._get_conv_output_size(
            input_channels, input_size
        )
        
        # 묻힘으로 가는 온연결 층
        self.fc = nn.Linear(self._conv_output_size, embedding_dim)
    
    def _get_conv_output_size(self, channels: int, size: int) -> int:
        """합성곱 층 뒤의 출력 크기를 셈한다."""
        with torch.no_grad():
            dummy = torch.zeros(1, channels, size, size)
            dummy = self._conv_forward(dummy)
            return dummy.view(1, -1).size(1)
    
    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """합성곱 층을 씌운다."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv4(x))
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력의 묻힘을 셈한다.
        
        인수:
            x: 입력 그림 (batch, channels, height, width)
        
        반환값:
            embeddings: (batch, embedding_dim)
        """
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        
        return x
```

### ResNet 등뼈를 쓴 오늘날의 부호기

```python
import torchvision.models as models

class ResNetSiameseEncoder(nn.Module):
    """
    미리 학습된 ResNet 등뼈를 쓰는 샴 부호기.
    """
    
    def __init__(
        self, 
        pretrained: bool = True,
        embedding_dim: int = 256
    ):
        super().__init__()
        
        # 미리 학습된 ResNet을 불러온다
        resnet = models.resnet18(pretrained=pretrained)
        
        # 마지막 가려내기 층을 없앤다
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 묻힘 공간으로 쏘아 넣는 머리
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """묻힘을 셈한다."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        
        # 묻힘을 L2로 고른다
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
```

## 온전한 샴 망

### 표준 구현

```python
class SiameseNetwork(nn.Module):
    """
    닮음 학습을 위한 온전한 샴 망.
    
    배운 묻힘을 견주어 두 입력이
    같은 부류에 드는지를 셈한다.
    """
    
    def __init__(self, encoder: nn.Module, use_distance: bool = True):
        """
        인수:
            encoder: 나누어 쓰는 부호기 망
            use_distance: True이면 거리를 내고, 아니면 닮음을 낸다
        """
        super().__init__()
        self.encoder = encoder
        self.use_distance = use_distance
        
        # 선택: 거리 위에 함수를 하나 더 배운다
        if hasattr(encoder, 'embedding_dim'):
            embed_dim = encoder.embedding_dim
        else:
            embed_dim = 4096  # 기본값
        
        self.similarity_network = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """입력 하나를 부호로 바꾼다."""
        return self.encoder(x)
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        두 입력 사이의 닮음을 셈한다.
        
        인수:
            x1: 첫 입력 배치 (batch, *input_shape)
            x2: 둘째 입력 배치 (batch, *input_shape)
        
        반환값:
            use_distance이면 L2 거리 (batch,)
            아니면 [0, 1] 안의 닮음 점수 (batch,)
        """
        # 두 입력을 모두 부호로 바꾼다
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        if self.use_distance:
            # L2 거리
            distance = F.pairwise_distance(z1, z2, p=2)
            return distance
        else:
            # 절대 차이 위에서 배운 닮음
            diff = torch.abs(z1 - z2)
            similarity = self.similarity_network(diff)
            return similarity.squeeze(-1)
    
    def predict_same_class(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        입력이 같은 부류에서 왔는지 맞힌다.
        
        인수:
            x1, x2: 입력 배치
            threshold: 판단 문턱값
        
        반환값:
            predictions: 참거짓 텐서 (batch,)
        """
        if self.use_distance:
            distances = self.forward(x1, x2)
            return distances < threshold
        else:
            similarities = self.forward(x1, x2)
            return similarities > threshold
```

## 손실 함수

### 대조 손실

샴 망을 익히는 표준 손실은 다음과 같다.

$$\mathcal{L}(z_1, z_2, y) = (1-y) \cdot \frac{1}{2} d^2 + y \cdot \frac{1}{2} \max(0, m - d)^2$$

여기서 닮은 쌍이면 $y=0$이고 닮지 않은 쌍이면 $y=1$이다.

```python
class ContrastiveLoss(nn.Module):
    """
    샴 망을 위한 대조 손실.
    
    닮은 쌍은 끌어당기고 닮지 않은 쌍은
    여백 너머로 밀어낸다.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        인수:
            margin: 닮지 않은 쌍의 가장 작은 거리
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        대조 손실을 셈한다.
        
        인수:
            z1: 첫 묻힘 (batch, embed_dim)
            z2: 둘째 묻힘 (batch, embed_dim)
            y: 이름표 - 닮으면 0, 닮지 않으면 1
        
        반환값:
            loss: 스칼라 손실 값
        """
        # 유클리드 거리를 셈한다
        distance = F.pairwise_distance(z1, z2, p=2)
        
        # 닮은 쌍의 손실: 거리를 가장 작게 한다
        similar_loss = (1 - y) * 0.5 * distance ** 2
        
        # 닮지 않은 쌍의 손실: 여백까지 거리를 가장 크게 한다
        dissimilar_loss = y * 0.5 * F.relu(self.margin - distance) ** 2
        
        loss = similar_loss + dissimilar_loss
        
        return loss.mean()

class MarginContrastiveLoss(nn.Module):
    """
    닮은 쌍과 닮지 않은 쌍에 따로 여백을 두는 변형.
    """
    
    def __init__(
        self, 
        pos_margin: float = 0.0, 
        neg_margin: float = 1.0
    ):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        distance = F.pairwise_distance(z1, z2, p=2)
        
        # 닮은 쌍은 pos_margin보다 가까워야 한다
        pos_loss = (1 - y) * F.relu(distance - self.pos_margin) ** 2
        
        # 닮지 않은 쌍은 neg_margin보다 멀어야 한다
        neg_loss = y * F.relu(self.neg_margin - distance) ** 2
        
        return (pos_loss + neg_loss).mean()
```

### 이진 교차 엔트로피 손실

닮음을 이진 분류로 다루는 다른 손실이다.

```python
class SiameseBCELoss(nn.Module):
    """
    샴 닮음 예측을 위한 이진 교차 엔트로피 손실.
    
    문제를 이진 분류로 다룬다.
    이 쌍은 같은 부류에서 왔는가?
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        similarity: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        인수:
            similarity: [0, 1] 안의 맞힌 닮음
            y: 이름표 - 같은 부류면 1, 다르면 0
        """
        # 선택: 이름표 매끄럽게 하기
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return F.binary_cross_entropy(similarity, y.float())
```

## 한 예시 분류

### 견주어 가려내기

한 예시 분류에서는 물음 그림을 부류마다의 보기 하나와 견준다.

```python
def one_shot_classify(
    model: SiameseNetwork,
    support_set: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    샴 망으로 한 예시 분류를 한다.
    
    인수:
        model: 익힌 샴 망
        support_set: 부류마다 보기 하나 (n_classes, *input_shape)
        support_labels: 부류 이름표 (n_classes,)
        query: 가려낼 물음 그림 (n_query, *input_shape)
        device: 셈할 장치
    
    반환값:
        predictions: 맞힌 부류 이름표 (n_query,)
    """
    model.eval()
    model = model.to(device)
    
    support_set = support_set.to(device)
    query = query.to(device)
    
    n_classes = support_set.size(0)
    n_query = query.size(0)
    
    predictions = []
    
    with torch.no_grad():
        # 받침 집합을 한 번만 부호로 바꾼다
        support_embeddings = model.forward_one(support_set)
        
        for i in range(n_query):
            query_embedding = model.forward_one(query[i:i+1])
            
            # 받침 보기마다의 거리를 셈한다
            distances = []
            for j in range(n_classes):
                dist = F.pairwise_distance(
                    query_embedding,
                    support_embeddings[j:j+1]
                )
                distances.append(dist.item())
            
            # 거리가 가장 작은 부류를 내놓는다
            pred_idx = np.argmin(distances)
            predictions.append(support_labels[pred_idx].item())
    
    return torch.tensor(predictions, device=device)

def one_shot_classify_efficient(
    model: SiameseNetwork,
    support_set: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor
) -> torch.Tensor:
    """
    배치 연산을 쓰는 효율적인 한 예시 분류.
    """
    model.eval()
    
    n_classes = support_set.size(0)
    n_query = query.size(0)
    
    with torch.no_grad():
        # 모든 보기를 부호로 바꾼다
        support_embeddings = model.forward_one(support_set)  # (C, D)
        query_embeddings = model.forward_one(query)  # (Q, D)
        
        # 모든 쌍별 거리 계산
        # (Q, C) 거리 행렬
        distances = torch.cdist(query_embeddings, support_embeddings, p=2)
        
        # 물음마다 가장 가까운 받침 보기를 얻는다
        nearest_idx = distances.argmin(dim=1)
        predictions = support_labels[nearest_idx]
    
    return predictions
```

## 학습 전략

### 쌍 뽑기

제대로 익히려면 닮은 쌍과 닮지 않은 쌍을 고르게 뽑아야 한다.

```python
import random
from collections import defaultdict

class PairSampler:
    """
    샴 망 학습을 위해 고르게 쌍을 뽑는다.
    """
    
    def __init__(
        self, 
        labels: torch.Tensor,
        positive_ratio: float = 0.5
    ):
        """
        인수:
            labels: 모든 부류 이름표
            positive_ratio: 뽑을 닮은 쌍의 비율
        """
        self.labels = labels
        self.positive_ratio = positive_ratio
        
        # 부류에서 첨자로 가는 대응을 만든다
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label.item()].append(idx)
        
        self.classes = list(self.class_indices.keys())
    
    def sample_pairs(self, batch_size: int):
        """
        쌍의 배치를 뽑는다.
        
        반환값:
            idx1, idx2: 짝지은 보기의 첨자
            labels: 같은 부류면 0, 다른 부류면 1
        """
        idx1, idx2, pair_labels = [], [], []
        
        n_positive = int(batch_size * self.positive_ratio)
        n_negative = batch_size - n_positive
        
        # 양의 쌍을 뽑는다(같은 부류)
        for _ in range(n_positive):
            # 보기가 둘 이상인 부류를 고른다
            valid_classes = [c for c in self.classes 
                          if len(self.class_indices[c]) >= 2]
            cls = random.choice(valid_classes)
            
            # 이 부류에서 서로 다른 보기 둘을 뽑는다
            i1, i2 = random.sample(self.class_indices[cls], 2)
            
            idx1.append(i1)
            idx2.append(i2)
            pair_labels.append(0)  # 같은 부류
        
        # 음의 쌍을 뽑는다(다른 부류)
        for _ in range(n_negative):
            # 서로 다른 부류 둘을 고른다
            cls1, cls2 = random.sample(self.classes, 2)
            
            i1 = random.choice(self.class_indices[cls1])
            i2 = random.choice(self.class_indices[cls2])
            
            idx1.append(i1)
            idx2.append(i2)
            pair_labels.append(1)  # 다른 부류
        
        # 뒤섞는다
        combined = list(zip(idx1, idx2, pair_labels))
        random.shuffle(combined)
        idx1, idx2, pair_labels = zip(*combined)
        
        return (
            torch.tensor(idx1),
            torch.tensor(idx2),
            torch.tensor(pair_labels)
        )

class HardPairMiner:
    """
    더 야무진 학습을 위한 실시간 어려운 쌍 캐기.
    """
    
    def __init__(self, model: SiameseNetwork, margin: float = 0.5):
        self.model = model
        self.margin = margin
    
    def mine_hard_pairs(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ):
        """
        어려운 양의 쌍과 음의 쌍을 찾는다.
        
        어려운 양의 쌍: 같은 부류인데 멀리 떨어져 있다
        어려운 음의 쌍: 다른 부류인데 가까이 있다
        """
        n = embeddings.size(0)
        
        # 모든 쌍별 거리 계산
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # 이름표 가리개를 만든다
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        hard_pos_pairs = []
        hard_neg_pairs = []
        
        for i in range(n):
            # 어려운 양의 보기: 같은 부류에서 거리가 가장 먼 것
            pos_mask = labels_eq[i].clone()
            pos_mask[i] = False  # 자기 자신 제외
            
            if pos_mask.any():
                pos_distances = distances[i][pos_mask]
                hardest_pos_idx = pos_mask.nonzero()[pos_distances.argmax()]
                
                if distances[i, hardest_pos_idx] > self.margin:
                    hard_pos_pairs.append((i, hardest_pos_idx.item(), 0))
            
            # 어려운 음의 보기: 다른 부류에서 거리가 가장 가까운 것
            neg_mask = ~labels_eq[i]
            
            if neg_mask.any():
                neg_distances = distances[i][neg_mask]
                hardest_neg_idx = neg_mask.nonzero()[neg_distances.argmin()]
                
                if distances[i, hardest_neg_idx] < self.margin:
                    hard_neg_pairs.append((i, hardest_neg_idx.item(), 1))
        
        return hard_pos_pairs, hard_neg_pairs
```

### 완전한 학습 루프

```python
class SiameseTrainer:
    """
    샴 망의 학습 되돌이.
    """
    
    def __init__(
        self,
        model: SiameseNetwork,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.train_labels = train_labels.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.pair_sampler = PairSampler(train_labels)
    
    def train_epoch(
        self, 
        n_iterations: int = 1000,
        batch_size: int = 32,
        log_interval: int = 100
    ):
        """
        한 세대를 학습한다.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for iteration in range(n_iterations):
            # 쌍을 뽑는다
            idx1, idx2, labels = self.pair_sampler.sample_pairs(batch_size)
            
            x1 = self.train_data[idx1]
            x2 = self.train_data[idx2]
            labels = labels.float().to(self.device)
            
            # 순전파
            z1 = self.model.encoder(x1)
            z2 = self.model.encoder(x2)
            
            # 손실을 계산한다
            loss = self.loss_fn(z1, z2, labels)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 정확도를 셈한다(살펴보기용)
            with torch.no_grad():
                distances = F.pairwise_distance(z1, z2)
                predictions = (distances > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += batch_size
            
            if (iteration + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                accuracy = correct / total
                print(f"Iteration {iteration+1}/{n_iterations} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
                total_loss = 0
                correct = 0
                total = 0
        
        return total_loss / max(n_iterations % log_interval, 1)
    
    def evaluate(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        n_way: int = 5,
        n_episodes: int = 100
    ):
        """
        한 예시 분류 성능을 평가한다.
        """
        self.model.eval()
        
        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # 서로 다른 부류를 얻는다
        unique_classes = torch.unique(test_labels)
        
        accuracies = []
        
        for _ in range(n_episodes):
            # n_way개의 부류를 뽑는다
            episode_classes = unique_classes[
                torch.randperm(len(unique_classes))[:n_way]
            ]
            
            # 부류마다 받침 1개와 물음 1개를 뽑는다
            support_data = []
            query_data = []
            query_labels = []
            
            for new_label, cls in enumerate(episode_classes):
                cls_mask = test_labels == cls
                cls_indices = cls_mask.nonzero(as_tuple=True)[0]
                
                perm = cls_indices[torch.randperm(len(cls_indices))]
                support_idx = perm[0]
                query_idx = perm[1] if len(perm) > 1 else perm[0]
                
                support_data.append(test_data[support_idx])
                query_data.append(test_data[query_idx])
                query_labels.append(new_label)
            
            support = torch.stack(support_data)
            query = torch.stack(query_data)
            query_labels = torch.tensor(query_labels, device=self.device)
            support_labels = torch.arange(n_way, device=self.device)
            
            # 분류
            predictions = one_shot_classify_efficient(
                self.model, support, support_labels, query
            )
            
            acc = (predictions == query_labels).float().mean().item()
            accuracies.append(acc)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        return mean_acc, std_acc
```

## 변형과 확장

### 세쌍 샴 망

샴 구조에 세쌍 손실을 곁들인다.

```python
class TripletSiameseNetwork(nn.Module):
    """
    세쌍 손실로 익힌 샴 망.
    """
    
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ):
        """
        세쌍의 묻힘을 셈한다.
        """
        z_a = self.encoder(anchor)
        z_p = self.encoder(positive)
        z_n = self.encoder(negative)
        
        return z_a, z_p, z_n
    
    def triplet_loss(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor,
        margin: float = 0.2
    ):
        """세쌍 손실을 셈한다."""
        z_a, z_p, z_n = self.forward(anchor, positive, negative)
        
        d_ap = F.pairwise_distance(z_a, z_p)
        d_an = F.pairwise_distance(z_a, z_n)
        
        loss = F.relu(d_ap - d_an + margin)
        
        return loss.mean()
```

### 영역을 넘나드는 샴 망

영역 적응 상황을 위한 것이다.

```python
class CrossDomainSiamese(nn.Module):
    """
    영역마다 따로 부호기를 두는 샴 망.
    
    서로 다른 영역의 보기를 견줄 때 쓸모 있다
    (이를테면 스케치와 사진 맞추기).
    """
    
    def __init__(
        self, 
        encoder_a: nn.Module, 
        encoder_b: nn.Module,
        shared_dim: int = 256
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        
        # 함께 쓰는 공간으로 쏘아 넣는다
        self.project_a = nn.Linear(encoder_a.embedding_dim, shared_dim)
        self.project_b = nn.Linear(encoder_b.embedding_dim, shared_dim)
    
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        두 영역의 보기를 견준다.
        """
        z_a = self.project_a(self.encoder_a(x_a))
        z_b = self.project_b(self.encoder_b(x_b))
        
        # L2로 고른다
        z_a = F.normalize(z_a, p=2, dim=1)
        z_b = F.normalize(z_b, p=2, dim=1)
        
        return z_a, z_b
```

## 요약

샴 망은 다음으로 한 예시 학습의 바탕이 되는 접근법을 준다.

1. **견주어 닮음 배우기**: 가중치를 나누어 쓰는 쌍둥이 망
2. **최근접 이웃 가려내기**: 가장 닮은 받침 보기를 찾아 물음을 가려낸다
3. **새 부류로 옮겨 가기**: 닮음 함수가 학습 부류 너머로 일반화된다

구현에서 살펴야 할 핵심은 다음과 같다.

- 학습 중에 양의 쌍과 음의 쌍을 고르게 맞추라
- 더 야무진 학습을 위해 어려운 음의 보기 캐기를 쓰라
- 더 나은 표현을 위해 미리 학습된 등뼈를 쓰는 것을 생각해 보라
- 학습 중에 대조 손실과 한 예시 정확도를 함께 살피라

## 참고 문헌

1. Bromley, J., et al. "Signature Verification using a Siamese Time Delay Neural Network." NeurIPS 1993.
2. Koch, G., et al. "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop 2015.
3. Chopra, S., et al. "Learning a Similarity Metric Discriminatively, with Application to Face Verification." CVPR 2005.
4. Schroff, F., et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR 2015.

## 연습문제

**연습문제 1.**
샴 망 구조와 대조 손실을 설명하라.

??? success "연습문제 1 풀이"
    샴 망은 두 입력을 (가중치를 나누어 쓰는) 똑같은 망으로 다루어 묻힘 $f(x_1), f(x_2)$을 낸다. 대조 손실은 $L = y\|f(x_1)-f(x_2)\|^2 + (1-y)\max(0, m - \|f(x_1)-f(x_2)\|)^2$이며, 같은 부류면 $y=1$, 다른 부류면 $y=0$이고 $m$은 여백이다.

---

**연습문제 2.**
거리 학습에서 대조 손실과 세쌍 손실을 견주어라.

??? success "연습문제 2 풀이"
    대조 손실: 보기의 쌍을 쓰며 같은 부류는 끌어당기고 다른 부류는 밀어낸다. 세쌍 손실: (닻, 양, 음) 세쌍을 쓰며 $L = \max(0, d(a,p) - d(a,n) + m)$이다. 세쌍 손실은 상대 거리를 살피므로 더 나은 묻힘을 낼 때가 많지만, 알맹이 있는 세쌍을 조심스레 캐내야 한다.

---

**연습문제 3.**
파이토치로 한 예시 검증을 위한 샴 망을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    class SiameseNet(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x1, x2):
            e1, e2 = self.backbone(x1), self.backbone(x2)
            return F.pairwise_distance(e1, e2)
    # 대조 손실
    loss = y * dist**2 + (1-y) * F.relu(margin - dist)**2
    ```

---

**연습문제 4.**
어려운 음의 보기 캐기란 무엇이며 샴 망을 익히는 데 왜 중요한가?

??? success "연습문제 4 풀이"
    어려운 음의 보기란 묻힘 공간에서 가까이 놓인 다른 부류의 쌍이다(모델이 헷갈려 한다). 아무렇게나 고른 음의 보기는 너무 쉬워서 기울기 신호를 거의 주지 못할 때가 많다. 어려운 음의 보기 캐기는 알맹이가 가장 많은 음의 보기를 골라 학습 효율을 크게 끌어올린다. 전략으로는 실시간 어려운 캐기(배치 안에서 가장 어려운 것), 반쯤 어려운 캐기(여백 안), 미리 해 두는 캐기가 있다.
