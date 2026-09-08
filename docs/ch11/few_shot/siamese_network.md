# 샴 망

한 예시 학습을 위한 샴 망. 샴 망은 보기 쌍 사이의 닮음 재기를 배운다.

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

## 1. 코드

```python
"""
한 예시 학습을 위한 샴 망

샴 망은 보기 쌍 사이의 닮음 재기를 배운다.
가중치를 나누어 쓰는 쌍둥이 망으로 입력을 묻은 다음
묻힘 사이의 닮음을 셈한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class SiameseEncoder(nn.Module):
    """
    샴 망을 위한 CNN 부호기.
    """
    def __init__(self, input_channels=1, hidden_dim=64, embedding_dim=128):
        super(SiameseEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 블록 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 블록 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 블록 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 블록 4
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, embedding_dim),
            nn.Sigmoid()  # 묻힘을 [0, 1]로 묶는다
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):
    """
    쌍 사이의 닮음을 셈하는 샴 망.
    """
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, x1, x2):
        """
        입력 쌍의 묻힘과 거리를 셈한다.
        
        인수:
            x1: 첫 입력 (batch_size, channels, height, width)
            x2: 둘째 입력 (batch_size, channels, height, width)
        
        반환값:
            distance: 묻힘 사이의 L1 거리
        """
        # 묻힘을 얻는다
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        
        # L1 거리를 셈한다
        distance = torch.abs(embedding1 - embedding2)
        
        return distance
    
    def predict_similarity(self, x1, x2):
        """
        닮음 점수를 맞힌다(0 = 다름, 1 = 같음).
        """
        distance = self.forward(x1, x2)
        # 거리를 닮음으로 바꾸는 마지막 층을 더한다
        similarity = torch.sigmoid(distance.sum(dim=1))
        return similarity


class ContrastiveLoss(nn.Module):
    """
    샴 망을 익히는 대조 손실.
    
    손실 = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2
    
    여기서 닮은 쌍이면 Y=0, 닮지 않은 쌍이면 Y=1이고,
    D은 묻힘 사이의 유클리드 거리이다.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distance, label):
        """
        인수:
            distance: 묻힘 사이의 유클리드 거리
            label: 닮은 쌍이면 0, 닮지 않은 쌍이면 1
        """
        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()


class TripletLoss(nn.Module):
    """
    세쌍 손실: 닻을 양의 보기 쪽으로 끌고 음의 보기에서 밀어낸다.
    
    손실 = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        인수:
            anchor: 닻 묻힘 (batch_size, embedding_dim)
            positive: 양의 묻힘(닻과 같은 부류)
            negative: 음의 묻힘(다른 부류)
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()


def train_siamese(model, x1, x2, labels, optimizer, criterion):
    """
    대조 손실을 쓰는 샴 망의 학습 걸음.
    
    인수:
        x1, x2: 입력 쌍
        labels: 닮으면 0, 닮지 않으면 1
    """
    model.train()
    optimizer.zero_grad()
    
    # 묻힘을 얻는다
    emb1 = model.encoder(x1)
    emb2 = model.encoder(x2)
    
    # 유클리드 거리를 셈한다
    distance = F.pairwise_distance(emb1, emb2)
    
    # 손실을 계산한다
    loss = criterion(distance, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def one_shot_classification(model, support_set, support_labels, query):
    """
    익힌 샴 망으로 한 예시 분류를 한다.
    
    인수:
        support_set: (n_classes, *input_shape) - 부류마다 보기 하나
        support_labels: (n_classes,) - 부류 이름표
        query: (n_query, *input_shape) - 가려낼 물음
    
    반환값:
        predictions: (n_query,) - 맞힌 부류 이름표
    """
    model.eval()
    n_classes = support_set.shape[0]
    n_query = query.shape[0]
    
    with torch.no_grad():
        # 받침 집합의 묻힘을 얻는다
        support_embeddings = model.encoder(support_set)
        
        predictions = []
        for i in range(n_query):
            # 이 물음의 묻힘을 얻는다
            query_embedding = model.encoder(query[i:i+1])
            
            # 모든 받침 보기까지의 거리를 셈한다
            distances = []
            for j in range(n_classes):
                dist = F.pairwise_distance(
                    query_embedding, 
                    support_embeddings[j:j+1]
                )
                distances.append(dist)
            
            distances = torch.stack(distances)
            
            # 거리가 가장 작은 부류를 내놓는다
            predicted_idx = torch.argmin(distances)
            predictions.append(support_labels[predicted_idx])
        
        return torch.tensor(predictions)


# 사용 예
if __name__ == "__main__":
    # 모델 생성
    encoder = SiameseEncoder(input_channels=1, hidden_dim=64, embedding_dim=128)
    model = SiameseNetwork(encoder)
    
    # 대조 손실로 익히기
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 학습 배치 보기
    batch_size = 32
    x1 = torch.randn(batch_size, 1, 28, 28)
    x2 = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 2, (batch_size,)).float()  # 0=닮음, 1=닮지 않음
    
    loss = train_siamese(model, x1, x2, labels, optimizer, criterion)
    print(f"Training loss: {loss:.4f}")
    
    # 한 예시 분류 보기
    n_classes = 5
    n_query = 10
    support_set = torch.randn(n_classes, 1, 28, 28)
    support_labels = torch.arange(n_classes)
    query = torch.randn(n_query, 1, 28, 28)
    
    predictions = one_shot_classification(model, support_set, support_labels, query)
    print(f"Predictions: {predictions}")
    
    # 대안: 세쌍 손실로 익히기
    triplet_criterion = TripletLoss(margin=1.0)
    anchor = torch.randn(batch_size, 1, 28, 28)
    positive = torch.randn(batch_size, 1, 28, 28)
    negative = torch.randn(batch_size, 1, 28, 28)
    
    anchor_emb = model.encoder(anchor)
    positive_emb = model.encoder(positive)
    negative_emb = model.encoder(negative)
    
    triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
    print(f"Triplet loss: {triplet_loss.item():.4f}")```

## 2. 논의

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 4개(`SiameseEncoder`, `SiameseNetwork`, `ContrastiveLoss`, `TripletLoss`)를 정한다. 클래스마다 서로 다른 부품을 감싸 코드를 모듈 방식으로 만들고 넓히기 쉽게 한다. `forward` 메서드가 파이토치가 자동 미분에 쓰는 계산 그래프를 정한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
`SiameseEncoder`의 앞먹임을 따라가며 텐서의 꼴을 좇아라. 기본 매개변수로 표본 4개짜리 배치를 넣었을 때 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`를 지금 값에서 3으로 바꾸어라. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$ 공식으로 합성곱과 풀링 층마다 그 뒤의 공간 차원을 다시 셈하라. 마지막 합성곱·풀링 층의 편 출력에 맞도록 첫 선형 층의 `in_features`를 고쳐라. `model = SiameseEncoder(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
`SiameseEncoder`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = SiameseEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.

## 정리하며

**다룬 것** — 샴 망

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 4개(`SiameseEncoder`, `SiameseNetwork`, `ContrastiveLoss`, `TripletLoss`)를 정한다.

핵심 클래스는 `SiameseEncoder`, `SiameseNetwork`, `ContrastiveLoss`, `TripletLoss`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
