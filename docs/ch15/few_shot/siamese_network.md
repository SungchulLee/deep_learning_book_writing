# 샴 네트워크

샴 네트워크는 웨이트를 공유하는 인코더로 두 입력을 임베딩하고, 같은 클래스는 가깝게 다른 클래스는 멀게 배치하도록 학습한다. 여기서는 대조 로스와 트리플렛 로스, 서포트의 최근접 이웃을 이용하는 원샷 분류를 살펴본다.

## 1. 코드

```python
"""
한 예시 학습을 위한 샴 망

샴 망은 보기 쌍 사이의 닮음 재기를 배운다.
웨이트를 나누어 쓰는 쌍둥이 망으로 입력을 묻은 다음
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
    샴 망을 위한 CNN 인코더.
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
            # 마지막 컨볼루션 켜의 자리 크기는 들임 크기에 따라 달라진다.
            # 1x1로 모아 두면 아래 선형 켜의 들임 수가 언제나 맞는다.
            nn.AdaptiveAvgPool2d(1),
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
        # 거리를 닮음으로 바꾸는 마지막 레이어를 더한다
        similarity = torch.sigmoid(distance.sum(dim=1))
        return similarity


class ContrastiveLoss(nn.Module):
    """
    샴 망을 익히는 대조 로스.
    
    로스 = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2
    
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
    세쌍 로스: 닻을 양의 보기 쪽으로 끌고 음의 보기에서 밀어낸다.
    
    로스 = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
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
    대조 로스를 쓰는 샴 망의 학습 스텝.
    
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
    
    # 로스를 계산한다
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
    
    # 대조 로스로 익히기
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 학습 배치 보기
    batch_size = 32
    # Koch의 샴 넷트웍은 105x105 Omniglot 그림을 바탕으로 한다
    x1 = torch.randn(batch_size, 1, 105, 105)
    x2 = torch.randn(batch_size, 1, 105, 105)
    labels = torch.randint(0, 2, (batch_size,)).float()  # 0=닮음, 1=닮지 않음
    
    loss = train_siamese(model, x1, x2, labels, optimizer, criterion)
    print(f"Training loss: {loss:.4f}")
    
    # 한 예시 분류 보기
    n_classes = 5
    n_query = 10
    support_set = torch.randn(n_classes, 1, 105, 105)
    support_labels = torch.arange(n_classes)
    query = torch.randn(n_query, 1, 105, 105)
    
    predictions = one_shot_classification(model, support_set, support_labels, query)
    print(f"Predictions: {predictions}")
    
    # 대안: 세쌍 로스로 익히기
    triplet_criterion = TripletLoss(margin=1.0)
    anchor = torch.randn(batch_size, 1, 105, 105)
    positive = torch.randn(batch_size, 1, 105, 105)
    negative = torch.randn(batch_size, 1, 105, 105)
    
    anchor_emb = model.encoder(anchor)
    positive_emb = model.encoder(positive)
    negative_emb = model.encoder(negative)
    
    triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
    print(f"Triplet loss: {triplet_loss.item():.4f}")
```

**출력:**

```
Training loss: 0.4928
Predictions: tensor([1, 2, 0, 1, 1, 2, 0, 0, 0, 3])
Triplet loss: 1.0038
```

## 2. 논의

학습 함수는 두 임베딩의 유클리드 거리 $d$를 계산한다. 이 코드의 라벨 규약은 0=같은 클래스, 1=다른 클래스이며 대조 로스는 $(1-y)d^2+y\max(m-d,0)^2$다. 다른 클래스 쌍은 거리가 마진 이상이면 로스가 0이 된다. 트리플렛 로스는 $\max(d(a,p)-d(a,n)+m,0)$이며, 여기서는 제곱 거리가 아닌 거리를 사용한다.

SiameseNetwork.forward는 스칼라 거리 대신 성분별 절댓값 차이를 반환한다. predict_similarity는 그 합에 sigmoid를 적용하므로 차이가 클수록 값이 커지고 최소값도 0.5다. 이름과 달리 같은 클래스일 확률로 해석할 수 없다. 학습 함수와 원샷 분류 함수는 이 메서드를 사용하지 않고 인코더의 유클리드 거리를 직접 계산한다.

주 예제는 105×105 흑백 입력에 맞춰져 있다. 큰 커널의 패딩 없는 컨볼루션 때문에 64×64 입력은 중간에 공간 크기가 부족해진다. 원샷 분류는 각 쿼리에서 가장 가까운 서포트의 라벨을 고르며, 여러 샷을 넣으면 클래스 평균이 아닌 표본 최근접 이웃 분류가 된다. 무작위 쌍과 무작위 라벨의 한 번 갱신으로 실제 유사도 학습 성능을 판단할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,1,105,105)의 컨볼루션·풀링 출력과 최종 임베딩 모양을 추적하라.

</div>

??? success "풀이"
    공간 크기는 105→96→48→42→21→18→9→6이다. 컨볼루션 출력 채널은 차례로 64,128,128,256이며 마지막 피처 맵은 (4,256,6,6)이다. 적응형 평균 풀링과 평탄화를 거쳐 (4,256), 선형 레이어와 sigmoid를 거쳐 (4,128)이 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
input_channels=3만 바꾸어 64×64 RGB 입력을 처리할 수 있는지 설명하라.

</div>

??? success "풀이"
    공간 크기는 64→55→27→21→10→7→3까지 줄어든 뒤 마지막 4×4 컨볼루션을 적용할 수 없다. 채널 변경 외에 커널·패딩·풀링 또는 입력 크기를 재설계해야 한다. 뒤의 적응형 풀링은 이미 발생한 컨볼루션 크기 오류를 해결하지 못한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 클래스 쌍과 다른 클래스 쌍의 대조 로스를 d=0.2, margin=1에서 계산하라.

</div>

??? success "풀이"
    같은 클래스는 y=0이므로 0.04, 다른 클래스는 y=1이므로 0.64다. 이 규약을 반대로 주면 학습 목적도 반대가 된다. 실제 데이터로 쌍을 만들 때 각 라벨과 원본 클래스의 관계를 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

공유 인코더의 거리 학습과 최근접 서포트 분류를 연결한다. 라벨 규약, 거리의 정의와 입력 크기를 확인하고 predict_similarity의 값을 유사도 확률로 해석하지 않는다.
