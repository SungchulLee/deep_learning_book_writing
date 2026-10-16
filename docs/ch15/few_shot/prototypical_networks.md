# 프로토타입 네트워크

프로토타입 네트워크는 서포트 표본의 임베딩을 클래스별로 평균 내고, 쿼리를 가장 가까운 평균에 배정한다. 에피소드마다 새 분류 기준을 계산하므로 테스트 클래스마다 별도의 출력 레이어를 학습할 필요가 없다. Snell 등의 「Prototypical Networks for Few-shot Learning」(2017)에 기반한 간단한 예제다.

## 1. 코드

```python
"""
소수 예시 학습을 위한 원형 망

참고: Snell 외, "Prototypical Networks for Few-shot Learning" (2017)

핵심 생각: 받침 보기의 묻힘을 평균 내어 부류마다 원형 표현을 셈한 다음,
그 원형까지의 거리를 바탕으로
물음을 가려낸다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvEncoder(nn.Module):
    """
    그림을 묻는 단순한 4레이어 컨볼루션 인코더.
    소수 예시 학습 논문에서 흔히 쓴다.
    """
    def __init__(self, input_channels=1, hidden_dim=64, output_dim=64):
        super(ConvEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(input_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, output_dim),
        )
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class PrototypicalNetwork(nn.Module):
    """
    N-클래스 K-예시 분류를 위한 원형 망.
    
    인수:
        encoder: 입력을 피처 공간에 묻는 뉴럴 넷트웍
    """
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, support, support_labels, query):
        """
        인수:
            support: (n_support, *input_shape) - 받침 집합 보기
            support_labels: (n_support,) - 받침 집합의 이름표
            query: (n_query, *input_shape) - 가려낼 물음 보기
        
        반환값:
            logits: (n_query, n_classes) - 가려내기 로짓
        """
        # 모든 보기를 묻는다
        n_classes = len(torch.unique(support_labels))
        n_support = support.shape[0]
        n_query = query.shape[0]
        
        # 효율적인 부호화를 위해 받침과 물음을 이어 붙인다
        all_examples = torch.cat([support, query], dim=0)
        embeddings = self.encoder(all_examples)
        
        # 다시 받침과 물음으로 쪼갠다
        support_embeddings = embeddings[:n_support]
        query_embeddings = embeddings[n_support:]
        
        # 부류마다 원형을 셈한다
        prototypes = self._compute_prototypes(support_embeddings, support_labels, n_classes)
        
        # 물음에서 원형까지의 거리를 셈한다
        logits = self._compute_logits(query_embeddings, prototypes)
        
        return logits
    
    def _compute_prototypes(self, embeddings, labels, n_classes):
        """
        받침 묻힘의 평균으로 부류마다 원형을 셈한다.
        """
        prototypes = []
        for c in range(n_classes):
            # 부류 c의 받침 보기를 모두 찾는다
            class_mask = (labels == c)
            class_embeddings = embeddings[class_mask]
            # 평균(원형)을 셈한다
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _compute_logits(self, query_embeddings, prototypes):
        """
        물음에서 원형까지의 유클리드 거리 제곱의 음수를 셈한다.
        거리의 음수가 로짓 노릇을 한다(가까울수록 확률이 높다).
        """
        # 퍼뜨리기를 위해 차원을 늘린다
        # query: (n_query, 1, embedding_dim)
        # prototypes: (1, n_classes, embedding_dim)
        query_expanded = query_embeddings.unsqueeze(1)
        prototypes_expanded = prototypes.unsqueeze(0)
        
        # 유클리드 거리의 제곱을 셈한다
        distances = torch.sum((query_expanded - prototypes_expanded) ** 2, dim=2)
        
        # 거리의 음수를 로짓으로 낸다
        return -distances


def train_step(model, support, support_labels, query, query_labels, optimizer):
    """
    원형 망의 학습 스텝 하나.
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


def evaluate(model, support, support_labels, query, query_labels):
    """
    소수 예시 과제에서 모델을 평가한다.
    """
    model.eval()
    with torch.no_grad():
        logits = model(support, support_labels, query)
        loss = F.cross_entropy(logits, query_labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


# 사용 예
if __name__ == "__main__":
    # 모델 생성
    encoder = ConvEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = PrototypicalNetwork(encoder)
    
    # 5-클래스 1-예시 과제 보기
    n_way = 5
    k_shot = 1
    n_query = 15
    
    # 흉내 데이터(batch_size, channels, height, width)
    support = torch.randn(n_way * k_shot, 1, 28, 28)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, 1, 28, 28)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # 순전파
    logits = model(support, support_labels, query)
    print(f"Logits shape: {logits.shape}")  # (15, 5)이어야 한다
    
    # 학습 보기
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_step(model, support, support_labels, query, query_labels, optimizer)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

**출력:**

```
Logits shape: torch.Size([15, 5])
Loss: 5.0646, Accuracy: 0.2667
```

## 2. 논의

클래스 $c$의 서포트 집합을 $S_c$라 하면 프로토타입은 $p_c=|S_c|^{-1}\sum_{x\in S_c}f_\theta(x)$다. 쿼리 $q$의 로짓은 $-\|f_\theta(q)-p_c\|^2$이며, 클래스 축 소프트맥스로 확률을 얻는다. 학습 중에는 쿼리 로스의 경사가 쿼리 임베딩과 서포트 평균을 거쳐 인코더에 전달된다.

코드는 고유 라벨의 개수를 센 다음 0부터 순회한다. 따라서 에피소드 라벨은 빈 클래스 없이 0부터 N−1까지 다시 매겨야 한다. 예를 들어 원래 라벨 3과 8을 그대로 전달하면 빈 집합의 평균을 계산하게 된다. 예제의 쿼리 15개는 클래스당 15개가 아니라 에피소드 전체 개수다.

28×28 입력은 네 번 풀링한 뒤 1×1이 되므로 임베딩 차원은 64다. 서포트와 쿼리를 합쳐 인코딩하므로 학습 모드의 BatchNorm 통계는 쿼리 집합에도 의존한다. 평가 모드에서는 저장된 통계를 사용한다. 이 차이를 평가 프로토콜에 명시해야 한다. 무작위 이미지와 라벨로 수행한 한 번의 갱신은 데이터 흐름을 보여 주며, 실제 퓨샷 분류 성능을 측정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치 4개, 입력 (4,1,28,28)일 때 ConvEncoder의 각 블록 출력과 최종 임베딩 모양을 계산하라.

</div>

??? success "연습문제 1 풀이"
    블록 출력은 (4,64,14,14), (4,64,7,7), (4,64,3,3), (4,64,1,1)이며 평탄화하면 (4,64)다. 각 블록의 패딩된 컨볼루션은 공간 크기를 유지하고 MaxPool2d(2)가 소수점을 버리며 절반으로 줄인다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력에서 필요한 변경과 임베딩 차원을 설명하라.

</div>

??? success "연습문제 2 풀이"
    input_channels=3으로 생성하면 첫 컨볼루션이 RGB를 받는다. 네 번 풀링하면 4×4가 되므로 출력은 (B,1024)다. 이 인코더에는 선형 레이어가 없으므로 in_features를 고칠 필요가 없다. 서포트와 쿼리의 입력 크기는 같아야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
한 클래스의 서포트 임베딩이 (0,0), (2,0)이고 다른 클래스의 프로토타입이 (4,0)일 때 쿼리 (2,0)의 로짓을 구하라.

</div>

??? success "연습문제 3 풀이"
    첫 프로토타입은 (1,0)이므로 로짓은 (-1,-4)다. 첫 클래스의 확률은 $e^{-1}/(e^{-1}+e^{-4})\approx0.953$이다. 평균과 제곱 거리는 임베딩 공간에서 계산하며 픽셀 평균을 사용하는 것은 아니다.

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

프로토타입은 에피소드의 서포트 평균이며 쿼리는 음의 제곱 거리로 분류한다. 라벨 재매핑, 인코더 출력 차원과 BatchNorm의 평가 방식을 함께 확인해야 한다.
