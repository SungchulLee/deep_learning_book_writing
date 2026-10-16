# 매칭 네트워크

매칭 네트워크는 쿼리와 각 서포트 표본의 유사도를 계산한 뒤, 그 웨이트로 서포트 라벨을 합산한다. 클래스별 평균을 먼저 만드는 프로토타입 방식과 달리 개별 표본이 예측에 기여한다. Vinyals 등의 「Matching Networks for One Shot Learning」(2016)에서 제안한 아이디어를 단순화한 구현이다.

## 1. 코드

```python
"""
한 예시 학습을 위한 맞춤 망

참고: Vinyals 외, "Matching Networks for One Shot Learning" (2016)

핵심 생각: 주의 장치로 물음 보기를 받침 집합과 견주어,
받침 보기마다 예측에 이바지하는 무게를 매긴다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class AttentionEncoder(nn.Module):
    """
    맞춤 망을 위한 주의 장치를 갖춘 인코더.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 마지막 출력 가져오기
        return self.fc(lstm_out[:, -1, :])


class SimpleEncoder(nn.Module):
    """
    그림을 위한 단순한 CNN 인코더.
    """
    def __init__(self, input_channels=1, hidden_dim=64, output_dim=64):
        super(SimpleEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)


class MatchingNetwork(nn.Module):
    """
    코사인 닮음 주의를 쓰는 맞춤 망.
    
    모델은 받침 집합 위의 주의 무게를 셈하여 물음을 가려내며,
    주의는 묻힘 사이의 코사인 닮음을 바탕으로 한다.
    """
    def __init__(self, encoder, use_full_context_embeddings=False):
        super(MatchingNetwork, self).__init__()
        self.encoder = encoder
        self.use_fce = use_full_context_embeddings
        
    def forward(self, support, support_labels, query):
        """
        인수:
            support: (n_support, *input_shape) - 받침 집합
            support_labels: (n_support,) - 원핫 또는 이름표 첨자
            query: (n_query, *input_shape) - 물음 집합
        
        반환값:
            predictions: (n_query, n_classes) - 예측 확률
        """
        # 받침과 물음을 부호로 바꾼다
        support_embeddings = self.encoder(support)
        query_embeddings = self.encoder(query)
        
        # 코사인 닮음을 위해 묻힘을 고른다
        support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        # 코사인 닮음으로 주의 무게를 셈한다
        # 꼴: (n_query, n_support)
        attention = torch.mm(query_embeddings, support_embeddings.t())
        attention = F.softmax(attention, dim=1)
        
        # 필요하면 support_labels를 원핫으로 바꾼다
        n_classes = support_labels.max().item() + 1
        if support_labels.dim() == 1:
            support_labels_one_hot = F.one_hot(support_labels, n_classes).float()
        else:
            support_labels_one_hot = support_labels.float()
        
        # 주의를 바탕으로 받침 이름표를 무게 두어 더한다
        # 꼴: (n_query, n_classes)
        predictions = torch.mm(attention, support_labels_one_hot)
        
        return predictions
    
    def predict(self, support, support_labels, query):
        """
        물음 보기의 부류 예측을 얻는다.
        """
        predictions = self.forward(support, support_labels, query)
        return torch.argmax(predictions, dim=1)


def cosine_distance(x, y):
    """
    두 묻힘 집합 사이의 코사인 거리를 셈한다.
    """
    # 정규화
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    
    # 코사인 닮음
    similarity = torch.mm(x, y.t())
    
    return 1 - similarity


def train_matching_network(model, support, support_labels, query, query_labels, optimizer):
    """
    맞춤 망의 학습 걸음.
    """
    model.train()
    optimizer.zero_grad()
    
    # 예측을 얻는다
    predictions = model(support, support_labels, query)
    
    # 로스를 셈한다(부드러운 목표를 쓰는 교차 엔트로피)
    query_labels_one_hot = F.one_hot(query_labels, predictions.shape[1]).float()
    loss = F.binary_cross_entropy(predictions, query_labels_one_hot)
    
    # 또는 딱딱한 이름표로 교차 엔트로피를 쓴다
    # loss = F.cross_entropy(predictions, query_labels)
    
    loss.backward()
    optimizer.step()
    
    # 정확도를 계산한다
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


def evaluate_matching_network(model, support, support_labels, query, query_labels):
    """
    맞춤 망을 평가한다.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(support, support_labels, query)
        query_labels_one_hot = F.one_hot(query_labels, predictions.shape[1]).float()
        loss = F.binary_cross_entropy(predictions, query_labels_one_hot)
        
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


# 사용 예
if __name__ == "__main__":
    # 모델 생성
    encoder = SimpleEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = MatchingNetwork(encoder)
    
    # 5-클래스 5-예시 과제 보기
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # 임시 데이터 만들기
    support = torch.randn(n_way * k_shot, 1, 28, 28)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, 1, 28, 28)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # 순전파
    predictions = model(support, support_labels, query)
    print(f"Predictions shape: {predictions.shape}")  # (15, 5)
    
    # 학습
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_matching_network(model, support, support_labels, query, query_labels, optimizer)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # 예측한다
    predicted_classes = model.predict(support, support_labels, query)
    print(f"Predicted classes: {predicted_classes}")
```

**출력:**

```
Predictions shape: torch.Size([15, 5])
Loss: 0.5001, Accuracy: 0.3333
Predicted classes: tensor([4, 2, 2, 3, 2, 4, 1, 4, 2, 2, 4, 2, 2, 2, 1])
```

## 2. 논의

인코더 출력에 L2 정규화를 적용하고 쿼리·서포트 사이의 코사인 유사도를 구한다. 서포트 축 소프트맥스를 $a_i(q)$라 하면 클래스 확률은 $p(c\mid q,S)=\sum_{i:y_i=c}a_i(q)$다. 반환값은 로짓이 아니라 합이 1인 확률이다. 클래스별 서포트 수가 다르면 합산되는 표본 수 자체도 예측에 영향을 준다.

이 구현은 use_full_context_embeddings 값을 저장하지만 실제 계산에 사용하지 않는다. AttentionEncoder도 주 예제에서 호출되지 않는다. 따라서 양방향 LSTM으로 서포트 문맥을 반영하는 원 논문의 전체 모델로 해석할 수 없다. 주 예제의 SimpleEncoder는 세 번 풀링하여 28×28을 3×3으로 줄이고 576차원 임베딩을 만든다.

학습 함수는 원핫 정답과 예측 확률의 이진 교차 엔트로피를 사용한다. 단일 라벨 다중 클래스 분류의 일반적인 음의 로그 우도와는 다른 목적함수다. 주석의 대안처럼 확률을 CrossEntropyLoss에 직접 넣으면 다시 log-softmax가 적용되므로 같은 로스가 되지 않는다. 확률을 이용한 음의 로그 우도는 정답 클래스 확률의 로그로 계산한다. 인덱스 라벨은 0부터 연속적으로 매기고, 원핫 입력은 열 개수가 클래스 수임을 확인한다. 무작위 입력의 출력 예시는 실제 인식 정확도가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
AttentionEncoder에 (4,10,input_dim)을 넣었을 때 LSTM과 최종 출력의 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"
    양방향 LSTM 출력은 (4,10,2×hidden_dim)이다. 마지막 시간 위치를 선택한 뒤 선형 레이어를 적용하므로 (4,output_dim)이 된다. 컨볼루션이나 풀링은 없다. 양방향 LSTM의 마지막 위치 출력은 두 방향의 최종 은닉 상태를 이어 붙인 것과 일반적으로 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
SimpleEncoder가 64×64 RGB 이미지를 받도록 설정하면 최종 임베딩 차원은 얼마인가?

</div>

??? success "연습문제 2 풀이"
    input_channels=3으로 생성한다. 세 번 풀링하면 64→32→16→8이므로 output_dim=64일 때 평탄화 출력은 4096차원이다. 선형 출력 레이어는 없으며 AttentionEncoder의 입력 규약과 혼동하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
서포트 수 M, 쿼리 수 Q, 임베딩 차원 d일 때 이 구현의 유사도 계산 비용을 설명하라.

</div>

??? success "연습문제 3 풀이"
    쿼리와 서포트의 행렬 곱은 O(QMd)이고 유사도 행렬의 저장 공간은 O(QM)이다. 이는 한 시퀀스 내부의 모든 토큰 쌍을 처리하는 셀프 어텐션과 구별한다. 클래스별 합산은 원핫 행렬과의 곱으로 수행하며 인코더 계산 비용은 별도다.

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

개별 서포트와의 코사인 유사도를 확률 웨이트로 바꾸고 라벨을 합산한다. 현재 코드는 전체 문맥 임베딩을 구현하지 않으며, 확률 출력과 로스 함수의 규약을 구분해야 한다.
