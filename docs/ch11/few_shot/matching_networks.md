# 맞춤 망

한 예시 학습을 위한 맞춤 망. 참고: Vinyals 외, "Matching Networks for One Shot Learning" (2016)

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

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
    맞춤 망을 위한 주의 장치를 갖춘 부호기.
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
    그림을 위한 단순한 CNN 부호기.
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
    
    # 손실을 셈한다(부드러운 목표를 쓰는 교차 엔트로피)
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
    
    # 5-갈래 5-예시 과제 보기
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
    print(f"Predicted classes: {predicted_classes}")```

## 2. 논의

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 3개(`AttentionEncoder`, `SimpleEncoder`, `MatchingNetwork`)를 정한다. 클래스마다 서로 다른 부품을 감싸 코드를 모듈 방식으로 만들고 넓히기 쉽게 한다. `forward` 메서드가 파이토치가 자동 미분에 쓰는 계산 그래프를 정한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
`AttentionEncoder`의 앞먹임을 따라가며 텐서의 꼴을 좇아라. 기본 매개변수로 표본 4개짜리 배치를 넣었을 때 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`를 지금 값에서 3으로 바꾸어라. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$ 공식으로 합성곱과 풀링 층마다 그 뒤의 공간 차원을 다시 셈하라. 마지막 합성곱·풀링 층의 편 출력에 맞도록 첫 선형 층의 `in_features`를 고쳐라. `model = AttentionEncoder(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인하라.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
`AttentionEncoder`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = AttentionEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.

## 정리하며

**다룬 것** — 맞춤 망

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 3개(`AttentionEncoder`, `SimpleEncoder`, `MatchingNetwork`)를 정한다.

핵심 클래스는 `AttentionEncoder`, `SimpleEncoder`, `MatchingNetwork`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
