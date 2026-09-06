# 원형 망

소수 예시 학습을 위한 원형 망. 참고: Snell 외, "Prototypical Networks for Few-shot Learning" (2017)

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

## 코드

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
    그림을 묻는 단순한 4층 합성곱 부호기.
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
    N-갈래 K-예시 분류를 위한 원형 망.
    
    인수:
        encoder: 입력을 특징 공간에 묻는 신경망
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
    원형 망의 학습 걸음 하나.
    """
    model.train()
    optimizer.zero_grad()
    
    # 순전파
    logits = model(support, support_labels, query)
    
    # 손실을 계산한다
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
    
    # 5-갈래 1-예시 과제 보기
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
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")```

## 논의

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 2개(`ConvEncoder`, `PrototypicalNetwork`)를 정한다. 클래스마다 서로 다른 부품을 감싸 코드를 모듈 방식으로 만들고 넓히기 쉽게 한다. `forward` 메서드가 파이토치가 자동 미분에 쓰는 계산 그래프를 정한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
`ConvEncoder`의 앞먹임을 따라가며 텐서의 꼴을 좇아라. 기본 매개변수로 표본 4개짜리 배치를 넣었을 때 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 합성곱 층의 `in_channels`를 지금 값에서 3으로 바꾸어라. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$ 공식으로 합성곱과 풀링 층마다 그 뒤의 공간 차원을 다시 셈하라. 마지막 합성곱·풀링 층의 편 출력에 맞도록 첫 선형 층의 `in_features`를 고쳐라. `model = ConvEncoder(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인하라.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
`ConvEncoder`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = ConvEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.
