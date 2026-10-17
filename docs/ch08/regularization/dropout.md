# 드롭아웃

드롭아웃은 학습 중에 뉴런 활성화의 일부를 무작위로 0으로 만드는 정규화 기법이다. 이는 뉴런의 공적응을 막고, 다른 뉴런들의 여러 무작위 부분집합과 함께 쓸 때에도 쓸모 있는 더 견고한 피처를 배우도록 뉴럴 넷트웍을 이끈다.

---

## 1. 수학적 정식화

### 기본적인 드롭아웃 연산

학습 중에 활성화 벡터가 $h \in \mathbb{R}^d$이고 드롭아웃 확률이 $p$인 레이어에 대해 다음과 같다.

$$
\tilde{h}_i = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{with probability } 1-p
\end{cases}
$$

$\frac{1}{1-p}$을 곱해 배율을 조정하면(**inverted dropout**이라 한다) 기댓값이 그대로 유지된다.

$$
\mathbb{E}[\tilde{h}_i] = p \cdot 0 + (1-p) \cdot \frac{h_i}{1-p} = h_i
$$

### 마스크로 나타내기

이진 마스크 $m \sim \text{Bernoulli}(1-p)^d$을 쓰면 다음과 같다.

$$
\tilde{h} = \frac{m \odot h}{1-p}
$$

여기서 $\odot$은 원소별 곱을 뜻한다.

### 드롭아웃을 쓰는 순전파

웨이트가 $W$, 바이어스가 $b$, 활성화가 $\sigma$인 순방향 레이어에 대해 다음과 같다.

**학습 시:**

$$
y = \sigma\left( W \cdot \frac{m \odot x}{1-p} + b \right)
$$

**추론 시:**

$$
y = \sigma(Wx + b)
$$

학습 중에 역 드롭아웃으로 배율을 조정했으므로 추론에서는 드롭아웃을 적용하지 않는다.

---

## 2. 이론적 해석

### 앙상블로서의 해석

마스크마다 서로 다른 연결 패턴을 사용하므로, 웨이트를 공유하는 여러 부분 모델을 학습하는 앙상블 관점으로 이해할 수 있다. 다만 독립 모델을 각각 학습한 앙상블과 같지는 않다. 비선형 뉴럴 넷트웍에서는 평가 모드의 한 번의 출력이 마스크별 예측 평균과 정확히 일치하지도 않는다.

- 단위가 $d$개인 뉴럴 넷트웍에는 가능한 드롭아웃 마스크가 $2^d$개 있다
- 학습의 각 단계는 이 앙상블에서 부분 뉴럴 넷트웍 하나를 뽑는다
- 평가 모드의 전체 뉴럴 넷트웍은 계산이 저렴한 결정적 대안이며, 일반적으로 마스크 예측의 정확한 산술평균은 아니다

### 베이즈적 해석

특정 우도·사전분포·정규화 항과 변분 분포를 가정하면 드롭아웃 학습을 근사 베이즈 추론으로 해석할 수 있다. 모든 드롭아웃 모델이 자동으로 보정된 사후 불확실성을 제공하는 것은 아니다. MC 예측의 분산도 관측 잡음과 모델 불확실성을 어떻게 정의했는지에 따라 해석한다.

!!! info "함께 볼 것: 불확실성 추정을 위한 몬테카를로 드롭아웃"
    드롭아웃의 베이즈적 해석은 불확실성 정량화의 강력한 기법인 **몬테카를로 드롭아웃**을 가능하게 한다. 추론 중에도 드롭아웃을 켠 채 순전파를 여러 번 돌리면 예측의 불확실성을 추정할 수 있다. 다음 내용은 **44.1절 몬테카를로 드롭아웃**을 보라.
    
    - 변분 추론의 전체 유도 ([이론](../../ch43/mc_dropout/theory.md))
    - 실전 구현 방식 (구현)
    - 표본 수렴 분석 ([수렴](../../ch43/mc_dropout/convergence.md))
    - 보정을 위한 드롭아웃 비율 선택 ([드롭아웃 비율](../../ch43/mc_dropout/dropout_rate.md))

### 잡음 주입의 관점

드롭아웃은 뉴럴 넷트웍에 곱셈적 잡음을 넣는다.

$$
\tilde{h} = h \odot \epsilon, \quad \epsilon_i \sim \begin{cases}
\frac{1}{1-p} & \text{with prob } 1-p \\
0 & \text{with prob } p
\end{cases}
$$

이 곱셈적 잡음의 분산은 $\text{Var}[\epsilon_i] = \frac{p}{1-p}$이다.

---

## 3. PyTorch 구현

### 내장 드롭아웃

```python
import torch
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    """드롭아웃 레이어를 갖춘 표준 뉴럴 넷트웍."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # 순서가 선형 → 활성화 → 드롭아웃이다. ReLU와 드롭아웃은
            # 자리를 바꾸어도 결과가 같지만(둘 다 0을 0으로 보낸다),
            # 시그모이드처럼 0을 0으로 보내지 않는 활성화에서는
            # 달라지므로 이 순서를 습관으로 두는 편이 낫다
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 출력 레이어 뒤에는 드롭아웃을 걸지 않는다. 로짓을 무작위로
        # 지우면 그 표본의 예측 자체가 망가지기 때문이다.
        # 드롭아웃은 은닉 표현에만 건다
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 학습 모드와 평가 모드.
# 두 줄을 잇달아 부르는 것은 쓰임을 보이려는 것일 뿐, 실제로는
# eval()만 남는다. 학습 루프에서는 에포크마다 둘을 오가야 한다
model = NetworkWithDropout(784, [512, 256], 10, dropout_rate=0.5)
model.train()  # 드롭아웃 켜짐
model.eval()   # 드롭아웃 꺼짐
```

### 드롭아웃 직접 구현하기

```python
class CustomDropout(nn.Module):
    """내부 동작을 보이는 직접 구현한 드롭아웃."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        
        # 이진 마스크 생성 (1 = 남김, 0 = 떨어뜨림)
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        
        # 역 드롭아웃 배율 조정과 함께 마스크 적용
        return x * mask / (1 - self.p)
```

### 드롭아웃의 변형

```python
# 1차원 드롭아웃 - 순차열용 (채널 전체를 떨어뜨린다)
dropout_1d = nn.Dropout1d(p=0.5)  # 입력: (batch, 채널, 길이)

# 2차원 공간 드롭아웃 - 이미지용 (피처 맵 전체를 떨어뜨린다)
dropout_2d = nn.Dropout2d(p=0.5)  # 입력: (batch, 채널, H, W)

# 3차원 공간 드롭아웃 - 비디오/3차원 데이터용
dropout_3d = nn.Dropout3d(p=0.5)  # 입력: (batch, 채널, D, H, W)

# 알파 드롭아웃 - SELU 활성화용 (자기 정규화 뉴럴 넷트웍)
alpha_dropout = nn.AlphaDropout(p=0.5)
```

### 불확실성을 위한 몬테카를로 드롭아웃

```python
class MCDropoutModel(nn.Module):
    """불확실성 추정을 위한 몬테카를로 드롭아웃을 지원하는 모델."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        MC 드롭아웃으로 불확실성과 함께 예측한다.
        
        반환값:
            mean: 평균 예측
            std: 표준편차 (인식적 불확실성)
        """
        # 여기가 이 기법의 전부다. 보통은 평가할 때 드롭아웃을 끄지만,
        # 일부러 켜 둔 채로 여러 번 예측한다. 매번 다른 부분망이 답하므로
        # 그 답들의 흩어짐이 곧 모델의 불확실성이 된다.
        # 이는 근사 베이즈 추론으로 볼 수 있다는 것이 Gal & Ghahramani의 결과다
        self.train()

        # 주의: 배치 정규화가 있는 모델이라면 이대로 두면 안 된다.
        # train() 이 배치 정규화의 이동 통계까지 갱신해 버리기 때문이다.
        # 그런 경우에는 드롭아웃 모듈만 골라 train()으로 되돌려야 한다.
        predictions = []
        with torch.no_grad():   # 그래디언트는 필요 없다. 순전파만 100번 한다
            for _ in range(n_samples):
                pred = self(x)   # 매번 다른 마스크가 뽑혀 답이 조금씩 다르다
                predictions.append(pred)

        # (n_samples, 배치, 출력) 꼴로 쌓은 뒤 표본 축으로 통계를 낸다
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        # 이 표준편차는 "모델이 몰라서 생기는" 인식적 불확실성이다.
        # 데이터 자체의 잡음(우연적 불확실성)은 여기에 잡히지 않으므로,
        # 둘을 모두 알고 싶으면 모델이 분산도 함께 내놓도록 해야 한다
        std = predictions.std(dim=0)

        return mean, std
```

---

## 4. 드롭아웃을 쓰는 학습

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_dropout(model, train_loader, val_loader, epochs=100, lr=0.001):
    """드롭아웃 정칙화로 모델을 학습시킨다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # ── 학습 단계 ────────────────────────────────────────────────
        # model.train()이 드롭아웃을 켜고 배치 정규화가 배치 통계를
        # 쓰도록 만든다. 에폭마다 다시 부르는 까닭은 아래 검증 단계에서
        # eval()로 바꿔 두었기 때문이다. 이 한 줄을 빼먹으면 두 번째
        # 에폭부터 드롭아웃 없이 학습하게 되어 정칙화가 사라진다
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()   # 지난 배치의 그래디언트를 지운다
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # item()으로 수만 꺼낸다. 텐서째 더하면 계산 그래프가
            # 에폭 내내 쌓여 메모리가 샌다
            train_loss += loss.item()
            # max(1)은 (최댓값, 그 자리)를 준다. 여기서는 자리만 쓴다
            _, predicted = outputs.max(1)
            # 마지막 배치가 작을 수 있으므로 배치 크기를 고정값이 아니라
            # 실제 크기로 센다
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # ── 검증 단계 ────────────────────────────────────────────────
        # eval()이 드롭아웃을 끄고 배치 정규화가 이동 통계를 쓰게 한다.
        # 이것이 없으면 검증할 때마다 답이 달라져 곡선을 믿을 수 없다
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        # no_grad는 eval()과 다른 일을 한다. eval()은 레이어의 동작을 바꾸고
        # no_grad는 그래디언트 추적을 끈다. 둘 다 필요하다
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        # 로스는 배치 수로, 정확도는 표본 수로 나눈다. 로스가 이미
        # 배치 안에서 평균이 나 있기 때문이다. 마지막 배치가 작으면
        # 이 평균은 정확한 표본 평균이 아니라 살짝 치우친 값이다
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)

    # 드롭아웃이 걸린 모델에서는 학습 로스가 검증 로스보다 높게 나오는
    # 일이 흔하다. 학습 때만 유닛이 꺼져 있기 때문이며, 고장이 아니다
    return history
```

---

## 5. 심화 기법

### 드롭커넥트

드롭커넥트는 활성화 전체가 아니라 개별 *웨이트*를 무작위로 0으로 만들어 드롭아웃을 일반화한다. 전체 수학적 정식화, 구현, 표준 드롭아웃과의 비교는 **[드롭커넥트](dropconnect.md)**를 보라.

### CNN을 위한 공간 드롭아웃

```python
class CNNWithSpatialDropout(nn.Module):
    """공간 드롭아웃을 쓰는 CNN (피처 맵 전체를 떨어뜨린다)."""
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # padding=1 이면 3x3 컨볼루션이 크기를 그대로 유지한다
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28x28 -> 14x14

            # Dropout2d는 화소 하나가 아니라 채널(피처 맵) 하나를 통째로 끈다.
            # 컨볼루션 출력은 이웃 화소끼리 값이 거의 같아서, 보통의 드롭아웃으로
            # 화소 몇 개를 꺼도 옆 화소가 그 정보를 그대로 들고 있어 효과가 없다.
            # 채널째 꺼야 비로소 "그 피처를 못 보게 하는" 일이 된다
            nn.Dropout2d(p=dropout_rate),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14x14 -> 7x7
            nn.Dropout2d(p=dropout_rate),
        )
        self.classifier = nn.Sequential(
            # 64채널 x 7 x 7 을 편 것이 입력 차원이다
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            # 여기서는 보통의 Dropout을 쓴다. 완전연결 레이어의 유닛들은
            # 서로 이웃해 있지 않아 값이 겹치지 않으므로, 하나씩 꺼도 효과가 있다
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # (배치, 64, 7, 7) -> (배치, 3136). size(0)를 써서 배치 크기는
        # 건드리지 않고 나머지 축만 편다
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### 트랜스포머에서의 드롭아웃

```python
class TransformerBlockWithDropout(nn.Module):
    """표준 위치에 드롭아웃을 둔 트랜스포머 블록."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 트랜스포머에는 드롭아웃이 들어가는 자리가 세 곳이다.
        #   (1) 어텐션 웨이트 위 — MultiheadAttention의 dropout 인자
        #   (2) 피드포워드 안쪽 — 아래 Sequential의 Dropout
        #   (3) 두 하위레이어의 출력 — dropout1, dropout2
        # 셋 다 비율이 같을 필요는 없지만 보통 하나로 맞춘다.

        # (1) 소프트맥스를 지난 어텐션 웨이트 일부를 끈다.
        # 특정 토큰 하나에만 기대는 것을 막는다
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 보통 d_ff = 4 * d_model 로 넓혔다가
            nn.ReLU(),
            nn.Dropout(dropout),        # (2) 넓어진 자리에서 끈다.
                                        # 파라메터가 가장 많이 몰린 곳이라 효과가 크다
            nn.Linear(d_ff, d_model)    # 다시 원래 차원으로 좁힌다
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # (3) 잔차로 더해지기 직전에 거는 드롭아웃. 하위레이어마다 따로 둔다
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 자기 어텐션: 질의·열쇠·값에 모두 같은 x를 넣는다
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)

        # 순서가 요점이다. 드롭아웃은 잔차로 더하기 "전"에 걸어야 한다.
        # 더한 뒤에 걸면 우회로(x)까지 끊겨 깊은 망에서 그래디언트가 흐르지 못한다.
        # 여기서는 정규화가 더하기 뒤에 오는 post-norm 방식을 쓴다
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x
```

---

## 6. 실무 지침

### 드롭아웃 비율의 출발점

| 구조 | 위치 | 대표적인 비율 |
|--------------|----------|--------------|
| 완전 연결 | 은닉 레이어 | 0.5 |
| CNN | 컨볼루션 레이어 뒤 | 0.2 - 0.3 |
| CNN | 마지막 완전 연결 레이어 앞 | 0.5 |
| RNN/LSTM | 레이어 사이 | 0.2 - 0.5 |
| 트랜스포머 | 어텐션/FFN | 0.1 |

### 드롭아웃을 쓸 때

1. 파라메터가 많은 **큰 뉴럴 넷트웍**
2. **학습 데이터가 적을 때**
3. **뚜렷한 과적합** (학습 성능이 검증 성능보다 훨씬 좋을 때)
4. **밀집레이어** (컨볼루션 레이어보다 효과가 크다)

### 효과와 강도를 다시 검토할 때

1. **아주 작은 뉴럴 넷트웍** — 성능을 해칠 수 있다
2. **배치 정규화와 함께 쓸 때** — 드롭아웃 비율을 낮게 쓴다
3. **데이터가 충분할 때** — 필요 없을 수 있다
4. **이미 강한 어그멘테이션을 쓰고 있을 때**

### 흔한 실수

1. **모드 전환을 잊기**: 언제나 `model.train()`과 `model.eval()`을 쓰라
2. **출력의 규약**: 분류 로짓이나 확률 뒤에 무작위 마스크를 적용하면 로스의 의미가 달라질 수 있다. 마지막 선형 레이어 앞의 은닉 피처에는 드롭아웃을 사용할 수 있다
3. **어디서나 같은 비율**: 레이어마다 다른 비율이 필요할 수 있다
4. **비율이 너무 높음**: 0.2~0.3에서 시작하고 필요하면 올린다

---

## 7. 다른 정규화와 결합하기

```python
class RegularizedNetwork(nn.Module):
    """드롭아웃에 배치 정규화와 웨이트 감쇠를 결합한 뉴럴 넷트웍."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        # 레이어 안의 순서가 Linear -> BatchNorm -> ReLU -> Dropout 인 것이 요점이다.
        #   BatchNorm이 Linear 바로 뒤: 엑티베이션 함수에 들어가기 전에 자를 맞춰야
        #     ReLU가 한쪽으로 치우쳐 죽는 것을 막는다.
        #   Dropout이 맨 뒤: BatchNorm 앞에 두면 꺼진 유닛까지 섞여 배치 통계가
        #     흔들리고, 그러면 추론 때 쓰는 이동 통계가 훈련 때와 어긋난다.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 마지막 레이어에는 정규화도 드롭아웃도 두지 않는다.
            # 출력은 클래스별 로짓이므로 손대면 예측 자체가 망가진다
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ── 정칙화 세 가지를 함께 건다 ──────────────────────────────────────
# 드롭아웃은 활성을, 배치 정규화는 레이어 입력의 분포를, 웨이트 감쇠는
# 웨이트의 크기를 다스린다. 막는 곳이 서로 달라 겹쳐 써도 괜찮다.
model = RegularizedNetwork(784, 256, 10)

# Adam이 아니라 AdamW를 쓴다. Adam에서는 weight_decay가 그래디언트에 더해져
# 적응적 러닝레이트에 나눠지므로 실제 감쇠 세기가 파라메터마다 달라진다.
# AdamW는 갱신 단계에서 따로 빼므로 의도한 세기가 그대로 걸린다
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
고정 활성화 $h_i$와 $0\leq p<1$에 대해 일반 마스킹과 inverted dropout의 기댓값을 비교하라.

</div>

??? success "풀이"
    일반 마스킹은 $\tilde h_i=m_i h_i$이므로 기댓값이 $(1-p)h_i$이다. inverted dropout은 $\tilde h_i=m_i h_i/(1-p)$로 학습 중 배율을 조절해 기댓값이 $h_i$가 된다. PyTorch의 일반 드롭아웃은 후자이며 평가 중에는 항등으로 동작한다. 비선형 레이어 전체의 출력 기댓값까지 일치한다는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
역 드롭아웃을 PyTorch로 바닥부터 구현하라.

</div>

??? success "풀이"
    ```python
    def inverted_dropout(x, p=0.5, training=True):
        if not training:
            return x
        mask = (torch.rand_like(x) > p).float()
        return x * mask / (1 - p)
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
드롭아웃을 앙상블로 보는 해석과 그 한계를 설명하라.

</div>

??? success "풀이"
    마스크마다 부분 모델이 정해지고 모든 모델이 웨이트를 공유한다. 가능한 마스크가 $2^d$개라는 사실이 그만큼의 독립 모델을 완전히 학습했다는 뜻은 아니다. 비선형 연산 때문에 마스크를 평균낸 레이어로 계산한 출력과 각 마스크의 출력 평균은 일반적으로 다르다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
드롭아웃 확률을 너비만으로 정할 수 없는 이유와 선택 절차를 설명하라.

</div>

??? success "풀이"
    과제, 데이터량, 다른 정규화, 정규화 레이어과의 상호작용이 모두 영향을 준다. 0을 포함한 몇 가지 확률을 같은 검증 조건에서 비교하고, 과소적합과 학습·평가 모드의 차이를 살펴 선택한다. 넓은 레이어라는 이유만으로 높은 확률이 항상 적절한 것은 아니다.

## 정리하며

드롭아웃은 학습 중 활성화에 무작위 마스크를 적용한다. inverted dropout은 고정 활성화의 기댓값을 보존하지만 전체 비선형 예측의 평균까지 보존하지는 않는다. 일반 평가에서는 끄고, MC 추론에서는 필요한 마스킹 모듈만 켜 다른 레이어의 상태가 바뀌지 않게 한다.

**참고 문헌**

1. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929-1958.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
3. Wan, L., et al. (2013). Regularization of Neural Networks using DropConnect. *ICML*.
