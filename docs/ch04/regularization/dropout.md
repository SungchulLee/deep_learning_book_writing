# 드롭아웃

드롭아웃은 학습 중에 뉴런 활성화의 일부를 무작위로 0으로 만드는 정칙화 기법이다. 이는 뉴런의 공적응을 막고, 다른 뉴런들의 여러 무작위 부분집합과 함께 쓸 때에도 쓸모 있는 더 견고한 특징을 배우도록 신경망을 이끈다.

---

## 1. 수학적 정식화

### 기본적인 드롭아웃 연산

학습 중에 활성화 벡터가 $h \in \mathbb{R}^d$이고 드롭아웃 확률이 $p$인 층에 대해 다음과 같다.

$$
\tilde{h}_i = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{with probability } 1-p
\end{cases}
$$

$\frac{1}{1-p}$을 곱해 배율을 조정하면(**역 드롭아웃**이라 한다) 기댓값이 그대로 유지된다.

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

가중치가 $W$, 편향이 $b$, 활성화가 $\sigma$인 순방향 층에 대해 다음과 같다.

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

드롭아웃은 지수적으로 많은 신경망의 앙상블을 학습시키는 것으로 볼 수 있다.

- 단위가 $d$개인 신경망에는 가능한 드롭아웃 마스크가 $2^d$개 있다
- 학습의 각 단계는 이 앙상블에서 부분 신경망 하나를 뽑는다
- 시험 시점에 전체 신경망이 앙상블의 평균 예측을 근사한다

### 베이즈적 해석

드롭아웃은 딥러닝에서 베이즈 추론을 근사한다. Gal과 Ghahramani(2016)는 드롭아웃 학습이 근사 사후분포와 가중치에 대한 참 사후분포 사이의 KL 발산의 근사를 최소화함을 보였다.

!!! info "함께 볼 것: 불확실성 추정을 위한 몬테카를로 드롭아웃"
    드롭아웃의 베이즈적 해석은 불확실성 정량화의 강력한 기법인 **몬테카를로 드롭아웃**을 가능하게 한다. 추론 중에도 드롭아웃을 켠 채 순전파를 여러 번 돌리면 예측의 불확실성을 추정할 수 있다. 다음 내용은 **33.2절 몬테카를로 드롭아웃**을 보라.
    
    - 변분 추론의 전체 유도 ([이론](../../ch39/mc_dropout/theory.md))
    - 실전 구현 방식 (구현)
    - 표본 수렴 분석 ([수렴](../../ch39/mc_dropout/convergence.md))
    - 보정을 위한 드롭아웃 비율 선택 ([드롭아웃 비율](../../ch39/mc_dropout/dropout_rate.md))

### 잡음 주입의 관점

드롭아웃은 신경망에 곱셈적 잡음을 넣는다.

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
    """드롭아웃 층을 갖춘 표준 신경망."""
    
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

# 학습 모드와 평가 모드
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

# 2차원 공간 드롭아웃 - 이미지용 (특징 맵 전체를 떨어뜨린다)
dropout_2d = nn.Dropout2d(p=0.5)  # 입력: (batch, 채널, H, W)

# 3차원 공간 드롭아웃 - 비디오/3차원 데이터용
dropout_3d = nn.Dropout3d(p=0.5)  # 입력: (batch, 채널, D, H, W)

# 알파 드롭아웃 - SELU 활성화용 (자기 정규화 신경망)
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
        with torch.no_grad():   # 기울기는 필요 없다. 순전파만 100번 한다
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
            optimizer.zero_grad()   # 지난 배치의 기울기를 지운다
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

        # no_grad는 eval()과 다른 일을 한다. eval()은 층의 동작을 바꾸고
        # no_grad는 기울기 추적을 끈다. 둘 다 필요하다
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        # 손실은 배치 수로, 정확도는 표본 수로 나눈다. 손실이 이미
        # 배치 안에서 평균이 나 있기 때문이다. 마지막 배치가 작으면
        # 이 평균은 정확한 표본 평균이 아니라 살짝 치우친 값이다
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)

    # 드롭아웃이 걸린 모델에서는 학습 손실이 검증 손실보다 높게 나오는
    # 일이 흔하다. 학습 때만 유닛이 꺼져 있기 때문이며, 고장이 아니다
    return history
```

---

## 5. 심화 기법

### 드롭커넥트

드롭커넥트는 활성화 전체가 아니라 개별 *가중치*를 무작위로 0으로 만들어 드롭아웃을 일반화한다. 전체 수학적 정식화, 구현, 표준 드롭아웃과의 비교는 **[드롭커넥트](dropconnect.md)**를 보라.

### CNN을 위한 공간 드롭아웃

```python
class CNNWithSpatialDropout(nn.Module):
    """공간 드롭아웃을 쓰는 CNN (특징 맵 전체를 떨어뜨린다)."""
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # padding=1 이면 3x3 합성곱이 크기를 그대로 유지한다
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28x28 -> 14x14

            # Dropout2d는 화소 하나가 아니라 채널(특징 맵) 하나를 통째로 끈다.
            # 합성곱 출력은 이웃 화소끼리 값이 거의 같아서, 보통의 드롭아웃으로
            # 화소 몇 개를 꺼도 옆 화소가 그 정보를 그대로 들고 있어 효과가 없다.
            # 채널째 꺼야 비로소 "그 특징을 못 보게 하는" 일이 된다
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
            # 여기서는 보통의 Dropout을 쓴다. 완전연결층의 유닛들은
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
        #   (1) 어텐션 가중치 위 — MultiheadAttention의 dropout 인자
        #   (2) 피드포워드 안쪽 — 아래 Sequential의 Dropout
        #   (3) 두 하위층의 출력 — dropout1, dropout2
        # 셋 다 비율이 같을 필요는 없지만 보통 하나로 맞춘다.

        # (1) 소프트맥스를 지난 어텐션 가중치 일부를 끈다.
        # 특정 토큰 하나에만 기대는 것을 막는다
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 보통 d_ff = 4 * d_model 로 넓혔다가
            nn.ReLU(),
            nn.Dropout(dropout),        # (2) 넓어진 자리에서 끈다.
                                        # 매개변수가 가장 많이 몰린 곳이라 효과가 크다
            nn.Linear(d_ff, d_model)    # 다시 원래 차원으로 좁힌다
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # (3) 잔차로 더해지기 직전에 거는 드롭아웃. 하위층마다 따로 둔다
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 자기 어텐션: 질의·열쇠·값에 모두 같은 x를 넣는다
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)

        # 순서가 요점이다. 드롭아웃은 잔차로 더하기 "전"에 걸어야 한다.
        # 더한 뒤에 걸면 우회로(x)까지 끊겨 깊은 망에서 기울기가 흐르지 못한다.
        # 여기서는 정규화가 더하기 뒤에 오는 post-norm 방식을 쓴다
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x
```

---

## 6. 실무 지침

### 권장 드롭아웃 비율

| 구조 | 위치 | 대표적인 비율 |
|--------------|----------|--------------|
| 완전 연결 | 은닉층 | 0.5 |
| CNN | 합성곱 층 뒤 | 0.2 - 0.3 |
| CNN | 마지막 완전 연결층 앞 | 0.5 |
| RNN/LSTM | 층 사이 | 0.2 - 0.5 |
| 트랜스포머 | 어텐션/FFN | 0.1 |

### 드롭아웃을 쓸 때

1. 매개변수가 많은 **큰 신경망**
2. **학습 데이터가 적을 때**
3. **뚜렷한 과적합** (학습 성능이 검증 성능보다 훨씬 좋을 때)
4. **밀집층** (합성곱 층보다 효과가 크다)

### 드롭아웃을 쓰지 말아야 할 때

1. **아주 작은 신경망** — 성능을 해칠 수 있다
2. **배치 정규화와 함께 쓸 때** — 드롭아웃 비율을 낮게 쓴다
3. **데이터가 충분할 때** — 필요 없을 수 있다
4. **이미 강한 증강을 쓰고 있을 때**

### 흔한 실수

1. **모드 전환을 잊기**: 언제나 `model.train()`과 `model.eval()`을 쓰라
2. **출력 뒤의 드롭아웃**: 마지막 층 앞에 드롭아웃을 두지 마라
3. **어디서나 같은 비율**: 층마다 다른 비율이 필요할 수 있다
4. **비율이 너무 높음**: 0.2~0.3에서 시작하고 필요하면 올린다

---

## 7. 다른 정칙화와 결합하기

```python
class RegularizedNetwork(nn.Module):
    """드롭아웃에 배치 정규화와 가중치 감쇠를 결합한 신경망."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        # 층 안의 순서가 Linear -> BatchNorm -> ReLU -> Dropout 인 것이 요점이다.
        #   BatchNorm이 Linear 바로 뒤: 활성 함수에 들어가기 전에 자를 맞춰야
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

            # 마지막 층에는 정규화도 드롭아웃도 두지 않는다.
            # 출력은 갈래별 로짓이므로 손대면 예측 자체가 망가진다
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ── 정칙화 세 가지를 함께 건다 ──────────────────────────────────────
# 드롭아웃은 활성을, 배치 정규화는 층 입력의 분포를, 가중치 감쇠는
# 가중치의 크기를 다스린다. 막는 곳이 서로 달라 겹쳐 써도 괜찮다.
model = RegularizedNetwork(784, 256, 10)

# Adam이 아니라 AdamW를 쓴다. Adam에서는 weight_decay가 기울기에 더해져
# 적응적 학습률에 나눠지므로 실제 감쇠 세기가 매개변수마다 달라진다.
# AdamW는 갱신 단계에서 따로 빼므로 의도한 세기가 그대로 걸린다
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 연습문제

**연습문제 1.**
학습 중 드롭아웃 확률이 $p$인 뉴런의 기대 출력을 유도하고, 시험 시점의 배율 인수를 설명하라.

??? success "연습문제 1 풀이"
    학습 중에는 드롭아웃 전 출력을 $y$이라 할 때 $\mathbb{E}[\tilde{y}] = (1-p)y$이다. 시험 시점에는 이에 맞추려고 $(1-p)$을 곱한다. $y_{\text{test}} = (1-p)y$이다. 동등하게, 역 드롭아웃(학습 중에 $1-p$으로 나눈다)을 쓰면 시험 시점에 배율을 조정할 필요가 없다.

---

**연습문제 2.**
역 드롭아웃을 PyTorch로 바닥부터 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def inverted_dropout(x, p=0.5, training=True):
        if not training:
            return x
        mask = (torch.rand_like(x) > p).float()
        return x * mask / (1 - p)
    ```

---

**연습문제 3.**
앙상블 평균의 관점에서 드롭아웃의 정칙화 효과를 설명하라.

??? success "연습문제 3 풀이"
    드롭아웃 마스크마다 서로 다른 부분 신경망이 정해진다. 드롭아웃으로 학습하면 지수적으로 많은 부분 신경망(뉴런이 $d$개면 $2^d$개)을 동시에 학습시키는 셈이다. 시험 시점에 모든 뉴런을 쓰면 이 부분 신경망들의 기하평균을 근사하게 되며, 이는 모델 앙상블과 비슷하다.

---

**연습문제 4.**
드롭아웃 비율의 선택은 층의 너비에 어떻게 달려 있는가? 대표적인 출발점은 무엇인가?

??? success "연습문제 4 풀이"
    대표적으로 은닉층에는 $p=0.5$(Srivastava 등, 2014), 입력층에는 $p=0.2$을 쓴다. 넓은 층일수록 (중복이 많아) 더 높은 드롭아웃을 견딜 수 있다. 최적 비율은 층의 용량과 데이터셋의 크기에 달려 있으므로 교차 검증을 권한다.

## 정리하며

이 마당은 수학적 정식화、이론적 해석、PyTorch 구현、드롭아웃을 쓰는 학습을 차례로 짚었다.

**참고 문헌**

1. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929-1958.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
3. Wan, L., et al. (2013). Regularization of Neural Networks using DropConnect. *ICML*.
