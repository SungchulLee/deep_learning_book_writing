# 드롭커넥트

드롭커넥트는 학습 중에 뉴런의 활성화 전체가 아니라 개별 *가중치*(연결)를 무작위로 0으로 만드는, 드롭아웃의 일반화이다. Wan 등(2013)이 제안했으며 가중치 수준에서 작동하여 더 세밀한 확률적 정칙화를 제공한다. 드롭아웃이 활성화를 가려 사실상 뉴런 전체를 순전파에서 빼는 반면, 드롭커넥트는 가중치 행렬의 개별 원소를 가려 각 뉴런이 매 순전파에 부분적으로 참여하게 한다.

---

## 1. 수학적 정식화

### 표준 완전 연결층

표준 완전 연결층은 다음을 계산한다.

$$
y = \sigma(Wx + b)
$$

여기서 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$은 가중치 행렬, $x \in \mathbb{R}^{d_{\text{in}}}$은 입력, $b \in \mathbb{R}^{d_{\text{out}}}$은 편향, $\sigma$은 비선형 함수이다.

### 드롭커넥트의 순전파

학습 중에 드롭커넥트는 각 항목을 독립적으로 뽑아 이진 마스크 $M \in \{0, 1\}^{d_{\text{out}} \times d_{\text{in}}}$을 만든다.

$$
M_{ij} \sim \text{Bernoulli}(1 - p)
$$

여기서 $p$은 떨어뜨릴 확률이다. 순전파는 다음이 된다.

$$
y = \sigma\left((M \odot W) x + b\right)
$$

여기서 $\odot$은 원소별(아다마르) 곱을 뜻한다.

### 역 드롭커넥트

역 드롭아웃과 마찬가지로 학습 중에 $\frac{1}{1-p}$으로 배율을 조정하여 추론에서는 다시 조정할 필요가 없게 한다.

$$
y_{\text{train}} = \sigma\left(\frac{M \odot W}{1-p} \, x + b\right)
$$

$$
y_{\text{inference}} = \sigma(Wx + b)
$$

### 기대 출력

역 배율 조정을 하면 가려진 층의 기대 출력이 가리지 않은 층과 같아진다.

$$
\mathbb{E}_M\left[\frac{M \odot W}{1-p}\right] = W
$$

이로써 학습과 추론의 동작이 일관되게 유지된다.

### 활성화 전 값의 분포

활성화 전 값 $u = (M \odot W) x$은 독립인 확률변수의 합이다. $j$번째 출력 단위에 대해 다음과 같다.

$$
u_j = \sum_{i=1}^{d_{\text{in}}} M_{ji} W_{ji} x_i
$$

각 항 $M_{ji} W_{ji} x_i$은 다음을 갖는다.

$$
\mathbb{E}[M_{ji} W_{ji} x_i] = (1-p) W_{ji} x_i
$$

$$
\text{Var}[M_{ji} W_{ji} x_i] = p(1-p)(W_{ji} x_i)^2
$$

중심극한정리에 의해 $d_{\text{in}}$이 충분히 크면 $u_j$은 근사적으로 정규분포를 따른다.

$$
u_j \;\dot{\sim}\; \mathcal{N}\!\left((1-p) \sum_i W_{ji} x_i, \;\; p(1-p) \sum_i (W_{ji} x_i)^2\right)
$$

원 논문은 효율적인 추론에 이 정규 근사를 쓴다(다만 실무에서는 역 배율 조정이 더 흔하다).

---

## 2. 드롭아웃과의 비교

### 마스크를 어디에 적용하는가

| 항목 | 드롭아웃 | 드롭커넥트 |
|--------|---------|-------------|
| 마스크의 대상 | 활성화 $h$ | 가중치 $W$ |
| 마스크의 모양 | 층당 $(d,)$ | 층당 $(d_{\text{out}}, d_{\text{in}})$ |
| 세밀함 | 뉴런 전체를 떨어뜨린다 | 개별 연결을 떨어뜨린다 |
| 가능한 부분 신경망 | 층당 $2^d$개 | 층당 $2^{d_{\text{out}} \times d_{\text{in}}}$개 |
| 희소성의 형태 | 구조적 (행/열 단위) | 비구조적 (임의의 항목) |

### 특수한 경우로서의 드롭아웃

드롭아웃은 마스크가 구조적인 형태를 갖는 드롭커넥트의 특수한 경우로 볼 수 있다. 드롭아웃에서는 뉴런 $i$이 떨어지면 뉴런 $i$에서 나가는 *모든* 가중치가 한꺼번에 0이 된다. 드롭커넥트에서는 각 가중치가 독립적으로 떨어진다. 형식적으로 가중치에 대한 드롭아웃 마스크는 다음과 같다.

$$
M_{ji}^{\text{dropout}} = m_i \quad \forall j, \qquad m_i \sim \text{Bernoulli}(1-p)
$$

반면 드롭커넥트의 마스크는 다음과 같다.

$$
M_{ji}^{\text{dropconnect}} \sim \text{Bernoulli}(1-p) \quad \text{independently for all } (j, i)
$$

### 암묵적 앙상블의 크기

입력이 $d_{\text{in}}$개, 출력이 $d_{\text{out}}$개인 층 하나에 대해 다음과 같다.

- **드롭아웃**은 $2^{d_{\text{in}}}$개의 부분 신경망을 암묵적으로 평균한다
- **드롭커넥트**는 $2^{d_{\text{in}} \times d_{\text{out}}}$개의 부분 신경망을 암묵적으로 평균한다

지수적으로 더 큰 이 앙상블은 학습 중 분산이 커지는 대가로 더 풍부한 정칙화를 제공한다.

---

## 3. PyTorch 구현

### 기본적인 드롭커넥트 층

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropConnect(nn.Module):
    """
    드롭커넥트: 학습 중에 개별 가중치를 무작위로 0으로 만든다.
    
    역 배율 조정을 쓰므로 추론에서는 손댈 것이 없다.
    
    인수:
        in_features: 각 입력 표본의 크기
        out_features: 각 출력 표본의 크기
        p: 각 가중치를 떨어뜨릴 확률 (기본값: 0.5)
        bias: True이면 학습 가능한 편향을 더한다 (기본값: True)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 p: float = 0.5, bias: bool = True):
        super().__init__()
        # p=1 을 막는 까닭은 아래에서 1-p 로 나누기 때문이다.
        # 0으로 나누면 무한대가 되어 학습이 곧바로 터진다
        if not 0 <= p < 1:
            raise ValueError(f"drop probability must be in [0, 1), got {p}")

        self.p = p
        # 가중치를 직접 만들지 않고 nn.Linear를 안에 품는다.
        # 초기화와 저장·불러오기를 PyTorch에 맡길 수 있어 손이 덜 간다
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            # 가중치에 대한 이진 마스크 뽑기.
            # 가중치와 같은 모양에 확률 (1-p) 를 채워 베르누이로 뽑으므로
            # 살릴 자리가 1, 끌 자리가 0이 된다.
            # 드롭아웃이 활성 한 줄을 끄는 데 견주어, 여기서는 연결
            # 하나하나를 따로 끄므로 만들어지는 부분망이 훨씬 다양하다
            mask = torch.bernoulli(
                torch.full_like(self.linear.weight, 1 - self.p)
            )

            # 역 배율 조정(inverted dropout).
            # 가중치의 (1-p)만 살아남아 출력의 기댓값이 그만큼 작아지므로
            # 학습할 때 미리 1/(1-p) 배로 키워 둔다. 그래서 아래 평가 경로에서는
            # 아무 보정 없이 원래 가중치를 그대로 쓸 수 있다
            effective_weight = self.linear.weight * mask / (1 - self.p)

            # self.linear(x)를 부르지 않고 F.linear에 가중치를 넘긴다.
            # self.linear.weight를 제자리에서 고치면 학습된 값이 망가진다
            return F.linear(x, effective_weight, self.linear.bias)

        # 평가 경로. 위에서 눈금을 맞춰 두었으므로 그대로 통과시킨다
        return self.linear(x)
```

### 감싸개로서의 드롭커넥트

```python
class DropConnectWrapper(nn.Module):
    """
    아무 nn.Linear 층이든 드롭커넥트 정칙화로 감싼다.
    
    인수:
        linear_layer: 이미 있는 nn.Linear 모듈
        p: 각 가중치를 떨어뜨릴 확률
    """
    
    def __init__(self, linear_layer: nn.Linear, p: float = 0.5):
        super().__init__()
        self.linear = linear_layer
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            mask = torch.bernoulli(
                torch.full_like(self.linear.weight, 1 - self.p)
            )
            weight = self.linear.weight * mask / (1 - self.p)
            return F.linear(x, weight, self.linear.bias)
        return self.linear(x)

def apply_dropconnect(model: nn.Module, p: float = 0.5) -> nn.Module:
    """
    모델 안의 모든 nn.Linear 층을 드롭커넥트로 감싼 판본으로 바꾼다.
    
    인수:
        model: 신경망
        p: 떨어뜨릴 확률
        
    반환값:
        모든 선형층에 드롭커넥트를 적용한 모델
    """
    # named_children()은 바로 아래 자식만 준다. named_modules()와 달리
    # 손자까지 훑지 않으므로, 아래에서 재귀로 직접 내려가야 한다.
    # 굳이 이렇게 하는 까닭은 setattr로 부모의 자리를 바꿔치기해야 하는데
    # 그러려면 "누구의 어떤 이름인지"를 알아야 하기 때문이다.
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 원래 층을 감싼 것으로 갈아 끼운다. 감싸개가 원래 층을
            # 그대로 품으므로 학습된 가중치는 잃지 않는다
            setattr(model, name, DropConnectWrapper(module, p=p))
        else:
            # nn.Linear가 아니면 그 안으로 내려가 같은 일을 되풀이한다.
            # 주의: 반환값을 쓰지 않아도 되는 까닭은 이 함수가 모델을
            # 제자리에서 고치기 때문이다. 곧 원본이 바뀐다
            apply_dropconnect(module, p)
    return model
```

### 드롭커넥트를 쓰는 신경망

```python
class NetworkWithDropConnect(nn.Module):
    """드롭커넥트 층을 쓰는 순방향 신경망."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, drop_prob=0.5):
        super().__init__()
        
        # 은닉층의 너비 목록을 받아 층을 프로그램으로 쌓는다.
        # 이렇게 해 두면 hidden_dims만 바꾸어 깊이와 너비를 자유롭게 바꿀 수 있다
        layers = []
        prev_dim = input_dim   # 다음 층의 입력 차원을 이어 나른다

        for hidden_dim in hidden_dims:
            layers.extend([
                # 드롭아웃이 활성(유닛의 출력)을 끄는 데 견주어,
                # 드롭커넥트는 가중치 하나하나를 끈다. 끄는 대상이
                # 훨씬 잘게 나뉘므로 같은 확률에서도 더 다양한 부분망이 만들어진다
                DropConnect(prev_dim, hidden_dim, p=drop_prob),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        # 출력층에는 드롭커넥트를 쓰지 않는다.
        # 여기서 가중치를 끄면 특정 갈래로 가는 길이 통째로 막혀
        # 그 갈래의 로짓이 무너진다
        layers.append(nn.Linear(prev_dim, output_dim))

        # 리스트를 별표로 풀어 Sequential에 넘긴다
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 사용 예: 784 -> 512 -> 256 -> 10
model = NetworkWithDropConnect(
    input_dim=784, hidden_dims=[512, 256], output_dim=10, drop_prob=0.5
)
```

### 불확실성을 위한 몬테카를로 드롭커넥트

MC 드롭아웃처럼 드롭커넥트도 추론에서 켜 둔 채로 예측의 불확실성을 추정할 수 있다.

```python
class MCDropConnectModel(nn.Module):
    """불확실성 추정을 위한 몬테카를로 드롭커넥트를 지원하는 모델."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, drop_prob=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                DropConnect(prev_dim, hidden_dim, p=drop_prob),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        MC 드롭커넥트로 불확실성과 함께 예측한다.
        
        인수:
            x: 입력 텐서
            n_samples: 확률적 순전파의 횟수
            
        반환값:
            mean: 표본에 걸친 평균 예측
            std: 표준편차 (인식적 불확실성)
        """
        # 평가할 때 보통은 확률적 요소를 끄지만, 여기서는 일부러 켠 채로
        # 여러 번 예측한다. 매번 다른 부분망이 답하므로 그 답들의 흩어짐이
        # 곧 모델의 불확실성이 된다. MC 드롭아웃과 같은 발상이며, 끄는
        # 대상이 활성이 아니라 가중치라는 점만 다르다.
        #
        # 주의: 이 모델에는 배치 정규화가 없어 안전하지만, 있다면
        # train()이 이동 통계까지 갱신해 버린다. 그런 경우에는
        # DropConnect 모듈만 골라 켜야 한다.
        self.train()

        predictions = []
        with torch.no_grad():   # 기울기는 필요 없다. 순전파만 되풀이한다
            for _ in range(n_samples):
                pred = self(x)   # 매번 다른 마스크가 뽑힌다
                predictions.append(pred)

        # (n_samples, 배치, 출력) 꼴로 쌓은 뒤 표본 축으로 통계를 낸다
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        # 이 표준편차는 모델이 몰라서 생기는 인식적 불확실성이다.
        # 데이터 자체의 잡음은 여기에 잡히지 않는다
        std = predictions.std(dim=0)

        return mean, std
```

---

## 4. 합성곱 층을 위한 드롭커넥트

드롭커넥트는 합성곱 필터에도 적용할 수 있다.

```python
class DropConnectConv2d(nn.Module):
    """
    합성곱 필터에 드롭커넥트를 적용한 Conv2d.
    
    핵의 각 가중치가 독립적으로 떨어진다.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, p=0.5, bias=True):
        super().__init__()
        self.p = p
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
    
    def forward(self, x):
        if self.training and self.p > 0:
            # 가중치와 같은 모양으로 확률 (1-p) 를 채운 뒤 베르누이로 뽑는다.
            # 결과는 살릴 자리가 1, 끌 자리가 0인 마스크다.
            # 마스크를 매 순전파마다 새로 뽑으므로 걸음마다 다른 부분망이 된다
            mask = torch.bernoulli(
                torch.full_like(self.conv.weight, 1 - self.p)
            )

            # (1-p)로 나누는 것이 "역 드롭아웃" 눈금 맞추기다.
            # 가중치의 (1-p)만 살아남아 출력의 기댓값이 그만큼 작아지므로
            # 미리 키워 둔다. 이렇게 해 두면 평가할 때 아무 보정도 하지
            # 않고 원래 가중치를 그대로 쓸 수 있다
            weight = self.conv.weight * mask / (1 - self.p)

            # self.conv를 부르지 않고 F.conv2d에 가중치를 직접 넘긴다.
            # self.conv.weight를 제자리에서 바꾸면 학습된 값이 망가지므로,
            # 마스크를 씌운 사본을 따로 넘기는 것이다
            return F.conv2d(
                x, weight, self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding
            )

        # 평가할 때는 보통의 합성곱. 위에서 이미 눈금을 맞춰 두었으므로
        # 여기서 (1-p)를 곱하는 보정이 필요 없다
        return self.conv(x)
```

---

## 5. 학습 예제

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_dropconnect(
    model, train_loader, val_loader, epochs=100, lr=0.001
):
    """드롭커넥트 모델을 학습시키고 드롭아웃 기준선과 비교한다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 학습 — 드롭커넥트 켜짐
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 — 드롭커넥트 꺼짐
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_correct/val_total:.4f}")
    
    return history
```

---

## 6. 계산에 대한 고려

드롭커넥트는 활성화 벡터만이 아니라 가중치 행렬 전체에 대해 마스크를 뽑으므로 표준 드롭아웃보다 계산 비용이 더 든다. 입력이 $d_{\text{in}}$개, 출력이 $d_{\text{out}}$개인 층에 대해 다음과 같다.

| 연산 | 드롭아웃 | 드롭커넥트 |
|-----------|---------|-------------|
| 마스크의 크기 | $d_{\text{in}}$ | $d_{\text{in}} \times d_{\text{out}}$ |
| 마스크 뽑기 | $O(d_{\text{in}})$ | $O(d_{\text{in}} \times d_{\text{out}})$ |
| 메모리 부담 | 무시할 만함 | 가중치 행렬에 비례 |
| 추론 | 표준과 같음 | 표준과 같음 |

실무에서는 마스크 생성이 GPU에서 잘 병렬화되고 가중치 행렬이 이미 메모리에 있으므로 대부분의 구조에서 부담이 크지 않다.

---

## 7. 실무 지침

### 드롭아웃 대신 드롭커넥트를 쓸 때

1. **매개변수가 많은 밀집층**: 드롭커넥트가 더 세밀한 정칙화를 제공한다
2. **드롭아웃으로 모자랄 때**: 드롭아웃을 세게 걸어도 모델이 여전히 과적합할 때
3. **불확실성 추정**: 더 큰 암묵적 앙상블이 더 잘 보정된 불확실성을 줄 수 있다
4. **실험**: 표준 드롭아웃의 결과가 만족스럽지 않을 때 시도해 볼 만하다

### 권장 떨어뜨림 비율

| 구조 | 드롭아웃 | 드롭커넥트 |
|-------------|---------|-------------|
| 완전 연결 | 0.5 | 0.5 |
| 합성곱 | 0.2 – 0.3 | 0.3 – 0.5 |
| 출력층 | 적용하지 않음 | 적용하지 않음 |

### 흔한 관행

- **모드 전환**: 드롭아웃과 마찬가지로 드롭커넥트를 켜고 끄려면 언제나 `model.train()`과 `model.eval()`을 쓴다
- **편향 항**: 보통 편향 매개변수에는 드롭커넥트를 적용하지 않는다
- **다른 정칙화와 결합하기**: 드롭커넥트는 가중치 감쇠와 함께 쓸 수 있다. 과적합이 계속되면 한쪽을 줄인다

---

## 연습문제

**연습문제 1.**
드롭아웃과 드롭커넥트의 차이를 수학적으로 설명하라.

??? success "연습문제 1 풀이"
    드롭아웃은 뉴런의 출력 전체를 0으로 만든다. $m \sim \text{Bernoulli}(1-p)^d$일 때 $\tilde{h} = m \odot h$이다. 드롭커넥트는 개별 가중치를 0으로 만든다. $M \sim \text{Bernoulli}(1-p)^{d_{\text{in}} \times d_{\text{out}}}$일 때 $\tilde{W} = M \odot W$이다. 드롭커넥트가 더 세밀하며, 확률변수가 드롭아웃의 $d_{\text{out}}$개에 비해 $d_{\text{in}} \times d_{\text{out}}$개이다.

---

**연습문제 2.**
드롭커넥트를 PyTorch로 구현하라.

??? success "연습문제 2 풀이"
    ```python
    class DropConnect(nn.Module):
        def __init__(self, layer, p=0.5):
            super().__init__()
            self.layer = layer
            self.p = p
        def forward(self, x):
            if self.training:
                mask = torch.bernoulli(torch.ones_like(self.layer.weight) * (1-self.p))
                return F.linear(x, self.layer.weight * mask / (1-self.p), self.layer.bias)
            return self.layer(x)
    ```

---

**연습문제 3.**
학습 중에 드롭커넥트가 드롭아웃보다 대체로 비싼 이유는 무엇인가?

??? success "연습문제 3 풀이"
    드롭커넥트는 표본마다, 가중치 행렬마다 크기가 $d_{\text{in}} \times d_{\text{out}}$인 마스크를 만들고 저장해야 한다. 드롭아웃은 크기가 $d_{\text{out}}$인 마스크만 있으면 된다. 계산 부담이 뉴런의 수가 아니라 가중치의 수에 비례한다.

---

**연습문제 4.**
드롭아웃과 드롭커넥트가 만드는 암묵적 앙상블의 이론적 표현력을 비교하라.

??? success "연습문제 4 풀이"
    드롭아웃은 (뉴런의 수를 $d$이라 할 때) $2^d$개 부분 신경망의 앙상블을 만든다. 드롭커넥트는 $2^{d_{\text{in}} \times d_{\text{out}}}$개를 만들어 지수적으로 더 큰 앙상블이 된다. 이 풍부한 앙상블은 더 강한 정칙화를 주지만 계산 비용이 더 든다.

## 정리하며

이 마당은 수학적 정식화、드롭아웃과의 비교、PyTorch 구현、합성곱 층을 위한 드롭커넥트을 차례로 짚었다.

**참고 문헌**

1. Wan, L., Zeiler, M., Zhang, S., Le Cun, Y., & Fergus, R. (2013). Regularization of Neural Networks using DropConnect. *Proceedings of the 30th International Conference on Machine Learning (ICML)*.
2. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(1), 1929-1958.
3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*.
