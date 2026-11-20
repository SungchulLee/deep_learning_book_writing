# MC 드롭아웃 비율 선택

---

## 1. 개관

드롭아웃 비율 $p$(유닛을 떨어뜨릴 확률)은 학습의 정칙화와 불확실성 근사의 성능에 함께 영향을 주는 중요한 하이퍼파라미터다. 이 글은 몬테카를로 드롭아웃 응용에서 드롭아웃 비율을 원리적으로 고르는 방법을 제시한다.

---

## 2. 이론 기초

### 사전분포와 사후분포의 트레이드오프

변분 추론의 관점에서 보면 드롭아웃 비율 $p$는 웨이트에 대한 사전분포를 암묵적으로 정한다. 드롭아웃 비율, 웨이트 감쇠 $\lambda$, 사전분포 사이의 관계는 이렇다.

$$
\lambda = \frac{p \ell^2}{2N\tau}
$$

여기서

- $\ell^2$은 사전분포의 길이 스케일
- $N$은 데이터 꾸러미 크기
- $\tau$는 모델의 정밀도(관측 잡음 분산의 역수)

**뜻하는 바**:

- **$p$가 클수록** → 정칙화가 강함 → 사후분포가 넓어짐 → 불확실성이 커짐
- **$p$가 작을수록** → 정칙화가 약함 → 사후분포가 좁아짐 → 불확실성이 작아짐

### 드롭아웃 분포의 분산

학습된 평균이 $m$인 웨이트 $w$에서 드롭아웃 분포는 다음을 지닌다.

$$
\mathbb{E}[w] = (1-p) \cdot m
$$

$$
\text{Var}[w] = p(1-p) \cdot m^2
$$

변동계수는

$$
\text{CV}[w] = \frac{\sqrt{\text{Var}[w]}}{\mathbb{E}[w]} = \sqrt{\frac{p}{1-p}}
$$

| $p$ | CV |
|-----|-----|
| 0.1 | 0.33 |
| 0.2 | 0.50 |
| 0.3 | 0.65 |
| 0.5 | 1.00 |
| 0.7 | 1.53 |

$p$가 클수록 웨이트의 변동이 커지고 따라서 예측 분포도 넓어진다.

### 출력 분산의 전파

드롭아웃 비율이 $p$인 레이어 하나 $y = \sigma(Wx + b)$에서 분산은 대략 이렇게 전파된다.

$$
\text{Var}[y_j] \approx \frac{p}{1-p} \cdot \mathbb{E}[y_j]^2 + (1-p) \cdot \text{Var}[\text{pre-activation}]
$$

깊은 네트워크에서는 이것이 레이어에 걸쳐 쌓이므로 깊은 레이어일수록 드롭아웃 비율에 더 민감해진다.

---

## 3. 레이어마다 다른 드롭아웃 비율

### 레이어별 지침

레이어 유형마다 알맞은 드롭아웃 비율이 다르다.

1. **앞쪽 레이어**(입력 가까이): 낮은 비율(0.1~0.3)
   - 안정적이어야 할 범용 특성을 학습한다
   - 너무 많이 떨어뜨리면 특성의 계층 구조가 무너진다

2. **가운데 레이어**: 중간 비율(0.3~0.5)
   - 정칙화와 정보 흐름 사이의 균형

3. **뒤쪽 레이어**(출력 가까이): 높은 비율(0.4~0.6)
   - 과적합되기 쉽다
   - 여기서의 MC 드롭아웃이 출력의 불확실성을 가장 직접 반영한다

4. **컨볼루션 레이어**: 낮은 비율(0.1~0.3)
   - 공간 구조를 지켜야 한다
   - 공간 드롭아웃(특성 맵을 채널째 떨어뜨리기)을 쓴다

5. **완전 연결 레이어**: 높은 비율(0.3~0.5)
   - 파라메터가 많아 과적합되기 더 쉽다

### 구조별 지침

```python
def get_recommended_dropout_rates(architecture: str) -> dict:
    """
    구조 클래스에 따라 즐겨 쓰는 드롭아웃 비율을 얻는다.
    
    켜 클래스를 드롭아웃 비율에 맞춘 사전을 돌려준다.
    """
    recommendations = {
        'mlp': {
            'hidden': 0.5,
            'input': 0.2,  # 골라 쓰며, 흔히 빼놓는다
        },
        'cnn': {
            'conv_early': 0.1,
            'conv_late': 0.25,
            'fc': 0.5,
        },
        'resnet': {
            'after_conv': 0.0,  # 배치 잣대 잡기가 정칙화를 맡는다
            'after_block': 0.2,
            'fc': 0.5,
        },
        'transformer': {
            'attention': 0.1,
            'ffn': 0.1,
            'embedding': 0.1,
        },
        'rnn': {
            'between_layers': 0.3,
            'recurrent': 0.0,  # 그 대신 변이 드롭아웃을 쓴다
            'output': 0.5,
        },
        'vae': {
            'encoder': 0.2,
            'decoder': 0.2,
            # 붙임말: 드롭아웃은 VAE의 숨은 밭을 흔든다
        }
    }
    return recommendations.get(architecture, {'default': 0.5})
```

---

## 4. 데이터에 따른 선택

### 데이터 꾸러미 크기와의 관계

데이터 꾸러미가 클수록 정칙화가 덜 필요하다.

$$
p_{\text{optimal}} \propto \frac{1}{\sqrt{N}}
$$

**경험적 지침**:

| 데이터 꾸러미 크기 | 권장 $p$ |
|--------------|-----------------|
| 1,000 미만 | 0.5 ~ 0.7 |
| 1,000 ~ 10,000 | 0.4 ~ 0.5 |
| 10,000 ~ 100,000 | 0.3 ~ 0.4 |
| 100,000 초과 | 0.1 ~ 0.3 |

### 모델 용량에 맞추기

모델이 클수록 정칙화가 더 필요하다.

$$
p_{\text{optimal}} \propto \log(\text{num\_params})
$$

```python
def suggest_dropout_from_model_size(
    num_params: int,
    dataset_size: int,
    base_rate: float = 0.5
) -> float:
    """
    모형과 자료의 크기로 드롭아웃 비율을 내놓는다.
    
    어림 규칙: 모형이 크고 자료가 작을수록 드롭아웃을 높인다
    """
    import math
    
    # 담는 힘 견줌: 자료 하나당 파라메터
    capacity_ratio = num_params / dataset_size
    
    # 잣대 값(겪어 본 값)
    if capacity_ratio > 100:
        scale = 1.2
    elif capacity_ratio > 10:
        scale = 1.0
    elif capacity_ratio > 1:
        scale = 0.8
    else:
        scale = 0.6
    
    suggested = base_rate * scale
    
    # 이치에 닿는 테두리로 자른다
    return max(0.1, min(0.7, suggested))
```

---

## 5. 캘리브레이션에 기반한 선택

### 드롭아웃 비율과 캘리브레이션

드롭아웃 비율은 불확실성의 캘리브레이션에 직접 영향을 준다. 너무 낮으면 모델이 과신하고, 너무 높으면 확신이 지나치게 낮아진다.

**기대 캘리브레이션 오차(ECE)**:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|
$$

여기서 $\text{acc}(b)$는 버킷 $b$의 정확도이고 $\text{conf}(b)$는 그 버킷의 평균 신뢰도다.

### 가장 좋은 캘리브레이션을 찾는 격자 탐색

```python
import torch
import numpy as np
from typing import List, Tuple

def calibration_grid_search(
    model_class,
    train_loader,
    val_loader,
    dropout_rates: List[float],
    mc_samples: int = 50,
    n_bins: int = 15,
    **model_kwargs
) -> Tuple[float, dict]:
    """
    따짐 꾸러미에서 ECE를 가장 작게 하는 드롭아웃 비율을 찾는다.
    
    Args:
        model_class: 지어 낼 모형 클래스
        train_loader: 익힘 자료
        val_loader: 따짐 자료  
        dropout_rates: 해 볼 드롭아웃 비율 목록
        mc_samples: 따질 때 쓸 MC 표본 수
        n_bins: 눈금 맞음 통의 수
        
    Returns:
        best_rate: 가장 좋은 드롭아웃 비율
        results: 비율을 자에 맞춘 사전
    """
    results = {}
    
    for p in dropout_rates:
        print(f"드롭아웃 비율 {p}으로 익히는 중")
        
        # 모형을 익힌다
        model = model_class(dropout_rate=p, **model_kwargs)
        train_model(model, train_loader)  # 익힘 함수는 따로 마련한다
        
        # 눈금 맞음을 따진다
        ece, mce, brier = evaluate_calibration(
            model, val_loader, mc_samples, n_bins
        )
        
        results[p] = {
            'ece': ece,
            'mce': mce,
            'brier': brier
        }
        
        print(f"  ECE: {ece:.4f}, MCE: {mce:.4f}, 브라이어: {brier:.4f}")
    
    # ECE로 가장 좋은 것을 고른다
    best_rate = min(results.keys(), key=lambda p: results[p]['ece'])
    
    return best_rate, results

def evaluate_calibration(
    model,
    data_loader,
    mc_samples: int = 50,
    n_bins: int = 15
) -> Tuple[float, float, float]:
    """
    MC 드롭아웃 미루어 봄으로 눈금 맞음 자를 셈한다.
    
    Returns:
        ece: 바라는 눈금 맞음 어긋남
        mce: 가장 큰 눈금 맞음 어긋남
        brier: 브라이어 점수
    """
    model.eval()
    model.enable_mc_dropout()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            # MC 드롭아웃 미루어 봄
            probs_samples = []
            for _ in range(mc_samples):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                probs_samples.append(probs)
            
            mean_probs = torch.stack(probs_samples).mean(dim=0)
            all_probs.append(mean_probs.cpu())
            all_labels.append(y.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 자를 셈한다
    confidences, predictions = all_probs.max(dim=1)
    accuracies = predictions.eq(all_labels).float()
    
    # 통 나누기
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            
            gap = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * gap
            mce = max(mce, gap.item())
    
    # 브라이어 점수
    one_hot = torch.zeros_like(all_probs)
    one_hot.scatter_(1, all_labels.unsqueeze(1), 1)
    brier = ((all_probs - one_hot) ** 2).sum(dim=1).mean()
    
    return ece.item(), mce, brier.item()
```

---

## 6. 불확실성 성능의 트레이드오프

### 날카로움 대 캘리브레이션

**날카로움**(sharpness)은 예측 분포가 얼마나 한곳에 집중되어 있는지를 잰다.

$$
\text{Sharpness} = -\mathbb{E}[\mathbb{H}[\hat{p}]] = \mathbb{E}\left[\sum_c \hat{p}_c \log \hat{p}_c\right]
$$

드롭아웃이 셀수록 분포는 덜 날카로워진다(더 넓어진다).

**트레이드오프**:

- $p$가 낮으면: 날카롭지만 과신할 수 있다
- $p$가 높으면: 캘리브레이션은 잘 되나 지나치게 불확실해질 수 있다

```python
def compute_sharpness(probs: torch.Tensor) -> float:
    """
    미루어 봄의 날카로움(음수 엔트로피)을 셈한다.
    클수록 더 자신하는, 날카로운 미루어 봄이다.
    """
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return -entropy.mean().item()

def compute_calibration_sharpness_curve(
    model,
    val_loader,
    dropout_rates: List[float],
    mc_samples: int = 50
) -> dict:
    """
    드롭아웃 비율마다 눈금 맞음과 날카로움을 셈한다.
    
    맞바꿈을 눈으로 보기에 좋다.
    """
    results = {'dropout_rate': [], 'ece': [], 'sharpness': []}
    
    for p in dropout_rates:
        # 드롭아웃 비율을 잠깐 바꾼다
        original_rates = []
        for m in model.modules():
            if hasattr(m, 'p'):
                original_rates.append(m.p)
                m.p = p
        
        # 따진다
        all_probs = []
        all_labels = []
        
        model.enable_mc_dropout()
        with torch.no_grad():
            for x, y in val_loader:
                samples = [torch.softmax(model(x), -1) for _ in range(mc_samples)]
                mean_probs = torch.stack(samples).mean(dim=0)
                all_probs.append(mean_probs)
                all_labels.append(y)
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        # 자
        ece, _, _ = evaluate_calibration(model, val_loader, mc_samples)
        sharpness = compute_sharpness(all_probs)
        
        results['dropout_rate'].append(p)
        results['ece'].append(ece)
        results['sharpness'].append(sharpness)
        
        # 본디 비율로 되돌린다
        idx = 0
        for m in model.modules():
            if hasattr(m, 'p'):
                m.p = original_rates[idx]
                idx += 1
    
    return results
```

### 예측 성능의 트레이드오프

학습하는 동안 드롭아웃이 세면 정확도가 떨어질 수 있다.

```python
def accuracy_dropout_sweep(
    model_class,
    train_loader,
    val_loader,
    dropout_rates: List[float],
    **kwargs
) -> dict:
    """
    맞음과 드롭아웃 비율의 맞바꿈을 잰다.
    """
    results = {'p': [], 'train_acc': [], 'val_acc': [], 'gap': []}
    
    for p in dropout_rates:
        model = model_class(dropout_rate=p, **kwargs)
        history = train_model(model, train_loader, val_loader)
        
        results['p'].append(p)
        results['train_acc'].append(history['train_acc'][-1])
        results['val_acc'].append(history['val_acc'][-1])
        results['gap'].append(
            history['train_acc'][-1] - history['val_acc'][-1]
        )
    
    return results
```

---

## 7. 콘크리트 드롭아웃: 드롭아웃 비율 학습하기

### 왜 하는가

$p$를 하이퍼파라미터로 다루는 대신 학습 중에 함께 학습할 수 있다. 콘크리트 드롭아웃(갈 등, 2017)은 이산적인 베르누이 드롭아웃을 미분 가능한 연속 분포로 완화(relaxation)한다.

### 콘크리트 분포

콘크리트(또는 검벨-소프트맥스) 완화는

$$
z = \sigma\left( \frac{1}{\tau} \left( \log \frac{p}{1-p} + \log \frac{u}{1-u} \right) \right)
$$

여기서 $u \sim \text{Uniform}(0, 1)$이고 $\tau$는 온도다.

$\tau \to 0$이면 이는 베르누이 표본에 다가간다. 학습하는 동안에는 그래디언트가 흐르도록 $\tau > 0$을 쓴다.

### 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcreteDropout(nn.Module):
    """
    드롭아웃 확률을 배울 수 있는 콘크리트 드롭아웃 켜.
    
    베르누이 드롭아웃을 콘크리트로 풀어 써서
    그래디언트 내림으로 드롭아웃 비율을 배운다.
    """
    
    def __init__(
        self,
        init_p: float = 0.5,
        temperature: float = 0.1,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        # 드롭아웃 확률의 배울 수 있는 로짓
        # p = sigmoid(logit_p)
        init_logit = torch.log(torch.tensor(init_p / (1 - init_p)))
        self.logit_p = nn.Parameter(init_logit)
        
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
    @property
    def p(self) -> torch.Tensor:
        """이제의 드롭아웃 확률."""
        return torch.sigmoid(self.logit_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        
        if self.training:
            # 콘크리트로 풀어 쓰기
            u = torch.rand_like(x).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + 
                 torch.log(p) - torch.log(1 - p)) / self.temperature
            )
            z = 1 - s
        else:
            # 미루어 볼 때는 굳은 드롭아웃
            z = torch.bernoulli(torch.full_like(x, 1 - p))
        
        # 뒤집은 드롭아웃이므로 1/(1-p)으로 잣대를 맞춘다
        return x * z / (1 - p + 1e-8)
    
    def regularization_loss(
        self,
        layer_weight: torch.Tensor,
        num_data: int
    ) -> torch.Tensor:
        """
        이 켜의 정칙화 잃음을 셈한다.
        
        다음이 든다:
        1. 짐 정칙화(드롭아웃으로 잣대를 맞춤)
        2. p의 엔트로피 정칙화(뻔하지 않은 p을 배우게 이끔)
        """
        p = self.p
        
        # 짐 정칙화: λ * (1-p) * ||W||^2
        weight_reg = self.weight_regularizer * (1 - p) * (layer_weight ** 2).sum()
        
        # 드롭아웃 정칙화: p이 0과 1에서 멀어지게 이끈다
        # 고른 베르누이에 대한 KL 갈림을 쓴다
        dropout_reg = self.dropout_regularizer * num_data * (
            p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)
        )
        
        return weight_reg + dropout_reg

class ConcreteDropoutLinear(nn.Module):
    """
    콘크리트 드롭아웃을 얹은 선형 켜.
    
    들임에 배울 수 있는 드롭아웃을 걸어 nn.Linear을 감싼다.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_p: float = 0.5,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = ConcreteDropout(
            init_p=init_p,
            weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.linear(x)
    
    def regularization_loss(self, num_data: int) -> torch.Tensor:
        return self.dropout.regularization_loss(self.linear.weight, num_data)
    
    @property
    def p(self) -> float:
        return self.dropout.p.item()

class ConcreteDropoutNetwork(nn.Module):
    """
    콘크리트 드롭아웃으로 드롭아웃 비율을 배우는 네트워크.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        init_p: float = 0.5,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(
                ConcreteDropoutLinear(
                    prev_dim, hidden_dim,
                    init_p=init_p,
                    weight_regularizer=weight_regularizer,
                    dropout_regularizer=dropout_regularizer
                )
            )
            prev_dim = hidden_dim
        
        # 날임 켜(드롭아웃 없음)
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)
    
    def regularization_loss(self, num_data: int) -> torch.Tensor:
        """콘크리트 드롭아웃 켜 모두의 온 정칙화 잃음."""
        reg = 0
        for layer in self.layers:
            reg = reg + layer.regularization_loss(num_data)
        return reg
    
    def get_dropout_rates(self) -> List[float]:
        """켜마다 배운 드롭아웃 비율을 얻는다."""
        return [layer.p for layer in self.layers]

def train_concrete_dropout(
    model: ConcreteDropoutNetwork,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3
):
    """
    정칙화를 제대로 갖춘 콘크리트 드롭아웃 익힘 돌기.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    num_data = len(train_loader.dataset)
    
    for epoch in range(epochs):
        model.train()
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            output = model(x)
            nll_loss = criterion(output, y)
            reg_loss = model.regularization_loss(num_data)
            
            loss = nll_loss + reg_loss
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            rates = model.get_dropout_rates()
            print(f"{epoch+1}판: 배운 드롭아웃 비율: {rates}")
    
    return model
```

---

## 8. 실전 지침

### 판단 틀

```
1. 구조마다의 기본값에서 비롯한다
   - MLP: 숨은 켜에 0.5
   - CNN: 컨볼루션 0.25, 온통 이은 켜 0.5
   - 트랜스포머: 0.1

2. 자료 꾸러미 크기에 맞춘다
   - 자료가 적으면(1만 미만): 0.1~0.2 올린다
   - 자료가 많으면(10만 넘음): 0.1~0.2 내린다

3. 눈금 맞음 자로 따진다
   - 지나치게 자신하면(ECE는 낮은데 틀린 미루어 봄을 자신하면): p을 올린다
   - 지나치게 머뭇거리면(맞았는데도 아리송함이 크면): p을 내린다

4. 일의 요건을 헤아린다
   - 목숨이 걸린 일: p을 높여 아리송함을 조심스럽게 잡는다
   - 제때 미루어 봄: p을 낮춰 MC 표본을 덜 쓴다

5. 저절로 맞추려면 콘크리트 드롭아웃을 쓴다
   - 가장 좋은 비율이 아리송할 때
   - 켜마다 다른 비율이 있어야 할 때
```

### 자주 빠지는 함정

1. **모든 레이어에 같은 비율:** 레이어 유형마다 다른 비율이 필요하다

2. **모델 용량을 잊음:** 큰 모델에는 드롭아웃이 더 필요하다

3. **학습과 추론의 불일치:** MC 드롭아웃은 학습 때와 같은 비율을 써야 한다

4. **캘리브레이션을 잊음:** 정확도만으로는 불확실성이 좋은지 알 수 없다

5. **작업을 고려하지 않음:** 분포 외 탐지는 분포 내 예측보다 $p$가 높은 편이 나을 수 있다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 2-레이어 뉴럴 네트워크에서, 평균장 변분 추론 · 라플라스 근사 · MC 드롭아웃이 저마다 내놓는 근사 사후분포의 형태를 견주어라.

</div>

??? success "풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$는 다룰 수 없다. 세 근사 방법은 저마다 다룰 수 있는 형태를 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 중심으로 하고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 암묵적으로 정의된다. 근사마다 참 사후분포 형태의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
드롭아웃 비율을 달리한 MC 드롭아웃에서 얻은 불확실성 추정의 캘리브레이션을 딥 앙상블과 비교하는 실험을 설계하여라. 쓸 지표와 그림을 밝혀라.

</div>

??? success "풀이"
    지표: (1) 버킷 15개의 기대 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음의 로그 우도(NLL), (4) 분포 외 탐지의 AUROC. 그림: 방법마다 관측 빈도를 예측 신뢰도에 대고 그린 신뢰도 다이어그램(reliability diagram). 절차: 모든 방법을 CIFAR-10(분포 내)에서 학습하고, CIFAR-10 테스트 데이터에서 캘리브레이션을, SVHN에서 분포 외 탐지를 평가한다. 사후(post-hoc) 온도 스케일링을 베이스라인으로 쓴다. 무작위 시드 5개에 걸친 평균과 표준편차를 보고한다. 캘리브레이션이 잘 된 방법은 신뢰도 다이어그램에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 네트워크의 예측 분산이 인식적 불확실성과 우연적 불확실성으로 분해됨을 증명하여라. 학습 데이터 크기 $N \to \infty$일 때 두 항이 어떻게 되는지 보여라.

</div>

??? success "풀이"
    예측 분산은 전분산 법칙으로 분해된다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 항은 데이터 생성 과정의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 항은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 부근으로 수렴하므로 $O(1/N)$으로 줄어든다. 극한에서는 우연적 불확실성만 남는다. 이 분해는 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 절의 불확실성 추정 방법을 트레이딩 시스템의 포지션 크기 결정에 어떻게 쓸 수 있는지 논하여라. 실용적인 판단 규칙을 제시하여라.

</div>

??? success "풀이"
    판단 규칙: 포지션 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측 수익률, $\sigma_e^2$을 인식적 분산이라 하자. 포지션은 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$는 위험 회피 계수다. 인식적 불확실성이 크면(낯선 시장 상황) 포지션을 줄이고, 작으면(익숙한 국면) 더 과감하게 거래한다. 여기에 더해 인식적 불확실성의 상한을 두고 그 위로는 거래하지 않게 해 기권(abstain)할 수 있다. 이 틀은 모델의 신뢰도로 스케일한 켈리 기준 방식의 포지션 크기 결정을 자연스럽게 구현한다. 워크포워드(walk-forward) 분석으로 백테스트해 $\lambda$를 보정한다. $\square$

## 정리하며

이 절에서는 개관, 이론 기초, 레이어마다 다른 드롭아웃 비율, 데이터에 따른 선택, 캘리브레이션 기반 선택, 콘크리트 드롭아웃을 살펴보았다.

**살펴볼 거리**

1. Gal, Y., Hron, J., & Kendall, A. (2017). Concrete Dropout. *NeurIPS*.

2. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.

3. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.

4. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis*.
