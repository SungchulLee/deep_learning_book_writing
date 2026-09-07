# MC 드롭아웃의 드롭아웃 비율 고르기
## 두루 보기

드롭아웃 비율 $p$(낱자리를 떨어뜨릴 낌새)은 익힘의 정칙화와 아리송함 어림의 됨됨이에 함께 걸리는 종요로운 하이퍼파라미터다. 이 글은 몬테카를로 드롭아웃 쓰임에서 드롭아웃 비율을 이치에 닿게 고르는 길을 준다.

## 이론 밑바탕

### 앞선 분포와 뒷분포의 맞바꿈

변이 미루어 봄의 눈으로 보면 드롭아웃 비율 $p$은 짐에 대한 앞선 분포를 넌지시 정한다. 드롭아웃 비율, 짐 줄이기 $\lambda$, 앞선 분포의 사이는 이렇다.

$$
\lambda = \frac{p \ell^2}{2N\tau}
$$

여기서

- $\ell^2$은 앞선 분포의 길이 잣대
- $N$은 자료 꾸러미 크기
- $\tau$은 모형의 촘촘함(살핌 흩어짐의 거꿀)

**뜻하는 바**:

- **$p$이 클수록** → 정칙화가 셈 → 뒷분포가 넓어짐 → 아리송함이 커짐
- **$p$이 작을수록** → 정칙화가 여림 → 뒷분포가 좁아짐 → 아리송함이 작아짐

### 드롭아웃 분포의 흩어짐

배운 평균이 $m$인 짐 $w$에서 드롭아웃 분포는 다음을 지닌다.

$$
\mathbb{E}[w] = (1-p) \cdot m
$$

$$
\text{Var}[w] = p(1-p) \cdot m^2
$$

바뀜 값은

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

$p$이 클수록 짐의 흔들림이 커지고 따라서 미루어 보는 분포도 넓어진다.

### 날임 흩어짐의 퍼짐

드롭아웃 비율이 $p$인 켜 하나 $y = \sigma(Wx + b)$에서 흩어짐은 어림잡아 이렇게 퍼진다.

$$
\text{Var}[y_j] \approx \frac{p}{1-p} \cdot \mathbb{E}[y_j]^2 + (1-p) \cdot \text{Var}[\text{pre-activation}]
$$

깊은 그물에서는 이것이 켜에 걸쳐 쌓이므로 깊은 켜일수록 드롭아웃 비율에 더 예민해진다.

## 켜마다 다른 드롭아웃 비율

### 켜에 값을 매기는 길

켜의 갈래마다 알맞은 드롭아웃 비율이 다르다.

1. **이른 켜**(들임 가까이): 낮은 비율(0.1~0.3)
   - 든든해야 할 두루 쓰이는 결을 배운다
   - 너무 많이 떨어뜨리면 결의 층이 무너진다

2. **가운데 켜**: 가운데 비율(0.3~0.5)
   - 정칙화와 소식 흐름 사이의 저울질

3. **늦은 켜**(날임 가까이): 높은 비율(0.4~0.6)
   - 지나치게 맞추기 쉽다
   - 여기서의 MC 드롭아웃이 날임의 아리송함을 가장 곧바로 담는다

4. **엮음 켜**: 낮은 비율(0.1~0.3)
   - 자리의 얼개를 지켜야 한다
   - 자리 드롭아웃(결 그림을 통째로 떨어뜨리기)을 쓴다

5. **온통 이은 켜**: 높은 비율(0.3~0.5)
   - 매개변수가 많아 지나치게 맞추기 더 쉽다

### 얼개마다의 길잡이

```python
def get_recommended_dropout_rates(architecture: str) -> dict:
    """
    얼개 갈래에 따라 즐겨 쓰는 드롭아웃 비율을 얻는다.
    
    켜 갈래를 드롭아웃 비율에 맞춘 사전을 돌려준다.
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
            'after_conv': 0.0,  # 묶음 잣대 잡기가 정칙화를 맡는다
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

## 자료에 매인 고르기

### 자료 꾸러미 크기와의 사이

자료 꾸러미가 클수록 정칙화가 덜 있어야 한다.

$$
p_{\text{optimal}} \propto \frac{1}{\sqrt{N}}
$$

**겪어 본 길잡이**:

| 자료 꾸러미 크기 | 즐겨 쓰는 $p$ |
|--------------|-----------------|
| 1,000 미만 | 0.5 ~ 0.7 |
| 1,000 ~ 10,000 | 0.4 ~ 0.5 |
| 10,000 ~ 100,000 | 0.3 ~ 0.4 |
| 100,000 넘음 | 0.1 ~ 0.3 |

### 모형 담는 힘에 맞추기

모형이 클수록 정칙화가 더 있어야 한다.

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
    
    # 담는 힘 견줌: 자료 하나당 매개변수
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

## 눈금 맞음에 기댄 고르기

### 드롭아웃 비율과 눈금 맞음

드롭아웃 비율은 아리송함의 눈금 맞음에 곧바로 걸린다. 너무 낮으면 지나치게 자신하고, 너무 높으면 지나치게 머뭇거린다.

**바라는 눈금 맞음 어긋남(ECE)**:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|
$$

여기서 $\text{acc}(b)$은 통 $b$의 맞음이고 $\text{conf}(b)$은 평균 자신함이다.

### 가장 좋은 눈금 맞음을 찾는 격자 뒤지기

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
        model_class: 지어 낼 모형 갈래
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

## 아리송함 됨됨이의 맞바꿈

### 날카로움 대 눈금 맞음

**날카로움**은 미루어 보는 분포가 얼마나 모였는지를 잰다.

$$
\text{Sharpness} = -\mathbb{E}[\mathbb{H}[\hat{p}]] = \mathbb{E}\left[\sum_c \hat{p}_c \log \hat{p}_c\right]
$$

드롭아웃이 셀수록 분포는 덜 날카롭고(더 넓고) 된다.

**맞바꿈**:

- $p$이 낮으면: 날카롭지만 지나치게 자신할 수 있다
- $p$이 높으면: 눈금은 잘 맞으나 지나치게 아리송할 수 있다

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

### 미루어 보는 됨됨이의 맞바꿈

익히는 동안 드롭아웃이 세면 맞음이 떨어질 수 있다.

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

## 콘크리트 드롭아웃: 드롭아웃 비율 배우기

### 왜 하는가

$p$을 하이퍼파라미터로 다루는 대신 익히는 동안 배울 수 있다. 콘크리트 드롭아웃(갈 등, 2017)은 베르누이 드롭아웃을 이어지게 풀어 쓴다.

### 콘크리트 분포

콘크리트(또는 검벨-소프트맥스) 풀어 쓰기는

$$
z = \sigma\left( \frac{1}{\tau} \left( \log \frac{p}{1-p} + \log \frac{u}{1-u} \right) \right)
$$

여기서 $u \sim \text{Uniform}(0, 1)$이고 $\tau$은 온도다.

$\tau \to 0$이면 이는 베르누이 표본에 다가간다. 익히는 동안에는 기울기가 흐르도록 $\tau > 0$을 쓴다.

### 짜보기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteDropout(nn.Module):
    """
    드롭아웃 낌새를 배울 수 있는 콘크리트 드롭아웃 켜.
    
    베르누이 드롭아웃을 콘크리트로 풀어 써서
    기울기 내림으로 드롭아웃 비율을 배운다.
    """
    
    def __init__(
        self,
        init_p: float = 0.5,
        temperature: float = 0.1,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5
    ):
        super().__init__()
        
        # 드롭아웃 낌새의 배울 수 있는 로짓
        # p = sigmoid(logit_p)
        init_logit = torch.log(torch.tensor(init_p / (1 - init_p)))
        self.logit_p = nn.Parameter(init_logit)
        
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
    @property
    def p(self) -> torch.Tensor:
        """이제의 드롭아웃 낌새."""
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
    콘크리트 드롭아웃으로 드롭아웃 비율을 배우는 그물.
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

## 참으로 쓸 만한 길

### 판단 틀

```
1. 얼개마다의 기본값에서 비롯한다
   - MLP: 숨은 켜에 0.5
   - CNN: 엮음 0.25, 온통 이은 켜 0.5
   - 변환기: 0.1

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

1. **어디나 같은 비율:** 켜 갈래마다 다른 비율이 있어야 한다

2. **모형 담는 힘을 잊음:** 큰 모형에는 드롭아웃이 더 있어야 한다

3. **익힘과 미루어 봄의 어긋남:** MC 드롭아웃이 익힘 때 비율을 쓰게 해야 한다

4. **눈금 맞음을 잊음:** 맞음만으로는 아리송함이 좋은지 알 수 없다

5. **일을 헤아리지 않음:** 밖 분포 알아내기는 분포 안 미루어 봄보다 $p$이 높은 편이 나을 수 있다

## 살펴볼 거리

1. Gal, Y., Hron, J., & Kendall, A. (2017). Concrete Dropout. *NeurIPS*.

2. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.

3. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.

4. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis*.

## 익힘 문제

**익힘 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "익힘 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**익힘 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "익힘 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**익힘 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "익힘 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**익힘 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "익힘 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$
