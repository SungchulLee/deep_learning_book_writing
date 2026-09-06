# 레이블 평활화
## 개요

레이블 평활화는 분류 모델을 위한 정칙화 기법으로, 딱딱한 원-핫 목표 벡터를 모든 클래스에 작은 확률 질량을 나누어 주는 부드러운 목표로 바꾼다. 모델이 참 클래스에 확률을 전부 몰아주지 못하게 하여 지나치게 확신하는 예측을 억제하고 모델의 보정을 개선하며, 가중치 기반 및 데이터 기반 정칙화를 보완하는 출력 쪽 정칙화로 작동한다.

## 수학적 정식화

### 딱딱한 목표와 부드러운 목표

클래스가 $K$개인 분류 문제에서 참 클래스가 $k$인 표본의 표준 원-핫 목표는 다음과 같다.

$$
y_i^{\text{hard}} = \begin{cases} 1 & \text{if } i = k \\ 0 & \text{if } i \neq k \end{cases}
$$

레이블 평활화는 이를 부드러운 목표로 바꾼다.

$$
y_i^{\text{smooth}} = \begin{cases} 1 - \varepsilon & \text{if } i = k \\ \frac{\varepsilon}{K - 1} & \text{if } i \neq k \end{cases}
$$

여기서 $\varepsilon \in [0, 1)$은 평활화 매개변수이다. 동등하게 다음과 같이 쓴다.

$$
y^{\text{smooth}} = (1 - \varepsilon) \, y^{\text{hard}} + \frac{\varepsilon}{K} \, \mathbf{1}
$$

동등한 두 가지 표현에 주목하라. 목표가 아닌 $K-1$개 클래스에 $\varepsilon$을 고르게 나누는 방식(첫째 형태)과, 원-핫 벡터를 모든 $K$개 클래스에 대한 균등분포와 섞는 방식(둘째 형태)이다. 첫째 형태는 $\sum_i y_i^{\text{smooth}} = 1$을 보장하며, 둘째 형태도 마찬가지이다.

### 레이블 평활화를 쓰는 교차 엔트로피

딱딱한 목표를 쓰는 표준 교차 엔트로피 손실은 다음과 같다.

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} y_i^{\text{hard}} \log p_i = -\log p_k
$$

레이블 평활화를 쓰면 다음과 같다.

$$
\mathcal{L}_{\text{LS}} = -\sum_{i=1}^{K} y_i^{\text{smooth}} \log p_i = -(1 - \varepsilon) \log p_k - \frac{\varepsilon}{K-1} \sum_{i \neq k} \log p_i
$$

이는 다음과 같이 분해할 수 있다.

$$
\mathcal{L}_{\text{LS}} = (1 - \varepsilon) \, \mathcal{L}_{\text{CE}} + \varepsilon \, \mathcal{L}_{\text{uniform}}
$$

여기서 $\mathcal{L}_{\text{uniform}} = -\frac{1}{K} \sum_{i=1}^{K} \log p_i$은 균등한 목표 분포에 대한 교차 엔트로피이다.

### KL 발산으로서의 해석

레이블 평활화 손실은 다음과 동등하다.

$$
\mathcal{L}_{\text{LS}} = (1 - \varepsilon) \, H(y^{\text{hard}}, p) + \varepsilon \, H(u, p)
$$

여기서 $H(\cdot, \cdot)$은 교차 엔트로피를, $u = \frac{1}{K} \mathbf{1}$은 균등분포를 뜻한다. $H(u, p) = \log K + D_{\text{KL}}(u \| p)$이므로 레이블 평활화 손실을 최소화하면 모델 출력과 균등분포 사이의 KL 발산에 벌점을 주게 되고, 이는 로짓이 한없이 커지는 것을 막는다.

### 로짓에 미치는 영향

레이블 평활화가 없으면 교차 엔트로피 손실은 옳은 클래스의 로짓 $z_k$을 다른 로짓에 견주어 $\infty$으로 몰아간다. 레이블 평활화를 쓰면 최적의 로짓 배치는 다음을 만족한다.

$$
z_k - z_j = \log\frac{(1 - \varepsilon)(K - 1)}{\varepsilon} \quad \text{for all } j \neq k
$$

이 유한한 간격은 모델이 한없이 확신하게 되는 것을 막고 로짓의 크기를 유계로 유지한다.

## 레이블 평활화가 통하는 이유

### 지나친 확신 막기

딱딱한 목표는 모델이 옳은 클래스에 대해 $p_k = 1$을 내놓도록 이끄는데, 이는 로짓이 한없이 커져야 함을 뜻한다. 그 결과 지나치게 확신하며 일반화가 나쁜 예측을 하게 되고, 학습 데이터에 레이블 잡음이나 모호한 예가 있을 때 특히 그렇다.

### 개선된 보정

잘 보정된 모델의 예측 확률은 경험적 빈도와 일치한다. 모델이 80%의 확신을 말하면 실제로도 80%만큼 맞아야 한다. 레이블 평활화는 딱딱한 목표가 부추기는 극단적인 확률값을 막아 보정을 개선한다.

### 로짓 크기의 암묵적 정칙화

레이블 평활화는 큰 로짓 값에 벌점을 주어 출력층에만 적용한 가중치 감쇠와 비슷하게 작동한다. 이 벌점은 모델이 확신하되 지나치지는 않은 예측을 내놓도록 이끈다.

### 표현에 대한 군집 효과

Müller 등(2019)은 레이블 평활화가 같은 클래스의 끝에서 둘째 층 표현이 더 촘촘히 모이고 다른 클래스의 표현으로부터 같은 거리에 놓이도록 이끌어, 더 짜임새 있고 전이하기 좋은 특징 공간을 만든다는 것을 보였다.

## PyTorch 구현

### 내장 교차 엔트로피 쓰기

PyTorch의 `CrossEntropyLoss`는 레이블 평활화를 바로 지원한다.

```python
import torch
import torch.nn as nn

# 내장 레이블 평활화 (PyTorch >= 1.10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 사용법
logits = model(inputs)       # (batch_size, num_classes)
loss = criterion(logits, targets)  # 목푯값은 클래스 인덱스이다 (LongTensor)
```

### 직접 구현하기

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    레이블 평활화를 쓰는 교차 엔트로피 손실.
    
    인수:
        epsilon: [0, 1) 안의 평활화 매개변수. 기본값: 0.1.
        reduction: 'mean', 'sum', 또는 'none'.
    """
    
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        인수:
            logits: 모델의 날 출력, 모양 (batch_size, num_classes)
            targets: 클래스 인덱스, 모양 (batch_size,)
            
        반환값:
            레이블 평활화된 교차 엔트로피 손실
        """
        num_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 딱딱한 목표 성분: -log p_k
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # 균등 성분: -(1/K) sum_i log p_i
        smooth_loss = -log_probs.mean(dim=-1)
        
        # 결합된 손실
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

### 부드러운 목표를 쓰는 레이블 평활화

목표가 이미 확률일 때(예: 지식 증류나 믹스업에서 온 경우) 다음과 같이 한다.

```python
class SoftTargetCrossEntropy(nn.Module):
    """
    부드러운 목표 분포를 받는 교차 엔트로피 손실.
    
    원-핫 목표와 부드러운 목표를 모두 지원한다. 필요하면 부드러운 목표 위에
    레이블 평활화를 더 얹을 수도 있다.
    """
    
    def __init__(self, epsilon: float = 0.0):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        인수:
            logits: 모양 (batch_size, num_classes)
            targets: 모양 (batch_size, num_classes) — 부드러운 확률 목표
        """
        num_classes = logits.size(-1)
        
        # 요청되면 추가 평활화 적용
        if self.epsilon > 0:
            targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
        
        return loss.mean()
```

### 학습 루프와의 통합

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_label_smoothing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epsilon: float = 0.1,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    레이블 평활화로 모델을 학습시킨다.
    
    인수:
        model: 분류 신경망
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        epsilon: 레이블 평활화 매개변수
        epochs: 학습 에포크 수
        lr: 학습률
        
    반환값:
        학습 이력
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=epsilon)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_confidence': []
    }
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        # 검증
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_confidences = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = nn.CrossEntropyLoss()(outputs, y_batch)  # 공정한 비교를 위한 딱딱한 교차 엔트로피
                
                probs = torch.softmax(outputs, dim=-1)
                max_probs, predicted = probs.max(1)
                
                val_loss += loss.item() * X_batch.size(0)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
                all_confidences.append(max_probs)
        
        avg_confidence = torch.cat(all_confidences).mean().item()
        
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        history['val_confidence'].append(avg_confidence)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Val Acc={val_correct/val_total:.4f}, "
                  f"Avg Confidence={avg_confidence:.4f}")
    
    return history
```

## 보정 재기

### 기대 보정 오차 (ECE)

```python
import numpy as np

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    기대 보정 오차를 계산한다.
    
    인수:
        probs: 양성/선택된 클래스에 대한 예측 확률, 모양 (n,)
        labels: 맞았는지를 나타내는 이진 지시자, 모양 (n,)
        n_bins: 보정에 쓸 구간의 개수
        
    반환값:
        ECE 값 (작을수록 잘 보정된 것이다)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            avg_confidence = probs[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()
            ece += (n_in_bin / len(probs)) * abs(avg_accuracy - avg_confidence)
    
    return ece

def evaluate_calibration(model, data_loader, device='cpu'):
    """데이터셋에서 모델의 보정을 평가한다."""
    model.eval()
    all_probs = []
    all_correct = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=-1)
            max_probs, predicted = probs.max(1)
            
            all_probs.append(max_probs.cpu().numpy())
            all_correct.append(predicted.eq(y_batch).cpu().numpy())
    
    probs = np.concatenate(all_probs)
    correct = np.concatenate(all_correct)
    
    ece = expected_calibration_error(probs, correct)
    print(f"ECE: {ece:.4f}")
    print(f"Accuracy: {correct.mean():.4f}")
    print(f"Mean confidence: {probs.mean():.4f}")
    
    return ece
```

## 특정 구조에서의 레이블 평활화

### 트랜스포머

레이블 평활화는 트랜스포머 학습의 표준 구성 요소이다. 원 논문 "Attention Is All You Need"는 $\varepsilon = 0.1$을 쓴다.

```python
class TransformerClassifier(nn.Module):
    """레이블 평활화를 쓰는 트랜스포머 기반 분류기."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, 
                 num_classes, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, 
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # 전역 평균 풀링
        return self.classifier(x)

# 표준 트랜스포머 학습 설정
model = TransformerClassifier(
    vocab_size=30000, d_model=512, n_heads=8,
    n_layers=6, num_classes=1000
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

### 지식 증류와의 관계

레이블 평활화는 자기 증류와 관련이 있다. 레이블 평활화가 만드는 부드러운 목표는 지식 증류에서 교사 모델이 내놓는 부드러운 목표와 닮았다. 다만 교사가 배운 분포 대신 목표가 아닌 클래스에 균등분포를 쓴다는 점이 다르다.

$$
y^{\text{distill}}_i = \begin{cases}
(1 - \varepsilon) + \varepsilon \, p_i^{\text{teacher}} & \text{if } i = k \\
\varepsilon \, p_i^{\text{teacher}} & \text{if } i \neq k
\end{cases}
$$

레이블 평활화는 $p^{\text{teacher}}$이 균등한 특수한 경우이다.

## 다른 정칙화와 결합하기

레이블 평활화는 학습 목표를 바꾸는 다른 기법들과 상호작용한다.

```python
def combined_augmentation_training_step(
    model, images, labels, criterion,
    use_mixup=True, mixup_alpha=0.2,
    label_smoothing=0.1
):
    """
    레이블 평활화와 믹스업을 결합한 학습 단계.
    
    믹스업은 이미 부드러운 목표를 만들므로 함께 쓸 때에는
    레이블 평활화를 줄인다.
    """
    num_classes = 10  # 예
    
    if use_mixup:
        # 믹스업이 부드러운 목표를 만든다 — 평활화를 줄여 쓴다
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        index = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # 믹스업으로 부드러운 목표 만들기
        targets_a = torch.zeros(images.size(0), num_classes, device=images.device)
        targets_a.scatter_(1, labels.unsqueeze(1), 1.0)
        targets_b = torch.zeros(images.size(0), num_classes, device=images.device)
        targets_b.scatter_(1, labels[index].unsqueeze(1), 1.0)
        
        soft_targets = lam * targets_a + (1 - lam) * targets_b
        
        # 가벼운 추가 평활화 적용
        reduced_epsilon = label_smoothing * 0.5
        soft_targets = (1 - reduced_epsilon) * soft_targets + reduced_epsilon / num_classes
        
        logits = model(mixed_images)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    else:
        logits = model(images)
        loss = criterion(logits, labels)  # criterion에 label_smoothing이 내장되어 있다
    
    return loss
```

## 실무 지침

### epsilon 고르기

| 상황 | 권장 $\varepsilon$ |
|----------|--------------------------|
| 표준 분류 | 0.1 |
| 잡음이 있는 레이블 | 0.1 – 0.3 |
| 클래스가 적을 때 ($K < 10$) | 0.05 – 0.1 |
| 클래스가 많을 때 ($K > 100$) | 0.1 – 0.2 |
| 믹스업/컷믹스와 함께 | 0.05 (줄임) |
| 지식 증류 | 0.0 – 0.05 (교사가 부드러운 목표를 준다) |

### 레이블 평활화를 쓸 때

1. **분류 과제**: 기본값으로 두어도 거의 언제나 이롭다
2. **지나치게 확신하는 모델**: 예측 확률의 보정이 나쁠 때
3. **잡음이 있는 레이블**: 잘못 붙은 레이블의 영향을 누그러뜨린다
4. **대규모 학습**: ImageNet과 언어 모델 학습의 표준 관행이다

### 레이블 평활화를 쓰지 말아야 할 때

1. **회귀 과제**: 해당되지 않는다(목표가 연속이다)
2. **클래스 불균형이 극심한 이진 과제**: 드문 클래스의 학습을 방해할 수 있다
3. **정확한 확신도가 필요할 때**: 레이블 평활화는 확신도를 조직적으로 낮춘다
4. **뒤이은 증류**: 암묵적 지식의 전달을 해칠 수 있다(Müller 등, 2019)

## 참고 문헌

1. Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*. (Introduced label smoothing.)
2. Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS*.
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*. (Uses $\varepsilon = 0.1$ for transformer training.)
4. Pereyra, G., et al. (2017). Regularizing Neural Networks by Penalizing Confident Output Distributions. *ICLR Workshop*.

## 연습문제

**연습문제 1.**
레이블 평활화의 목표 분포와 그것이 교차 엔트로피 손실에 미치는 영향을 유도하라.

??? success "연습문제 1 풀이"
    평활화된 목표는 $y'_k = (1-\epsilon)y_k + \epsilon/K$이다. 참 클래스 $c$에 대해서는 $y'_c = 1 - \epsilon + \epsilon/K = 1 - \epsilon(K-1)/K$이고, 다른 클래스에 대해서는 $y'_k = \epsilon/K$이다. 손실은 표준 교차 엔트로피와 균등 교차 엔트로피의 혼합이 된다.

---

**연습문제 2.**
레이블 평활화가 모델의 보정을 개선하는 방식을 설명하라.

??? success "연습문제 2 풀이"
    딱딱한 레이블은 로짓을 $\pm\infty$ 쪽으로 밀어 모델을 지나치게 확신하게 만든다. 레이블 평활화는 틀린 클래스에도 약간의 확률 질량을 두도록 보상하여 지나친 확신에 벌점을 주며, 그 결과 더 잘 보정된 확률 추정(예측 확신도가 실제 정확도와 맞는)을 얻는다.

---

**연습문제 3.**
내장 `label_smoothing` 매개변수를 쓰지 않고 PyTorch에서 레이블 평활화를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def label_smoothing_loss(logits, targets, epsilon=0.1):
        K = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1 - epsilon) * nll + epsilon * smooth).mean()
    ```

---

**연습문제 4.**
레이블 평활화를 쓰지 말아야 할 때는 언제인가?

??? success "연습문제 4 풀이"
    (1) 참 레이블이 이미 부드럽거나 확률적일 때, (2) 뒤따르는 과제가 잘 보정된 확률을 요구할 때(레이블 평활화는 특정 보정 지표에서 오히려 보정을 망칠 수 있다), (3) 지식 증류를 쓸 때(교사의 부드러운 목표가 이미 평활화를 제공한다) 레이블 평활화를 피하라.
