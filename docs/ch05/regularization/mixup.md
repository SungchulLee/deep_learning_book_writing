# 믹스업

믹스업은 학습 예와 그 레이블 쌍의 *볼록 결합*으로 신경망을 학습시키는 데이터 증강 겸 정칙화 기법이다. 기존 데이터 점 사이에 놓이는 가상의 학습 표본을 만들어 모델이 학습 예 사이에서 선형에 가깝게 행동하도록 이끌며, 그 결과 결정 경계가 더 매끄러워지고 일반화가 좋아지며 예측의 보정도 나아진다.

---

## 1. 수학적 정식화

### 핵심 연산

두 학습 예 $(x_i, y_i)$과 $(x_j, y_j)$이 주어지면 믹스업은 가상의 예를 만든다.

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

여기서 혼합 계수 $\lambda$은 베타분포에서 뽑는다.

$$
\lambda \sim \text{Beta}(\alpha, \alpha), \quad \alpha > 0
$$

초매개변수 $\alpha$은 보간의 강도를 조절한다. $\alpha \to 0$이면 $\lambda$이 0과 1에 몰리고(섞이지 않는다), $\alpha \to \infty$이면 $\lambda$이 0.5에 몰린다(가장 많이 섞인다).

### 근방 위험 최소화

표준적인 경험적 위험 최소화(ERM)는 학습 분포가 학습 점을 중심으로 하는 델타 함수의 모임이라고 가정한다.

$$
\mathcal{L}_{\text{ERM}} = \frac{1}{n} \sum_{i=1}^{n} \ell\left(f(x_i), y_i\right)
$$

믹스업은 각 학습 점 둘레에 근방 분포를 정의하는 **근방 위험 최소화(VRM)**를 구현한다. 믹스업의 근방은 모든 볼록 결합의 모임이다.

$$
\mathcal{L}_{\text{Mixup}} = \mathbb{E}_{\lambda \sim \text{Beta}(\alpha, \alpha)} \left[ \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \ell\left(f(\lambda x_i + (1-\lambda) x_j), \; \lambda y_i + (1-\lambda) y_j\right) \right]
$$

실무에서는 $n^2$개의 쌍을 모두 계산하지 않고 미니배치 안에서 무작위 순열로 쌍을 뽑는다.

### 베타분포의 성질

베타$(\alpha, \alpha)$ 분포는 0.5를 중심으로 대칭이다.

| $\alpha$ | 분포의 모양 | 섞임의 양상 |
|----------|-------------------|-----------------|
| $\alpha \to 0$ | 0과 1에 몰림 | 거의 섞이지 않음 (ERM으로 돌아감) |
| $\alpha = 0.2$ | U자 모양 | 대개 한쪽 표본, 이따금 섞임 |
| $\alpha = 1.0$ | [0, 1] 위의 균등분포 | 모든 혼합 비율이 똑같이 가능 |
| $\alpha = 2.0$ | 0.5 둘레의 종 모양 | 대개 같은 비중으로 섞임 |

분류에서는 보통 $\alpha \in [0.1, 0.4]$이 가장 잘 통하며, 클래스 경계를 지나치게 흐리지 않으면서 충분한 정칙화를 준다.

### 레이블의 표현

믹스업은 레이블을 연속 벡터로 나타낼 수 있어야 한다. 클래스가 $K$개인 분류에서는 섞기 전에 레이블을 원-핫 벡터로 바꾼다.

$$
y_i = e_{c_i} \in \mathbb{R}^K, \quad \tilde{y} = \lambda \, e_{c_i} + (1-\lambda) \, e_{c_j}
$$

그런 다음 교차 엔트로피로 이 부드러운 목표에 대한 손실을 계산한다.

$$
\ell(\tilde{y}, p) = -\sum_{k=1}^{K} \tilde{y}_k \log p_k = -\lambda \log p_{c_i} - (1-\lambda) \log p_{c_j}
$$

---

## 2. 믹스업이 통하는 이유

### 선형 보간이라는 사전 지식

믹스업은 모델이 학습 예 사이에서 근사적으로 선형으로 행동해야 한다는 사전 지식을 강제한다.

$$
f(\lambda x_i + (1-\lambda) x_j) \approx \lambda f(x_i) + (1-\lambda) f(x_j)
$$

이는 강하지만 이로운 귀납 편향으로, 결정 경계를 더 매끄럽게 하고 학습 점 사이의 진동을 줄인다.

### 정칙화 효과

믹스업은 개별 학습 예를 외우는 모델의 능력을 줄여 정칙화 장치로 작동한다. 신경망은 의미 있는 보간을 뒷받침하는 표현을 배워야 하며, 이는 더 단순하고 더 잘 일반화되는 특징을 선호하게 만든다.

### 보정의 개선

딱딱한 0/1 레이블 대신 부드러운 목표의 연속 분포로 학습하므로 믹스업은 출력 확률이 더 잘 보정된 모델을 만든다. 즉 예측된 확신도가 참 정확도에 더 가깝게 들어맞는다.

### 기울기 분석

Thulasidasan 등(2019)은 믹스업이 야코비 행렬 $\frac{\partial f(x)}{\partial x}$의 노름을 줄여, 입력의 섭동에 대한 모델의 민감도를 제한하는 야코비 정칙화의 한 형태로 작동함을 보였다.

---

## 3. PyTorch 구현

### 기본적인 믹스업

```python
import torch
import torch.nn as nn
import numpy as np

def mixup_data(x: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.2) -> tuple:
    """
    데이터 배치에 믹스업을 적용한다.
    
    인수:
        x: 입력 배치, 모양 (batch_size, ...)
        y: 레이블(클래스 인덱스), 모양 (batch_size,)
        alpha: 베타분포의 매개변수. 클수록 더 많이 섞인다.
        
    반환값:
        mixed_x: 섞인 입력
        y_a: 원래 레이블
        y_b: 순열을 적용한 레이블
        lam: 혼합 계수
    """
    # Beta(alpha, alpha)에서 섞는 비율을 뽑는다. 대칭이라 평균은 늘 0.5지만
    # 모양이 alpha에 달렸다. alpha가 작으면(0.2 따위) 0이나 1 가까이 몰려
    # "거의 원본"인 표본이 많아지고, alpha가 1이면 균등분포가 되어
    # 절반씩 섞인 표본이 많아진다
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0   # 믹스업을 끈 것과 같다

    batch_size = x.size(0)
    # 새 배치를 따로 불러오지 않고 같은 배치를 뒤섞어 제 자신과 짝짓는다.
    # 데이터 로더를 건드리지 않아도 되므로 어디에나 끼워 넣기 쉽다
    index = torch.randperm(batch_size, device=x.device)

    # 입력은 여기서 바로 섞는다
    mixed_x = lam * x + (1 - lam) * x[index]
    # 레이블은 섞지 않고 두 벌을 그대로 돌려준다. 분류에서는 y가
    # 클래스 번호(정수)라 lam*y_a + (1-lam)*y_b 가 뜻이 없기 때문이다.
    # 대신 아래 mixup_criterion에서 손실을 그 비율로 섞는다
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """
    믹스업 손실을 두 표준 손실의 가중 결합으로 계산한다.
    
    인수:
        criterion: 바탕이 되는 손실 함수 (예: CrossEntropyLoss)
        pred: 모델의 예측
        y_a: 첫째 레이블 집합
        y_b: 둘째 레이블 집합
        lam: 혼합 계수
    """
    # 손실을 섞는 것이 레이블을 섞는 것과 같아지는 까닭.
    # 교차 엔트로피는 레이블에 대해 선형이므로
    #   CE(pred, lam*y_a + (1-lam)*y_b) = lam*CE(pred, y_a) + (1-lam)*CE(pred, y_b)
    # 가 성립한다. 그래서 원-핫을 만들어 섞지 않고도 같은 결과를 얻는다.
    #
    # 주의: 이 등식은 교차 엔트로피처럼 레이블에 선형인 손실에서만
    # 성립한다. 초점 손실이나 다른 비선형 손실에는 그대로 쓸 수 없다
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### 완전한 학습 루프

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_mixup(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 0.2,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    믹스업 증강으로 모델을 학습시킨다.
    
    인수:
        model: 신경망
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        alpha: 믹스업 보간의 세기
        epochs: 학습 에포크 수
        lr: 학습률
        
    반환값:
        학습 이력
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 믹스업을 쓰는 학습
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 믹스업 적용.
            # 컷믹스의 학습 루프와 달리 확률로 켜고 끄지 않고 모든
            # 배치에 건다. 다만 베타분포에서 뽑은 lam이 0이나 1에
            # 가까운 배치는 사실상 원본이므로, 깨끗한 이미지도
            # 자연히 섞여 들어간다
            mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha)
            
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
            loss.backward()
            optimizer.step()
            # 이 손실은 섞인 이름표에 대한 값이라 아래 검증 손실과
            # 같은 자로 잰 값이 아니다. 두 곡선을 겹쳐 그리면 안 된다
            train_loss += loss.item()
        
        # 검증 (믹스업 없음).
        # 믹스업은 학습에만 건다. 시험에서 만날 것은 섞이지 않은
        # 이미지이므로, 평가는 그 조건에서 해야 뜻이 있다
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

## 4. 믹스업의 변형

### 다양체 믹스업

다양체 믹스업은 입력 공간에서 섞는 대신 무작위로 고른 층의 은닉 표현에 보간을 적용한다.

```python
class ManifoldMixupModel(nn.Module):
    """
    무작위 은닉층에서 다양체 믹스업을 지원하는 모델.
    
    참고: Verma 등, "Manifold Mixup: Better Representations by
               은닉 상태 사이 메우기"(ICML 2019)
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        # 인덱싱을 위해 층을 목록으로 만든다
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x, mixup_layer=None, lam=None, index=None):
        """
        다양체 믹스업을 선택적으로 쓰는 순전파.
        
        인수:
            x: 입력 텐서
            mixup_layer: 믹스업을 적용할 층의 인덱스 (None이면 믹스업 없음)
            lam: 혼합 계수
            index: 배치에 대한 순열 인덱스
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 고른 층에서 믹스업 적용
            if mixup_layer is not None and i == mixup_layer:
                x = lam * x + (1 - lam) * x[index]
        
        return self.output(x)

def train_step_manifold_mixup(model, X_batch, y_batch, criterion, 
                               optimizer, alpha=0.2):
    """다양체 믹스업을 쓰는 학습 단계 하나."""
    optimizer.zero_grad()
    
    # 섞을 층을 무작위로 고르기
    n_layers = len(model.layers)
    mixup_layer = np.random.randint(0, n_layers + 1)  # +1은 입력 공간을 포함한다
    
    # 혼합 계수 뽑기
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # 순열 만들기
    batch_size = X_batch.size(0)
    index = torch.randperm(batch_size, device=X_batch.device)
    
    if mixup_layer == 0:
        # 입력 공간에서의 믹스업
        mixed_x = lam * X_batch + (1 - lam) * X_batch[index]
        outputs = model(mixed_x)
    else:
        # 은닉층에서의 믹스업
        outputs = model(X_batch, mixup_layer=mixup_layer - 1, 
                       lam=lam, index=index)
    
    # 섞인 레이블
    loss = lam * criterion(outputs, y_batch) + (1 - lam) * criterion(outputs, y_batch[index])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 배치 수준의 믹스업 전략

```python
class BatchMixup:
    """
    여러 짝짓기 전략을 갖춘 유연한 믹스업.
    """
    
    def __init__(self, alpha=0.2, strategy='random'):
        """
        인수:
            alpha: 베타분포의 매개변수
            strategy: 짝짓기 전략 — 'random', 'cross_class', 'same_class'
        """
        self.alpha = alpha
        self.strategy = strategy
    
    def __call__(self, x, y):
        if self.alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        
        if self.strategy == 'random':
            index = torch.randperm(batch_size, device=x.device)
        
        elif self.strategy == 'cross_class':
            # 각 표본을 다른 클래스의 표본과 짝짓기
            index = self._cross_class_permutation(y)
        
        elif self.strategy == 'same_class':
            # 각 표본을 같은 클래스의 표본과 짝짓기.
            # 이 경우 y_a와 y_b가 같아 이름표는 섞이지 않는다. 즉
            # 정칙화가 아니라 같은 클래스 안에서 새 표본을 지어내는
            # 쪽에 가깝고, 결정 경계를 매끄럽게 하는 믹스업 본래의
            # 효과는 사라진다
            index = self._same_class_permutation(y)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # 컷믹스와 갈리는 지점이다. 컷믹스는 조각을 오려 붙여 어느
        # 화소든 두 이미지 가운데 하나에서 오지만, 믹스업은 화소마다
        # 두 이미지를 겹쳐 반투명하게 만든다. 그래서 lam은 넓이의
        # 비율이 아니라 밝기의 비율이며, 다시 계산할 일도 없다
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam
    
    def _cross_class_permutation(self, y):
        """서로 다른 클래스를 짝짓는 순열을 만든다."""
        batch_size = y.size(0)
        index = torch.randperm(batch_size, device=y.device)
        
        # 되도록 클래스를 가로질러 짝지으려 시도한다.
        # 같은 클래스끼리 섞으면 이름표가 그대로라 배울 것이 없으므로
        # 그런 짝을 풀어 준다. 배치가 한 클래스로만 채워졌거나 클래스가
        # 둘뿐이면 풀 상대가 없어 그대로 남는다. 보장이 아니라 발견법이다
        for i in range(batch_size):
            if y[i] == y[index[i]]:
                # 클래스가 다른 교환 상대 찾기.
                # 앞의 컷믹스판과 달리 i+1부터 훑는다. 이미 손본 앞쪽을
                # 다시 건드리지 않으므로 고쳐 놓은 짝이 도로 망가지지 않는다
                for j in range(i + 1, batch_size):
                    if y[i] != y[index[j]] and y[j] != y[index[i]]:
                        index[i], index[j] = index[j].clone(), index[i].clone()
                        break
        return index
    
    def _same_class_permutation(self, y):
        """같은 클래스를 짝짓는 순열을 만든다."""
        batch_size = y.size(0)
        index = torch.arange(batch_size, device=y.device)
        
        # 클래스 안에서 섞기
        for c in y.unique():
            class_mask = (y == c).nonzero(as_tuple=True)[0]
            if len(class_mask) > 1:
                perm = class_mask[torch.randperm(len(class_mask))]
                index[class_mask] = perm
        
        return index
```

### 회귀를 위한 믹스업

믹스업은 레이블 섞는 방식을 전혀 바꾸지 않고도 회귀 과제에 바로 적용된다.

```python
def mixup_regression(x, y, alpha=0.2):
    """
    회귀 과제를 위한 믹스업.
    
    회귀의 목푯값은 이미 연속이므로 레이블 섞기는
    단순한 보간이 된다.
    """
    # 섞는 비율 lam을 Beta(alpha, alpha)에서 뽑는다.
    # alpha가 작으면(0.2 따위) 뽑히는 값이 0이나 1 가까이 몰려
    # "거의 원본"인 표본이 많아진다. alpha가 1이면 균등분포가 되어
    # 절반씩 섞인 표본이 많아진다. 곧 alpha가 섞기의 세기를 정한다
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0   # alpha=0이면 섞지 않는다(믹스업을 끈 것과 같다)

    # 배치를 뒤섞은 인덱스. 새 배치를 따로 만들지 않고 같은 배치를
    # 제 자신과 짝지어 섞는 것이 믹스업의 요령이다
    index = torch.randperm(x.size(0), device=x.device)

    # 입력과 목푯값을 같은 lam으로 섞는다. 두 곳에 같은 비율을 써야
    # "입력을 섞으면 답도 그만큼 섞인다"는 선형성 가정이 지켜진다
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    # 분류에서는 레이블이 원-핫이라 손실 쪽에서 lam으로 나누어 셈해야 하지만,
    # 회귀는 목푯값이 이미 연속이므로 이렇게 곧바로 섞으면 끝난다
    return mixed_x, mixed_y
```

---

## 5. 믹스업을 다른 기법과 결합하기

### 믹스업 + 레이블 평활화

둘 다 부드러운 목표를 만든다. 함께 쓸 때에는 레이블 평활화 매개변수를 줄인다.

```python
def mixup_with_label_smoothing(model, x, y, alpha=0.2, epsilon=0.05):
    """
    믹스업을 가벼운 레이블 평활화와 결합한다.
    
    믹스업이 이미 레이블을 부드럽게 하므로 epsilon을 줄여 쓴다.
    """
    num_classes = 10  # 필요에 따라 조정하라
    
    # 믹스업
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # 믹스업에서 온 부드러운 목표
    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    y_onehot_perm = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot_perm.scatter_(1, y[index].unsqueeze(1), 1.0)
    
    soft_targets = lam * y_onehot + (1 - lam) * y_onehot_perm
    
    # 추가 레이블 평활화 적용
    soft_targets = (1 - epsilon) * soft_targets + epsilon / num_classes
    
    # 손실을 계산한다
    logits = model(mixed_x)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    
    return loss
```

### 믹스업 + 컷믹스

배치마다 믹스업과 컷믹스 중 하나를 무작위로 고른다.

```python
def mixup_or_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, 
                     cutmix_prob=0.5):
    """배치마다 믹스업이나 컷믹스 중 하나를 무작위로 적용한다."""
    if np.random.random() < cutmix_prob:
        # 컷믹스 적용 (구현은 cutmix.md 참고)
        return cutmix_data(x, y, alpha=cutmix_alpha)
    else:
        return mixup_data(x, y, alpha=mixup_alpha)
```

---

## 6. 실무 지침

### 초매개변수 선택

| 매개변수 | 권장 범위 | 비고 |
|-----------|------------------|-------|
| $\alpha$ (CIFAR-10) | 0.2 – 1.0 | 데이터셋이 작을수록 크게 |
| $\alpha$ (ImageNet) | 0.1 – 0.4 | 0.2가 표준 |
| $\alpha$ (텍스트) | 0.1 – 0.2 | 이산 데이터에는 작게 |
| $\alpha$ (회귀) | 0.1 – 0.4 | 분류와 비슷 |

### 믹스업을 쓸 때

1. **학습 데이터가 적을 때**: 데이터가 부족할 때 믹스업이 가장 이롭다
2. **지나치게 확신하는 모델**: 모델의 예측 보정이 나쁠 때
3. **클래스가 많은 분류**: 부드러운 목표가 세밀한 구분에 도움이 된다
4. **적대적 견고성이 필요할 때**: 믹스업은 작은 섭동에 대한 견고성을 높인다

### 믹스업을 피할 때

1. **물체 검출/분할**: 공간적 레이블 섞기가 간단하지 않다(대신 컷믹스를 고려하라)
2. **$\alpha$이 아주 클 때**: 심하게 섞인 예는 최적화기를 혼란스럽게 할 수 있다
3. **짜임새가 강한 데이터**: 보간이 의미를 보존하지 못할 수 있는 경우

### 평가에 대한 참고

평가 시점에는 믹스업을 **결코 적용하지 않는다**. 모델은 손대지 않은 깨끗한 데이터로 평가한다.

---

## 연습문제

**연습문제 1.**
믹스업의 학습 목적 함수를 유도하고 왜 $\lambda \sim \text{Beta}(\alpha, \alpha)$인지 설명하라.

??? success "연습문제 1 풀이"
    믹스업은 가상의 학습 예를 만든다. $\lambda \sim \text{Beta}(\alpha, \alpha)$일 때 $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$이다. 베타분포는 대칭이며 $\alpha$으로 조절된다. $\alpha \to 0$이면 섞이지 않고, $\alpha = 1$이면 균등하며, $\alpha \to \infty$이면 언제나 반반으로 섞인다.

---

**연습문제 2.**
PyTorch 학습 루프에서 믹스업을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def mixup_data(x, y, alpha=0.2):
        lam = torch.distributions.Beta(alpha, alpha).sample()
        idx = torch.randperm(x.size(0))
        mixed_x = lam * x + (1 - lam) * x[idx]
        return mixed_x, y, y[idx], lam

    mixed_x, y_a, y_b, lam = mixup_data(x, y)
    loss = lam * criterion(model(mixed_x), y_a) + (1-lam) * criterion(model(mixed_x), y_b)
    ```

---

**연습문제 3.**
근방 위험 최소화의 관점에서 믹스업이 정칙화 장치로 작동하는 방식을 설명하라.

??? success "연습문제 3 풀이"
    표준적인 ERM은 경험적 분포(학습 점에 놓인 점질량)에 대한 위험을 최소화한다. 믹스업은 점질량을 '근방'(쌍의 볼록 결합)으로 바꾸어 경험적 분포를 매끄럽게 한다. 이는 학습 점 사이에서 선형적인 행동을 이끌어 학습된 함수의 복잡도를 줄인다.

---

**연습문제 4.**
믹스업, 컷믹스, 다양체 믹스업을 비교하라. 각각은 언제 가장 알맞은가?

??? success "연습문제 4 풀이"
    믹스업은 화소 공간에서의 전역 보간으로 단순하고 효과적이다. 컷믹스는 한 이미지의 직사각형 조각을 다른 이미지에 붙여 공간 구조를 보존한다. 다양체 믹스업은 은닉층 공간에서 보간하여 더 추상적인 변이를 포착한다. 물체 검출에는 컷믹스가, 세밀한 인식에는 다양체 믹스업이 가장 잘 통한다.

## 정리하며

이 마당은 수학적 정식화、믹스업이 통하는 이유、PyTorch 구현、믹스업의 변형을 차례로 짚었다.

**참고 문헌**

1. Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
2. Verma, V., et al. (2019). Manifold Mixup: Better Representations by Interpolating Hidden States. *ICML*.
3. Thulasidasan, S., et al. (2019). On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks. *NeurIPS*.
4. Chapelle, O., Weston, J., Bottou, L., & Vapnik, V. (2001). Vicinal Risk Minimization. *NeurIPS*.
