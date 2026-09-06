# 컷아웃
## 개요

컷아웃(무작위 지우기라고도 한다)은 학습 중에 입력 이미지의 정사각형 또는 직사각형 영역을 무작위로 가리는 데이터 증강 기법이다. 이미지의 일부를 가림으로써 모델이 변별적인 조각 하나에 매달리지 않고 더 넓은 범위의 공간 특징에 기대게 하여 견고성과 일반화를 높인다.

## 수학적 정식화

### 핵심 연산

이미지 $x \in \mathbb{R}^{C \times H \times W}$이 주어지면 컷아웃은 무작위로 놓인 직사각형 영역 안에서는 0이고 그 밖에서는 1인 이진 마스크 $\mathbf{M} \in \{0, 1\}^{H \times W}$을 만든다.

$$
\tilde{x} = \mathbf{M} \odot x
$$

여기서 $\odot$은 채널 방향으로 방송되는 원소별 곱을 뜻한다. 레이블 $y$은 그대로 남는데, 이 점이 레이블까지 바꾸는 컷믹스와 컷아웃을 가른다.

### 마스크 만들기

1. 마스크의 중심을 균등하게 뽑는다. $(c_x, c_y) \sim \text{Uniform}([0, W] \times [0, H])$
2. 마스크의 크기 $s$을 정한다(변의 길이를 고정하거나 어떤 범위에서 뽑는다)
3. 이미지 경계로 잘라 낸 경계 상자를 계산한다.

$$
x_1 = \max(0,\; c_x - \lfloor s/2 \rfloor), \quad x_2 = \min(W,\; c_x + \lfloor s/2 \rfloor)
$$

$$
y_1 = \max(0,\; c_y - \lfloor s/2 \rfloor), \quad y_2 = \min(H,\; c_y + \lfloor s/2 \rfloor)
$$

4. $\mathbf{M}[y_1:y_2, x_1:x_2] = 0$으로 두고 나머지 성분은 1로 남긴다

중심을 이미지 경계 근처나 바깥까지 포함해 어디서나 뽑을 수 있게 하면 실효 마스크 크기가 달라지므로 가림의 강도에 자연스러운 변이가 생긴다.

### 채움값

원래 컷아웃 논문은 가려진 영역을 0(정규화 뒤의 평균 화소값)으로 채운다. 무작위 지우기는 이를 일반화하여 무작위 화소값, 채널별 평균, 또는 상수 채움을 허용한다.

$$
\tilde{x}_{c, y_1:y_2, x_1:x_2} = \begin{cases}
0 & \text{(zero fill — original Cutout)} \\
\mu_c & \text{(per-channel mean fill)} \\
\text{Uniform}(0, 1) & \text{(random fill — Random Erasing)}
\end{cases}
$$

### 관련 방법들과의 비교

| 방법 | 마스크의 대상 | 채움값 | 레이블이 바뀌는가? | 주된 이점 |
|--------|------------|------------|----------------|-------------|
| 컷아웃 | 직사각형을 0/상수로 | 0 또는 평균 | 아니다 | 공간적 견고성을 강제한다 |
| 무작위 지우기 | 직사각형을 무작위 값으로 | 무작위 화소 | 아니다 | 조각을 외우는 것을 막는다 |
| 컷믹스 | 다른 이미지의 직사각형 | 다른 표본의 내용 | 그렇다 (비례) | 낭비되는 화소가 없다 |
| 믹스업 | 전역 혼합 | 두 이미지의 가중합 | 그렇다 (비례) | 매끄러운 결정 경계 |

## 컷아웃이 통하는 이유

### 국소 조각에 대한 의존 막기

신경망은 변별력이 아주 큰 국소 조각 몇 개에 기대는 것만으로도 높은 학습 정확도를 얻을 수 있다. 컷아웃은 그런 조각을 무작위로 지워 모델이 입력의 전체 공간 범위를 쓰는 중복된 표현을 갖추도록 강제한다.

### 정칙화 효과

컷아웃은 입력 잡음 주입의 한 형태로 볼 수 있다. 기대 손실로 학습하는 모델 $f$에 대해 다음과 같다.

$$
\mathcal{L}_{\text{cutout}} = \mathbb{E}_{\mathbf{M}}\left[\ell(f(\mathbf{M} \odot x), y)\right]
$$

이는 모델이 가능한 모든 가림 형태 아래에서 손실을 최소화하도록 이끌어, 어느 한 공간 영역에 지나치게 기대는 것에 벌점을 준다.

### 드롭아웃과의 관계

컷아웃은 때때로 "입력에 대한 공간 드롭아웃"이라고 불린다. 표준 드롭아웃이 개별 원소를 무작위로 0으로 만드는 반면 컷아웃은 이어진 직사각형 영역을 0으로 만들어 이미지의 공간 구조에 더 잘 맞는다. 이는 공간 조각이 아니라 특징 맵 전체를 떨어뜨리는 `Dropout2d`(공간 드롭아웃)과 관련은 있지만 다르다.

### 개선된 물체 위치 파악

물체의 부분적인 모습으로 학습하면 모델은 부위만 보고도 물체를 알아보도록 배운다. 모델이 가장 변별적인 영역 하나에 기댈 수 없으므로 약지도 물체 위치 파악이 개선됨이 알려져 있다.

## PyTorch 구현

### 컷아웃 변환 직접 만들기

```python
import torch
import numpy as np

class Cutout:
    """
    이미지 텐서에서 정사각형 조각 하나 이상을 무작위로 가린다.
    
    참고: DeVries & Taylor, "Improved Regularization of CNNs with Cutout"
    
    인수:
        n_holes: 잘라 낼 조각의 수
        length: 각 정사각형 조각의 변 길이
        fill_value: 가려진 영역을 채울 값 (기본값: 0.0)
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16, 
                 fill_value: float = 0.0):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        텐서 이미지에 컷아웃을 적용한다.
        
        인수:
            img: 모양이 (C, H, W)인 텐서 이미지
            
        반환값:
            컷아웃 영역이 적용된 이미지
        """
        h, w = img.shape[-2:]
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            # 중심 뽑기
            cy = np.random.randint(h)
            cx = np.random.randint(w)
            
            # 상자 계산 (잘라 냄)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            
            mask[..., y1:y2, x1:x2] = 0
        
        if self.fill_value == 0.0:
            return img * mask
        else:
            return img * mask + self.fill_value * (1 - mask)
```

### PyTorch의 내장 RandomErasing 쓰기

PyTorch는 채움값과 가로세로비를 설정할 수 있게 컷아웃을 일반화한 `transforms.RandomErasing`을 제공한다.

```python
import torchvision.transforms as T

# 기본 컷아웃 (0으로 채움)
cutout_transform = T.RandomErasing(
    p=0.5,           # 적용할 확률
    scale=(0.02, 0.33),  # 지울 이미지 넓이의 비율
    ratio=(0.3, 3.3),    # 가로세로비의 범위
    value=0,              # 채움값 (0 = 0으로 채움)
    inplace=False
)

# 무작위 지우기 (무작위 화소로 채움)
random_erasing_transform = T.RandomErasing(
    p=0.5,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value='random'   # 무작위 화소값으로 채우기
)

# 컷아웃을 쓰는 완전한 학습 파이프라인
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),  # ToTensor 뒤에 적용된다
])
```

### 여러 선택지를 갖는 설정 가능한 컷아웃

```python
class FlexibleCutout:
    """
    모양, 채움값, 적용 확률을 설정할 수 있는 컷아웃.
    
    인수:
        p: 컷아웃을 적용할 확률
        n_holes: 뚫을 구멍의 수
        min_length: 각 구멍의 최소 변 길이
        max_length: 각 구멍의 최대 변 길이
        fill_mode: 'zero', 'mean', 'random', 또는 실수 값
    """
    
    def __init__(self, p: float = 0.5, n_holes: int = 1,
                 min_length: int = 8, max_length: int = 24,
                 fill_mode: str = 'zero'):
        self.p = p
        self.n_holes = n_holes
        self.min_length = min_length
        self.max_length = max_length
        self.fill_mode = fill_mode
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return img
        
        C, H, W = img.shape
        result = img.clone()
        
        for _ in range(self.n_holes):
            length = np.random.randint(self.min_length, self.max_length + 1)
            
            cy = np.random.randint(H)
            cx = np.random.randint(W)
            
            y1 = max(0, cy - length // 2)
            y2 = min(H, cy + length // 2)
            x1 = max(0, cx - length // 2)
            x2 = min(W, cx + length // 2)
            
            if self.fill_mode == 'zero':
                result[:, y1:y2, x1:x2] = 0
            elif self.fill_mode == 'mean':
                for c in range(C):
                    result[c, y1:y2, x1:x2] = img[c].mean()
            elif self.fill_mode == 'random':
                result[:, y1:y2, x1:x2] = torch.rand(C, y2-y1, x2-x1)
            elif isinstance(self.fill_mode, (int, float)):
                result[:, y1:y2, x1:x2] = self.fill_mode
        
        return result
```

### 배치 수준의 컷아웃

효율을 위해 배치 수준에서 컷아웃을 적용한다.

```python
class BatchCutout:
    """
    배치 전체에 한 번에 컷아웃을 적용한다 (GPU에 알맞다).
    
    인수:
        n_holes: 이미지마다의 구멍 수
        length: 각 구멍의 변 길이
        p: 이미지마다의 적용 확률
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16, p: float = 0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p
    
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        인수:
            batch: (B, C, H, W) 텐서
            
        반환값:
            컷아웃이 적용된 배치
        """
        B, C, H, W = batch.shape
        
        # 어떤 이미지에 컷아웃을 적용할지 정하기
        apply_mask = torch.rand(B, device=batch.device) < self.p
        
        result = batch.clone()
        
        for _ in range(self.n_holes):
            # 배치 전체에 대한 무작위 중심
            cy = torch.randint(0, H, (B,), device=batch.device)
            cx = torch.randint(0, W, (B,), device=batch.device)
            
            y1 = torch.clamp(cy - self.length // 2, 0, H)
            y2 = torch.clamp(cy + self.length // 2, 0, H)
            x1 = torch.clamp(cx - self.length // 2, 0, W)
            x2 = torch.clamp(cx + self.length // 2, 0, W)
            
            # 이미지마다 마스크 만들기
            mask = torch.ones(B, 1, H, W, device=batch.device)
            for b in range(B):
                if apply_mask[b]:
                    mask[b, :, y1[b]:y2[b], x1[b]:x2[b]] = 0
            
            result = result * mask
        
        return result
```

## 학습 예제

```python
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train_with_cutout(
    model: nn.Module,
    cutout_length: int = 16,
    cutout_n_holes: int = 1,
    epochs: int = 200,
    lr: float = 0.1
) -> dict:
    """
    컷아웃 증강과 함께 CIFAR-10에서 CNN을 학습시킨다.
    
    인수:
        model: CNN 모델
        cutout_length: 컷아웃 조각의 변 길이
        cutout_n_holes: 이미지마다의 조각 수
        epochs: 학습 에포크 수
        lr: 처음 학습률
    """
    # 컷아웃을 쓰는 데이터 파이프라인
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
        Cutout(n_holes=cutout_n_holes, length=cutout_length),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    train_set = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=train_transform)
    val_set = datasets.CIFAR10('./data', train=False, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, 
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(correct / total)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={correct/total:.4f}")
    
    return history
```

## 컷아웃과 무작위 지우기

| 특징 | 컷아웃 (DeVries & Taylor) | 무작위 지우기 (Zhong 등) |
|---------|--------------------------|-------------------------------|
| 모양 | 크기가 고정된 정사각형 | 크기와 가로세로비가 변함 |
| 채움 | 0 (상수) | 무작위 화소값 |
| 크기 조절 | 변의 길이 $s$ | 넓이 비율 $[s_l, s_h]$ |
| 가로세로비 | 1:1 (정사각형) | 설정 가능한 범위 |
| PyTorch 내장 | 없음 (직접 만든 변환) | `transforms.RandomErasing` |

무작위 채움은 모델이 지워진 영역을 알아내는 법을 배우지 못하게 하므로(정규화 뒤의 0처럼 채움값이 상수이면 알아내기 쉽다) 요즘 파이프라인에서는 대체로 무작위 지우기를 선호한다.

## 초매개변수 선택

### 컷아웃의 크기

마스크의 크기가 가장 중요한 초매개변수이다. 원 논문의 지침은 다음과 같다.

| 데이터셋 | 이미지 크기 | 권장 컷아웃 길이 |
|---------|-----------|--------------------------|
| CIFAR-10 | 32×32 | 16 (이미지 너비의 50%) |
| CIFAR-100 | 32×32 | 8 (이미지 너비의 25%) |
| SVHN | 32×32 | 20 (이미지 너비의 62.5%) |
| ImageNet | 224×224 | `scale=(0.02, 0.33)`으로 `RandomErasing`을 쓴다 |

### 구멍의 개수

- **구멍 1개**: 표준 설정으로 대부분의 과제에 충분하다
- **구멍 2~3개**: 아주 크거나 복잡한 이미지에 추가 정칙화를 줄 수 있다
- **구멍이 너무 많으면**: 정보를 너무 많이 없애 학습을 해친다

### 적용 확률

- $p = 0.5$: `RandomErasing`의 표준값
- $p = 1.0$: 원래 컷아웃 논문에서 쓴 값(언제나 적용)
- 다른 강한 증강과 함께 쓸 때는 $p$을 낮춘다

## 다른 기법과 결합하기

컷아웃은 다른 대부분의 정칙화 방법과 서로 보완한다.

```python
# 컷아웃 + 표준 증강 + 가중치 감쇠 + 드롭아웃
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),
])

model = SomeCNN(dropout_rate=0.3)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

컷믹스는 가림 효과를 포함하면서 섞인 레이블과 정보가 담긴 채움 내용까지 주므로, 컷믹스를 쓸 때에는 보통 컷아웃이 필요 없다. 자세한 내용은 **[컷믹스](cutmix.md)**를 보라.

## 실무 지침

### 컷아웃을 쓸 때

1. **이미지 분류**: CNN을 위한 강력한 기본 증강이다
2. **작은 데이터셋**: 데이터가 적을 때 크게 나아진다
3. **국소 특징에 기대는 모델**: 모델이 특정 조각에 과적합한다고 의심될 때
4. **표준 파이프라인의 일부로**: 위험이 적고 넣기 쉽다

### 컷아웃을 피할 때

1. **이미 컷믹스를 쓰고 있을 때**: 컷믹스가 컷아웃의 이점을 모두 포함한다
2. **아주 작은 이미지**: 이미지가 이미 작으면 컷아웃이 정보를 너무 많이 없앨 수 있다
3. **온전한 공간 정보가 필요한 과제**: 모든 화소가 중요한 경우(예: 세심한 처리 없는 조밀 예측)

### 평가

검증이나 시험 중에는 컷아웃을 **결코 적용하지 않는다**. 언제나 손대지 않은 깨끗한 이미지로 평가하라.

## 참고 문헌

1. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. *arXiv:1708.04552*.
2. Zhong, Z., et al. (2020). Random Erasing Data Augmentation. *AAAI*.
3. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
4. Singh, K. K., & Lee, Y. J. (2017). Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization. *ICCV*.

## 연습문제

**연습문제 1.**
컷아웃 알고리즘과 그 초매개변수를 설명하라.

??? success "연습문제 1 풀이"
    컷아웃은 입력 이미지에서 크기가 $L \times L$인 정사각형 영역을 무작위로 골라 0(또는 평균 화소값)으로 채운다. 그 영역은 이미지 경계 바깥까지 뻗을 수 있다. 핵심 초매개변수는 조각의 크기 $L$이며, CIFAR-10에서는 보통 16(이미지 크기의 절반)이다.

---

**연습문제 2.**
컷아웃을 `torchvision.transforms`의 사용자 정의 변환으로 구현하라.

??? success "연습문제 2 풀이"
    ```python
    class Cutout:
        def __init__(self, length):
            self.length = length
        def __call__(self, img):
            h, w = img.size(1), img.size(2)
            y, x = torch.randint(h, (1,)), torch.randint(w, (1,))
            y1 = max(0, y - self.length//2)
            y2 = min(h, y + self.length//2)
            x1 = max(0, x - self.length//2)
            x2 = min(w, x + self.length//2)
            img[:, y1:y2, x1:x2] = 0
            return img
    ```

---

**연습문제 3.**
정칙화 기법으로서 컷아웃과 드롭아웃의 관계를 설명하라.

??? success "연습문제 3 풀이"
    둘 다 표현의 일부를 무작위로 가린다. 드롭아웃은 개별 뉴런을, 컷아웃은 공간 영역을 가린다. 컷아웃은 공간 상관 때문에 개별 화소에 대한 드롭아웃이 너무 잘게 쪼개지는 이미지를 위해 설계되었다. 컷아웃은 신경망이 부분적인 관측에서 배우도록 강제하여 가림에 대한 견고성을 높인다.

---

**연습문제 4.**
이미지 해상도가 다를 때 컷아웃의 조각 크기는 얼마로 해야 하는가?

??? success "연습문제 4 풀이"
    어림 규칙으로 조각의 크기는 이미지 한 변의 25~50%로 한다. CIFAR-10(32x32)은 $L=16$, ImageNet(224x224)은 $L=112$이다. 너무 작으면 정칙화 효과가 미미하고, 너무 크면 정보를 너무 많이 없애 학습 신호를 해친다.
