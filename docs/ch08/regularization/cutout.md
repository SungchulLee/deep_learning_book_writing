# 컷아웃

컷아웃은 트레인 이미지의 무작위 영역을 가리고 원래 라벨을 유지하는 어그멘테이션 기법이다. 일부 영역에 대한 의존을 줄이려는 방법이며, 가린 뒤에도 라벨이 유효할 만큼 크기와 확률을 정해야 한다. 무작위 지우기(Random Erasing)와는 영역·채움값의 설정이 다르다.

## 1. 수학적 정식화

### 핵심 연산

이미지 $x \in \mathbb{R}^{C \times H \times W}$가 주어지면 컷아웃은 무작위로 놓인 직사각형 영역 안에서는 0이고 그 밖에서는 1인 이진 마스크 $\mathbf{M} \in \{0, 1\}^{H \times W}$를 만든다.

$$
\tilde{x} = \mathbf{M} \odot x
$$

여기서 $\odot$은 채널 방향으로 브로드캐스팅되는 원소별 곱을 뜻한다. 라벨 $y$는 그대로 남는데, 이 점이 라벨까지 바꾸는 컷믹스와 컷아웃을 가른다.

### 마스크 만들기

1. 마스크의 중심을 균등분포에서 샘플링한다. $(c_x, c_y) \sim \text{Uniform}([0, W] \times [0, H])$
2. 마스크의 크기 $s$를 정한다(변의 길이를 고정하거나 어떤 범위에서 샘플링한다)
3. 이미지 경계로 잘라 낸 경계 상자를 계산한다.

$$
x_1 = \max(0,\; c_x - \lfloor s/2 \rfloor), \quad x_2 = \min(W,\; c_x + \lfloor s/2 \rfloor)
$$

$$
y_1 = \max(0,\; c_y - \lfloor s/2 \rfloor), \quad y_2 = \min(H,\; c_y + \lfloor s/2 \rfloor)
$$

4. $\mathbf{M}[y_1:y_2, x_1:x_2] = 0$으로 두고 나머지 성분은 1로 남긴다

중심을 이미지 경계 근처까지 포함해 이미지 안 어디서나 샘플링하면, 경계에 걸린 마스크는 이미지 밖으로 나간 만큼 잘린다. 그래서 실제로 가려지는 넓이가 표본마다 달라지고 가림의 강도에 자연스러운 변이가 생긴다.

### 채움값

원래 컷아웃 논문은 가려진 영역을 0으로 채운다. 입력을 채널 평균을 빼고 표준편차로 나누어 정규화한 뒤라면 0은 평균 픽셀값에 해당한다. 무작위 지우기는 이를 일반화하여 무작위 픽셀값, 채널별 평균, 또는 상수 채움을 허용한다.

$$
\tilde{x}_{c, y_1:y_2, x_1:x_2} = \begin{cases}
0 & \text{(zero fill — original Cutout)} \\
\mu_c & \text{(per-channel mean fill)} \\
\text{Uniform}(0, 1) & \text{(random fill — Random Erasing)}
\end{cases}
$$

### 관련 방법들과의 비교

| 방법 | 마스크의 대상 | 채움값 | 라벨이 바뀌는가? | 주된 이점 |
|--------|------------|------------|----------------|-------------|
| 컷아웃 | 직사각형을 0/상수로 | 0 또는 평균 | 아니다 | 공간적 강건성을 강제한다 |
| 무작위 지우기 | 직사각형을 무작위 값으로 | 무작위 픽셀 | 아니다 | 패치 암기를 막는다 |
| 컷믹스 | 다른 이미지의 직사각형 | 다른 표본의 내용 | 그렇다 (비례) | 낭비되는 픽셀이 없다 |
| 믹스업 | 전역 혼합 | 두 이미지의 가중합 | 그렇다 (비례) | 매끄러운 결정 경계 |

---

## 2. 컷아웃이 통하는 이유

### 국소 패치에 대한 의존 막기

뉴럴 네트워크는 변별력이 아주 큰 국소 패치 몇 개에 기대는 것만으로도 높은 트레인 정확도를 얻을 수 있다. 컷아웃은 그런 패치를 무작위로 지워 모델이 입력의 전체 공간 범위를 쓰는 중복된 표현을 갖추도록 강제한다.

### 정칙화 효과

컷아웃은 입력 잡음 주입의 한 형태로 볼 수 있다. 모델 $f$는 무작위 마스크 $\mathbf{M}$에 대한 기대 로스를 최소화하도록 트레인된다. $\ell$은 표본 하나의 로스(예: 교차 엔트로피)이다.

$$
\mathcal{L}_{\text{cutout}} = \mathbb{E}_{\mathbf{M}}\left[\ell(f(\mathbf{M} \odot x), y)\right]
$$

이는 모델이 가능한 모든 가림 형태 아래에서 로스를 최소화하도록 이끌어, 어느 한 공간 영역에 지나치게 기대는 것에 패널티를 준다.

### 드롭아웃과의 관계

컷아웃은 때때로 "입력에 대한 공간 드롭아웃"이라고 불린다. 표준 드롭아웃이 개별 원소를 무작위로 0으로 만드는 반면 컷아웃은 이어진 직사각형 영역을 0으로 만들어 이미지의 공간 구조에 더 잘 맞는다. 이는 공간 패치가 아니라 채널 하나의 피처 맵을 통째로 0으로 만드는 `Dropout2d`(공간 드롭아웃)와 관련은 있지만 다르다.

### 개선된 물체 위치 파악

물체의 부분적인 모습으로 트레인하면 모델은 부위만 보고도 물체를 알아보도록 배운다. 모델이 가장 변별적인 영역 하나에 기댈 수 없으므로, 약지도(weakly supervised, 상자 없이 이미지 라벨만 주는) 객체 위치 파악이 개선된다는 보고가 있다.

---

## 3. PyTorch 구현

이 쪽의 코드는 클래스와 함수의 정의만 싣는다. 모델을 트레인하지 않으므로 출력 블록이 없고, 컷아웃이 정확도를 얼마나 올리는지에 대한 수치도 이 쪽에서는 재지 않는다.

### 컷아웃 변환 직접 만들기

```python
import torch
import numpy as np

class Cutout:
    """
    이미지 텐서에서 정사각형 패치 하나 이상을 무작위로 가린다.
    
    참고: DeVries & Taylor, "Improved Regularization of CNNs with Cutout"
    
    인수:
        n_holes: 잘라 낼 패치의 수
        length: 각 정사각형 패치의 변 길이
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
            img: 형태가 (C, H, W)인 텐서 이미지
            
        반환값:
            컷아웃 영역이 적용된 이미지
        """
        h, w = img.shape[-2:]
        # 구멍을 직접 뚫지 않고 마스크를 만들어 두었다가 마지막에 한 번
        # 곱한다. 구멍이 여러 개일 때 겹쳐도 문제가 없고, 원본을
        # 제자리에서 고치지 않아도 되기 때문이다
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            # 중심 샘플링.
            # 중심을 이미지 안 아무 데나 잡으므로 구멍이 가장자리에
            # 걸리면 아래에서 잘려 length보다 작아진다. 원 논문도 이렇게
            # 두는데, 구멍이 늘 이미지 안에 온전히 들어가면 가장자리가
            # 가운데보다 덜 가려져 편향이 생기기 때문이다
            cy = np.random.randint(h)
            cx = np.random.randint(w)
            
            # 상자 계산 (잘라 냄)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            
            mask[..., y1:y2, x1:x2] = 0
        
        # 채움값이 0이면 곱셈 한 번으로 끝난다. 아래 일반형과 결과가 같지만
        # 덧셈을 건너뛰는 지름길이다.
        # 주의: 이 변환은 Normalize "뒤"에 놓이므로, 0은 검은색이 아니라
        # 정규화 공간의 원점, 곧 데이터셋의 채널 평균값이다. 평균으로
        # 채우는 것이 검정으로 채우는 것보다 나은 까닭은, 그 자리가
        # 통계적으로 아무 정보도 주지 않는 값이 되기 때문이다
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

# 무작위 지우기 (무작위 픽셀로 채움)
random_erasing_transform = T.RandomErasing(
    p=0.5,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value='random'   # 무작위 픽셀값으로 채우기
)

# 컷아웃을 쓰는 완전한 트레인 파이프라인
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
        # 확률 p로만 건다. 모든 이미지에 걸면 모델이 "늘 가려진 이미지"에
        # 맞춰져 온전한 이미지를 볼 때 오히려 손해가 된다
        if np.random.random() > self.p:
            return img

        C, H, W = img.shape
        result = img.clone()   # 원본을 건드리지 않는다

        for _ in range(self.n_holes):
            length = np.random.randint(self.min_length, self.max_length + 1)

            # 중심을 이미지 안 아무 데서나 고른다. 구멍이 가장자리를
            # 넘어가면 아래 clamp에서 잘려 실제 구멍이 작아지는데,
            # 이는 의도된 것이다. 물체가 가장자리에 있을 때도 견디게 한다
            cy = np.random.randint(H)
            cx = np.random.randint(W)

            # 이미지 밖으로 나가지 않도록 자른다
            y1 = max(0, cy - length // 2)
            y2 = min(H, cy + length // 2)
            x1 = max(0, cx - length // 2)
            x2 = min(W, cx + length // 2)

            if self.fill_mode == 'zero':
                # 정규화 뒤라면 0이 곧 채널 평균이므로 이것이 가장 무난하다.
                # 정규화 앞이라면 0은 검은색이라 분포를 밀어 버린다
                result[:, y1:y2, x1:x2] = 0
            elif self.fill_mode == 'mean':
                # 이 이미지 자신의 채널 평균으로 채운다. 정규화 전후를
                # 가리지 않고 안전하지만 채널마다 따로 계산해야 한다
                for c in range(C):
                    result[c, y1:y2, x1:x2] = img[c].mean()
            elif self.fill_mode == 'random':
                # 잡음으로 채운다. 가장 세게 흔드는 방식이다
                result[:, y1:y2, x1:x2] = torch.rand(C, y2-y1, x2-x1)
            elif isinstance(self.fill_mode, (int, float)):
                result[:, y1:y2, x1:x2] = self.fill_mode

        # 주의: 구멍끼리 겹칠 수 있다. n_holes를 크게 잡아도 실제로
        # 가려지는 넓이는 그만큼 늘지 않는다
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
        
        # 어떤 이미지에 컷아웃을 적용할지 정하기.
        # 위의 Cutout은 이미지 한 장에 언제나 적용하는 변환이지만, 이쪽은
        # 이미지마다 확률 p로 켜고 끈다. 그래서 한 배치 안에 가려진
        # 이미지와 깨끗한 이미지가 섞인다
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
            
            # 이미지마다 마스크 만들기.
            # 채널 축이 1이라 곱할 때 C로 브로드캐스팅된다. 세 채널을 똑같이
            # 가려야 하므로 채널마다 마스크를 따로 둘 이유가 없다.
            # 중심 샘플링은 텐서 연산으로 한꺼번에 했지만 잘라 내기는
            # 이미지마다 폭이 달라 결국 반복문이 남는다. "GPU에 알맞다"고
            # 했어도 이 부분은 배치 크기만큼 파이썬 반복을 돈다
            mask = torch.ones(B, 1, H, W, device=batch.device)
            for b in range(B):
                if apply_mask[b]:
                    mask[b, :, y1[b]:y2[b], x1[b]:x2[b]] = 0
            
            result = result * mask
        
        return result
```

---

## 4. 트레인 예제

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
    컷아웃 어그멘테이션과 함께 CIFAR-10에서 CNN을 트레인시킨다.
    
    인수:
        model: CNN 모델
        cutout_length: 컷아웃 패치의 변 길이
        cutout_n_holes: 이미지마다의 패치 수
        epochs: 트레인 에포크 수
        lr: 초기 러닝레이트
    """
    # 컷아웃을 쓰는 데이터 파이프라인
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
        # Cutout이 맨 뒤, 그것도 Normalize 뒤에 온다. 텐서를 받는
        # 변환이라 ToTensor 뒤여야 하고, 가린 자리의 0이 채널 평균이
        # 되도록 Normalize 뒤여야 한다.
        # RandomCrop이 앞에 있는 것도 중요하다. 자르기가 먼저 끝나야
        # 뚫어 놓은 구멍이 잘려 나가지 않는다
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
                              # num_workers 를 0으로 둔다. 1 이상이면 맥과 윈도가 워커 프로세스를
                              # **새로 띄우며**(spawn) 이 스크립트를 처음부터 다시 읽는다. 맨 바깥에서 데이터를
                              # 돌리고 있으면 워커가 또 워커를 띄워 끝내 죽는다. 1 이상을 쓰려면 돌리는
                              # 부분을 모두 `if __name__ == "__main__":` 안으로 넣어야 한다.
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256)
    
    criterion = nn.CrossEntropyLoss()
    # CIFAR-10의 표준 레시피다. Adam이 아니라 모멘텀을 넣은 SGD를
    # 쓰는데, 이 데이터셋에서는 잘 조율한 SGD가 최종 정확도에서
    # 여전히 앞선다. lr=0.1은 이 조합에서만 알맞은 큰 값이다
    optimizer = optim.SGD(model.parameters(), lr=lr, 
                          momentum=0.9, weight_decay=5e-4)
    # 러닝레이트를 코사인 곡선으로 0까지 떨어뜨린다. T_max를 전체 에포크로
    # 맞추어 두어야 마지막 에포크에서 정확히 0에 닿는다.
    # 주의: weight_decay와 컷아웃이 함께 걸려 있으므로, 컷아웃만의
    # 효과를 보려면 컷아웃만 끈 베이스라인과 비교해야 한다
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 트레인
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

---

## 5. 컷아웃과 무작위 지우기

| 항목 | 컷아웃 (DeVries & Taylor) | 무작위 지우기 (Zhong 등) |
|---------|--------------------------|-------------------------------|
| 모양 | 크기가 고정된 정사각형 | 크기와 가로세로비가 변함 |
| 채움 | 0 (상수) | 무작위 픽셀값 |
| 크기 조절 | 변의 길이 $s$ | 넓이 비율 $[s_l, s_h]$ |
| 가로세로비 | 1:1 (정사각형) | 설정 가능한 범위 |
| PyTorch 내장 | 없음 (직접 만든 변환) | `transforms.RandomErasing` |

무작위 채움도 원본 이미지와 통계가 다르면 모델이 가린 영역을 알아낼 수 있다. 채움 방식의 우열은 과제에 따라 달라진다. torchvision RandomErasing은 상수 채움도 지원하므로 무작위 픽셀만을 쓰는 연산으로 한정하지 않는다.

---

## 6. 하이퍼파라메터 선택

### 컷아웃의 크기

마스크의 크기가 가장 중요한 하이퍼파라메터다. 아래는 논문이나 구현에서 사용하는 탐색 출발점의 예이다. 서로 다른 기법의 설정까지 포함하므로 모든 행을 컷아웃 원 논문의 보편적 권장값으로 읽지 않는다.

| 데이터셋 | 이미지 크기 | 권장 컷아웃 길이 |
|---------|-----------|--------------------------|
| CIFAR-10 | 32×32 | 16 (이미지 너비의 50%) |
| CIFAR-100 | 32×32 | 8 (이미지 너비의 25%) |
| SVHN | 32×32 | 20 (이미지 너비의 62.5%) |
| ImageNet | 224×224 | `scale=(0.02, 0.33)`으로 `RandomErasing`을 쓴다 |

### 구멍의 개수

- **구멍 1개**: 표준 설정으로 대부분의 과제에 충분하다
- **구멍 2~3개**: 아주 크거나 복잡한 이미지에 추가 정칙화를 줄 수 있다
- **구멍이 너무 많으면**: 정보를 너무 많이 없애 트레인을 해친다

### 적용 확률

- $p = 0.5$: `RandomErasing`의 표준값
- $p = 1.0$: 원래 컷아웃 논문에서 쓴 값(언제나 적용)
- 다른 강한 어그멘테이션과 함께 쓸 때는 $p$를 낮춘다

---

## 7. 다른 기법과 결합하기

컷아웃은 다른 대부분의 정칙화 방법과 서로 보완한다.

```python
# 정칙화 세 가지를 함께 건다. 서로 다른 곳을 막으므로 함께 쓸 수 있다.
#   - 어그멘테이션(컷아웃 포함): 입력을 흔든다
#   - 드롭아웃: 뉴럴 네트워크 안의 엑티베이션 값을 흔든다
#   - 웨이트 감쇠: 웨이트의 크기를 누른다
train_transform = transforms.Compose([
    # 4픽셀 패딩하고 32x32로 다시 자른다. 물체의 위치를 조금씩 옮겨
    # 위치에 기대지 않도록 만든다
    transforms.RandomCrop(32, padding=4),

    # 좌우 뒤집기. CIFAR의 물체는 좌우가 바뀌어도 같은 클래스이므로 안전하다
    # (숫자나 글자였다면 써서는 안 된다)
    transforms.RandomHorizontalFlip(),

    # 밝기와 대비를 흔들어 조명 조건에 덜 민감하게 만든다
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    transforms.ToTensor(),   # PIL 이미지를 [0,1] 텐서로

    # CIFAR-10의 채널별 평균과 표준편차. 정규화는 반드시 ToTensor 뒤에 온다
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),

    # 이것이 컷아웃이다. 정규화 뒤에 두는 까닭은 value=0이 "정규화된
    # 스케일에서의 0", 곧 평균값을 뜻하게 하기 위해서다. 정규화 앞에 두면
    # 검은색을 칠하는 셈이 되어 데이터 분포를 다른 쪽으로 밀어 버린다
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),
])

# SomeCNN은 자리표시자다. 드롭아웃을 가진 임의의 CNN으로 바꿔 넣는다
model = SomeCNN(dropout_rate=0.3)

# AdamW를 쓰는 까닭. 보통의 Adam에서는 weight_decay가 그래디언트에 더해져
# 적응적 러닝레이트에 나눠지므로 실제 감쇠 세기가 파라메터마다 달라진다.
# AdamW는 감쇠를 갱신 단계에서 따로 빼므로 의도한 대로 걸린다
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

컷믹스는 가림 효과를 포함하면서 섞인 라벨과 정보가 담긴 채움 내용까지 주므로, 컷믹스를 쓸 때에는 보통 컷아웃이 필요 없다. 자세한 내용은 **[컷믹스](cutmix.md)**를 보라.

---

## 8. 실무 지침

### 컷아웃을 쓸 때

1. **이미지 분류**: CNN을 위한 강력한 기본 어그멘테이션이다
2. **작은 데이터셋**: 데이터가 적을 때 도움이 되는 경우가 많다
3. **국소 피처에 기대는 모델**: 모델이 특정 패치에 과적합한다고 의심될 때
4. **표준 파이프라인의 일부로**: 위험이 적고 넣기 쉽다

### 컷아웃을 피할 때

1. **이미 컷믹스를 쓰고 있을 때**: 가림 강도가 겹칠 수 있으므로 추가 효과를 비교한다
2. **아주 작은 이미지**: 이미지가 이미 작으면 컷아웃이 정보를 너무 많이 없앨 수 있다
3. **온전한 공간 정보가 필요한 과제**: 모든 픽셀이 중요한 경우(예: 세심한 처리 없는 조밀 예측, 곧 분할처럼 픽셀마다 라벨을 예측하는 과제)

### 평가

원본 분포의 검증과 시험에서는 트레인용 무작위 컷아웃을 끈다. 가림 강건성을 측정하는 별도 평가에서는 가림 조건과 원본 성능을 함께 보고한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
컷아웃 알고리즘과 그 하이퍼파라메터를 설명하라.

</div>

??? success "풀이"
    컷아웃은 입력 이미지에서 크기가 $L \times L$인 정사각형 영역을 무작위로 골라 0(또는 평균 픽셀값)으로 채운다. 그 영역은 이미지 경계 바깥까지 뻗을 수 있다. 핵심 하이퍼파라메터는 패치의 크기 $L$이며, CIFAR-10에서는 보통 16(이미지 크기의 절반)이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
컷아웃을 `torchvision.transforms`의 사용자 정의 변환으로 구현하라.

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정칙화 기법으로서 컷아웃과 드롭아웃의 관계를 설명하라.

</div>

??? success "풀이"
    둘 다 표현의 일부를 무작위로 가린다. 드롭아웃은 개별 뉴런을, 컷아웃은 공간 영역을 가린다. 이미지는 이웃 픽셀끼리 상관이 커서 개별 픽셀을 떨어뜨려도 주변 픽셀로 쉽게 메워진다. 그래서 컷아웃은 이어진 영역을 통째로 가린다. 컷아웃은 뉴럴 네트워크가 부분적인 관측에서 배우도록 강제하여 가림에 대한 강건성을 높인다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
다른 해상도에 컷아웃 크기를 옮길 때 무엇을 검증해야 하는가?

</div>

??? success "풀이"
    픽셀 길이와 이미지 대비 면적, 객체 크기와 위치를 함께 본다. 해상도 비율만으로 크기를 정하면 작은 객체를 완전히 지울 수 있다. 여러 크기·확률을 검증하고 가려진 뒤에도 원래 라벨이 합리적인지 확인한다. 224 해상도에서 길이 112가 항상 적절한 것은 아니다.

## 정리하며

컷아웃은 연속 공간 영역을 가리는 입력 어그멘테이션이다. 개별 원소 드롭아웃이나 채널 전체를 가리는 Dropout2d와 구분한다. 채움값과 마스크 크기를 입력 정규화·객체 크기에 맞춰 선택하고 라벨 보존 여부를 확인한다.

**참고 문헌**

1. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. *arXiv:1708.04552*.
2. Zhong, Z., et al. (2020). Random Erasing Data Augmentation. *AAAI*.
3. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
4. Singh, K. K., & Lee, Y. J. (2017). Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization. *ICCV*.
